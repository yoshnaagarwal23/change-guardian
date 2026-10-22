"""Grounded knowledge: SOPs and past Remedy tickets.

Retrieval is restricted to verified internal sources ("golden data"), never the web.
TF-IDF is enough here: the vocabulary is small and operational, it is deterministic,
and it costs nothing per query. Swap in embeddings only if evaluation shows misses.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Domain synonyms so "OOM killer" finds "memory" remedies, etc. Applied to both sides.
_EXPANSIONS = {
    r"\boom\b|out[- ]of[- ]memory|garbage collection|\bgc\b|heap": " memory ",
    r"heartbeat|oneagent|activegate|telemetry": " agent connectivity monitoring ",
    r"thread pool|worker": " process worker capacity ",
    r"queue_depth|queue depth": " queue backlog ",
    r"authentication|signature|brute[- ]force|login": " credential token authentication security ",
    r"connection pool|sockettimeout": " connection pool database ",
    r"packet loss|network interface|health check": " host unreachable network ",
    r"latency|p95|response time|slowdown": " latency slowdown downstream ",
    r"\bcpu\b": " cpu ",
    r"disk|/var/log|volume": " disk logs ",
}

_STATUS_RANK = {"Resolved": 0, "Closed": 1, "In Progress": 2, "Assigned": 3}


def expand(text: str) -> str:
    low = text.lower()
    extra = [syn for pattern, syn in _EXPANSIONS.items() if re.search(pattern, low)]
    return low + " " + " ".join(extra)


@dataclass
class Remedy:
    name: str
    action: str

    @property
    def text(self) -> str:
        return f"{self.name}: {self.action}"


@dataclass
class Sop:
    sop_id: str
    title: str
    event_type: str
    severity: str
    diagnostic_steps: list[str] = field(default_factory=list)
    remedies: list[Remedy] = field(default_factory=list)

    @property
    def all_text(self) -> list[str]:
        return self.diagnostic_steps + [r.text for r in self.remedies]


def parse_sop(text: str) -> Sop:
    """Parse one SOP Markdown file (the format in data/sops/)."""
    header = re.search(r"^# (SOP-\d+):\s*(.+)$", text, re.M)
    if not header:
        raise ValueError("SOP is missing a '# SOP-XXX: Title' header")

    def meta(key: str) -> str:
        m = re.search(rf"\*\*{key}:\*\*\s*(.+)", text)
        return m.group(1).strip() if m else ""

    diag_block = text.split("## Diagnostic Steps", 1)[-1].split("## Remediation Steps", 1)[0]
    diagnostic = [re.sub(r"^\d+\.\s*", "", line).strip()
                  for line in diag_block.splitlines() if re.match(r"^\d+\.", line.strip())]
    remedy_block = text.split("## Remediation Steps", 1)[-1]
    remedies = [Remedy(m.group(1).strip(), m.group(2).strip())
                for m in re.finditer(r"^- \*\*(.+?):\*\*\s*(.+)$", remedy_block, re.M)]
    return Sop(header.group(1), header.group(2).strip(), meta("Event Type"),
               meta("Severity Level"), diagnostic, remedies)


def load_sops(sop_dir: Path) -> dict[str, Sop]:
    sops = [parse_sop(p.read_text()) for p in sorted(Path(sop_dir).glob("SOP-*.md"))]
    return {s.sop_id: s for s in sops}


def rank_texts(texts: list[str], query: str, k: int | None = None) -> list[tuple[str, float]]:
    """Rank short texts (SOP steps, remedies) by relevance to a diagnosis. Stable on ties."""
    docs = [expand(t) for t in texts]
    vec = TfidfVectorizer().fit(docs + [expand(query)])
    sims = cosine_similarity(vec.transform([expand(query)]), vec.transform(docs))[0]
    ranked = sorted(range(len(texts)), key=lambda i: -sims[i])
    return [(texts[i], float(sims[i])) for i in ranked[:k]]


def rank_remedies(sop: Sop, query: str, k: int = 2) -> list[tuple[Remedy, float]]:
    """Which remediation branch of the SOP fits this diagnosis best."""
    by_text = {r.text: r for r in sop.remedies}
    return [(by_text[t], s) for t, s in rank_texts([r.text for r in sop.remedies], query, k)]


class TicketIndex:
    """Search past Remedy tickets as precedents, one result per distinct root cause."""

    def __init__(self, tickets: pd.DataFrame):
        self.tickets = tickets.reset_index(drop=True)
        docs = (self.tickets["summary"] + " " + self.tickets["root_cause"]).map(expand)
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(docs)
        self.matrix = self.vec.transform(docs)

    def search(self, query: str, event_type: str | None = None, k: int = 3,
               exclude: set[str] | None = None) -> list[dict]:
        df = self.tickets.assign(score=cosine_similarity(
            self.vec.transform([expand(query)]), self.matrix)[0])
        if event_type is not None and (df["event_type"] == event_type).any():
            df = df[df["event_type"] == event_type]
        if exclude:
            df = df[~df["incident_id"].isin(exclude)]
        df = df.assign(status_rank=df["status"].map(_STATUS_RANK).fillna(9))

        hits = []
        for root_cause, group in df.groupby("root_cause", sort=False):
            best = group.sort_values(["status_rank", "timestamp"], ascending=[True, False]).iloc[0]
            hits.append({
                "incident_id": best["incident_id"],
                "event_type": best["event_type"],
                "summary": best["summary"],
                "root_cause": root_cause,
                "remediation": best["remediation"],
                "status": best["status"],
                "precedent_count": int(len(group)),
                "score": round(float(group["score"].max()), 3),
            })
        return sorted(hits, key=lambda h: -h["score"])[:k]
