"""LoRA fine-tuning of a small open model to write Change Guardian answers.

Same script on a laptop and on the university cluster; only the config changes.

  uv run python src/train_lora.py --config configs/mac.yaml
  uv run python src/train_lora.py --config configs/mac.yaml --max-steps 5   # smoke test
  uv run python src/train_lora.py --config configs/cluster.yaml --eval-base

Writes the adapter to `output_dir`, metrics to `output_dir/metrics.json`, and logs
params/metrics to MLflow (./mlruns; view with `uv run mlflow ui`).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import yaml

from metrics import score_generation, summarize
from sft_data import OUT as SFT_DIR
from sft_data import build_dataset, label_mask, read_jsonl, write_jsonl

REQUIRED = ["model_id", "lora", "learning_rate", "epochs", "output_dir"]


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if "base" in cfg:  # simple inheritance: cluster/sweep configs extend another file
        cfg = {**load_config(Path(path).parent / cfg.pop("base")), **cfg}
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise KeyError(f"{path} is missing {missing}")
    return cfg


def pick_device(torch) -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_splits(cfg: dict) -> tuple[list[dict], list[dict]]:
    train_p, test_p = SFT_DIR / "train.jsonl", SFT_DIR / "test.jsonl"
    if not (train_p.exists() and test_p.exists()):
        from tools import TriageTools
        train, test = build_dataset(TriageTools.from_disk(), cfg.get("holdout_frac", 0.2), cfg.get("seed", 42))
        write_jsonl(train, train_p)
        write_jsonl(test, test_p)
    return read_jsonl(train_p), read_jsonl(test_p)


def encode(tokenizer, example: dict, max_len: int) -> dict:
    prompt = tokenizer.apply_chat_template(example["messages"][:-1], tokenize=False, add_generation_prompt=True)
    answer = example["messages"][-1]["content"] + tokenizer.eos_token
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = prompt_ids + tokenizer(answer, add_special_tokens=False)["input_ids"]
    labels = label_mask(prompt_ids, full_ids)
    # Truncate from the left of the prompt so the answer is always kept.
    return {"input_ids": full_ids[-max_len:], "labels": labels[-max_len:]}


def batches(rows: list[dict], size: int, pad_id: int, torch, seed: int):
    rows = rows[:]
    random.Random(seed).shuffle(rows)
    for i in range(0, len(rows), size):
        chunk = rows[i:i + size]
        width = max(len(r["input_ids"]) for r in chunk)
        ids = [r["input_ids"] + [pad_id] * (width - len(r["input_ids"])) for r in chunk]
        labels = [r["labels"] + [-100] * (width - len(r["labels"])) for r in chunk]
        mask = [[1] * len(r["input_ids"]) + [0] * (width - len(r["input_ids"])) for r in chunk]
        yield (torch.tensor(ids), torch.tensor(labels), torch.tensor(mask))


def eval_set(test_rows: list[dict], cfg: dict) -> list[dict]:
    """All held-out alert types (the generalization test) + a sample of held-out tickets."""
    alerts = [r for r in test_rows if r.get("kind") == "alert"]
    tickets = [r for r in test_rows if r.get("kind", "ticket") == "ticket"]
    return alerts[: cfg.get("eval_alerts", len(alerts))] + tickets[: cfg.get("eval_samples", len(tickets))]


def evaluate_model(model, tokenizer, examples: list[dict], device: str, torch, max_new_tokens: int) -> dict:
    """Flat metrics per example kind, e.g. alert_consistency, ticket_rouge_l."""
    model.eval()
    by_kind: dict[str, list[dict]] = {}
    for ex in examples:
        prompt = tokenizer.apply_chat_template(ex["messages"][:-1], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        by_kind.setdefault(ex.get("kind", "ticket"), []).append(
            score_generation(text, ex["gold"], ex["sources"], ex.get("diagnosis"), ex.get("key_step")))
    return {f"{kind}_{k}": v for kind, scores in by_kind.items()
            for k, v in summarize(scores).items() if v is not None}


def evaluate_adapter(cfg: dict, adapter: str | None) -> dict:
    """Score an existing adapter (or the base model when adapter is None) without training."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = cfg.get("device") or pick_device(torch)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"])
    model = AutoModelForCausalLM.from_pretrained(cfg["model_id"]).to(device)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter).to(device)
    _, test_rows = load_splits(cfg)
    return evaluate_model(model, tokenizer, eval_set(test_rows, cfg), device, torch,
                          cfg.get("max_new_tokens", 256))


def train(cfg: dict, max_steps: int | None = None, eval_base: bool = False) -> dict:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

    torch.manual_seed(cfg.get("seed", 42))
    device = cfg.get("device") or pick_device(torch)
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(cfg.get("precision", "fp32"), torch.float32)
    model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], dtype=dtype).to(device)

    train_rows, test_rows = load_splits(cfg)
    eval_rows = eval_set(test_rows, cfg)
    results: dict = {"config": cfg, "device": device}

    if eval_base:
        results["base"] = evaluate_model(model, tokenizer, eval_rows, device, torch, cfg.get("max_new_tokens", 256))
        print("base model:", results["base"])

    lora = cfg["lora"]
    model = get_peft_model(model, LoraConfig(
        r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"], task_type="CAUSAL_LM"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%) on {device}")

    encoded = [encode(tokenizer, ex, cfg.get("max_len", 1024)) for ex in train_rows]
    bs, accum = cfg.get("batch_size", 2), cfg.get("grad_accum", 4)
    steps_per_epoch = math.ceil(len(encoded) / bs / accum)
    total_steps = min(max_steps or 10**9, steps_per_epoch * cfg["epochs"])
    optim = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                              lr=float(cfg["learning_rate"]), weight_decay=cfg.get("weight_decay", 0.0))
    sched = get_linear_schedule_with_warmup(optim, int(cfg.get("warmup_ratio", 0.05) * total_steps), total_steps)

    tracker = _mlflow(cfg)
    step, losses, start = 0, [], time.time()
    model.train()
    for epoch in range(cfg["epochs"]):
        for i, (ids, labels, mask) in enumerate(batches(encoded, bs, tokenizer.pad_token_id, torch,
                                                         cfg.get("seed", 42) + epoch)):
            loss = model(input_ids=ids.to(device), attention_mask=mask.to(device), labels=labels.to(device)).loss
            (loss / accum).backward()
            losses.append(loss.item())
            if (i + 1) % accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            optim.zero_grad()
            step += 1
            avg = sum(losses[-accum:]) / accum
            if tracker:
                tracker.log_metric("train_loss", avg, step=step)
            if step % cfg.get("log_every", 5) == 0 or step == 1:
                print(f"epoch {epoch} step {step}/{total_steps} loss {avg:.4f}")
            if step >= total_steps:
                break
        if step >= total_steps:
            break

    results["train"] = {"steps": step, "final_loss": sum(losses[-accum:]) / max(1, min(accum, len(losses))),
                        "minutes": round((time.time() - start) / 60, 2)}
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    results["adapter"] = evaluate_model(model, tokenizer, eval_rows, device, torch, cfg.get("max_new_tokens", 256))
    print("fine-tuned:", results["adapter"])

    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2, default=str))
    if tracker:
        for k, v in results["adapter"].items():
            tracker.log_metric(f"eval_{k}", v)
        tracker.end_run()
    return results


def _mlflow(cfg: dict):
    if not cfg.get("mlflow", True):
        return None
    try:
        import mlflow
    except ImportError:
        return None
    mlflow.set_experiment(cfg.get("experiment", "change-guardian-lora"))
    mlflow.start_run(run_name=Path(cfg["output_dir"]).name)
    flat = {k: v for k, v in cfg.items() if not isinstance(v, (dict, list))}
    flat.update({f"lora_{k}": v for k, v in cfg["lora"].items() if not isinstance(v, list)})
    mlflow.log_params(flat)
    return mlflow


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/mac.yaml"))
    parser.add_argument("--max-steps", type=int, help="stop early (smoke test)")
    parser.add_argument("--eval-base", action="store_true", help="also score the untuned model")
    parser.add_argument("--output-dir", help="override output_dir")
    parser.add_argument("--eval-only", metavar="ADAPTER", nargs="?", const="",
                        help="score an existing adapter (or the base model if no path) and exit")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.eval_only is not None:
        print(json.dumps(evaluate_adapter(cfg, args.eval_only or None), indent=2))
        return
    train(cfg, args.max_steps, args.eval_base)


if __name__ == "__main__":
    main()
