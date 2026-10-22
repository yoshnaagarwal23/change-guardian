"""SOP parsing, SOP routing, and grounded ticket retrieval."""
from knowledge import parse_sop


def test_parse_sop_reads_header_steps_and_remedies(sample_sop_text):
    # Act
    sop = parse_sop(sample_sop_text)
    # Assert
    assert sop.sop_id == "SOP-004"
    assert sop.event_type == "Resource"
    assert len(sop.diagnostic_steps) == 4
    assert [r.name for r in sop.remedies] == [
        "Disk Cleanup / Volume Expansion", "Resource Reallocation", "Process Isolation"]


def test_every_sop_file_parses(tools):
    assert set(tools.sops) == {f"SOP-00{i}" for i in range(1, 7)}
    for sop in tools.sops.values():
        assert sop.diagnostic_steps and sop.remedies


def test_find_sop_follows_the_agreed_event_mapping(tools):
    assert tools.find_sop("Availability").sop_id == "SOP-001"
    assert tools.find_sop("Resource").sop_id == "SOP-004"
    assert tools.find_sop("Custom Alert").sop_id == "SOP-006"
    # No client SOP covers agent/heartbeat connectivity yet -> documentation gap.
    assert tools.find_sop("Monitoring Unavailable") is None


def test_rank_remedies_picks_the_branch_that_matches_the_diagnosis(tools):
    sop = tools.find_sop("Resource")
    disk = tools.rank_remedies(sop, "Disk space capacity remaining less than 5% on /var/log.")
    cpu = tools.rank_remedies(sop, "CPU usage saturation sustained at 98.4% for 15 mins.")
    assert disk[0][0].name == "Disk Cleanup / Volume Expansion"
    assert cpu[0][0].name == "Resource Reallocation"


def test_rank_steps_puts_the_relevant_diagnostic_step_first(tools):
    from knowledge import rank_texts
    sop = tools.find_sop("Resource")
    best = rank_texts(sop.diagnostic_steps, "Disk space capacity remaining less than 5% on /var/log.")[0][0]
    assert "Disk IOPS" in best


import pytest  # noqa: E402


@pytest.mark.parametrize("diagnosis,ticket_cause,expected", [
    ("Process crashed due to out-of-memory exception.",
     "Linux kernel Out-Of-Memory (OOM) killer terminated the primary gateway process.", True),
    ("Database connection pool exhausted. SocketTimeoutException.",
     "Database connection pool exhausted due to unindexed query deployed in recent patch.", True),
    ("Health check endpoint timeout after 5000ms.",
     "Linux kernel Out-Of-Memory (OOM) killer terminated the primary gateway process.", False),
    ("Disk space capacity remaining less than 5% on /var/log.",
     "Extensive garbage collection (GC) cycles taking up 95% of active CPU cycles.", False),
])
def test_precedent_matches_only_when_it_describes_the_same_failure(diagnosis, ticket_cause, expected):
    from knowledge import precedent_matches
    assert precedent_matches(diagnosis, ticket_cause) is expected


def test_similar_tickets_are_deduplicated_precedents_of_the_same_event_type(tools):
    hits = tools.find_similar_tickets("Process crashed due to out-of-memory exception.", "Availability")
    assert hits, "expected at least one precedent"
    assert all(h["event_type"] == "Availability" for h in hits)
    # Remedy tickets are templated: one precedent per distinct root cause, with a count.
    assert len({h["root_cause"] for h in hits}) == len(hits)
    assert hits[0]["precedent_count"] > 1
    assert "OOM" in hits[0]["root_cause"] or "Out-Of-Memory" in hits[0]["root_cause"]


def test_similar_ticket_search_can_exclude_an_incident(tools):
    first = tools.find_similar_tickets("brute-force login", "Custom Alert", k=1)[0]
    again = tools.find_similar_tickets("brute-force login", "Custom Alert", k=1,
                                       exclude={first["incident_id"]})[0]
    assert again["incident_id"] != first["incident_id"]
