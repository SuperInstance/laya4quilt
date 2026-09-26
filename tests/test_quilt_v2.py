"""Quilt ledger v2 — canonical chain alignment with the fleet substrate.

FAIL-first: written against the fleet spec (Quilt Charter: fnv1a-64 canonical
chain) BEFORE the v2 implementation. The v1 ledger chained with fnv1a-32 and
its docs claimed that was "the same chain algorithm the fleet's reference
quilt kernel uses" — false per spec. These tests pin the correction.

Cross-verification is done with the fleet algorithm reimplemented INLINE
(the fleet offsets/primes), not by importing another repo: this test file
must stand alone in the laya repo.
"""
import json

from laya.quilt import QuiltLedger, canonical

# Fleet substrate constants (quilt_doctor/substrate.py, Quilt Charter fnv1a-64)
FLEET_OFFSET = 0xcbf29ce484222325
FLEET_PRIME = 0x100000001b3
MASK64 = (1 << 64) - 1


def fleet_fnv1a(s: str) -> str:
    h = FLEET_OFFSET
    for b in s.encode("utf-8"):
        h = ((h ^ b) * FLEET_PRIME) & MASK64
    return "%016x" % h


def decider(answers):
    return lambda state, questions: {"answers": answers, "model": "test"}


def test_fnv1a64_mode_matches_fleet_chain_algorithm():
    led = QuiltLedger(actor="crab", engine="test", chain="fnv1a64")
    row = led.decide({"s": 1}, [{"q": "a"}], decider({"a": 1}))
    assert len(row["row_hash"]) == 16  # fleet-width hash
    body = {k: v for k, v in row.items() if k != "row_hash"}
    assert row["row_hash"] == fleet_fnv1a(canonical(body))


def test_fnv1a32_legacy_mode_still_available():
    led = QuiltLedger(actor="crab", engine="test")  # default stays legacy
    row = led.decide({"s": 1}, [{"q": "a"}], decider({"a": 1}))
    assert len(row["row_hash"]) == 8


def test_unknown_chain_refused():
    try:
        QuiltLedger(actor="crab", engine="test", chain="fnv1a128")
    except ValueError as exc:
        assert "chain" in str(exc)
    else:
        raise AssertionError("unknown chain width must raise ValueError")


def test_verify_replays_fnv1a64_chain():
    led = QuiltLedger(actor="crab", engine="test", chain="fnv1a64")
    led.decide({"s": 1}, [{"q": "a"}], decider({"a": 1}))
    led.decide({"s": 2}, [{"q": "b"}], decider({"b": 2}))
    ok, bad = led.verify()
    assert ok and bad is None


def test_append_jsonl_roundtrips_and_pins_tamper():
    import tempfile, os
    led = QuiltLedger(actor="crab", engine="test", chain="fnv1a64")
    led.decide({"s": 1}, [{"q": "a"}], decider({"a": 1}))
    led.decide({"s": 2}, [{"q": "b"}], decider({"b": 2}))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "quilt.jsonl")
        led.append_jsonl(path)
        lines = [json.loads(l) for l in open(path) if l.strip()]
        assert len(lines) == 2
        # independent replay with the fleet algorithm over canonical rows
        prev = "0" * 16
        for line in lines:
            body = {k: v for k, v in line.items() if k != "row_hash"}
            assert line["row_hash"] == fleet_fnv1a(canonical(body))
            assert line["chain_prev"] == prev
            prev = line["row_hash"]
        # tamper the second row: replay pins it
        lines[1]["payload"]["answers"]["b"] = 999
        body = {k: v for k, v in lines[1].items() if k != "row_hash"}
        assert lines[1]["row_hash"] != fleet_fnv1a(canonical(body))


def test_append_jsonl_is_append_only_across_instances():
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "quilt.jsonl")
        a = QuiltLedger(actor="crab", engine="test", chain="fnv1a64")
        a.decide({"s": 1}, [{"q": "a"}], decider({"a": 1}))
        a.append_jsonl(path)
        b = QuiltLedger(actor="crab", engine="test", chain="fnv1a64")
        b.append_jsonl(path)  # loads chain state from disk, continues it
        row = b.decide({"s": 2}, [{"q": "b"}], decider({"b": 2}))
        assert row["chain_prev"] == a.rows[-1]["row_hash"]
        b.append_jsonl(path)  # persist the continuation
        lines = [json.loads(l) for l in open(path) if l.strip()]
        assert len(lines) == 2


def test_forget_books_citation_row_and_deletes_nothing():
    led = QuiltLedger(actor="crab", engine="test", chain="fnv1a64")
    row = led.decide({"s": 1}, [{"q": "a"}], decider({"a": 1}))
    n_before = len(led.rows)
    led.forget(row["row_hash"], reason="superseded by policy v2")
    assert len(led.rows) == n_before + 1
    forget = led.rows[-1]
    assert forget["op"] == "FORGET"
    assert forget["payload"]["target"] == row["row_hash"]
    assert forget["payload"]["reason"] == "superseded by policy v2"
    ok, _ = led.verify()  # chain still verifies with the FORGET row
    assert ok


def test_forget_unknown_target_is_refused():
    led = QuiltLedger(actor="crab", engine="test", chain="fnv1a64")
    out = led.forget("f" * 16, reason="nothing there")
    assert out["op"] == "REFUSED"
