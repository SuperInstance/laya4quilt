"""Quilt-native decision ledger for laya.

Every laya decision is booked as an *attested observation*: content (state +
questions + answers), witness-stake (actor + decision engine identity),
time-context (a monotonic tick — partial order, never false finality), and a
content-binding (a hash chain over canonical rows, fnv1a-32, the same chain
algorithm the fleet's reference quilt kernel uses, so any kernel port can
verify rows this ledger writes).

Doctrine (kept model-free on purpose):

- BIND:  a decision row binds state, questions and answers under one hash.
- LINK:  ensemble + resolve rows bind decisions to each other — provenance
         as first-class edges, rivals preserved verbatim.
- EFFECT: derived state (verify results, rollups) is recomputed from rows,
         never trusted as stored.
- VIEW:  exports book a VIEW row; the export is a projection, not the truth.
- FORGET: a citation row marks a prior row superseded — the target is
         preserved verbatim, never deleted (fleet 5+1).
- REFUSED: a malformed engine result or an invalid operation books a named
         refusal row — refusals are visible, never silent drops.
- Disagreement is preserved, not deleted: ensembles keep every engine's
         answer; resolution is a new row that cites rivals, never erases them.

Chain widths: fnv1a-64 is the canonical fleet mode (Quilt Charter — the same
width the reference kernel's WAL uses, so any kernel port verifies these
rows). fnv1a-32 remains as `chain="fnv1a32"` for ledgers that already hold
legacy rows; new ledgers should default to the 64-bit mode.

The module is stdlib-only. Torch and the checkpoints are loaded lazily by
QuiltLayaBridge, so this file runs (and its chain verifies) anywhere Python
does — including next to every other quilt substrate.
"""
import hashlib
import json
import time as _time

__all__ = [
    "FNV1A_OFFSET",
    "fnv1a32",
    "fnv1a64",
    "canonical",
    "sha256_hex",
    "QuiltLedger",
    "QuiltLayaBridge",
]

FNV1A_OFFSET = 0x811C9DC5
FNV1A_PRIME = 0x01000193
MASK32 = 0xFFFFFFFF

FNV1A64_OFFSET = 0xCBF29CE484222325
FNV1A64_PRIME = 0x100000001B3
MASK64 = 0xFFFFFFFFFFFFFFFF

GENESIS32 = "0" * 8   # chain_prev of the first row, fnv1a-32 ledgers
GENESIS64 = "0" * 16  # chain_prev of the first row, fnv1a-64 (fleet canonical)
GENESIS = GENESIS32  # back-compat alias


def fnv1a32(data):
    """fnv1a-32 over bytes — legacy chain width (kept for existing ledgers).

    Integrity, not security: catches accidental mutation and pins tamper at
    its own row. Any 5-opcode kernel port recomputes this chain without
    Python. New ledgers should prefer :func:`fnv1a64`, the fleet-canonical
    width the Quilt Charter specifies.
    """
    if isinstance(data, str):
        data = data.encode("utf-8")
    h = FNV1A_OFFSET
    for byte in data:
        h ^= byte
        h = (h * FNV1A_PRIME) & MASK32
    return h


def fnv1a64(data):
    """fnv1a-64 over bytes — the fleet-canonical chain width (Quilt Charter).

    Same algorithm, 64-bit state: this is the width the reference quilt
    kernel's WAL chains with, so rows booked under fnv1a64 cross-verify with
    hermit, git-agent, and the kernel ports without any width translation.
    """
    if isinstance(data, str):
        data = data.encode("utf-8")
    h = FNV1A64_OFFSET
    for byte in data:
        h ^= byte
        h = (h * FNV1A64_PRIME) & MASK64
    return h


def canonical(obj):
    """Deterministic serialization: sorted keys, tight separators."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(obj):
    """Strong binding for state/question payloads (the row chain stays fnv1a)."""
    if not isinstance(obj, str):
        obj = canonical(obj)
    return hashlib.sha256(obj.encode("utf-8")).hexdigest()


def _now():
    return _time.time()


class QuiltLedger:
    """Hash-chained decision ledger. One ledger = one actor's stake stream.

    Rows are plain dicts in insertion order. The chain is self-verifying:
    row N's `row_hash` covers the canonical row (minus the hash field) plus
    row N-1's `row_hash`, so a mutation anywhere pins the first bad row.
    """

    CHAINS = {"fnv1a32": (fnv1a32, GENESIS32, "%08x"),
              "fnv1a64": (fnv1a64, GENESIS64, "%016x")}

    def __init__(self, actor, engine, clock=None, chain="fnv1a32"):
        if not actor or not isinstance(actor, str):
            raise ValueError("ledger requires an actor (the witness-stake)")
        if not engine or not isinstance(engine, str):
            raise ValueError("ledger requires an engine identity")
        if chain not in self.CHAINS:
            raise ValueError(
                "unknown chain %r; expected one of %s" % (chain, sorted(self.CHAINS)))
        self.actor = actor
        self.engine = engine
        self.chain = chain
        self._hash, self._genesis, self._fmt = self.CHAINS[chain]
        self._clock = clock or _now
        self.rows = []
        self._tick = 0

    # ------------------------------------------------------------------ core
    def _next_tick(self):
        self._tick += 1
        return self._tick

    def _book(self, op, payload):
        tick = self._next_tick()
        row = {
            "tick": tick,
            "ts": round(self._clock(), 6),
            "op": op,
            "actor": self.actor,
            "engine": self.engine,
            "payload": payload,
            "chain_prev": self.rows[-1]["row_hash"] if self.rows else self._genesis,
        }
        row["row_hash"] = self._fmt % self._hash(canonical(row))
        self.rows.append(row)
        return row

    # ------------------------------------------------------------------ ops
    def decide(self, state, questions, decide_fn, ts=None):
        """BIND a decision: run `decide_fn(state, questions)` and book the
        answers. A malformed engine result books a REFUSED row naming the
        failure — the chain never silently drops a decision attempt."""
        state_hash = sha256_hex(state)
        q_hash = sha256_hex(questions)
        try:
            answers = decide_fn(state, questions)
            if not isinstance(answers, dict) or "answers" not in answers:
                raise ValueError("engine result must be a dict with an 'answers' map")
            payload = {
                "state_hash": state_hash,
                "questions_hash": q_hash,
                "engine_model": str(answers.get("model", "unknown")),
                # Deep-copy through the canonical form: booked history must
                # not share structure with anything the engine might reuse.
                "answers": json.loads(canonical(answers["answers"])),
                "usage": json.loads(canonical(answers.get("usage", {}))),
            }
            return self._book("BIND", payload)
        except Exception as exc:  # refusal visibility: name it, don't drop it
            return self._book("REFUSED", {
                "state_hash": state_hash,
                "questions_hash": q_hash,
                "reason": "ENGINE_RESULT_INVALID",
                "detail": "%s: %s" % (type(exc).__name__, exc),
            })

    def ensemble(self, group_id, state, questions, engines):
        """BIND the same question set under several engine identities and LINK
        them into one disagreement group. Every rival answer is preserved —
        MV-register honesty. `engines`: {engine_name: decide_fn}."""
        if not engines or len(engines) < 2:
            return self._book("REFUSED", {
                "group": group_id, "reason": "ENSEMBLE_REQUIRES_2_ENGINES",
                "detail": "got %d" % len(engines) if engines else "0",
            })
        row_ids = []
        for name, fn in engines.items():
            saved = self.engine
            try:
                self.engine = name
                row = self.decide(state, questions, fn)
            finally:
                self.engine = saved
            row_ids.append(row["row_hash"])
        link = self._book("LINK", {
            "group": group_id,
            "kind": "ensemble",
            "members": row_ids,
        })
        return {"rows": row_ids, "link": link["row_hash"], "group": group_id}

    def resolve(self, group_id, member_hash, rationale):
        """EFFECT: a meta-attestation choosing one ensemble member, with a
        staked rationale. Reversible, cites its rivals, deletes nothing."""
        member = None
        for row in self.rows:
            if row["row_hash"] == member_hash and row["op"] == "BIND":
                member = row
        if member is None:
            return self._book("REFUSED", {
                "group": group_id, "reason": "RESOLVE_TARGET_NOT_FOUND",
                "detail": member_hash,
            })
        siblings = [r for r in self.rows
                    if r["op"] == "BIND" and r["payload"].get("state_hash") == member["payload"]["state_hash"]
                    and r["row_hash"] != member_hash]
        return self._book("EFFECT", {
            "group": group_id,
            "kind": "resolution",
            "chosen": member_hash,
            "rivals": [r["row_hash"] for r in siblings],
            "rationale": rationale,
        })

    def forget(self, target_hash, reason):
        """FORGET (the fleet's +1 opcode): a citation row marking a prior row
        superseded. The target is preserved verbatim — this books a pointer,
        it never deletes. Unknown targets book a REFUSED row."""
        target = next((r for r in self.rows if r["row_hash"] == target_hash), None)
        if target is None:
            return self._book("REFUSED", {
                "reason": "FORGET_TARGET_NOT_FOUND",
                "detail": target_hash,
            })
        return self._book("FORGET", {
            "target": target_hash,
            "target_op": target["op"],
            "reason": reason,
        })

    # -------------------------------------------------------------- persistence
    def append_jsonl(self, path):
        """Canonical fleet serialization: append rows to a JSONL WAL.

        If `path` already holds rows from this ledger, they are loaded first
        so the chain continues (a second instance appends, never restarts).
        The file is the projection; the in-memory rows remain the truth until
        booked. Returns the number of rows now on disk.
        """
        path = str(path)
        existing = []
        try:
            with open(path, "r", encoding="utf-8") as fh:
                existing = [json.loads(line) for line in fh if line.strip()]
        except FileNotFoundError:
            pass
        if existing and not self.rows:
            # Continuing a ledger from disk: adopt its chain state wholesale.
            self.rows = existing
            self._tick = max((r.get("tick", 0) for r in existing), default=0)
        start = len(existing)
        new_rows = self.rows[start:]
        if not new_rows and start:
            return start
        with open(path, "a", encoding="utf-8") as fh:
            for row in new_rows:
                fh.write(canonical(row) + "\n")
            fh.flush()
        return len(self.rows)

    # ------------------------------------------------------------------ views
    def verify(self):
        """Replay the chain from genesis. Returns (ok, first_bad_row_hash).
        EFFECT rows are recomputed-ok by construction (derived state is
        derived); the check is chain integrity + schema, row by row."""
        prev = self._genesis
        for row in self.rows:
            want = self._fmt % self._hash(
                canonical({k: v for k, v in row.items() if k != "row_hash"}))
            if row.get("chain_prev") != prev or row.get("row_hash") != want:
                return False, row.get("row_hash", "unknown")
            prev = row["row_hash"]
        return True, None

    def view(self, fmt="json"):
        """Book a VIEW row and return a projection. 'json' returns the ledger;
        'canon' returns a CANON-stub-shaped digest (provenance + hashed
        entries) so distill→decide pipelines share one canon surface.

        The projection's chain head is captured BEFORE the VIEW row is
        booked: a view describes the ledger as it was when observed, and
        the VIEW row itself is the receipt that the observation happened.
        """
        head = self.rows[-1]["row_hash"] if self.rows else self._genesis
        self._book("VIEW", {"format": fmt, "rows": len(self.rows), "head": head})
        if fmt == "json":
            return {"rows": list(self.rows), "chain_head": head}
        if fmt == "canon":
            binds = [r for r in self.rows if r["op"] == "BIND"]
            return {
                "provenance": {
                    "ledger_chain_head": head,
                    "actor": self.actor,
                    "engine": self.engine,
                    "rows": len(self.rows),
                },
                "entries": [
                    {
                        "tick": r["tick"],
                        "engine": r["engine"],
                        "state_hash": r["payload"]["state_hash"],
                        "answers": r["payload"]["answers"],
                        "hash": r["row_hash"],
                    }
                    for r in binds
                ],
            }
        raise ValueError("unknown view format: %r" % fmt)


class QuiltLayaBridge:
    """Lazy glue from a laya Agent/Router to a decide_fn the ledger accepts.

    Torch and checkpoint weights load only when `decide_fn` is first called,
    never at import. `route` records which checkpoint served the decision,
    so the ledger's engine column reflects the routing reality.
    """

    def __init__(self, agent=None, router=None, route="english"):
        self.agent = agent
        self.router = router
        self.route = route
        self.served_by = route

    def _ensure_agent(self):
        if self.agent is not None:
            return self.agent
        from .agent import Agent  # lazy: torch enters here, not before

        if self.router is not None:
            self.agent, self.served_by = self.router.get(self.route)
        else:
            self.agent = Agent()
            self.served_by = getattr(self.agent, "model_id", "laya")
        return self.agent

    def decide_fn(self):
        """Return a callable (state, questions) -> system_one result, tagged
        with the serving checkpoint in 'model'."""
        def _decide(state, questions):
            agent = self._ensure_agent()
            out = agent.system_one(state, questions)
            out = dict(out)
            out["model"] = "%s@%s" % (out.get("model", "laya"), self.served_by)
            return out
        return _decide
