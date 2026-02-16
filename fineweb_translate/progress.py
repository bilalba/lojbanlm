"""JSONL-based progress tracking with resumability."""

import json
from datetime import datetime, timezone
from pathlib import Path

from .config import FAILURES_PATH, MAX_RETRIES, PROGRESS_PATH, TRANSLATIONS_PATH


class ProgressTracker:
    """Append-only progress tracker backed by JSONL files.

    Three files:
    - progress.jsonl: every attempt (audit log, used for resume)
    - translations.jsonl: accepted translations only (primary output)
    - failures.jsonl: chunks that exhausted all retries
    """

    def __init__(
        self,
        progress_path: Path = PROGRESS_PATH,
        translations_path: Path = TRANSLATIONS_PATH,
        failures_path: Path = FAILURES_PATH,
    ):
        self._progress_path = progress_path
        self._translations_path = translations_path
        self._failures_path = failures_path
        # chunk_id -> {"status", "pass_number", "grammar_pass_rate"}
        self._state: dict[str, dict] = {}
        self._counts = {"accepted": 0, "rejected": 0, "failed": 0}
        self._load_existing()

    def _load_existing(self):
        """Read existing progress file to rebuild in-memory state."""
        for p in [self._progress_path, self._translations_path, self._failures_path]:
            p.parent.mkdir(parents=True, exist_ok=True)

        if not self._progress_path.exists():
            return

        with open(self._progress_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                chunk_id = entry["chunk_id"]
                self._state[chunk_id] = entry

        # Recount
        self._counts = {"accepted": 0, "rejected": 0, "failed": 0}
        for entry in self._state.values():
            status = entry.get("status", "")
            if status == "accepted":
                self._counts["accepted"] += 1
            elif status == "failed":
                self._counts["failed"] += 1
            elif status == "rejected":
                self._counts["rejected"] += 1

    def is_done(self, chunk_id: str) -> bool:
        """True if chunk has been accepted or exhausted all retries."""
        if chunk_id not in self._state:
            return False
        status = self._state[chunk_id].get("status")
        return status in ("accepted", "failed")

    def needs_retry(self, chunk_id: str) -> int | None:
        """Return the next pass number if chunk needs retry, else None."""
        if chunk_id not in self._state:
            return None  # Not yet attempted
        entry = self._state[chunk_id]
        if entry["status"] == "accepted":
            return None
        if entry["status"] == "failed":
            return None
        pass_num = entry.get("pass_number", 1)
        if pass_num < MAX_RETRIES + 1:  # passes are 1-indexed
            return pass_num + 1
        return None

    def record(
        self,
        chunk_id: str,
        status: str,  # "accepted", "rejected", "failed"
        pass_number: int,
        model: str,
        grammar_pass_rate: float | None,
        english: str = "",
        lojban: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
    ):
        """Record a translation attempt."""
        timestamp = datetime.now(timezone.utc).isoformat()

        progress_entry = {
            "chunk_id": chunk_id,
            "status": status,
            "pass_number": pass_number,
            "model": model,
            "grammar_pass_rate": grammar_pass_rate,
            "timestamp": timestamp,
        }
        self._state[chunk_id] = progress_entry

        # Append to progress log
        with open(self._progress_path, "a") as f:
            f.write(json.dumps(progress_entry) + "\n")

        if status == "accepted":
            self._counts["accepted"] += 1
            # Write full translation to output
            translation_entry = {
                "chunk_id": chunk_id,
                "english": english,
                "lojban": lojban,
                "model": model,
                "pass_number": pass_number,
                "grammar_pass_rate": grammar_pass_rate,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "timestamp": timestamp,
            }
            with open(self._translations_path, "a") as f:
                f.write(json.dumps(translation_entry, ensure_ascii=False) + "\n")

        elif status == "failed":
            self._counts["failed"] += 1
            failure_entry = {
                "chunk_id": chunk_id,
                "english": english,
                "last_lojban": lojban,
                "last_model": model,
                "last_pass": pass_number,
                "last_grammar_pass_rate": grammar_pass_rate,
                "timestamp": timestamp,
            }
            with open(self._failures_path, "a") as f:
                f.write(json.dumps(failure_entry, ensure_ascii=False) + "\n")

        else:
            self._counts["rejected"] += 1

    def get_stats(self) -> dict:
        """Return summary counts."""
        return {
            "total_seen": len(self._state),
            "accepted": self._counts["accepted"],
            "rejected": self._counts["rejected"],
            "failed": self._counts["failed"],
            "pending_retry": sum(
                1 for e in self._state.values()
                if e.get("status") == "rejected"
            ),
        }

    def print_stats(self):
        s = self.get_stats()
        print(f"Progress: {s['accepted']} accepted, {s['failed']} failed, "
              f"{s['pending_retry']} pending retry, {s['total_seen']} total")
