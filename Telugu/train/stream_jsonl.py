"""
stream_jsonl.py

Streams a .jsonl file directly off the Hugging Face Hub (no full-file
download to disk) and yields the text of each record, one at a time.

Handles the messy realities of scraped/self-collected corpora:
  - lines that aren't valid JSON at all (truncated writes, encoding glitches)
  - records missing the expected text field, or using a different field name
  - records where the text field is null / not a string / empty after strip
  - blank lines

Nothing here loads the whole file into memory: we iterate the HTTP response
line by line, so this is safe for files far larger than available RAM.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Iterator, Optional, Sequence

import requests
from huggingface_hub import hf_hub_url

logger = logging.getLogger(__name__)


@dataclass
class StreamStats:
    """Running counters so you can report data-cleaning stats in Phase 1/2 report."""
    lines_seen: int = 0
    json_errors: int = 0
    missing_text_field: int = 0
    empty_after_strip: int = 0
    yielded: int = 0

    def as_dict(self) -> dict:
        return {
            "lines_seen": self.lines_seen,
            "json_errors": self.json_errors,
            "missing_text_field": self.missing_text_field,
            "empty_after_strip": self.empty_after_strip,
            "yielded": self.yielded,
        }


def _extract_text(record: dict, field_candidates: Sequence[str]) -> Optional[str]:
    """
    Pull the text out of a record whose schema may vary line-to-line.

    Tries each candidate field name in order. Returns None if none of them
    yield a non-empty string, so the caller can skip the record and count it.
    """
    for field_name in field_candidates:
        value = record.get(field_name)
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                return stripped
        # some scraped datasets nest text under e.g. {"data": {"text": ...}}
        elif isinstance(value, dict):
            nested = value.get("text")
            if isinstance(nested, str) and nested.strip():
                return nested.strip()
    return None


def stream_hf_jsonl(
    repo_id: str,
    filename: str,
    repo_type: str = "dataset",
    field_candidates: Sequence[str] = ("text",),
    revision: str = "main",
    stats: Optional[StreamStats] = None,
    chunk_size: int = 1 << 20,
) -> Iterator[str]:
    """
    Stream a .jsonl file from a Hugging Face Hub repo, yielding cleaned text.

    Args:
        repo_id: e.g. "iconically-mine/LMA-indivudal-project"
        filename: path within the repo, e.g. "Nepalese/.../telugu_train.jsonl"
        repo_type: "dataset" or "model" — must match how the repo is hosted.
        field_candidates: field names to try for the text content, in order.
        revision: branch/tag/commit to read from.
        stats: optional StreamStats instance to accumulate counters into
               (pass the same object across calls if you want a running total).
        chunk_size: byte chunk size for the underlying HTTP stream.

    Yields:
        str: cleaned text for each valid record. Invalid/empty records are
        skipped silently but counted in `stats`.

    Note:
        Uses HTTP range-streaming via `requests` with stream=True — the file
        is never written to local disk, and memory use stays flat regardless
        of file size.
    """
    if stats is None:
        stats = StreamStats()

    url = hf_hub_url(repo_id=repo_id, filename=filename, repo_type=repo_type, revision=revision)

    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        # iter_lines decodes on the fly and splits on newlines without
        # buffering the whole response body.
        for raw_line in resp.iter_lines(chunk_size=chunk_size, decode_unicode=True):
            if not raw_line:
                continue  # blank line
            stats.lines_seen += 1

            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError:
                stats.json_errors += 1
                continue

            if not isinstance(record, dict):
                stats.json_errors += 1
                continue

            text = _extract_text(record, field_candidates)
            if text is None:
                stats.missing_text_field += 1
                continue
            if not text:
                stats.empty_after_strip += 1
                continue

            stats.yielded += 1
            yield text

    logger.info("stream_hf_jsonl(%s) done: %s", filename, stats.as_dict())