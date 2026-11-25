"""
Data ingestion layer.

Streams posts from a directory of JSONL files (one file per forum thread).
Each line in a JSONL file is a single post object. Yields posts one at a
time so the pipeline never loads the full dataset into memory.

Also exposes helpers for building the set of already-processed message_ids
from an output file (used for resume-on-failure).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Generator, Iterator, Set

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = {"message_id", "content"}


def _is_valid_post(record: object, path: Path, lineno: int, id_field: str = "message_id") -> bool:
    if not isinstance(record, dict):
        logger.warning("%s line %d: not a JSON object, skipping", path.name, lineno)
        return False
    missing = {id_field, "content"} - record.keys()
    if missing:
        logger.warning(
            "%s line %d: missing fields %s, skipping", path.name, lineno, missing
        )
        return False
    if not isinstance(record.get("content"), str) or not record["content"].strip():
        logger.debug("%s line %d: empty content, skipping", path.name, lineno)
        return False
    return True


def _validate_task(record: object, path: Path, lineno: int, id_field: str) -> None:
    if not isinstance(record, dict):
        raise ValueError(f"{path}:{lineno}: expected a JSON object")
    task_id = record.get(id_field)
    if type(task_id) not in (str, int) or (isinstance(task_id, str) and not task_id.strip()):
        raise ValueError(f"{path}:{lineno}: expected a nonempty string or integer in {id_field}")
    if not isinstance(record.get("content"), str):
        raise ValueError(f"{path}:{lineno}: expected string content")


def iter_posts(
    data_dir: str | Path,
    *,
    glob: str = "*.jsonl",
    skip_ids: Set[int] | None = None,
    id_field: str = "message_id",
    strict_tasks: bool = False,
) -> Generator[dict, None, None]:
    """Read a JSON file or a directory of JSONL files."""
    input_path = Path(data_dir)
    skip = skip_ids or set()

    if input_path.is_file():
        if input_path.suffix.lower() != ".json":
            raise ValueError("Input file must have a .json extension")

        with input_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        if isinstance(data, dict):
            records = [data]
        elif isinstance(data, list):
            records = data
        else:
            raise ValueError(
                "JSON input must be an object or an array of objects"
            )

        for index, record in enumerate(records, start=1):
            if strict_tasks:
                _validate_task(record, input_path, index, id_field)
            elif not _is_valid_post(record, input_path, index, id_field=id_field):
                continue
            if record[id_field] in skip:
                continue
            yield record

        return

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    files = sorted(input_path.glob(glob))
    if not files:
        if strict_tasks:
            raise ValueError(f"No files matching {glob!r} in {input_path}")
        logger.warning("No files matching %r found in %s", glob, input_path)
        return

    for path in files:
        with path.open("r", encoding="utf-8") as fh:
            for lineno, raw_line in enumerate(fh, start=1):
                if not raw_line.strip():
                    continue

                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    if strict_tasks:
                        raise ValueError(f"{path}:{lineno}: invalid JSON") from exc
                    logger.warning(
                        "%s line %d: invalid JSON (%s), skipping",
                        path.name, lineno, exc,
                    )
                    continue

                if strict_tasks:
                    _validate_task(record, path, lineno, id_field)
                elif not _is_valid_post(record, path, lineno, id_field=id_field):
                    continue
                if record[id_field] in skip:
                    continue

                yield record

# def iter_posts(
#     data_dir: str | Path,
#     *,
#     glob: str = "*.jsonl",
#     skip_ids: Set[int] | None = None,
# ) -> Generator[dict, None, None]:
#     """
#     Yield valid post dicts from every JSONL file in *data_dir*.

#     Parameters
#     ----------
#     data_dir:
#         Directory containing the JSONL thread files.
#     glob:
#         File pattern to match. Defaults to ``*.jsonl``.
#     skip_ids:
#         Set of message_ids already processed. Posts whose message_id appears
#         here are silently skipped (resume support).
#     """
#     data_dir = Path(data_dir)
#     if not data_dir.is_dir():
#         raise FileNotFoundError(f"Data directory not found: {data_dir}")

#     skip = skip_ids or set()
#     files = sorted(data_dir.glob(glob))
#     if not files:
#         logger.warning("No files matching %r found in %s", glob, data_dir)
#         return

#     for path in files:
#         logger.debug("Ingesting %s", path.name)
#         try:
#             with path.open("r", encoding="utf-8", errors="replace") as fh:
#                 for lineno, raw_line in enumerate(fh, start=1):
#                     raw_line = raw_line.strip()
#                     if not raw_line:
#                         continue
#                     try:
#                         record = json.loads(raw_line)
#                     except json.JSONDecodeError as e:
#                         logger.warning(
#                             "%s line %d: JSON parse error (%s), skipping",
#                             path.name, lineno, e,
#                         )
#                         continue
#                     if not _is_valid_post(record, path, lineno):
#                         continue
#                     mid = record.get("message_id")
#                     if mid in skip:
#                         continue
#                     yield record
#         except OSError as e:
#             logger.error("Cannot read %s: %s — skipping file", path, e)


def load_processed_ids(output_path: str | Path) -> Set[int]:
    """
    Return the set of message_ids already written to *output_path*.

    Used on startup to resume an interrupted run. Non-existent or empty
    files return an empty set.
    """
    output_path = Path(output_path)
    ids: Set[int] = set()
    if not output_path.exists():
        return ids
    with output_path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                record = json.loads(raw_line)
                mid = record.get("message_id")
                if mid is not None:
                    ids.add(mid)
            except json.JSONDecodeError:
                continue
    logger.info("Loaded %d already-processed message_ids from %s", len(ids), output_path)
    return ids


def count_posts(data_dir: str | Path, glob: str = "*.jsonl", id_field: str = "message_id") -> int:
    """Count the total number of valid posts across all files (for progress bars)."""
    total = 0
    for _ in iter_posts(data_dir, glob=glob, id_field=id_field):
        total += 1
    return total
