"""
Router — wires the heuristic pre-filter to the LLM classifier.

Consumes a stream of posts, applies the prefilter, writes filtered-out
posts to the audit log, and yields LLM-classified results for any post
that passes the heuristic gate.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Generator, Iterable

from prefilter import passes_prefilter
from classify_llm import LLMClassifier

logger = logging.getLogger(__name__)


def _write_jsonl(record: dict[str, Any], path: Path) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")

def _clean_data(content: str) -> str:
    """
    Remove all content in between <LINK> and </LINK> tags, and between <QUOTE> and </QUOTE> tags.
    """
    import re

    # Remove <LINK>...</LINK> blocks
    content = re.sub(r"<link.*?<\/link>", "", content, flags=re.DOTALL)
    # Remove <QUOTE>...</QUOTE> blocks
    content = re.sub(r"<quote.*?<\/quote>", "", content, flags=re.DOTALL)
    content = re.sub(r"<video.*?>", "", content, flags=re.DOTALL)
    content = re.sub(r"<image.*?>", "", content, flags=re.DOTALL)
    return content.strip()

def route_posts(
    posts: Iterable[dict[str, Any]],
    classifier: LLMClassifier,
    *,
    audit_log_path: Path | None = None,
    batch_size: int = 32,
    use_prefilter: bool = True,
    context_window: int = 1,
) -> Generator[dict[str, Any], None, None]:
    """
    Route each post through the two-stage filter.

    Stage 1 — heuristic prefilter (cheap, high recall).
    Stage 2 — LLM classifier (expensive, high precision).

    Posts that fail stage 1 are written to *audit_log_path* (if given) with
    label="prefilter_negative" so a sample can be audited for leakage.

    Posts that pass stage 1 are batched and sent to the LLM. Classified
    records are yielded to the caller.

    Parameters
    ----------
    posts:
        Iterable of raw post dicts from the ingest layer.
    classifier:
        Configured LLMClassifier instance.
    audit_log_path:
        Path for the prefilter-negative JSONL audit log.
    batch_size:
        Number of posts to accumulate before a single LLM batch call.
        Larger batches amortise HTTP overhead; smaller batches reduce
        memory pressure and give finer progress granularity.
    context_window:
        Number of previous posts in the same thread to include in the
        LLM prompt for each classified post.
    """
    pending: list[dict[str, Any]] = []
    thread_history: dict[Any, deque[dict[str, Any]]] = defaultdict(
        lambda: deque(maxlen=context_window)
    )

    def _flush(batch: list[dict[str, Any]]) -> Generator[dict[str, Any], None, None]:
        results = classifier.classify_batch(batch)
        yield from results

    for post in posts:
        thread_key = post.get("thread_id", "__no_thread__")
        previous_posts = list(thread_history[thread_key])
        
        # Adding additional measure to scrub <LINK> and <QUOTE> blocks from content
        post["content"] = _clean_data(post.get("content", ""))

        if use_prefilter:
            ok, matched_pattern = passes_prefilter(post)
            if not ok:
                if audit_log_path:
                    audit_record = {
                        "message_id": post.get("message_id"),
                        "thread_id": post.get("thread_id"),
                        "thread_subject": post.get("thread_title"),
                        "label": "prefilter_negative",
                    }
                    _write_jsonl(audit_record, audit_log_path)
                thread_history[thread_key].append(
                    {
                        "message_id": post.get("message_id"),
                        "author": post.get("author"),
                        "thread_subject": post.get("thread_title"),
                        "content": post.get("content"),
                    }
                )
                continue
            enriched = dict(post)
            enriched["prefilter_matched"] = matched_pattern
        else:
            enriched = dict(post)
            enriched["prefilter_matched"] = None

        enriched["previous_posts"] = [dict(prev) for prev in previous_posts]
        pending.append(enriched)
        thread_history[thread_key].append(
            {
                "message_id": post.get("message_id"),
                "author": post.get("author"),
                "thread_subject": post.get("thread_title"),
                "content": post.get("content"),
            }
        )

        if len(pending) >= batch_size:
            yield from _flush(pending)
            pending.clear()

    if pending:
        yield from _flush(pending)
