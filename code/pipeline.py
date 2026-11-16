"""
Main pipeline entry point.

Usage
-----
    python pipeline.py \\
        --data-dir /path/to/jsonl/threads \\
        --output    results/flagged.jsonl \\
        --audit-log results/audit_prefilter.jsonl \\
        --vllm-url  http://localhost:8000/v1 \\
        --model     Qwen/Qwen3-30B-A3B \\
        --batch-size 32

Resume support
--------------
If the run is interrupted, re-run with the same --output path.
Already-processed message_ids are loaded from the output file on startup
and skipped during ingestion. No work is duplicated.

Output schema (one JSON object per line)
-----------------------------------------
{
  "message_id": int,
  "thread_id": int,
  "user_id": int,
  "author": str,
  "created": str,
  "url": str,
  "thread_title": str,
  "label": "negative" | "positive" | "error",
  "confidence": "low" | "medium" | "high",
  "reasoning": str,
  "prefilter_matched": str | null,
  "llm_error": str | null
}
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from tqdm import tqdm

from classify_llm import LLMClassifier
from ingest import iter_posts, load_processed_ids
from route import route_posts

logger = logging.getLogger(__name__)


def _configure_logging(log_level: str, log_file: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        handlers=handlers,
    )


def _write_result(record: dict, fh) -> None:
    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    fh.flush()


def run_pipeline(
    data_dir: Path,
    output_path: Path,
    audit_log_path: Path,
    vllm_url: str,
    model: str,
    batch_size: int,
    context_window: int,
    max_retries: int,
    thinking_budget: int,
    dry_run: bool,
    use_prefilter: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_log_path.parent.mkdir(parents=True, exist_ok=True)

    processed_ids = load_processed_ids(output_path)
    logger.info(
        "Resuming from %d previously processed posts.", len(processed_ids)
    )

    classifier = LLMClassifier(
        base_url=vllm_url,
        model=model,
        max_retries=max_retries,
        thinking_budget=thinking_budget,
    )

    posts = iter_posts(data_dir, skip_ids=processed_ids)

    if dry_run:
        if use_prefilter:
            logger.info("DRY RUN — prefilter statistics only, no LLM calls.")
            total = passed = 0
            from prefilter import passes_prefilter
            for post in tqdm(posts, desc="prefilter dry-run", unit="post"):
                total += 1
                ok, _ = passes_prefilter(post)
                if ok:
                    passed += 1
            logger.info(
                "Dry run complete. %d / %d posts passed prefilter (%.1f%%)",
                passed, total, 100 * passed / max(total, 1),
            )
        else:
            total = 0
            for _ in tqdm(posts, desc="dry-run bypassing prefilter", unit="post"):
                total += 1
            logger.info(
                "Dry run complete. prefilter bypassed; %d posts would be sent to the LLM.",
                total,
            )
        return

    flagged_count = 0
    total_count = 0

    with output_path.open("a", encoding="utf-8") as out_fh:
        classified = route_posts(
            posts,
            classifier,
            audit_log_path=audit_log_path,
            batch_size=batch_size,
            use_prefilter=use_prefilter,
            context_window=context_window,
        )

        with tqdm(
            classified,
            desc="classifying",
            unit="post",
            mininterval=2.0,
        ) as pbar:
            for record in pbar:
                total_count += 1
                _write_result(record, out_fh)
                if record.get("label") == "positive":
                    flagged_count += 1
                pbar.set_postfix(
                    flagged=flagged_count,
                    rate=f"{flagged_count/max(total_count,1)*100:.1f}%",
                )

    logger.info(
        "Pipeline complete. %d posts sent to LLM; %d flagged (%.2f%%).",
        total_count,
        flagged_count,
        100 * flagged_count / max(total_count, 1),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-stage harassment/doxing filter for forum JSONL data."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        type=Path,
        # help="Directory containing input *.jsonl thread files.",
        help="Path to a JSON file or a directory containing JSONL thread files.",
    )
    parser.add_argument(
        "--output",
        default=Path("results/classified.jsonl"),
        type=Path,
        help="Output JSONL path. Appended to for resume support. (default: results/classified.jsonl)",
    )
    parser.add_argument(
        "--audit-log",
        default=Path("results/audit_prefilter.jsonl"),
        type=Path,
        help="JSONL log of posts dropped by the prefilter for periodic recall auditing.",
    )
    parser.add_argument(
        "--vllm-url",
        default="http://localhost:8000/v1",
        help="vLLM OpenAI-compatible endpoint URL.",
    )
    parser.add_argument(
        "--model",
        default="qwen3-32b",
        help="Model name as registered in vLLM.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of posts per LLM batch call.",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=0,
        help="Number of previous posts in the same thread to include in the LLM prompt.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="LLM call retries before marking a post as error.",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=0,
        help="Token budget for Qwen3 thinking mode (0 = disabled).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional path to write log output.",
    )
    parser.add_argument(
        "--skip-prefilter",
        action="store_true",
        help="Do not apply the regex prefilter; send all posts directly to the LLM.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run only the prefilter and report statistics; skip LLM calls.",
    )

    args = parser.parse_args()
    if args.context_window < 0:
        parser.error("--context-window must be >= 0")
    _configure_logging(args.log_level, args.log_file)

    run_pipeline(
        data_dir=args.data_dir,
        output_path=args.output,
        audit_log_path=args.audit_log,
        vllm_url=args.vllm_url,
        model=args.model,
        batch_size=args.batch_size,
        context_window=args.context_window,
        max_retries=args.max_retries,
        thinking_budget=args.thinking_budget,
        dry_run=args.dry_run,
        use_prefilter=not args.skip_prefilter,
    )


if __name__ == "__main__":
    main()
