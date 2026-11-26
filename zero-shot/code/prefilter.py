"""
Heuristic pre-filter — stage 1 of the two-stage pipeline.

Design goal: **very high recall, low precision**.
Anything that passes goes to the LLM; anything that fails is logged
and periodically audited for false-negative leakage.

The filter errs heavily toward keeping posts. A single keyword match
is enough to pass. We aim for ~0.95 recall on true positives; precision
is irrelevant at this stage.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lexicons
# ---------------------------------------------------------------------------

# Direct harassment-coordination vocabulary.
_HARASSMENT_LEXICON: list[str] = [
    r"\bdox(?:x(?:ing|ed|es|er)?|ing|ed|er)?\b",
    r"\bswat?(?:ing|ed|er)?\b",
    r"\brain\b",                      # "raid", "raiders"
    r"\braiding\b",
    r"\braiders?\b",
    r"\bstalk(?:ing|er)?\b",
    r"\bharass(?:ing|ment|ed)?\b",
    r"\bmass.?report",
    r"\breport.?(?:their|them|his|her|the)\b",
    r"\bbomb(?:ard|ing)?\b",          # "bombed their DMs"
    r"\bspam(?:ming|med)?\b",
    r"\bintimidation\b",
    r"\bthreaten(?:ing)?\b",
    r"\bblackmail(?:ing)?\b",
]

# PII / doxing vocabulary.
_DOXING_LEXICON: list[str] = [
    r"\bphone.?(?:number|num)\b",
    r"\bhome.?address\b",
    r"\bstreet.?address\b",
    r"\bwhere.{0,15}(?:lives?|lives?|works?|located)\b",
    r"\bfull.?name\b",
    r"\breal.?name\b",
    r"\bI[Pv]{1,2}.?address\b",
    r"\bsocial.?security\b",
    r"\bSSN\b",
    r"\btheir.{0,10}(?:address|location|deets|info)\b",
    r"\bfind(?:ing)?.{0,10}(?:address|location|number|info)\b",
    r"\bpost(?:ing|ed)?.{0,10}(?:address|location|number|info)\b",
    r"\bshare.{0,10}(?:address|location|number|info)\b",
    r"\bprivate.?info(?:rmation)?\b",
    r"\bpersonal.?info(?:rmation)?\b",
]

# First-person commitment patterns coupled with target references.
# These are deliberately broad; "I will" alone would FP massively, but
# combined with a target pronoun or proper noun they signal intent.
_ACTION_COMMITMENT: list[str] = [
    r"\bI(?:'ll| will| am going to| plan to| intend to).{0,50}(?:them|him|her|they)\b",
    r"\bwe(?:'ll| will| are going to).{0,50}(?:them|him|her|they)\b",
    r"\beveryone.{0,30}(?:should|go|must|needs? to).{0,30}(?:report|harass|dm|message|contact)\b",
    r"\bgo (?:report|harass|message|after|get) (?:them|him|her)\b",
    r"\blet(?:'s| us).{0,30}(?:report|harass|dox|raid|send)\b",
    r"\bsomebody.{0,20}(?:should|needs? to|must).{0,20}(?:report|harass|dox|find)\b",
    r"\bsomeone.{0,20}(?:should|needs? to|must).{0,20}(?:report|harass|dox|find)\b",
    r"\byou should.{0,30}(?:report|harass|dox|message|contact)\b",
    r"\bteach (?:them|him|her).{0,20}(?:lesson|who)\b",
]

# PII value patterns — phone numbers, email addresses, partial addresses.
_PII_PATTERNS: list[str] = [
    # Phone numbers (various formats)
    r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    # Email addresses
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
    # Street address fragments
    r"\b\d{1,5}\s+[A-Za-z]{2,}\s+(?:St(?:reet)?|Ave(?:nue)?|Rd|Road|Blvd|Dr(?:ive)?|Ln|Lane|Ct|Court|Pl(?:ace)?|Way)\b",
]

# Compile everything once at module load.
_ALL_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE | re.DOTALL)
    for p in (
        _HARASSMENT_LEXICON
        + _DOXING_LEXICON
        + _ACTION_COMMITMENT
        + _PII_PATTERNS
    )
]


def _get_text(post: dict[str, Any]) -> str:
    """Concatenate content and thread_title for matching."""
    parts = [post.get("content", ""), post.get("thread_title", "")]
    return " ".join(p for p in parts if p)


def passes_prefilter(post: dict[str, Any]) -> tuple[bool, str | None]:
    """
    Return (True, matching_pattern) if the post should be sent to the LLM,
    (False, None) otherwise.

    The second element of the tuple is the first matching pattern string,
    useful for audit logging.
    """
    text = _get_text(post)
    for pattern in _ALL_PATTERNS:
        m = pattern.search(text)
        if m:
            return True, pattern.pattern
    return False, None


def prefilter_batch(
    posts: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Split a batch of posts into (passed, filtered_out).

    Posts in *passed* should proceed to the LLM.
    Posts in *filtered_out* are assumed negative and should be logged.
    """
    passed: list[dict[str, Any]] = []
    filtered: list[dict[str, Any]] = []
    for post in posts:
        ok, pattern = passes_prefilter(post)
        if ok:
            post = dict(post)
            post["prefilter_matched"] = pattern
            passed.append(post)
        else:
            filtered.append(post)
    return passed, filtered
