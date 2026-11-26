"""
Tests for classify_llm.py.

Uses unittest.mock to stub the vLLM endpoint — no real model required.
Tests cover: schema validation, correct parsing, retry behaviour,
error handling, and user-message construction.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from classify_llm import (
    ClassificationResult,
    LLMClassifier,
    _build_user_message,
    _parse_response,
    LABEL_VALUES,
    CONFIDENCE_VALUES,
)


# ---------------------------------------------------------------------------
# Unit tests for helpers
# ---------------------------------------------------------------------------

class TestBuildUserMessage:
    def test_includes_thread_title_and_author(self):
        post = {
            "thread_title": "Drama thread",
            "author": "user123",
            "content": "Some content here",
        }
        msg = _build_user_message(post)
        assert "[Thread: Drama thread]" in msg
        assert "[Author: user123]" in msg
        assert "Some content here" in msg

    def test_no_thread_title(self):
        post = {"author": "anon", "content": "hello"}
        msg = _build_user_message(post)
        assert "[Thread:" not in msg
        assert "[Author: anon]" in msg

    def test_missing_fields_dont_crash(self):
        msg = _build_user_message({})
        assert isinstance(msg, str)


class TestParseResponse:
    def _make_json(self, label="none", confidence="high", reasoning="ok"):
        return json.dumps({"label": label, "confidence": confidence, "reasoning": reasoning})

    def test_valid_none(self):
        result = _parse_response(self._make_json("none", "high", "Nothing actionable."))
        assert result.label == "none"
        assert result.confidence == "high"

    def test_valid_harassment_plan(self):
        result = _parse_response(self._make_json("harassment_plan", "medium", "Call to raid."))
        assert result.label == "harassment_plan"

    def test_valid_doxing(self):
        result = _parse_response(self._make_json("doxing", "low", "Solicits address."))
        assert result.label == "doxing"

    def test_invalid_label_raises(self):
        bad = json.dumps({"label": "toxic", "confidence": "high", "reasoning": "x"})
        with pytest.raises(ValueError, match="label must be one of"):
            _parse_response(bad)

    def test_invalid_confidence_raises(self):
        bad = json.dumps({"label": "none", "confidence": "very_sure", "reasoning": "x"})
        with pytest.raises(ValueError, match="confidence must be one of"):
            _parse_response(bad)

    def test_non_json_raises(self):
        with pytest.raises(ValueError, match="non-JSON"):
            _parse_response("not json at all")

    def test_missing_field_raises(self):
        bad = json.dumps({"label": "none", "confidence": "high"})
        with pytest.raises(ValueError):
            _parse_response(bad)


class TestClassificationResult:
    def test_all_valid_labels(self):
        for label in LABEL_VALUES:
            r = ClassificationResult(label=label, confidence="high", reasoning="ok")
            assert r.label == label

    def test_all_valid_confidences(self):
        for conf in CONFIDENCE_VALUES:
            r = ClassificationResult(label="none", confidence=conf, reasoning="ok")
            assert r.confidence == conf

    def test_invalid_label(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ClassificationResult(label="bad_label", confidence="high", reasoning="x")


# ---------------------------------------------------------------------------
# Integration-style tests using mocked OpenAI client
# ---------------------------------------------------------------------------

def _make_mock_response(label="none", confidence="high", reasoning="No call to action."):
    """Build a mock openai ChatCompletion response."""
    content = json.dumps({"label": label, "confidence": confidence, "reasoning": reasoning})
    choice = SimpleNamespace(message=SimpleNamespace(content=content))
    return SimpleNamespace(choices=[choice])


class TestLLMClassifier:
    POST = {
        "message_id": 42,
        "thread_id": 101,
        "author": "testuser",
        "thread_title": "Some thread",
        "content": "I hate that person.",
    }

    def _make_classifier(self):
        clf = LLMClassifier(base_url="http://fake:8000/v1", model="fake-model")
        return clf

    def test_classify_post_success(self):
        clf = self._make_classifier()
        clf._client.chat.completions.create = MagicMock(
            return_value=_make_mock_response("none", "high", "Plain expression of dislike.")
        )
        result = clf.classify_post(self.POST)
        assert result.label == "none"
        assert result.confidence == "high"

    def test_classify_post_harassment(self):
        clf = self._make_classifier()
        clf._client.chat.completions.create = MagicMock(
            return_value=_make_mock_response("harassment_plan", "high", "Clear call to raid.")
        )
        result = clf.classify_post(self.POST)
        assert result.label == "harassment_plan"

    def test_classify_post_retries_on_bad_json(self):
        """Model returns garbage twice then valid JSON — should succeed on third attempt."""
        clf = self._make_classifier()
        bad = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="not-json"))])
        good = _make_mock_response("doxing", "medium", "Published PII.")
        clf._client.chat.completions.create = MagicMock(side_effect=[bad, bad, good])
        clf.retry_delay = 0
        result = clf.classify_post(self.POST)
        assert result.label == "doxing"
        assert clf._client.chat.completions.create.call_count == 3

    def test_classify_post_exhausts_retries(self):
        """All retries return garbage — should raise ValueError."""
        clf = self._make_classifier()
        bad = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="bad"))])
        clf._client.chat.completions.create = MagicMock(return_value=bad)
        clf.retry_delay = 0
        with pytest.raises(ValueError, match="All 3 attempts failed"):
            clf.classify_post(self.POST)

    def test_classify_batch_success(self):
        clf = self._make_classifier()
        clf._client.chat.completions.create = MagicMock(
            return_value=_make_mock_response("none", "high", "Nothing.")
        )
        posts = [dict(self.POST, message_id=i) for i in range(3)]
        results = clf.classify_batch(posts)
        assert len(results) == 3
        assert all(r["label"] == "none" for r in results)
        assert all(r["llm_error"] is None for r in results)

    def test_classify_batch_partial_failure(self):
        """One post fails all retries; others succeed. Batch returns error record, not exception."""
        clf = self._make_classifier()
        bad = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="bad"))])
        good = _make_mock_response("none", "high", "ok")
        # 3 posts: first succeeds, second fails all retries (3 calls), third succeeds
        clf._client.chat.completions.create = MagicMock(
            side_effect=[good, bad, bad, bad, good]
        )
        clf.retry_delay = 0
        posts = [dict(self.POST, message_id=i) for i in range(3)]
        results = clf.classify_batch(posts)
        assert results[0]["label"] == "none"
        assert results[1]["label"] == "error"
        assert results[1]["llm_error"] is not None
        assert results[2]["label"] == "none"

    def test_original_post_fields_preserved(self):
        clf = self._make_classifier()
        clf._client.chat.completions.create = MagicMock(
            return_value=_make_mock_response()
        )
        result = clf.classify_batch([self.POST])[0]
        assert result["message_id"] == 42
        assert result["thread_id"] == 101
        assert result["author"] == "testuser"

    def test_system_prompt_sent(self):
        """The system prompt must be present in every API call."""
        from classify_llm import SYSTEM_PROMPT
        clf = self._make_classifier()
        clf._client.chat.completions.create = MagicMock(
            return_value=_make_mock_response()
        )
        clf.classify_post(self.POST)
        call_kwargs = clf._client.chat.completions.create.call_args
        messages = call_kwargs[1]["messages"]
        system_msgs = [m for m in messages if m["role"] == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0]["content"] == SYSTEM_PROMPT

    def test_guided_json_in_extra_body(self):
        """guided_json schema must be passed so vLLM constrains output."""
        from classify_llm import RESPONSE_SCHEMA
        clf = self._make_classifier()
        clf._client.chat.completions.create = MagicMock(
            return_value=_make_mock_response()
        )
        clf.classify_post(self.POST)
        call_kwargs = clf._client.chat.completions.create.call_args[1]
        assert call_kwargs["extra_body"]["guided_json"] == RESPONSE_SCHEMA

    def test_temperature_zero(self):
        clf = self._make_classifier()
        clf._client.chat.completions.create = MagicMock(
            return_value=_make_mock_response()
        )
        clf.classify_post(self.POST)
        call_kwargs = clf._client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.0
