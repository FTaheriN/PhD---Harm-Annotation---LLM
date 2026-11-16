import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

openai_mod = ModuleType("openai")
openai_mod.OpenAI = MagicMock()
sys.modules["openai"] = openai_mod

pydantic_mod = ModuleType("pydantic")

class DummyBaseModel:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)


def field_validator(name):
    def decorator(fn):
        return fn
    return decorator

pydantic_mod.BaseModel = DummyBaseModel
pydantic_mod.ValidationError = ValueError
pydantic_mod.field_validator = field_validator
sys.modules["pydantic"] = pydantic_mod

from route import route_posts


class DummyClassifier:
    def __init__(self):
        self.batches = []

    def classify_batch(self, batch):
        self.batches.append([dict(post) for post in batch])
        return [
            {
                **{k: v for k, v in post.items() if k != "previous_posts"},
                "label": "none",
                "confidence": "high",
                "reasoning": "LLM received the post.",
                "llm_error": None,
            }
            for post in batch
        ]


def test_route_posts_bypasses_prefilter(tmp_path):
    posts = [
        {"message_id": 1, "thread_id": 10, "content": "dox them"},
        {"message_id": 2, "thread_id": 10, "content": "hello world"},
    ]
    classifier = DummyClassifier()
    audit_log_path = tmp_path / "audit.jsonl"

    results = list(
        route_posts(
            posts,
            classifier,
            audit_log_path=audit_log_path,
            batch_size=2,
            use_prefilter=False,
        )
    )

    assert len(results) == 2
    assert classifier.batches == [
        [
            {**posts[0], "prefilter_matched": None, "previous_posts": []},
            {**posts[1], "prefilter_matched": None, "previous_posts": []},
        ]
    ]
    assert all(record["prefilter_matched"] is None for record in results)
    assert not audit_log_path.exists()


def test_route_posts_adds_previous_post_context(tmp_path):
    posts = [
        {"message_id": 1, "thread_id": 10, "author": "alice", "content": "hello"},
        {"message_id": 2, "thread_id": 10, "author": "bob", "content": "reply"},
        {"message_id": 3, "thread_id": 20, "author": "carol", "content": "other thread"},
        {"message_id": 4, "thread_id": 10, "author": "dave", "content": "follow up"},
    ]
    classifier = DummyClassifier()

    results = list(
        route_posts(
            posts,
            classifier,
            audit_log_path=None,
            batch_size=2,
            use_prefilter=False,
            context_window=1,
        )
    )

    assert len(results) == 4
    assert classifier.batches[0][0]["previous_posts"] == []
    assert classifier.batches[0][1]["previous_posts"] == [
        {"message_id": 1, "author": "alice", "content": "hello"}
    ]
    assert classifier.batches[1][0]["previous_posts"] == []
    assert classifier.batches[1][1]["previous_posts"] == [
        {"message_id": 2, "author": "bob", "content": "reply"}
    ]
    assert all("previous_posts" not in record for record in results)
