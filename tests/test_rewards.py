import pytest
from src.tasks.math_task import GSM8KTask


class TestGSM8KReward:
    def test_extract_boxed_answer(self):
        task = GSM8KTask()
        text = "Therefore, \\boxed{42} is the answer."
        assert task._extract_answer(text) == "42"

    def test_extract_hashtag_answer(self):
        task = GSM8KTask()
        text = "Step 3: add them\n#### 15"
        assert task._extract_answer(text) == "15"

    def test_no_answer_returns_none(self):
        task = GSM8KTask()
        text = "Just some reasoning with no answer format."
        assert task._extract_answer(text) is None
