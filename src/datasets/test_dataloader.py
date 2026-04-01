"""
tests/test_dataloader.py

Run: python -m pytest tests/test_dataloader.py -v
"""

import pytest
from src.datasets.qa_datasets import (
    extract_gsm8k, extract_commonsenseqa,
    extract_arc, extract_strategyqa,
    extract_wiki_multihop, difficulty_score
)


class TestExtractors:

    def test_gsm8k_extracts_number(self):
        sample = {
            "question": "A train goes 60 mph for 2 hours. How far?",
            "answer":   "The train travels 60 x 2 = 120 miles.\n#### 120"
        }
        result = extract_gsm8k(sample)
        assert result["answer"] == "120"
        assert result["category"] == "math"

    def test_gsm8k_no_hash(self):
        sample = {"question": "Q?", "answer": "42"}
        result = extract_gsm8k(sample)
        assert result["answer"] == "42"

    def test_commonsenseqa_letter_key(self):
        sample = {
            "question":  "Where do dogs sleep?",
            "answerKey": "B",
            "choices": {
                "label": ["A", "B", "C"],
                "text":  ["car", "dog bed", "tree"]
            }
        }
        result = extract_commonsenseqa(sample)
        assert result["answer"] == "dog bed"

    def test_arc_letter_key(self):
        sample = {
            "question":  "What is photosynthesis?",
            "answerKey": "C",
            "choices": {
                "label": ["A", "B", "C", "D"],
                "text":  ["Breathing", "Digestion", "Making food from light", "Swimming"]
            }
        }
        result = extract_arc(sample)
        assert result["answer"] == "Making food from light"

    def test_arc_numeric_key(self):
        sample = {
            "question":  "Science question?",
            "answerKey": "2",
            "choices": {
                "label": ["1", "2", "3", "4"],
                "text":  ["wrong", "correct", "wrong", "wrong"]
            }
        }
        result = extract_arc(sample)
        assert result["answer"] == "correct"

    def test_strategyqa_true(self):
        sample = {"question": "Did Einstein win the Nobel Prize?", "answer": True}
        result = extract_strategyqa(sample)
        assert result["answer"] == "yes"

    def test_strategyqa_false(self):
        sample = {"question": "Was Tesla born in France?", "answer": False}
        result = extract_strategyqa(sample)
        assert result["answer"] == "no"

    def test_wiki_multihop(self):
        sample = {
            "question": "In which city did the physicist born in Warsaw work?",
            "answer":   "Paris"
        }
        result = extract_wiki_multihop(sample)
        assert result["answer"] == "Paris"
        assert result["category"] == "factual_multihop"


class TestDifficultyScore:

    def test_simple_question_scores_zero(self):
        score = difficulty_score("What is the capital of France?")
        assert score == 0

    def test_multihop_question_scores_higher(self):
        score = difficulty_score(
            "Which city did the scientist who discovered radium move to after Warsaw?"
        )
        assert score >= 1

    def test_all_scores_valid_range(self):
        questions = [
            "What is 2+2?",
            "Who led the team that also won the championship?",
            "Which country, which later became a republic, was before that a monarchy?",
        ]
        for q in questions:
            score = difficulty_score(q)
            assert score in (0, 1, 2), f"Invalid score {score} for: {q}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])