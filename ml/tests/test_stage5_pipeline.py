#!/usr/bin/env python3
"""
Unit Tests for Stage 5 Dataset Engineering Pipeline.
Validates end-to-end record annotation, preservation of immutable identifiers,
structural field integrity, and benchmark consistency.
"""

import json
from pathlib import Path
import sys
import unittest

# Ensure ml root is in sys.path
ml_root = Path(__file__).resolve().parents[1]
if str(ml_root) not in sys.path:
    sys.path.insert(0, str(ml_root))

from src.preprocessing.behavioral_features import BehavioralFeatureExtractor
from src.preprocessing.topic_classifier import DeterministicTopicClassifier
from src.preprocessing.context_features import ContextFeatureExtractor
from scripts.build_stage5_dataset import annotate_record


class TestStage5Pipeline(unittest.TestCase):
    def setUp(self):
        self.behav_ext = BehavioralFeatureExtractor()
        self.topic_cls = DeterministicTopicClassifier()
        self.ctx_ext = ContextFeatureExtractor()

        # Synthetic Stage 4-style candidate example
        self.mock_stage4_record = {
            "messages": [
                {"role": "system", "content": "You are Vivek."},
                {"role": "user", "content": "Khelega kya valorant?"},
                {"role": "assistant", "content": "Aaja bhai lobby me hu"},
            ],
            "metadata": {
                "example_id": "conv_9999_tgt42_ctx1",
                "conversation_id": "conv_9999",
                "target_message_id": 42,
                "timestamp": "2025-02-14 20:00:00",
                "speaker": "Vivek",
                "context_turns": 1,
            },
        }

    def test_record_annotation_preserves_identifiers(self):
        annotated = annotate_record(
            record=self.mock_stage4_record,
            behavioral_extractor=self.behav_ext,
            topic_classifier=self.topic_cls,
            context_extractor=self.ctx_ext,
        )

        # Invariant preservation
        self.assertEqual(annotated["conversation_id"], "conv_9999")
        self.assertEqual(annotated["target_message_id"], 42)
        self.assertEqual(annotated["target_response"], "Aaja bhai lobby me hu")
        self.assertEqual(len(annotated["context"]), 1)
        self.assertEqual(annotated["context"][0]["content"], "Khelega kya valorant?")

        # Required Stage 5 metadata additions
        self.assertIn("behavioral_metadata", annotated)
        self.assertIn("topic_metadata", annotated)
        self.assertIn("context_metadata", annotated)

        # Behavioral metadata checks
        b_meta = annotated["behavioral_metadata"]
        self.assertEqual(b_meta["language"], "hinglish")
        self.assertIn("tone", b_meta)
        self.assertIn("response_type", b_meta)
        self.assertIn("response_length", b_meta)
        self.assertIn("emoji_count", b_meta)
        self.assertIn("has_emoji", b_meta)
        self.assertIn("has_slang", b_meta)
        self.assertIn("is_question", b_meta)

        # Topic metadata checks
        t_meta = annotated["topic_metadata"]
        self.assertEqual(t_meta["primary_topic"], "gaming")
        self.assertIn("confidence", t_meta)
        self.assertIn("scores", t_meta)

        # Context metadata checks
        c_meta = annotated["context_metadata"]
        self.assertEqual(c_meta["context_depth"], 1)
        self.assertEqual(c_meta["number_of_turns"], 1)
        self.assertTrue(c_meta["is_strict_alternation"])
        self.assertEqual(c_meta["last_user_message"], "Khelega kya valorant?")

    def test_dataset_files_exist_and_non_empty(self):
        stage5_dir = ml_root / "data" / "stage5"
        for split_file in ["stage5_train.jsonl", "val.jsonl", "test.jsonl"]:
            p = stage5_dir / "training" / split_file
            self.assertTrue(p.is_file(), f"Expected file {p} to exist")
            self.assertGreater(p.stat().st_size, 0)

        amb_path = stage5_dir / "benchmarks" / "ambiguous_prompts.jsonl"
        self.assertTrue(amb_path.is_file(), f"Expected file {amb_path} to exist")
        self.assertGreater(amb_path.stat().st_size, 0)

        report_path = stage5_dir / "reports" / "stage5_dataset_audit.md"
        self.assertTrue(report_path.is_file(), f"Expected report {report_path} to exist")
        self.assertGreater(report_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
