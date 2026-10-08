#!/usr/bin/env python3
"""
Unit Tests for Stage 5 Context Feature Extraction.
Tests context depth calculation, speaker alternation dynamics,
speaker sequence tracking, and conversation state inference.
"""

from pathlib import Path
import sys
import unittest

# Ensure ml root is in sys.path
ml_root = Path(__file__).resolve().parents[1]
if str(ml_root) not in sys.path:
    sys.path.insert(0, str(ml_root))

from src.preprocessing.context_features import (
    ContextFeatureExtractor,
    ContextFeatures,
)


class TestContextFeatureExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = ContextFeatureExtractor()

    def test_empty_context(self):
        feat = self.extractor.extract_features([], target_response="haan")
        self.assertEqual(feat.context_depth, 0)
        self.assertEqual(feat.number_of_turns, 0)
        self.assertTrue(feat.is_strict_alternation)
        self.assertEqual(feat.speaker_sequence, [])
        self.assertEqual(feat.last_user_message, "")

    def test_single_turn_context(self):
        ctx = [{"role": "user", "content": "Aaja bhai"}]
        feat = self.extractor.extract_features(ctx, target_response="Aaya")
        self.assertEqual(feat.context_depth, 1)
        self.assertEqual(feat.number_of_turns, 1)
        self.assertTrue(feat.is_strict_alternation)
        self.assertEqual(feat.speaker_sequence, ["user"])
        self.assertEqual(feat.last_user_message, "Aaja bhai")
        self.assertEqual(feat.conversation_state, "opening")

    def test_multi_turn_strict_alternation(self):
        ctx = [
            {"role": "user", "content": "Khelega kya?"},
            {"role": "assistant", "content": "Abhi to assignment kar raha hu"},
            {"role": "user", "content": "Kab tak free hoga?"},
        ]
        feat = self.extractor.extract_features(ctx, target_response="7 baje tak")
        self.assertEqual(feat.context_depth, 3)
        self.assertEqual(feat.number_of_turns, 2)
        self.assertTrue(feat.is_strict_alternation)
        self.assertEqual(feat.speaker_alternation_rate, 1.0)
        self.assertEqual(feat.speaker_sequence, ["user", "assistant", "user"])
        self.assertEqual(feat.last_user_message, "Kab tak free hoga?")
        self.assertEqual(feat.conversation_state, "inquiry")

    def test_non_strict_alternation(self):
        # Two consecutive user messages
        ctx = [
            {"role": "user", "content": "bhai sun"},
            {"role": "user", "content": "urgent kaam hai"},
            {"role": "assistant", "content": "bol kya hua"},
            {"role": "user", "content": "laptop chahiye"},
        ]
        feat = self.extractor.extract_features(ctx, target_response="le ja")
        self.assertEqual(feat.context_depth, 4)
        self.assertFalse(feat.is_strict_alternation)
        self.assertLess(feat.speaker_alternation_rate, 1.0)

    def test_conversation_state_banter(self):
        ctx = [{"role": "user", "content": "kya kar raha hai bsdk"}]
        feat = self.extractor.extract_features(ctx, target_response="tu bata saale")
        self.assertEqual(feat.conversation_state, "banter")

    def test_conversation_state_closing(self):
        ctx = [
            {"role": "user", "content": "chal mai so raha hu gn"},
        ]
        feat = self.extractor.extract_features(ctx, target_response="gn bhai")
        self.assertEqual(feat.conversation_state, "closing")

    def test_conversation_state_agreement(self):
        ctx = [
            {"role": "user", "content": "sham ko 6 baje ground pe milte hai"},
        ]
        feat = self.extractor.extract_features(ctx, target_response="done theek hai")
        self.assertEqual(feat.conversation_state, "agreement")

    def test_to_dict_serialization(self):
        ctx = [{"role": "user", "content": "Hi"}]
        feat = self.extractor.extract_features(ctx, target_response="Yo")
        d = feat.to_dict()
        self.assertIn("context_depth", d)
        self.assertIn("number_of_turns", d)
        self.assertIn("speaker_alternation_rate", d)
        self.assertIn("is_strict_alternation", d)
        self.assertIn("speaker_sequence", d)
        self.assertIn("last_user_message", d)
        self.assertIn("conversation_state", d)


if __name__ == "__main__":
    unittest.main()
