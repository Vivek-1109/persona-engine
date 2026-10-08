#!/usr/bin/env python3
"""
Unit Tests for Stage 5 Behavioral Feature Extraction.
Tests language detection, tone classification, response type taxonomy,
response length categorization, emoji analysis, and slang detection.
"""

from pathlib import Path
import sys
import unittest

# Ensure ml root is in sys.path
ml_root = Path(__file__).resolve().parents[1]
if str(ml_root) not in sys.path:
    sys.path.insert(0, str(ml_root))

from src.preprocessing.behavioral_features import (
    BehavioralFeatureExtractor,
    BehavioralFeatures,
    extract_emojis,
    count_emojis,
)


class TestBehavioralFeatureExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = BehavioralFeatureExtractor()

    def test_language_detection(self):
        self.assertEqual(self.extractor.detect_language("haan bhai mai bhi wahi bol raha tha"), "hinglish")
        self.assertEqual(self.extractor.detect_language("I will definitely be there by five pm"), "english")
        self.assertEqual(self.extractor.detect_language("हां भाई मैं भी आ रहा हूं"), "hindi")
        self.assertEqual(self.extractor.detect_language("12345 !!!"), "unknown")

    def test_tone_detection(self):
        # Teasing / Banter
        self.assertEqual(self.extractor.detect_tone("are bsdk tu kab sudhrega", has_emoji=False), "teasing")
        # Humorous
        self.assertEqual(self.extractor.detect_tone("bhai sach me lol xdd ye kya tha", has_emoji=False), "humorous")
        # Supportive
        self.assertEqual(self.extractor.detect_tone("tension mat le ho jayega sab theek", has_emoji=False), "supportive")
        # Serious
        self.assertEqual(self.extractor.detect_tone("urgent hospital emergency case hai", has_emoji=False), "serious")
        # Uncertain
        self.assertEqual(self.extractor.detect_tone("shayad kal aa saku pakka nahi pata", has_emoji=False), "uncertain")
        # Casual
        self.assertEqual(self.extractor.detect_tone("aaja bhai room pe hi hu", has_emoji=False), "casual")
        # Neutral
        self.assertEqual(self.extractor.detect_tone("ok.", has_emoji=False), "neutral")

    def test_response_type_detection(self):
        # Question
        self.assertEqual(self.extractor.detect_response_type("tu kab aa raha hai?", preceding_user_text=""), "question")
        # Greeting
        self.assertEqual(self.extractor.detect_response_type("hello", preceding_user_text=""), "greeting")
        # Acknowledgement
        self.assertEqual(self.extractor.detect_response_type("ha theek hai bhai", preceding_user_text=""), "acknowledgement")
        # Suggestion
        self.assertEqual(self.extractor.detect_response_type("chal sham ko dekh le", preceding_user_text=""), "suggestion")
        # Invitation
        self.assertEqual(self.extractor.detect_response_type("aaja bhai", preceding_user_text=""), "invitation")
        # Refusal
        self.assertEqual(self.extractor.detect_response_type("nahi aa paunga abhi", preceding_user_text=""), "refusal")
        # Answer
        self.assertEqual(self.extractor.detect_response_type("5 baje", preceding_user_text="kab aayega?"), "answer")
        # Default statement
        self.assertEqual(self.extractor.detect_response_type("mai laptop band kar raha tha abhi", preceding_user_text=""), "statement")

    def test_response_length_categorization(self):
        self.assertEqual(self.extractor.extract_response_length("ok").category, "very_short")
        self.assertEqual(self.extractor.extract_response_length("haan theek hai").category, "short")
        self.assertEqual(self.extractor.extract_response_length("aaj sham ko 6 baje ground pe milte hai sab log").category, "medium")
        long_text = "bhai sun aaj assignment submit karna tha par portal open nahi ho raha aur sir bol rahe hai kal subah tak ka time hai bas to jaldi se file bhej de"
        self.assertEqual(self.extractor.extract_response_length(long_text).category, "long")

    def test_emoji_extraction(self):
        text = "bhai kya scene hai 😂🔥"
        emojis = extract_emojis(text)
        self.assertEqual(emojis, ["😂", "🔥"])
        self.assertEqual(count_emojis(text), 2)
        self.assertEqual(len(extract_emojis("plain text")), 0)
        self.assertEqual(count_emojis("plain text"), 0)

    def test_slang_detection(self):
        self.assertTrue(self.extractor.has_slang("sahi hai yaar bsdk"))
        self.assertTrue(self.extractor.has_slang("bc kya kar raha hai"))
        self.assertTrue(self.extractor.has_slang("jugaad kar lenge"))
        self.assertFalse(self.extractor.has_slang("good morning sir"))

    def test_full_feature_extraction(self):
        feat = self.extractor.extract_features(
            text="haan bhai aaja valorant khele 😂",
            preceding_user_text="khelega kya aaj?"
        )
        self.assertIsInstance(feat, BehavioralFeatures)
        self.assertEqual(feat.language, "hinglish")
        self.assertTrue(feat.has_emoji)
        self.assertEqual(feat.emoji_count, 1)
        self.assertTrue(feat.has_slang)
        self.assertFalse(feat.is_question)
        self.assertIn("char_count", feat.response_length.to_dict())
        self.assertIn("word_count", feat.response_length.to_dict())
        self.assertIn("category", feat.response_length.to_dict())

        # Dictionary conversion
        d = feat.to_dict()
        self.assertEqual(d["language"], feat.language)
        self.assertEqual(d["tone"], feat.tone)
        self.assertEqual(d["response_type"], feat.response_type)


if __name__ == "__main__":
    unittest.main()
