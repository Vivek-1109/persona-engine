#!/usr/bin/env python3
"""
Unit Tests for Stage 5 Behavioral Feature Extraction.
Tests language detection, tone classification, response type taxonomy,
response length categorization, emoji analysis, and slang detection.
Includes regression tests for Unicode skin-tone emoji modifiers and context-based response typing.
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

    # =========================================================================
    # 7 TONE TESTS (Semantic Tone Classification)
    # =========================================================================
    def test_tone_short_casual_response(self):
        # Short response with casual peer particle 'bhai' must be casual, NOT neutral
        self.assertEqual(self.extractor.detect_tone("aaja bhai", has_emoji=False), "casual")

    def test_tone_short_neutral_response(self):
        # Short factual / numerical response without slang or affect is neutral
        self.assertEqual(self.extractor.detect_tone("2 baje", has_emoji=False), "neutral")
        self.assertEqual(self.extractor.detect_tone("ok.", has_emoji=False), "neutral")
        self.assertEqual(self.extractor.detect_tone("nahi", has_emoji=False), "neutral")

    def test_tone_teasing_response(self):
        self.assertEqual(self.extractor.detect_tone("are bsdk tu kab sudhrega", has_emoji=False), "teasing")

    def test_tone_humorous_response(self):
        self.assertEqual(self.extractor.detect_tone("bhai sach me lol xdd ye kya tha", has_emoji=False), "humorous")
        self.assertEqual(self.extractor.detect_tone("mast joke tha 😂", has_emoji=True), "humorous")

    def test_tone_uncertain_response(self):
        self.assertEqual(self.extractor.detect_tone("shayad kal aa saku pakka nahi pata", has_emoji=False), "uncertain")

    def test_tone_serious_response(self):
        self.assertEqual(self.extractor.detect_tone("urgent hospital emergency case hai", has_emoji=False), "serious")
        self.assertEqual(self.extractor.detect_tone("deadline aaj raat ki hai zaroori", has_emoji=False), "serious")

    def test_tone_supportive_response(self):
        self.assertEqual(self.extractor.detect_tone("tension mat le ho jayega sab theek", has_emoji=False), "supportive")
        self.assertEqual(self.extractor.detect_tone("all the best bhai chill kar", has_emoji=False), "supportive")

    # =========================================================================
    # 10 RESPONSE TYPE TESTS (Conversational Function & Context)
    # =========================================================================
    def test_response_type_short_answer(self):
        # 1. Short answer following an inquiry
        self.assertEqual(
            self.extractor.detect_response_type("5 baje", preceding_user_text="kab aayega?"),
            "answer",
        )

    def test_response_type_long_answer(self):
        # 2. Long answer following an inquiry (must NOT be misclassified as statement)
        long_ans = "portal pe jaake pdf upload kar dena sir ne date extend kar di hai"
        self.assertEqual(
            self.extractor.detect_response_type(long_ans, preceding_user_text="assignment kaise submit kare?"),
            "answer",
        )

    def test_response_type_long_statement(self):
        # 3. Long statement where preceding turn is NOT an inquiry
        long_stmt = "haan mai to rajai se nikal hi nahi raha aur subah se chai pee raha hu"
        self.assertEqual(
            self.extractor.detect_response_type(long_stmt, preceding_user_text="aaj bahut thand hai"),
            "statement",
        )

    def test_response_type_short_statement(self):
        # 4. Short statement where preceding turn is NOT an inquiry (must NOT be misclassified as answer)
        self.assertEqual(
            self.extractor.detect_response_type("mai offline tha", preceding_user_text="kal match dekha"),
            "statement",
        )

    def test_response_type_question(self):
        # 5. Question response
        self.assertEqual(
            self.extractor.detect_response_type("tu kab aa raha hai?", preceding_user_text=""),
            "question",
        )

    def test_response_type_acknowledgement(self):
        # 6. Acknowledgement
        self.assertEqual(
            self.extractor.detect_response_type("ok theek hai bhai", preceding_user_text=""),
            "acknowledgement",
        )

    def test_response_type_refusal(self):
        # 7. Refusal
        self.assertEqual(
            self.extractor.detect_response_type("nahi aa paunga abhi", preceding_user_text=""),
            "refusal",
        )

    def test_response_type_invitation(self):
        # 8. Invitation
        self.assertEqual(
            self.extractor.detect_response_type("aaja discord pe", preceding_user_text=""),
            "invitation",
        )

    def test_response_type_suggestion(self):
        # 9. Suggestion
        self.assertEqual(
            self.extractor.detect_response_type("link se download kar le", preceding_user_text=""),
            "suggestion",
        )

    def test_response_type_reaction(self):
        # 10. Reaction
        self.assertEqual(
            self.extractor.detect_response_type("😂", preceding_user_text=""),
            "reaction",
        )
        self.assertEqual(
            self.extractor.detect_response_type("...", preceding_user_text=""),
            "reaction",
        )

    # =========================================================================
    # EMOJI TESTS & SKIN-TONE MODIFIER REGRESSION TESTS
    # =========================================================================
    def test_emoji_skin_tone_modifiers_coalescing(self):
        # Regression test for skin-tone modifiers (U+1F3FB .. U+1F3FF)
        self.assertEqual(count_emojis("👍🏻"), 1)
        self.assertEqual(count_emojis("👍🏽"), 1)
        self.assertEqual(count_emojis("😂"), 1)
        self.assertEqual(count_emojis("👽"), 1)
        self.assertEqual(count_emojis("🗿"), 1)

        # Multi-emoji string with skin-tone modifier
        multi_text = "Nahi aa paunga 👍🏻 😂 🔥"
        emojis = extract_emojis(multi_text)
        self.assertEqual(len(emojis), 3)
        self.assertEqual(count_emojis(multi_text), 3)

        # Plain text
        self.assertEqual(count_emojis("plain text"), 0)
        self.assertEqual(len(extract_emojis("plain text")), 0)

    def test_response_length_categorization(self):
        self.assertEqual(self.extractor.extract_response_length("ok").category, "very_short")
        self.assertEqual(self.extractor.extract_response_length("haan theek hai").category, "short")
        self.assertEqual(self.extractor.extract_response_length("aaj sham ko 6 baje ground pe milte hai sab log").category, "medium")
        long_text = "bhai sun aaj assignment submit karna tha par portal open nahi ho raha aur sir bol rahe hai kal subah tak ka time hai bas to jaldi se file bhej de"
        self.assertEqual(self.extractor.extract_response_length(long_text).category, "long")

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

        d = feat.to_dict()
        self.assertEqual(d["language"], feat.language)
        self.assertEqual(d["tone"], feat.tone)
        self.assertEqual(d["response_type"], feat.response_type)


if __name__ == "__main__":
    unittest.main()
