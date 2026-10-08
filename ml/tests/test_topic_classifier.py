#!/usr/bin/env python3
"""
Unit Tests for Stage 5 Deterministic Topic Classifier.
Tests keyword-based classification across all 9 deterministic categories:
gaming, college, movies, technology, plans, casual_chat, sports, social, other.
Also tests context-aware topic resolution for terse utterances.
"""

from pathlib import Path
import sys
import unittest

# Ensure ml root is in sys.path
ml_root = Path(__file__).resolve().parents[1]
if str(ml_root) not in sys.path:
    sys.path.insert(0, str(ml_root))

from src.preprocessing.topic_classifier import (
    DeterministicTopicClassifier,
    TopicClassification,
)


class TestDeterministicTopicClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = DeterministicTopicClassifier()

    def test_topic_gaming(self):
        result = self.classifier.classify("aaja bhai valorant khele discord pe")
        self.assertEqual(result.primary_topic, "gaming")
        self.assertGreater(result.confidence, 0.4)

    def test_topic_college(self):
        result = self.classifier.classify("dean sir ne assignment submit karne bola hai hostel me")
        self.assertEqual(result.primary_topic, "college")

    def test_topic_movies(self):
        result = self.classifier.classify("bhai stree 2 ka trailer dekha kya mast movie thi")
        self.assertEqual(result.primary_topic, "movies")

    def test_topic_technology(self):
        result = self.classifier.classify("mera laptop update ho raha hai windows reinstall kar raha")
        self.assertEqual(result.primary_topic, "technology")

    def test_topic_plans(self):
        result = self.classifier.classify("kal subah 8 baje milte hai station pe chalte hai")
        self.assertEqual(result.primary_topic, "plans")

    def test_topic_casual_chat(self):
        result = self.classifier.classify("theek hai bhai baad me baat karte hai")
        self.assertEqual(result.primary_topic, "casual_chat")

    def test_topic_sports(self):
        result = self.classifier.classify("aaj kohli ne century maari kya india match jeet gayi")
        self.assertEqual(result.primary_topic, "sports")

    def test_topic_social(self):
        result = self.classifier.classify("happy birthday bhai party kab de raha hai")
        self.assertEqual(result.primary_topic, "social")

    def test_topic_other(self):
        # Arbitrary abstract text not matching any specific colloquial category
        result = self.classifier.classify("quantum entanglement occurs when subatomic particles interact closely")
        self.assertEqual(result.primary_topic, "other")

    def test_context_awareness_for_terse_responses(self):
        # Terse response "Aaya" would be casual_chat/plans on its own,
        # but with gaming context "Valorant khelega kya aaja", gaming must win.
        result = self.classifier.classify(
            text="Aaya",
            context_text="user: Valorant khelega kya bhai aaja lobby me"
        )
        self.assertEqual(result.primary_topic, "gaming")

    def test_output_structure_and_dict(self):
        result = self.classifier.classify("bgmi khele?")
        self.assertIsInstance(result, TopicClassification)
        self.assertIn("gaming", result.scores)
        self.assertIn("college", result.scores)
        d = result.to_dict()
        self.assertEqual(d["primary_topic"], "gaming")
        self.assertIsInstance(d["confidence"], float)
        self.assertIsInstance(d["scores"], dict)

    def test_context_weighting_game_aaja(self):
        # Target has "Aaja" (movement/plans keyword), but preceding prompt has "Game aaja"
        # Preceding context + topic evidence should overcome target "aaja"
        result = self.classifier.classify(
            text="Aaja",
            context_text="user: Game aaja",
            preceding_user_text="Game aaja"
        )
        self.assertEqual(result.primary_topic, "gaming")

    def test_gaming_room_code_hex(self):
        result = self.classifier.classify("Ruk dobara banata hu 2628a4")
        self.assertEqual(result.primary_topic, "gaming")

    def test_gaming_kills_and_bande_maare(self):
        result = self.classifier.classify(
            text="Pil bhi gya",
            context_text="Kitne bande maare bhai",
            preceding_user_text="Kitne bande maare bhai"
        )
        self.assertEqual(result.primary_topic, "gaming")

    def test_college_physics_newton(self):
        result = self.classifier.classify("Maa chudaye newton physics class test")
        self.assertEqual(result.primary_topic, "college")


if __name__ == "__main__":
    unittest.main()
