#!/usr/bin/env python3
"""
Persona Engine — Stage 3 Data Pipeline Unit Tests
Tests parsing, multiline message handling, speaker detection, conversation segmentation,
candidate extraction, behavioral annotation, and zero-leakage splitting using synthetic fixtures.
"""

from pathlib import Path
import sys
import unittest

# Ensure ml root is in sys.path
ml_root = Path(__file__).resolve().parents[1]
if str(ml_root) not in sys.path:
    sys.path.insert(0, str(ml_root))

from src.data.whatsapp_parser import WhatsAppParser, ParsedMessage
from src.data.conversation_segmenter import ConversationSegmenter
from src.data.candidate_extractor import CandidateExtractor
from src.data.behavioral_annotator import BehavioralAnnotator
from src.data.conversation_splitter import ConversationSplitter


class TestWhatsAppParser(unittest.TestCase):
    """Tests for WhatsAppParser using synthetic fixtures."""

    def test_single_and_multiline_parsing(self):
        sample_export = [
            "2/14/25, 5:58 PM - Messages and calls are end-to-end encrypted. *Learn more*\n",
            "2/14/25, 6:00 PM - UserA: Hello there\n",
            "Line two of message from UserA\n",
            "Line three of message from UserA\n",
            "2/14/25, 6:02 PM - PersonaX: Hi! How are you?\n",
            "2/14/25, 6:05 PM - UserA: <Media omitted>\n",
            "2/14/25, 6:06 PM - PersonaX: This message was deleted\n",
            "2/14/25, 6:07 PM - UserA: https://example.com\n",
        ]

        parser = WhatsAppParser(persona_speaker="PersonaX")
        messages = parser.parse_lines(sample_export)

        self.assertEqual(len(messages), 6)
        self.assertEqual(len(parser.unparsed_lines), 0)

        # System message
        self.assertEqual(messages[0].message_type, "system")
        self.assertEqual(messages[0].speaker, "SYSTEM")

        # Multiline text message
        self.assertEqual(messages[1].speaker, "UserA")
        self.assertEqual(messages[1].message_type, "text")
        self.assertIn("Line two of message from UserA", messages[1].text)
        self.assertIn("Line three of message from UserA", messages[1].text)

        # Persona message
        self.assertEqual(messages[2].speaker, "PersonaX")
        self.assertTrue(messages[2].is_persona)
        self.assertEqual(messages[2].message_type, "text")

        # Media message
        self.assertEqual(messages[3].message_type, "media")

        # Deleted message
        self.assertEqual(messages[4].message_type, "deleted")

        # URL only message
        self.assertEqual(messages[5].message_type, "url")

    def test_narrow_nobrk_space_parsing(self):
        # WhatsApp iOS exports often have \u202f before PM
        line = "2/14/25, 5:58\u202fPM - Alice: Test message"
        parser = WhatsAppParser(persona_speaker="Alice")
        msgs = parser.parse_lines([line])
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0].speaker, "Alice")
        self.assertEqual(msgs[0].text, "Test message")


class TestConversationSegmenter(unittest.TestCase):
    """Tests for conversation segmentation and turn merging."""

    def test_inactivity_gap_segmentation(self):
        msgs = [
            ParsedMessage(
                timestamp="2025-02-14 10:00:00",
                raw_timestamp="2/14/25, 10:00 AM",
                speaker="Partner",
                text="Morning!",
                message_type="text",
                line_number=1,
            ),
            ParsedMessage(
                timestamp="2025-02-14 10:05:00",
                raw_timestamp="2/14/25, 10:05 AM",
                speaker="Persona",
                text="Morning bro",
                message_type="text",
                line_number=2,
                is_persona=True,
            ),
            # 3 hour gap (180 mins) -> should start second conversation
            ParsedMessage(
                timestamp="2025-02-14 13:05:00",
                raw_timestamp="2/14/25, 1:05 PM",
                speaker="Partner",
                text="Lunch?",
                message_type="text",
                line_number=3,
            ),
            ParsedMessage(
                timestamp="2025-02-14 13:06:00",
                raw_timestamp="2/14/25, 1:06 PM",
                speaker="Persona",
                text="Aaja canteen",
                message_type="text",
                line_number=4,
                is_persona=True,
            ),
        ]

        segmenter = ConversationSegmenter(persona_speaker="Persona", gap_minutes=120)
        sessions = segmenter.segment(msgs)

        self.assertEqual(len(sessions), 2)
        self.assertEqual(sessions[0].conversation_id, "conv_0001")
        self.assertEqual(sessions[1].conversation_id, "conv_0002")

    def test_burst_merging(self):
        msgs = [
            ParsedMessage(
                timestamp="2025-02-14 10:00:00",
                raw_timestamp="2/14/25, 10:00 AM",
                speaker="Partner",
                text="Kya kar raha?",
                message_type="text",
                line_number=1,
            ),
            ParsedMessage(
                timestamp="2025-02-14 10:01:00",
                raw_timestamp="2/14/25, 10:01 AM",
                speaker="Partner",
                text="Free hai?",
                message_type="text",
                line_number=2,
            ),
            ParsedMessage(
                timestamp="2025-02-14 10:02:00",
                raw_timestamp="2/14/25, 10:02 AM",
                speaker="Persona",
                text="Haa",
                message_type="text",
                line_number=3,
                is_persona=True,
            ),
        ]

        segmenter = ConversationSegmenter(persona_speaker="Persona", gap_minutes=120)
        sessions = segmenter.segment(msgs)

        self.assertEqual(len(sessions), 1)
        turns = sessions[0].turns
        self.assertEqual(len(turns), 2)
        self.assertEqual(turns[0].role, "user")
        self.assertEqual(turns[0].text, "Kya kar raha?\nFree hai?")
        self.assertEqual(turns[1].role, "assistant")
        self.assertEqual(turns[1].text, "Haa")


class TestCandidateExtractor(unittest.TestCase):
    """Tests for candidate extraction and context generation."""

    def test_context_window_extraction(self):
        msgs = [
            ParsedMessage(
                timestamp="2025-02-14 10:00:00",
                raw_timestamp="",
                speaker="Partner",
                text="Msg 1",
                message_type="text",
                line_number=1,
            ),
            ParsedMessage(
                timestamp="2025-02-14 10:01:00",
                raw_timestamp="",
                speaker="Persona",
                text="Reply 1",
                message_type="text",
                line_number=2,
                is_persona=True,
            ),
            ParsedMessage(
                timestamp="2025-02-14 10:02:00",
                raw_timestamp="",
                speaker="Partner",
                text="Msg 2",
                message_type="text",
                line_number=3,
            ),
            ParsedMessage(
                timestamp="2025-02-14 10:03:00",
                raw_timestamp="",
                speaker="Persona",
                text="Reply 2",
                message_type="text",
                line_number=4,
                is_persona=True,
            ),
        ]

        segmenter = ConversationSegmenter(persona_speaker="Persona")
        sessions = segmenter.segment(msgs)

        extractor = CandidateExtractor(
            system_prompt="Test system prompt",
            context_turn_sizes=[1, 2],
            persona_name="Persona",
        )
        candidates = extractor.extract_all(sessions)

        # For Reply 1: only 1-turn context (Msg 1)
        # For Reply 2: 1-turn (Msg 2) and 2-turn (Msg 1 -> Reply 1 -> Msg 2)
        self.assertEqual(len(candidates), 3)

        # Verify last message is ALWAYS assistant
        for c in candidates:
            self.assertEqual(c["messages"][-1]["role"], "assistant")
            self.assertEqual(c["messages"][0]["role"], "system")
            self.assertEqual(c["messages"][1]["role"], "user")


class TestBehavioralAnnotator(unittest.TestCase):
    """Tests for deterministic annotation heuristics."""

    def setUp(self):
        self.annotator = BehavioralAnnotator()

    def test_language_detection(self):
        self.assertEqual(self.annotator.detect_language("kya kar raha hai bhai"), "hinglish")
        self.assertEqual(self.annotator.detect_language("This is a simple english sentence"), "english")
        self.assertEqual(self.annotator.detect_language("नमस्ते आप कैसे हैं"), "hindi")

    def test_tone_detection(self):
        self.assertEqual(self.annotator.detect_tone("bhai comedy ho gyi 😂", has_emoji=True), "humorous")
        self.assertEqual(self.annotator.detect_tone("bsdk chup baith", has_emoji=False), "teasing")
        self.assertEqual(self.annotator.detect_tone("pata nhi shayad ho", has_emoji=False), "uncertain")

    def test_response_type_detection(self):
        self.assertEqual(self.annotator.detect_response_type("kya lagta hai?"), "question")
        self.assertEqual(self.annotator.detect_response_type("haan"), "acknowledgement")
        self.assertEqual(self.annotator.detect_response_type("nhi"), "refusal")
        self.assertEqual(self.annotator.detect_response_type("😂"), "reaction")

    def test_pii_flagging(self):
        is_pii, cats = self.annotator.detect_sensitive_pii("Call me on 9876543210")
        self.assertTrue(is_pii)
        self.assertIn("phone", cats)

        is_pii2, cats2 = self.annotator.detect_sensitive_pii("Email me at user@test.com")
        self.assertTrue(is_pii2)
        self.assertIn("email", cats2)

        is_clean, _ = self.annotator.detect_sensitive_pii("aaja room pe")
        self.assertFalse(is_clean)


class TestConversationSplitter(unittest.TestCase):
    """Tests for zero-leakage conversation splitting."""

    def test_leak_free_split(self):
        # Create synthetic candidates belonging to 10 distinct conversations
        examples = []
        for i in range(10):
            conv_id = f"conv_{i:04d}"
            for ex_idx in range(3):
                examples.append({
                    "messages": [
                        {"role": "user", "content": f"Q {ex_idx}"},
                        {"role": "assistant", "content": f"A {ex_idx}"},
                    ],
                    "metadata": {
                        "example_id": f"{conv_id}_ex{ex_idx}",
                        "conversation_id": conv_id,
                    },
                })

        splitter = ConversationSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)
        train_ex, val_ex, test_ex, stats = splitter.split(examples)

        self.assertTrue(stats.is_leak_free)
        self.assertEqual(stats.total_conversations, 10)
        self.assertEqual(len(train_ex) + len(val_ex) + len(test_ex), 30)

        train_convs = {ex["metadata"]["conversation_id"] for ex in train_ex}
        val_convs = {ex["metadata"]["conversation_id"] for ex in val_ex}
        test_convs = {ex["metadata"]["conversation_id"] for ex in test_ex}

        self.assertEqual(len(train_convs & val_convs), 0)
        self.assertEqual(len(train_convs & test_convs), 0)
        self.assertEqual(len(val_convs & test_convs), 0)


if __name__ == "__main__":
    unittest.main()
