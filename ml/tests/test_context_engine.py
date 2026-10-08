"""
Unit tests for Persona Engine Context Engine (Stage 6B).
Verifies deterministic runtime context analysis across test scenarios A through I.
"""

from pathlib import Path
import sys
import unittest

# Ensure repository root and ml root in sys.path
ml_root = Path(__file__).resolve().parents[1]
repo_root = ml_root.parent
for p in [str(repo_root), str(ml_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from ml.src.context.context_analyzer import ContextAnalyzer
from ml.src.context.context_schema import (
    AmbiguityLevel,
    ConversationState,
    TopicCategory,
    UserIntent,
)


class TestContextEngine(unittest.TestCase):
    """Test suite covering Scenarios A through I for Context Engine."""

    def test_scenario_a_gaming(self):
        """A. Gaming: 'bgmi khelega?' -> topic=gaming, intent=invitation, ambiguity=medium."""
        messages = [
            {"role": "user", "content": "bgmi khelega?"}
        ]
        analysis = ContextAnalyzer.analyze(messages)
        self.assertEqual(analysis.topic, TopicCategory.GAMING)
        self.assertIn(analysis.user_intent, [UserIntent.INVITATION, UserIntent.QUESTION])
        self.assertIn(analysis.ambiguity, [AmbiguityLevel.LOW, AmbiguityLevel.MEDIUM])
        self.assertEqual(analysis.context_depth, 1)

    def test_scenario_b_assignment_to_gaming(self):
        """B. Assignment -> Gaming: contextual transition with activity extraction."""
        messages = [
            {"role": "user", "content": "free hai kya?"},
            {"role": "assistant", "content": "assignment submit kar raha tha"},
            {"role": "user", "content": "Khelega?"}
        ]
        analysis = ContextAnalyzer.analyze(messages)
        self.assertEqual(analysis.topic, TopicCategory.GAMING)
        self.assertIsNotNone(analysis.recent_activity)
        self.assertIn("assignment", analysis.recent_activity.lower())
        self.assertNotEqual(analysis.ambiguity, AmbiguityLevel.LOW)
        self.assertEqual(analysis.context_depth, 3)

    def test_scenario_c_canteen(self):
        """C. Canteen: 'canteen me milte hai?' -> 'Aaja' -> plans/social, not generic override."""
        messages = [
            {"role": "user", "content": "canteen me milte hai?"},
            {"role": "assistant", "content": "5 min me pohochta hu"},
            {"role": "user", "content": "Aaja"}
        ]
        analysis = ContextAnalyzer.analyze(messages)
        self.assertIn(analysis.topic, [TopicCategory.PLANS, TopicCategory.SOCIAL])
        self.assertIsNotNone(analysis.recent_activity)
        self.assertIn("canteen", analysis.recent_activity.lower())
        self.assertEqual(analysis.ambiguity, AmbiguityLevel.HIGH)

    def test_scenario_d_technology(self):
        """D. Technology: laptop specs and battery backup -> topic=technology."""
        messages = [
            {"role": "user", "content": "acer nitro le liya"},
            {"role": "assistant", "content": "Sahi hai"},
            {"role": "user", "content": "battery backup kaisa hai?"}
        ]
        analysis = ContextAnalyzer.analyze(messages)
        self.assertEqual(analysis.topic, TopicCategory.TECHNOLOGY)
        self.assertEqual(analysis.user_intent, UserIntent.INFORMATION)
        self.assertEqual(analysis.ambiguity, AmbiguityLevel.LOW)

    def test_scenario_e_college(self):
        """E. College: attendance and dean inquiry -> topic=college."""
        messages = [
            {"role": "user", "content": "kal attendance kitni hai?"},
            {"role": "assistant", "content": "pata nahi"},
            {"role": "user", "content": "Dean kuch bola?"}
        ]
        analysis = ContextAnalyzer.analyze(messages)
        self.assertEqual(analysis.topic, TopicCategory.COLLEGE)
        self.assertEqual(analysis.user_intent, UserIntent.QUESTION)

    def test_scenario_f_ambiguous_standalone(self):
        """F. Ambiguous standalone: 'Aaja' -> ambiguity=high."""
        messages = [
            {"role": "user", "content": "Aaja"}
        ]
        analysis = ContextAnalyzer.analyze(messages)
        self.assertEqual(analysis.ambiguity, AmbiguityLevel.HIGH)
        self.assertEqual(analysis.context_depth, 1)
        self.assertIsNone(analysis.recent_activity)

    def test_scenario_g_ambiguous_contextual(self):
        """G. Ambiguous contextual: 'game khelega?' -> 'haan' -> 'Aaja' -> topic=gaming."""
        messages = [
            {"role": "user", "content": "game khelega?"},
            {"role": "assistant", "content": "haan"},
            {"role": "user", "content": "Aaja"}
        ]
        analysis = ContextAnalyzer.analyze(messages)
        self.assertEqual(analysis.topic, TopicCategory.GAMING)
        self.assertEqual(analysis.ambiguity, AmbiguityLevel.HIGH)

    def test_scenario_h_context_depth(self):
        """H. Context depth: verifies accurate count for 1, 3, 7, and 11 messages."""
        for count in [1, 3, 7, 11]:
            dummy_messages = [
                {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
                for i in range(count)
            ]
            analysis = ContextAnalyzer.analyze(dummy_messages)
            self.assertEqual(analysis.context_depth, count)

    def test_scenario_i_speaker_alternation(self):
        """I. Speaker alternation: checks strictly alternating vs consecutive speakers."""
        # Strictly alternating (user -> assistant -> user -> assistant)
        alt_msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "how are you?"},
            {"role": "assistant", "content": "good"}
        ]
        alt_analysis = ContextAnalyzer.analyze(alt_msgs)
        self.assertEqual(alt_analysis.speaker_alternation_rate, 1.0)

        # Consecutive identical speakers (user -> user -> user)
        consec_msgs = [
            {"role": "user", "content": "hi"},
            {"role": "user", "content": "sun na"},
            {"role": "user", "content": "kaha hai?"}
        ]
        consec_analysis = ContextAnalyzer.analyze(consec_msgs)
        self.assertEqual(consec_analysis.speaker_alternation_rate, 0.0)

    # =========================================================================
    # Stage 6F: Scenarios A through W for Intent & Dialogue-Act Refinement
    # =========================================================================

    def test_6f_scenario_a_direct_question(self):
        """A. Direct questions: 'Kaise karu?' -> question."""
        analysis = ContextAnalyzer.analyze([{"role": "user", "content": "Kaise karu?"}])
        self.assertEqual(analysis.user_intent, UserIntent.QUESTION)

    def test_6f_scenario_b_question_vs_invitation(self):
        """B. Question vs invitation: 'Khelega?' -> invitation."""
        analysis = ContextAnalyzer.analyze([{"role": "user", "content": "Khelega?"}])
        self.assertEqual(analysis.user_intent, UserIntent.INVITATION)

    def test_6f_scenario_c_invitation(self):
        """C. Invitation: 'Canteen chalega?' -> invitation."""
        analysis = ContextAnalyzer.analyze([{"role": "user", "content": "Canteen chalega?"}])
        self.assertEqual(analysis.user_intent, UserIntent.INVITATION)

    def test_6f_scenario_d_embedded_invitation(self):
        """D. Embedded invitation: 'bhai game khelega to aa ja' -> invitation."""
        analysis = ContextAnalyzer.analyze([{"role": "user", "content": "bhai game khelega to aa ja"}])
        self.assertEqual(analysis.user_intent, UserIntent.INVITATION)

    def test_6f_scenario_e_multi_sentence_invitation(self):
        """E. Multi-sentence invitation: 'Assignment kar raha hu. Game khelega?' -> invitation."""
        analysis = ContextAnalyzer.analyze([{"role": "user", "content": "Assignment kar raha hu. Game khelega?"}])
        self.assertEqual(analysis.user_intent, UserIntent.INVITATION)

    def test_6f_scenario_f_information(self):
        """F. Information: 'Dean ne attendance ka notice nikala hai.' -> information."""
        analysis = ContextAnalyzer.analyze([{"role": "user", "content": "Dean ne attendance ka notice nikala hai."}])
        self.assertEqual(analysis.user_intent, UserIntent.INFORMATION)

    def test_6f_scenario_g_answer_request(self):
        """G. Answer request: 'Iska solution kya hai?' -> answer_request."""
        analysis = ContextAnalyzer.analyze([{"role": "user", "content": "Iska solution kya hai?"}])
        self.assertEqual(analysis.user_intent, UserIntent.ANSWER_REQUEST)

    def test_6f_scenario_h_planning(self):
        """H. Planning: 'Kal 5 baje milte hain.' -> planning."""
        analysis = ContextAnalyzer.analyze([{"role": "user", "content": "Kal 5 baje milte hain."}])
        self.assertEqual(analysis.user_intent, UserIntent.PLANNING)

    def test_6f_scenario_i_agreement(self):
        """I. Agreement: 'Haan bhai.' -> agreement."""
        analysis = ContextAnalyzer.analyze([{"role": "user", "content": "Haan bhai."}])
        self.assertEqual(analysis.user_intent, UserIntent.AGREEMENT)

    def test_6f_scenario_j_contextual_agreement(self):
        """J. Contextual agreement: 'Kal 5 baje milte hain.' -> 'Thik' -> agreement."""
        messages = [
            {"role": "assistant", "content": "Kal 5 baje milte hain."},
            {"role": "user", "content": "Thik"}
        ]
        analysis = ContextAnalyzer.analyze(messages)
        self.assertEqual(analysis.user_intent, UserIntent.AGREEMENT)

    def test_6f_scenario_k_disagreement(self):
        """K. Disagreement: 'Nhi bhai.' -> disagreement."""
        analysis = ContextAnalyzer.analyze([{"role": "user", "content": "Nhi bhai."}])
        self.assertEqual(analysis.user_intent, UserIntent.DISAGREEMENT)

    def test_6f_scenario_l_reaction(self):
        """L. Reaction: '😂' -> reaction."""
        analysis = ContextAnalyzer.analyze([{"role": "user", "content": "😂"}])
        self.assertEqual(analysis.user_intent, UserIntent.REACTION)

    def test_6f_scenario_m_casual_chat(self):
        """M. Casual chat: 'Kya haal hai?' -> casual_chat."""
        analysis = ContextAnalyzer.analyze([{"role": "user", "content": "Kya haal hai?"}])
        self.assertEqual(analysis.user_intent, UserIntent.CASUAL_CHAT)

    def test_6f_scenario_n_ambiguous_standalone_kya(self):
        """N. Ambiguous standalone: 'Kya' -> unknown."""
        analysis = ContextAnalyzer.analyze([{"role": "user", "content": "Kya"}])
        self.assertEqual(analysis.user_intent, UserIntent.UNKNOWN)

    def test_6f_scenario_o_ambiguous_standalone_aaja(self):
        """O. Ambiguous standalone: 'Aaja' -> ambiguity=HIGH, intent=invitation."""
        analysis = ContextAnalyzer.analyze([{"role": "user", "content": "Aaja"}])
        self.assertEqual(analysis.ambiguity, AmbiguityLevel.HIGH)
        self.assertEqual(analysis.user_intent, UserIntent.INVITATION)

    def test_6f_scenario_p_contextual_aaja_gaming(self):
        """P. Contextual Aaja: Gaming context -> topic=gaming, intent=invitation."""
        messages = [
            {"role": "user", "content": "bgmi khelenge?"},
            {"role": "assistant", "content": "haan lobby me hu"},
            {"role": "user", "content": "Aaja"}
        ]
        analysis = ContextAnalyzer.analyze(messages)
        self.assertEqual(analysis.topic, TopicCategory.GAMING)
        self.assertEqual(analysis.user_intent, UserIntent.INVITATION)

    def test_6f_scenario_q_contextual_aaja_canteen(self):
        """Q. Contextual Aaja: Canteen context -> topic in plans/social, intent=invitation."""
        messages = [
            {"role": "user", "content": "canteen chale?"},
            {"role": "assistant", "content": "haan chalte hain"},
            {"role": "user", "content": "Aaja"}
        ]
        analysis = ContextAnalyzer.analyze(messages)
        self.assertIn(analysis.topic, [TopicCategory.PLANS, TopicCategory.SOCIAL])
        self.assertEqual(analysis.user_intent, UserIntent.INVITATION)

    def test_6f_scenario_r_contextual_aaja_college(self):
        """R. Contextual Aaja: College context -> topic=college, intent=invitation."""
        messages = [
            {"role": "user", "content": "assignment complete kare?"},
            {"role": "assistant", "content": "haan library aa raha hu"},
            {"role": "user", "content": "Aaja"}
        ]
        analysis = ContextAnalyzer.analyze(messages)
        self.assertEqual(analysis.topic, TopicCategory.COLLEGE)
        self.assertEqual(analysis.user_intent, UserIntent.INVITATION)

    def test_6f_scenario_s_multi_sentence_question(self):
        """S. Multi-sentence question: 'Dean ne notice nikala hai. Attendance ka kya scene hai?' -> question."""
        analysis = ContextAnalyzer.analyze([
            {"role": "user", "content": "Dean ne notice nikala hai. Attendance ka kya scene hai?"}
        ])
        self.assertEqual(analysis.user_intent, UserIntent.QUESTION)

    def test_6f_scenario_t_multi_sentence_information(self):
        """T. Multi-sentence information: 'Battery drain ho rahi hai. Kal service center jana padega.' -> information."""
        analysis = ContextAnalyzer.analyze([
            {"role": "user", "content": "Battery drain ho rahi hai. Kal service center jana padega."}
        ])
        self.assertEqual(analysis.user_intent, UserIntent.INFORMATION)

    def test_6f_scenario_u_speaker_alternation(self):
        """U. Speaker alternation: Ensure user intent evaluates the user utterance, not assistant."""
        messages = [
            {"role": "user", "content": "Kaise karu?"},
            {"role": "assistant", "content": "bata raha hu wait kar"}
        ]
        analysis = ContextAnalyzer.analyze(messages)
        self.assertEqual(analysis.user_intent, UserIntent.QUESTION)
        self.assertEqual(analysis.last_user_message, "Kaise karu?")

    def test_6f_scenario_v_unknown(self):
        """V. Unknown: Genuinely unsupported gibberish utterance -> unknown."""
        analysis = ContextAnalyzer.analyze([{"role": "user", "content": "zzzqwx 12389"}])
        self.assertEqual(analysis.user_intent, UserIntent.UNKNOWN)

    def test_6f_scenario_w_determinism(self):
        """W. Determinism: Identical inputs yield identical outputs over repeated runs."""
        messages = [
            {"role": "user", "content": "Assignment kar raha hu. Game khelega?"}
        ]
        outputs = [ContextAnalyzer.analyze(messages).user_intent for _ in range(25)]
        self.assertTrue(all(out == UserIntent.INVITATION for out in outputs))


if __name__ == "__main__":
    unittest.main()
