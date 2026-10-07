"""
Persona Engine — ML Foundation Test Suite
Validates configuration loading, dataset operations, cleaning, validation,
ChatML example building, analysis, evaluation, and interface contracts.
"""

import sys
import unittest
from pathlib import Path

# Ensure ml root is on sys.path
ml_root = Path(__file__).resolve().parents[1]
if str(ml_root) not in sys.path:
    sys.path.insert(0, str(ml_root))

from src.analysis import DatasetAnalyzer, analyze_dataset
from src.annotation import AnnotationValidator, MessageAnnotation, create_annotated_conversation
from src.config import Config, load_config
from src.data import (
    DataCleaner,
    DatasetSplitter,
    DatasetValidator,
    clean_conversations,
    load_jsonl,
    split_conversations,
    validate_dataset_file,
)
from src.evaluation import (
    PersonaEvaluator,
    calculate_emoji_consistency,
    calculate_length_similarity,
    calculate_punctuation_alignment,
    calculate_vocabulary_overlap,
)
from src.inference import GenerationParams, PersonaGenerator
from src.preprocessing import (
    ConversationBuilder,
    TextNormalizer,
    build_training_examples,
    normalize_speaker,
    normalize_text,
)
from src.training import PersonaTrainer, TrainingConfig


class TestMLFoundation(unittest.TestCase):

    def test_01_config_loader(self):
        """Verify YAML configs load, merge, and support environment overrides."""
        cfg = load_config()
        self.assertIsNotNone(cfg)
        self.assertEqual(cfg.project.name, "persona-engine")
        self.assertTrue(Path(cfg.paths.sample_data).exists())

        # Test colab environment override
        colab_cfg = load_config(environment="colab")
        self.assertEqual(colab_cfg.environment, "colab")

    def test_02_sample_dataset_validation(self):
        """Verify sample dataset passes all schema and business validations."""
        sample_path = ml_root / "data" / "sample" / "sample_conversations.jsonl"
        self.assertTrue(sample_path.exists())

        validator = DatasetValidator()
        res = validator.validate_file(sample_path)
        self.assertTrue(res.is_valid, f"Validation failed with errors: {res.issues}")
        self.assertEqual(res.error_count, 0)
        self.assertGreater(res.total_conversations, 0)
        self.assertGreater(res.total_messages, 0)

    def test_03_validator_catches_invalid_data(self):
        """Verify validator flags invalid speaker labels and empty messages."""
        invalid_conversations = [
            {
                "conversation_id": "bad-001",
                "messages": [
                    {"speaker": "alien", "text": "Hello"},
                    {"speaker": "persona", "text": ""},
                ],
            }
        ]
        validator = DatasetValidator()
        res = validator.validate_dataset(invalid_conversations)
        self.assertFalse(res.is_valid)
        self.assertGreater(res.error_count, 0)

    def test_04_personality_preserving_cleaner(self):
        """Verify cleaner removes blank turns while preserving emojis and slang."""
        raw = [
            {
                "conversation_id": "test-clean",
                "messages": [
                    {"speaker": "user", "text": "   kya haal hai bhai? 😂   "},
                    {"speaker": "persona", "text": ""},  # should be filtered
                    {"speaker": "persona", "text": "sahi hu yaar! mast chal raha hai 🚀"},
                ],
            }
        ]
        cleaner = DataCleaner()
        cleaned = cleaner.clean_dataset(raw)
        self.assertEqual(len(cleaned), 1)
        msgs = cleaned[0]["messages"]
        self.assertEqual(len(msgs), 2)
        # Verify emojis and Hinglish are preserved
        self.assertIn("😂", msgs[0]["text"])
        self.assertIn("🚀", msgs[1]["text"])
        self.assertEqual(msgs[0]["text"], "kya haal hai bhai? 😂")

    def test_05_dataset_splitter(self):
        """Verify deterministic train/val/test splitting by conversation ID."""
        records = [
            {"conversation_id": f"conv-{i}", "messages": [{"speaker": "user", "text": "hi"}, {"speaker": "persona", "text": "yo"}]}
            for i in range(10)
        ]
        splitter = DatasetSplitter(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)
        splits = splitter.split(records)
        self.assertEqual(len(splits.train), 8)
        self.assertEqual(len(splits.validation), 1)
        self.assertEqual(len(splits.test), 1)

    def test_06_text_and_speaker_normalization(self):
        """Verify normalizer handles whitespace and maps speaker aliases."""
        normalizer = TextNormalizer()
        norm_text = normalizer.normalize("  Arre   bhai   kya  scene  hai?  ")
        self.assertEqual(norm_text, "Arre bhai kya scene hai?")

        self.assertEqual(normalize_speaker("me"), "persona")
        self.assertEqual(normalize_speaker("human"), "user")
        self.assertEqual(normalize_speaker("PERSONA"), "persona")

    def test_07_conversation_builder_chatml(self):
        """Verify chat formatted training example generation."""
        conv = {
            "conversation_id": "conv-test",
            "persona_id": "p1",
            "messages": [
                {"speaker": "user", "text": "Chai chalega?"},
                {"speaker": "persona", "text": "Haan bilkul! tapri pe mil ☕"},
            ],
        }
        builder = ConversationBuilder(chat_template="chatml")
        examples = builder.build_examples_from_conversation(conv)
        self.assertEqual(len(examples), 1)
        ex = examples[0]
        self.assertIn("<|im_start|>user\nChai chalega?<|im_end|>", ex["text"])
        self.assertIn("<|im_start|>assistant\nHaan bilkul! tapri pe mil ☕<|im_end|>", ex["text"])
        self.assertEqual(ex["conversation_id"], "conv-test")

    def test_08_annotation_models_and_validation(self):
        """Verify annotation models and validation against annotation schema."""
        ann = MessageAnnotation(
            topic="chai",
            intent="joke",
            emotion="happy",
            tone="playful",
            humor=True,
            sarcasm=False,
            language="hinglish",
        )
        conv = create_annotated_conversation(
            conversation_id="ann-001",
            messages=[
                {"speaker": "user", "text": "Chai?"},
                {"speaker": "persona", "text": "Haan ☕", "annotations": ann.to_dict()},
            ],
        )
        validator = AnnotationValidator()
        self.assertTrue(validator.validate(conv))

    def test_09_dataset_analyzer(self):
        """Verify analyzer produces correct metrics on sample conversations."""
        sample_path = ml_root / "data" / "sample" / "sample_conversations.jsonl"
        analyzer = DatasetAnalyzer()
        report = analyzer.analyze_file(sample_path)
        self.assertEqual(report.total_conversations, 120)
        self.assertEqual(report.total_messages, 478)
        self.assertGreater(report.emoji_total_count, 0)
        self.assertGreater(report.hinglish_ratio, 0.4)
        self.assertIn("naata", report.messages_by_speaker)
        self.assertIn("vivek", report.messages_by_speaker)

    def test_10_evaluation_metrics_and_evaluator(self):
        """Verify evaluation metric calculations and evaluator reporting."""
        len_sim = calculate_length_similarity("kuch nahi bhai", "kuch nahi bhai chill")
        self.assertGreater(len_sim, 0.5)

        vocab_sim = calculate_vocabulary_overlap("chai tapri pe", "tapri pe chai")
        self.assertEqual(vocab_sim, 1.0)

        emoji_cons = calculate_emoji_consistency("kuch nahi 😂", "mast hai 😎")
        self.assertEqual(emoji_cons, 1.0)

        evaluator = PersonaEvaluator()
        report = evaluator.evaluate([
            {
                "prompt": "chai chalega?",
                "generated": "haan bhai chalte hai ☕",
                "reference": "haan tapri pe milte hai ☕",
            }
        ])
        self.assertEqual(report.total_evaluated, 1)
        self.assertGreater(report.overall_style_score, 0.0)

    def test_11_persona_generator_interface(self):
        """Verify generator lifecycle: load, generate, unload."""
        generator = PersonaGenerator()
        generator.load_model(mock=True)
        self.assertTrue(generator.is_model_loaded)

        resp = generator.generate([{"speaker": "user", "text": "hello"}])
        self.assertIsInstance(resp, str)
        self.assertIn("Persona Model Response Placeholder", resp)

        generator.unload_model()
        self.assertFalse(generator.is_model_loaded)

    def test_12_persona_trainer_contract(self):
        """Verify trainer dataset preparation and dry-run lifecycle."""
        trainer = PersonaTrainer()
        sample_path = ml_root / "data" / "sample" / "sample_conversations.jsonl"
        status = trainer.prepare_dataset(sample_path)
        self.assertEqual(status["status"], "READY")
        self.assertGreater(status["num_examples"], 0)

        res = trainer.train(dry_run=True)
        self.assertEqual(res["status"], "DRY_RUN_PASSED")


if __name__ == "__main__":
    unittest.main()
