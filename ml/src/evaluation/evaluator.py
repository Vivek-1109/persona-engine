"""
Persona Engine — Evaluation Engine
Aggregates and computes behavioral and stylistic evaluation metrics over a test suite.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .metrics import (
    calculate_emoji_consistency,
    calculate_length_similarity,
    calculate_punctuation_alignment,
    calculate_vocabulary_overlap,
)


@dataclass
class EvaluationSample:
    prompt: str
    generated_text: str
    reference_text: str
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class EvaluationReport:
    total_evaluated: int = 0
    mean_length_similarity: float = 0.0
    mean_vocabulary_overlap: float = 0.0
    mean_emoji_consistency: float = 0.0
    mean_punctuation_alignment: float = 0.0
    overall_style_score: float = 0.0
    samples: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary_text(self) -> str:
        lines = [
            "==================================================",
            "PERSONA ENGINE — EVALUATION REPORT",
            "==================================================",
            f"Samples Evaluated         : {self.total_evaluated}",
            f"Length Similarity         : {self.mean_length_similarity:.3f}",
            f"Vocabulary Overlap        : {self.mean_vocabulary_overlap:.3f}",
            f"Emoji Consistency         : {self.mean_emoji_consistency:.3f}",
            f"Punctuation Alignment     : {self.mean_punctuation_alignment:.3f}",
            "--------------------------------------------------",
            f"OVERALL STYLE SCORE       : {self.overall_style_score:.3f} / 1.000",
            "==================================================",
        ]
        return "\n".join(lines)


class PersonaEvaluator:
    """
    Evaluator to score generated model responses against reference persona messages.
    """

    def evaluate_pair(self, generated: str, reference: str) -> Dict[str, float]:
        """Calculates metric scores for a single generated vs reference pair."""
        len_sim = calculate_length_similarity(generated, reference)
        vocab_sim = calculate_vocabulary_overlap(generated, reference)
        emoji_cons = calculate_emoji_consistency(generated, reference)
        punct_align = calculate_punctuation_alignment(generated, reference)

        return {
            "length_similarity": len_sim,
            "vocabulary_overlap": vocab_sim,
            "emoji_consistency": emoji_cons,
            "punctuation_alignment": punct_align,
            "composite_score": round((len_sim + vocab_sim + emoji_cons + punct_align) / 4.0, 4),
        }

    def evaluate(
        self,
        pairs: List[Dict[str, str]],
    ) -> EvaluationReport:
        """
        Evaluates a list of dictionaries with 'generated' and 'reference' keys
        (and optional 'prompt' key).
        """
        if not pairs:
            return EvaluationReport()

        n = len(pairs)
        len_sims, vocab_sims, emoji_conss, punct_aligns = [], [], [], []
        sample_results: List[Dict[str, Any]] = []

        for p in pairs:
            gen = p.get("generated", "")
            ref = p.get("reference", "")
            prompt = p.get("prompt", "")

            m = self.evaluate_pair(gen, ref)
            len_sims.append(m["length_similarity"])
            vocab_sims.append(m["vocabulary_overlap"])
            emoji_conss.append(m["emoji_consistency"])
            punct_aligns.append(m["punctuation_alignment"])

            sample_results.append({
                "prompt": prompt,
                "generated": gen,
                "reference": ref,
                "metrics": m,
            })

        mean_len = round(sum(len_sims) / n, 4)
        mean_vocab = round(sum(vocab_sims) / n, 4)
        mean_emoji = round(sum(emoji_conss) / n, 4)
        mean_punct = round(sum(punct_aligns) / n, 4)
        overall = round((mean_len + mean_vocab + mean_emoji + mean_punct) / 4.0, 4)

        return EvaluationReport(
            total_evaluated=n,
            mean_length_similarity=mean_len,
            mean_vocabulary_overlap=mean_vocab,
            mean_emoji_consistency=mean_emoji,
            mean_punctuation_alignment=mean_punct,
            overall_style_score=overall,
            samples=sample_results,
        )

    def save_report(
        self,
        report: EvaluationReport,
        output_file: Union[str, Path],
    ) -> Path:
        """Saves evaluation report to JSON."""
        out = Path(output_file)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        return out
