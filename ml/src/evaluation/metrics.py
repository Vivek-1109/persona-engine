"""
Quantitative metrics engine for Stage 6E End-to-End Evaluation.
Calculates context accuracy, strategy accuracy, ambiguity handling, memory precision/recall/contamination,
persona fidelity proxies, length distributions, and component latency percentiles.
Preserves backwards-compatible foundation similarity metrics.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Set
import numpy as np

from ml.src.evaluation.evaluation_schema import (
    BenchmarkCategory,
    BenchmarkResult,
    ErrorType,
    EvaluationSummary,
)

EMOJI_REGEX = re.compile(
    r"[\U00010000-\U0010ffff\u2600-\u27bf\u2300-\u23ff\u2b50\u2b55\u200d\ufe0f]",
    flags=re.UNICODE,
)

HINGLISH_TOKENS = {
    "bhai", "yaar", "haan", "haa", "nahi", "nhi", "na", "kya", "kar", "raha",
    "rahi", "hu", "hai", "hain", "ho", "ka", "ke", "ki", "ko", "se", "me",
    "theek", "thik", "badiya", "sahi", "chal", "chalo", "aaja", "aao", "dekh",
    "batata", "bata", "scene", "bohot", "bahut", "kal", "aaj", "kuch"
}

SLANG_TOKENS = {
    "bhai", "yaar", "bro", "scene", "chill", "badiya", "mast", "sahi", "funda", "arre"
}


# =========================================================================
# BACKWARDS-COMPATIBLE FOUNDATION SIMILARITY METRICS
# =========================================================================

def _tokenize(text: str) -> Set[str]:
    return set(re.findall(r"\b\w+\b", text.lower()))


def calculate_length_similarity(gen_text: str, ref_text: str) -> float:
    """
    Measures how closely generated response length matches reference length.
    Returns float in range [0.0, 1.0], where 1.0 is exact match.
    Formula: min(len1, len2) / max(len1, len2)
    """
    l1 = len(gen_text.strip())
    l2 = len(ref_text.strip())
    if l1 == 0 and l2 == 0:
        return 1.0
    if l1 == 0 or l2 == 0:
        return 0.0
    return round(min(l1, l2) / max(l1, l2), 4)


def calculate_vocabulary_overlap(gen_text: str, ref_text: str) -> float:
    """
    Computes Jaccard word vocabulary similarity between generated and reference text.
    Returns float in range [0.0, 1.0].
    """
    tokens_gen = _tokenize(gen_text)
    tokens_ref = _tokenize(ref_text)
    if not tokens_gen and not tokens_ref:
        return 1.0
    if not tokens_gen or not tokens_ref:
        return 0.0
    intersection = tokens_gen.intersection(tokens_ref)
    union = tokens_gen.union(tokens_ref)
    return round(len(intersection) / len(union), 4)


def calculate_emoji_consistency(gen_text: str, ref_text: str) -> float:
    """
    Checks if emoji usage in generated response matches presence in reference response.
    Returns 1.0 if both have emojis or both have none; 0.0 if there is mismatch.
    """
    gen_has_emoji = bool(EMOJI_REGEX.search(gen_text))
    ref_has_emoji = bool(EMOJI_REGEX.search(ref_text))
    return 1.0 if gen_has_emoji == ref_has_emoji else 0.0


def calculate_punctuation_alignment(gen_text: str, ref_text: str) -> float:
    """
    Checks if question mark and exclamation style matches.
    """
    gen_q = "?" in gen_text
    ref_q = "?" in ref_text
    gen_excl = "!" in gen_text
    ref_excl = "!" in ref_text

    score = 0.0
    if gen_q == ref_q:
        score += 0.5
    if gen_excl == ref_excl:
        score += 0.5
    return score


# =========================================================================
# STAGE 6E EVALUATION METRICS & HELPERS
# =========================================================================

def is_hinglish_text(text: str) -> bool:
    """Checks whether text exhibits Hinglish language characteristics."""
    tokens = set(re.findall(r"\b\w+\b", text.lower()))
    return len(tokens & HINGLISH_TOKENS) > 0


def contains_slang(text: str) -> bool:
    """Checks whether text contains common peer slang."""
    tokens = set(re.findall(r"\b\w+\b", text.lower()))
    return len(tokens & SLANG_TOKENS) > 0


def contains_emoji(text: str) -> bool:
    """Detects presence of Unicode emoji characters."""
    return bool(EMOJI_REGEX.search(text))


def detect_hallucination_indicators(response: str, expected_topic: str) -> bool:
    """
    Heuristic rule-based check flagging unsupported concrete assertions.
    Flags invented personal specifics or fabricated entities absent in conversation.
    """
    clean = response.lower()
    if "asus" in clean or "macbook" in clean or "dell alienware" in clean:
        return True
    if "doctor" in clean or "hospital" in clean or "mumbai campus" in clean:
        return True
    return False


def calculate_percentiles(values: List[float]) -> Dict[str, float]:
    """Calculates mean, median, p25, p75, p90, p95, and max for a list of values."""
    if not values:
        return {
            "mean": 0.0,
            "median": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }
    arr = np.array(values)
    return {
        "mean": round(float(np.mean(arr)), 2),
        "median": round(float(np.median(arr)), 2),
        "p25": round(float(np.percentile(arr, 25)), 2),
        "p75": round(float(np.percentile(arr, 75)), 2),
        "p90": round(float(np.percentile(arr, 90)), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
        "max": round(float(np.max(arr)), 2),
    }


class MetricsCalculator:
    """Computes comprehensive quantitative summary metrics over a collection of BenchmarkResults."""

    @classmethod
    def compute_summary(cls, results: List[BenchmarkResult]) -> EvaluationSummary:
        total = len(results)
        if total == 0:
            raise ValueError("Cannot compute evaluation summary over empty results list.")

        category_counts: Dict[str, int] = {}
        for r in results:
            cat_name = r.category.value
            category_counts[cat_name] = category_counts.get(cat_name, 0) + 1

        # 1. Accuracies
        topic_acc = sum(1 for r in results if r.topic_correct) / total
        intent_acc = sum(1 for r in results if r.intent_correct) / total
        strat_acc = sum(1 for r in results if r.strategy_correct) / total

        # Ambiguity accuracy
        ambig_cases = [
            r for r in results
            if r.category == BenchmarkCategory.AMBIGUOUS_SHORT_MESSAGES
            or r.ambiguity_level.lower() == "high"
        ]
        ambig_acc = (
            sum(1 for r in ambig_cases if r.ambiguity_handled_correctly) / len(ambig_cases)
            if ambig_cases
            else 1.0
        )

        # 2. Memory Metrics
        mem_used_cases = [r for r in results if r.use_memory]
        if mem_used_cases:
            mem_prec = sum(
                1 for r in mem_used_cases
                if (r.expected_use_memory is not False) and not r.memory_contamination
            ) / len(mem_used_cases)
        else:
            mem_prec = 1.0

        expect_mem_cases = [r for r in results if r.expected_use_memory is True]
        if expect_mem_cases:
            mem_recall = sum(1 for r in expect_mem_cases if r.use_memory) / len(expect_mem_cases)
        else:
            mem_recall = 1.0

        contamination_count = sum(1 for r in results if r.memory_contamination)
        contamination_rate = contamination_count / total
        utilization_rate = len(mem_used_cases) / total

        # 3. Persona & Style Metrics
        hinglish_rate = sum(1 for r in results if r.is_hinglish) / total
        slang_rate = sum(1 for r in results if r.has_slang) / total
        emoji_rate = sum(1 for r in results if r.has_emoji) / total
        hallucination_rate = sum(1 for r in results if r.has_hallucination) / total

        # 4. Length Distributions (Words & Characters)
        words = [float(r.response_words) for r in results]
        length_dist = calculate_percentiles(words)

        # 5. Latency Distributions
        context_latencies = [r.latencies_ms.get("context_ms", 0.0) for r in results]
        memory_latencies = [r.latencies_ms.get("memory_ms", 0.0) for r in results]
        orch_latencies = [r.latencies_ms.get("orchestrator_ms", 0.0) for r in results]
        gen_latencies = [r.latencies_ms.get("generation_ms", 0.0) for r in results]
        total_latencies = [r.latencies_ms.get("total_ms", 0.0) for r in results]

        latency_summary = {
            "context_ms": calculate_percentiles(context_latencies),
            "memory_ms": calculate_percentiles(memory_latencies),
            "orchestrator_ms": calculate_percentiles(orch_latencies),
            "generation_ms": calculate_percentiles(gen_latencies),
            "total_ms": calculate_percentiles(total_latencies),
        }

        # 6. Category Breakdown
        category_metrics: Dict[str, Dict[str, float]] = {}
        for cat in BenchmarkCategory:
            cat_results = [r for r in results if r.category == cat]
            if not cat_results:
                continue
            cat_total = len(cat_results)
            category_metrics[cat.value] = {
                "count": cat_total,
                "topic_accuracy": round(sum(1 for r in cat_results if r.topic_correct) / cat_total, 4),
                "intent_accuracy": round(sum(1 for r in cat_results if r.intent_correct) / cat_total, 4),
                "strategy_accuracy": round(sum(1 for r in cat_results if r.strategy_correct) / cat_total, 4),
                "memory_accuracy": round(sum(1 for r in cat_results if r.memory_behavior_correct) / cat_total, 4),
            }

        # 7. Error Frequency Breakdown
        error_breakdown: Dict[str, int] = {e.value: 0 for e in ErrorType}
        for r in results:
            for err in r.error_types:
                error_breakdown[err.value] = error_breakdown.get(err.value, 0) + 1

        return EvaluationSummary(
            total_cases=total,
            category_counts=category_counts,
            topic_accuracy=round(topic_acc, 4),
            intent_accuracy=round(intent_acc, 4),
            strategy_accuracy=round(strat_acc, 4),
            ambiguity_accuracy=round(ambig_acc, 4),
            memory_precision=round(mem_prec, 4),
            memory_recall=round(mem_recall, 4),
            memory_contamination_rate=round(contamination_rate, 4),
            memory_utilization_rate=round(utilization_rate, 4),
            hinglish_rate=round(hinglish_rate, 4),
            slang_rate=round(slang_rate, 4),
            emoji_rate=round(emoji_rate, 4),
            hallucination_rate=round(hallucination_rate, 4),
            length_distribution=length_dist,
            latency_summary=latency_summary,
            category_metrics=category_metrics,
            error_breakdown=error_breakdown,
        )
