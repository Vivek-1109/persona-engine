"""
Benchmark Runner for Intent & Dialogue-Act Classifier (Stage 6F).
Evaluates 60 controlled test cases across 10 intent categories, multi-turn contexts,
multi-sentence structures, and ambiguous utterances.
Computes overall accuracy, per-intent accuracy, confusion matrix, and latency profile.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional

# Ensure ml and repo root in sys.path
ml_root = Path(__file__).resolve().parents[1]
repo_root = ml_root.parent
for p in [str(repo_root), str(ml_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from ml.src.context.context_schema import UserIntent
from ml.src.context.intent_analyzer import IntentAnalyzer


@dataclass
class IntentTestCase:
    case_id: str
    category: str  # Primary test category
    expected_intent: UserIntent
    conversation: List[Dict[str, str]]
    is_ambiguous: bool = False
    is_multiturn: bool = False
    is_multisentence: bool = False
    notes: str = ""


BENCHMARK_CASES: List[IntentTestCase] = [
    # -------------------------------------------------------------------------
    # 1. QUESTION (6 cases)
    # -------------------------------------------------------------------------
    IntentTestCase(
        case_id="Q_01",
        category="question",
        expected_intent=UserIntent.QUESTION,
        conversation=[{"role": "user", "content": "Exam kab se start ho rahe hain?"}],
        notes="Standard direct factual question",
    ),
    IntentTestCase(
        case_id="Q_02",
        category="question",
        expected_intent=UserIntent.QUESTION,
        conversation=[{"role": "user", "content": "Tu kal college aayega kya?"}],
        notes="Confirmation question with trailing particle",
    ),
    IntentTestCase(
        case_id="Q_03",
        category="question",
        expected_intent=UserIntent.QUESTION,
        conversation=[{"role": "user", "content": "Kyu bhai kya hua?"}],
        notes="Compound causal question",
    ),
    IntentTestCase(
        case_id="Q_04",
        category="question",
        expected_intent=UserIntent.QUESTION,
        conversation=[
            {"role": "assistant", "content": "laptop format kiya tha"},
            {"role": "user", "content": "ab sahi chal raha hai?"}
        ],
        is_multiturn=True,
        notes="Multi-turn inquiry following assistant update",
    ),
    IntentTestCase(
        case_id="Q_05",
        category="question",
        expected_intent=UserIntent.QUESTION,
        conversation=[{"role": "user", "content": "konsa course enroll karu is semester?"}],
        notes="Selection question with 'konsa'",
    ),
    IntentTestCase(
        case_id="Q_06",
        category="question",
        expected_intent=UserIntent.QUESTION,
        conversation=[{"role": "user", "content": "koi acchi movie bata dekhne ke liye"}],
        notes="Recommendation inquiry",
    ),

    # -------------------------------------------------------------------------
    # 2. ANSWER REQUEST (6 cases)
    # -------------------------------------------------------------------------
    IntentTestCase(
        case_id="AR_01",
        category="answer_request",
        expected_intent=UserIntent.ANSWER_REQUEST,
        conversation=[{"role": "user", "content": "Iska solution kya hai?"}],
        notes="Explicit solution request",
    ),
    IntentTestCase(
        case_id="AR_02",
        category="answer_request",
        expected_intent=UserIntent.ANSWER_REQUEST,
        conversation=[{"role": "user", "content": "Ye bug kaise fix hoga bhai?"}],
        notes="Fix solicitation",
    ),
    IntentTestCase(
        case_id="AR_03",
        category="answer_request",
        expected_intent=UserIntent.ANSWER_REQUEST,
        conversation=[{"role": "user", "content": "Isme kya karna chahiye ab?"}],
        notes="Advisory solicitation",
    ),
    IntentTestCase(
        case_id="AR_04",
        category="answer_request",
        expected_intent=UserIntent.ANSWER_REQUEST,
        conversation=[
            {"role": "assistant", "content": "docker build fail ho gaya"},
            {"role": "user", "content": "kaise solve kare isko?"}
        ],
        is_multiturn=True,
        notes="Multi-turn problem solving request",
    ),
    IntentTestCase(
        case_id="AR_05",
        category="answer_request",
        expected_intent=UserIntent.ANSWER_REQUEST,
        conversation=[{"role": "user", "content": "batao iska step by step approach kya hai"}],
        notes="Methodology request",
    ),
    IntentTestCase(
        case_id="AR_06",
        category="answer_request",
        expected_intent=UserIntent.ANSWER_REQUEST,
        conversation=[{"role": "user", "content": "bhai help chahiye code me"}],
        notes="Assistance request",
    ),

    # -------------------------------------------------------------------------
    # 3. INVITATION (6 cases)
    # -------------------------------------------------------------------------
    IntentTestCase(
        case_id="INV_01",
        category="invitation",
        expected_intent=UserIntent.INVITATION,
        conversation=[{"role": "user", "content": "Khelega?"}],
        notes="Direct terse gaming invitation",
    ),
    IntentTestCase(
        case_id="INV_02",
        category="invitation",
        expected_intent=UserIntent.INVITATION,
        conversation=[{"role": "user", "content": "canteen chale?"}],
        notes="Social hanging out invitation",
    ),
    IntentTestCase(
        case_id="INV_03",
        category="invitation",
        expected_intent=UserIntent.INVITATION,
        conversation=[{"role": "user", "content": "bhai game khelega to aa ja"}],
        notes="Embedded conditional invitation",
    ),
    IntentTestCase(
        case_id="INV_04",
        category="invitation",
        expected_intent=UserIntent.INVITATION,
        conversation=[{"role": "user", "content": "Assignment kar raha hu. Game khelega?"}],
        is_multisentence=True,
        notes="Multi-sentence statement leading into invitation",
    ),
    IntentTestCase(
        case_id="INV_05",
        category="invitation",
        expected_intent=UserIntent.INVITATION,
        conversation=[
            {"role": "user", "content": "free hai?"},
            {"role": "assistant", "content": "haan bol"},
            {"role": "user", "content": "valorant aaja"}
        ],
        is_multiturn=True,
        notes="Multi-turn invitation setup",
    ),
    IntentTestCase(
        case_id="INV_06",
        category="invitation",
        expected_intent=UserIntent.INVITATION,
        conversation=[{"role": "user", "content": "chai peene chalte hain 5 min me"}],
        notes="Social snack invitation",
    ),

    # -------------------------------------------------------------------------
    # 4. AGREEMENT (6 cases)
    # -------------------------------------------------------------------------
    IntentTestCase(
        case_id="AGR_01",
        category="agreement",
        expected_intent=UserIntent.AGREEMENT,
        conversation=[{"role": "user", "content": "Haan bhai."}],
        notes="Direct affirmative acknowledgement",
    ),
    IntentTestCase(
        case_id="AGR_02",
        category="agreement",
        expected_intent=UserIntent.AGREEMENT,
        conversation=[
            {"role": "assistant", "content": "Kal 5 baje milte hain."},
            {"role": "user", "content": "Thik"}
        ],
        is_multiturn=True,
        is_ambiguous=True,
        notes="Contextual agreement to timing plan",
    ),
    IntentTestCase(
        case_id="AGR_03",
        category="agreement",
        expected_intent=UserIntent.AGREEMENT,
        conversation=[
            {"role": "assistant", "content": "movie dekhne chale?"},
            {"role": "user", "content": "done bhai"}
        ],
        is_multiturn=True,
        notes="Agreement to peer invitation",
    ),
    IntentTestCase(
        case_id="AGR_04",
        category="agreement",
        expected_intent=UserIntent.AGREEMENT,
        conversation=[{"role": "user", "content": "bilkul sahi baat hai"}],
        notes="Strong concurrence",
    ),
    IntentTestCase(
        case_id="AGR_05",
        category="agreement",
        expected_intent=UserIntent.AGREEMENT,
        conversation=[{"role": "user", "content": "chal theek hai"}],
        notes="Casual conversational acceptance",
    ),
    IntentTestCase(
        case_id="AGR_06",
        category="agreement",
        expected_intent=UserIntent.AGREEMENT,
        conversation=[
            {"role": "assistant", "content": "theek hai kal milte hain"},
            {"role": "user", "content": "bye bhai goodnight"}
        ],
        is_multiturn=True,
        notes="Closing agreement handshake",
    ),

    # -------------------------------------------------------------------------
    # 5. DISAGREEMENT (6 cases)
    # -------------------------------------------------------------------------
    IntentTestCase(
        case_id="DIS_01",
        category="disagreement",
        expected_intent=UserIntent.DISAGREEMENT,
        conversation=[{"role": "user", "content": "Nhi bhai."}],
        notes="Direct negation",
    ),
    IntentTestCase(
        case_id="DIS_02",
        category="disagreement",
        expected_intent=UserIntent.DISAGREEMENT,
        conversation=[{"role": "user", "content": "aisa nahi hai bilkul bhi"}],
        notes="Refutation of claim",
    ),
    IntentTestCase(
        case_id="DIS_03",
        category="disagreement",
        expected_intent=UserIntent.DISAGREEMENT,
        conversation=[{"role": "user", "content": "galat logic hai bhai code me"}],
        notes="Factual dispute",
    ),
    IntentTestCase(
        case_id="DIS_04",
        category="disagreement",
        expected_intent=UserIntent.DISAGREEMENT,
        conversation=[{"role": "user", "content": "nahi yaar bore lag raha hai"}],
        notes="Preference rejection",
    ),
    IntentTestCase(
        case_id="DIS_05",
        category="disagreement",
        expected_intent=UserIntent.DISAGREEMENT,
        conversation=[{"role": "user", "content": "rehne de mat kar abhi"}],
        notes="Dissuasion / command to cancel",
    ),
    IntentTestCase(
        case_id="DIS_06",
        category="disagreement",
        expected_intent=UserIntent.DISAGREEMENT,
        conversation=[
            {"role": "assistant", "content": "75% attendance criteria hat gaya"},
            {"role": "user", "content": "galat bol raha hai tu"}
        ],
        is_multiturn=True,
        notes="Contradicting assistant statement",
    ),

    # -------------------------------------------------------------------------
    # 6. INFORMATION (6 cases)
    # -------------------------------------------------------------------------
    IntentTestCase(
        case_id="INF_01",
        category="information",
        expected_intent=UserIntent.INFORMATION,
        conversation=[{"role": "user", "content": "Dean ne attendance ka notice nikala hai."}],
        notes="Institutional announcement report",
    ),
    IntentTestCase(
        case_id="INF_02",
        category="information",
        expected_intent=UserIntent.INFORMATION,
        conversation=[{"role": "user", "content": "bhai battery bahut jaldi drain ho rahi"}],
        notes="Device diagnostic report",
    ),
    IntentTestCase(
        case_id="INF_03",
        category="information",
        expected_intent=UserIntent.INFORMATION,
        conversation=[{"role": "user", "content": "code me null pointer error aa raha hai"}],
        notes="Software bug report",
    ),
    IntentTestCase(
        case_id="INF_04",
        category="information",
        expected_intent=UserIntent.INFORMATION,
        conversation=[{"role": "user", "content": "steam pe download ho gaya game"}],
        notes="Task completion report",
    ),
    IntentTestCase(
        case_id="INF_05",
        category="information",
        expected_intent=UserIntent.INFORMATION,
        conversation=[{"role": "user", "content": "kal 9 baje class hai attendance mandatory hai"}],
        notes="Schedule announcement",
    ),
    IntentTestCase(
        case_id="INF_06",
        category="information",
        expected_intent=UserIntent.INFORMATION,
        conversation=[{"role": "user", "content": "Battery drain ho rahi hai. Kal service center jana padega."}],
        is_multisentence=True,
        notes="Multi-sentence status update",
    ),

    # -------------------------------------------------------------------------
    # 7. PLANNING (6 cases)
    # -------------------------------------------------------------------------
    IntentTestCase(
        case_id="PLN_01",
        category="planning",
        expected_intent=UserIntent.PLANNING,
        conversation=[{"role": "user", "content": "Kal 5 baje milte hain."}],
        notes="Direct schedule proposal",
    ),
    IntentTestCase(
        case_id="PLN_02",
        category="planning",
        expected_intent=UserIntent.PLANNING,
        conversation=[{"role": "user", "content": "kal kab milte hain?"}],
        notes="Timing inquiry for meeting",
    ),
    IntentTestCase(
        case_id="PLN_03",
        category="planning",
        expected_intent=UserIntent.PLANNING,
        conversation=[{"role": "user", "content": "weekend ka kya plan hai?"}],
        notes="Itinerary coordination",
    ),
    IntentTestCase(
        case_id="PLN_04",
        category="planning",
        expected_intent=UserIntent.PLANNING,
        conversation=[{"role": "user", "content": "lunch break me chalte hain 1 baje"}],
        notes="Time-anchored departure plan",
    ),
    IntentTestCase(
        case_id="PLN_05",
        category="planning",
        expected_intent=UserIntent.PLANNING,
        conversation=[{"role": "user", "content": "Chal baad me baat karta hu"}],
        notes="Parting timing coordination",
    ),
    IntentTestCase(
        case_id="PLN_06",
        category="planning",
        expected_intent=UserIntent.PLANNING,
        conversation=[{"role": "user", "content": "ping me whenever you are free to deploy"}],
        notes="Asynchronous action scheduling",
    ),

    # -------------------------------------------------------------------------
    # 8. REACTION (6 cases)
    # -------------------------------------------------------------------------
    IntentTestCase(
        case_id="RCT_01",
        category="reaction",
        expected_intent=UserIntent.REACTION,
        conversation=[{"role": "user", "content": "😂"}],
        notes="Pure emoji laughter",
    ),
    IntentTestCase(
        case_id="RCT_02",
        category="reaction",
        expected_intent=UserIntent.REACTION,
        conversation=[{"role": "user", "content": "sach me yaar"}],
        notes="Exclamation of empathy / surprise",
    ),
    IntentTestCase(
        case_id="RCT_03",
        category="reaction",
        expected_intent=UserIntent.REACTION,
        conversation=[{"role": "user", "content": "bahut maza aaya sabke sath"}],
        notes="Reflective affective reaction",
    ),
    IntentTestCase(
        case_id="RCT_04",
        category="reaction",
        expected_intent=UserIntent.REACTION,
        conversation=[{"role": "user", "content": "arre yaar ye kya hai"}],
        notes="Frustrated outburst",
    ),
    IntentTestCase(
        case_id="RCT_05",
        category="reaction",
        expected_intent=UserIntent.REACTION,
        conversation=[{"role": "user", "content": "lmao bsdk"}],
        notes="Colloquial peer banter",
    ),
    IntentTestCase(
        case_id="RCT_06",
        category="reaction",
        expected_intent=UserIntent.REACTION,
        conversation=[{"role": "user", "content": "waah bhai zabardast"}],
        notes="Praise exclamation",
    ),

    # -------------------------------------------------------------------------
    # 9. CASUAL CHAT (6 cases)
    # -------------------------------------------------------------------------
    IntentTestCase(
        case_id="CAS_01",
        category="casual_chat",
        expected_intent=UserIntent.CASUAL_CHAT,
        conversation=[{"role": "user", "content": "Kya haal hai?"}],
        notes="Standard greeting chitchat",
    ),
    IntentTestCase(
        case_id="CAS_02",
        category="casual_chat",
        expected_intent=UserIntent.CASUAL_CHAT,
        conversation=[{"role": "user", "content": "bas badiya chal raha"}],
        notes="Pleasant status check-in",
    ),
    IntentTestCase(
        case_id="CAS_03",
        category="casual_chat",
        expected_intent=UserIntent.CASUAL_CHAT,
        conversation=[{"role": "user", "content": "aur bata kaisa hai"}],
        notes="Opening follow-up",
    ),
    IntentTestCase(
        case_id="CAS_04",
        category="casual_chat",
        expected_intent=UserIntent.CASUAL_CHAT,
        conversation=[{"role": "user", "content": "sab badhiya yahan"}],
        notes="Light chit-chat assertion",
    ),
    IntentTestCase(
        case_id="CAS_05",
        category="casual_chat",
        expected_intent=UserIntent.CASUAL_CHAT,
        conversation=[{"role": "user", "content": "kuch khaas nahi bas bore ho raha"}],
        notes="Boredom banter",
    ),
    IntentTestCase(
        case_id="CAS_06",
        category="casual_chat",
        expected_intent=UserIntent.CASUAL_CHAT,
        conversation=[{"role": "user", "content": "chill scene hai aaj"}],
        notes="Casual weekend mood description",
    ),

    # -------------------------------------------------------------------------
    # 10. UNKNOWN & AMBIGUOUS (6 cases)
    # -------------------------------------------------------------------------
    IntentTestCase(
        case_id="UNK_01",
        category="unknown",
        expected_intent=UserIntent.UNKNOWN,
        conversation=[{"role": "user", "content": "Kya"}],
        is_ambiguous=True,
        notes="Standalone ambiguous word without punctuation or context",
    ),
    IntentTestCase(
        case_id="UNK_02",
        category="unknown",
        expected_intent=UserIntent.UNKNOWN,
        conversation=[{"role": "user", "content": "zzzqwx 12389"}],
        notes="Unsupported gibberish token",
    ),
    IntentTestCase(
        case_id="UNK_03",
        category="unknown",
        expected_intent=UserIntent.UNKNOWN,
        conversation=[{"role": "user", "content": "..."}],
        is_ambiguous=True,
        notes="Punctuation-only empty ellipsis",
    ),
    IntentTestCase(
        case_id="AMB_01",
        category="ambiguous",
        expected_intent=UserIntent.INVITATION,
        conversation=[{"role": "user", "content": "Aaja"}],
        is_ambiguous=True,
        notes="Standalone 'Aaja' invitation interpretation",
    ),
    IntentTestCase(
        case_id="AMB_02",
        category="ambiguous",
        expected_intent=UserIntent.INVITATION,
        conversation=[
            {"role": "user", "content": "bgmi khelenge?"},
            {"role": "assistant", "content": "haan lobby me hu"},
            {"role": "user", "content": "Aaja"}
        ],
        is_multiturn=True,
        is_ambiguous=True,
        notes="Contextual 'Aaja' in gaming session",
    ),
    IntentTestCase(
        case_id="AMB_03",
        category="ambiguous",
        expected_intent=UserIntent.INVITATION,
        conversation=[
            {"role": "user", "content": "canteen chale?"},
            {"role": "assistant", "content": "haan 5 min me nikalte hain"},
            {"role": "user", "content": "Aaja"}
        ],
        is_multiturn=True,
        is_ambiguous=True,
        notes="Contextual 'Aaja' in canteen outing",
    ),
]


def run_intent_benchmark() -> Dict[str, any]:
    """Runs the 60-case intent benchmark and returns aggregated results."""
    total = len(BENCHMARK_CASES)
    correct_count = 0
    per_intent_totals = Counter()
    per_intent_correct = Counter()
    confusion = defaultdict(Counter)

    ambiguous_total = 0
    ambiguous_correct = 0
    multiturn_total = 0
    multiturn_correct = 0
    multisentence_total = 0
    multisentence_correct = 0

    latencies_ms = []

    for case in BENCHMARK_CASES:
        exp = case.expected_intent.value
        per_intent_totals[exp] += 1

        t0 = time.perf_counter()
        pred_intent = IntentAnalyzer.analyze(case.conversation)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

        pred = pred_intent.value
        confusion[exp][pred] += 1

        is_corr = (pred == exp)
        if is_corr:
            correct_count += 1
            per_intent_correct[exp] += 1

        if case.is_ambiguous:
            ambiguous_total += 1
            if is_corr:
                ambiguous_correct += 1

        if case.is_multiturn:
            multiturn_total += 1
            if is_corr:
                multiturn_correct += 1

        if case.is_multisentence:
            multisentence_total += 1
            if is_corr:
                multisentence_correct += 1

    overall_acc = round(correct_count / total * 100.0, 2)
    per_intent_acc = {
        intent: round(per_intent_correct[intent] / per_intent_totals[intent] * 100.0, 2)
        for intent in per_intent_totals
    }

    latencies_sorted = sorted(latencies_ms)
    mean_lat = round(sum(latencies_ms) / len(latencies_ms), 4)
    p95_lat = round(latencies_sorted[int(len(latencies_sorted) * 0.95)], 4)

    results = {
        "total_cases": total,
        "correct_cases": correct_count,
        "overall_accuracy_pct": overall_acc,
        "per_intent_accuracy": per_intent_acc,
        "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
        "ambiguous_accuracy_pct": round(ambiguous_correct / ambiguous_total * 100.0, 2) if ambiguous_total else 0.0,
        "multiturn_accuracy_pct": round(multiturn_correct / multiturn_total * 100.0, 2) if multiturn_total else 0.0,
        "multisentence_accuracy_pct": round(multisentence_correct / multisentence_total * 100.0, 2) if multisentence_total else 0.0,
        "mean_latency_ms": mean_lat,
        "p95_latency_ms": p95_lat,
    }

    return results


if __name__ == "__main__":
    res = run_intent_benchmark()
    print("==================================================")
    print("STAGE 6F — INTENT BENCHMARK RESULTS")
    print("==================================================")
    print(f"Total Cases             : {res['total_cases']}")
    print(f"Correct Cases           : {res['correct_cases']}")
    print(f"Overall Accuracy        : {res['overall_accuracy_pct']}%")
    print(f"Ambiguous Accuracy      : {res['ambiguous_accuracy_pct']}%")
    print(f"Multi-Turn Accuracy     : {res['multiturn_accuracy_pct']}%")
    print(f"Multi-Sentence Accuracy : {res['multisentence_accuracy_pct']}%")
    print(f"Mean Latency            : {res['mean_latency_ms']} ms")
    print(f"P95 Latency             : {res['p95_latency_ms']} ms")
    print("--------------------------------------------------")
    print("Per-Intent Accuracy:")
    for intent, acc in sorted(res['per_intent_accuracy'].items()):
        print(f"  {intent:16}: {acc}%")
    print("==================================================")
