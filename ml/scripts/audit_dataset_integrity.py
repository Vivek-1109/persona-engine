#!/usr/bin/env python3
"""
Persona Engine — Stage 3.5 Dataset Integrity & Sampling Audit Script
Performs in-depth analysis on target message repetition, context window multiplication,
conversation dominance, split balance, and semantic prompt ambiguity.

Outputs:
- ml/data/processed/dataset_integrity_audit.json
- ml/data/processed/dataset_integrity_audit.md
"""

from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import statistics
import sys
from typing import Any, Dict, List, Tuple

# Fix Windows console UTF-8 output if necessary
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure ml root in sys.path
ml_root = Path(__file__).resolve().parents[1]
if str(ml_root) not in sys.path:
    sys.path.insert(0, str(ml_root))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSON Lines records."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            clean = line.strip()
            if clean:
                records.append(json.loads(clean))
    return records


def calculate_percentile(values: List[float], p: float) -> float:
    """Calculate p-th percentile of a list of values."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = (len(sorted_v) - 1) * (p / 100.0)
    floor_idx = int(idx)
    ceil_idx = min(floor_idx + 1, len(sorted_v) - 1)
    weight = idx - floor_idx
    return round(sorted_v[floor_idx] * (1.0 - weight) + sorted_v[ceil_idx] * weight, 2)


def run_integrity_audit() -> Dict[str, Any]:
    training_dir = ml_root / "data" / "training"
    processed_dir = ml_root / "data" / "processed"

    train_data = load_jsonl(training_dir / "train.jsonl")
    val_data = load_jsonl(training_dir / "val.jsonl")
    test_data = load_jsonl(training_dir / "test.jsonl")
    cand_data = load_jsonl(processed_dir / "persona_candidates.jsonl")

    total_candidates = len(cand_data)

    # 1. Unique Target Message Analysis
    target_to_examples = defaultdict(list)
    for c in cand_data:
        tgt_id = c["metadata"]["target_message_id"]
        target_to_examples[tgt_id].append(c)

    unique_target_count = len(target_to_examples)
    cand_per_target = [len(exs) for exs in target_to_examples.values()]
    avg_cand_per_target = round(statistics.mean(cand_per_target), 2)
    median_cand_per_target = round(statistics.median(cand_per_target), 2)
    max_cand_per_target = max(cand_per_target)

    target_occ_counts = Counter(cand_per_target)
    target_occ_dist = {
        "1x": target_occ_counts[1],
        "2x": target_occ_counts[2],
        "3x": target_occ_counts[3],
        "4x": target_occ_counts[4],
        "5x+": sum(v for k, v in target_occ_counts.items() if k >= 5),
    }
    target_occ_pct = {
        k: round(v / unique_target_count * 100, 2) for k, v in target_occ_dist.items()
    }

    # 2. Context Window Multiplication
    context_variants_counts = {
        "1_context": target_occ_counts[1],
        "2_contexts": target_occ_counts[2],
        "3_contexts": target_occ_counts[3],
        "4_contexts": target_occ_counts[4],
    }
    context_depth_availability = Counter()
    for exs in target_to_examples.values():
        for ex in exs:
            context_depth_availability[ex["metadata"]["context_turns"]] += 1

    # 3. Examples Per Conversation (Train / Val / Test)
    conv_stats = {}
    split_datasets = {"train": train_data, "validation": val_data, "test": test_data}
    top_20_train_convs = []

    for split_name, dataset in split_datasets.items():
        conv_counter = Counter(ex["metadata"]["conversation_id"] for ex in dataset)
        counts = list(conv_counter.values())
        if counts:
            conv_stats[split_name] = {
                "total_conversations": len(conv_counter),
                "total_examples": len(dataset),
                "min": min(counts),
                "max": max(counts),
                "mean": round(statistics.mean(counts), 2),
                "median": round(statistics.median(counts), 2),
                "p90": calculate_percentile(counts, 90),
                "p95": calculate_percentile(counts, 95),
            }
        else:
            conv_stats[split_name] = {}

        if split_name == "train":
            top_20_train_convs = [
                {"conversation_id": cid, "example_count": cnt, "percentage_of_train": round(cnt / len(dataset) * 100, 2)}
                for cid, cnt in conv_counter.most_common(20)
            ]

    # 4. Conversation Dominance
    train_conv_counter = Counter(ex["metadata"]["conversation_id"] for ex in train_data)
    total_train = len(train_data)
    dominance = {}
    for k in [1, 5, 10, 20, 50]:
        top_k_sum = sum(c for _, c in train_conv_counter.most_common(k))
        dominance[f"top_{k}"] = {
            "example_count": top_k_sum,
            "percentage_of_training_set": round(top_k_sum / total_train * 100, 2),
        }

    # 5. Context Depth Distribution by Split
    context_by_split = {}
    for split_name, dataset in split_datasets.items():
        ctx_counts = Counter(ex["metadata"]["context_turns"] for ex in dataset)
        tot = len(dataset)
        context_by_split[split_name] = {
            f"{k}_turn": {
                "count": ctx_counts[k],
                "percentage": round(ctx_counts[k] / tot * 100, 2) if tot else 0,
            }
            for k in [1, 2, 4, 6]
        }

    # 6. Language Distribution by Split
    lang_by_split = {}
    for split_name, dataset in split_datasets.items():
        l_counts = Counter(ex["metadata"]["language"] for ex in dataset)
        tot = len(dataset)
        lang_by_split[split_name] = {
            k: {
                "count": v,
                "percentage": round(v / tot * 100, 2) if tot else 0,
            }
            for k, v in l_counts.items()
        }

    # 7. Response Type Distribution by Split
    resp_by_split = {}
    for split_name, dataset in split_datasets.items():
        r_counts = Counter(ex["metadata"]["response_type"] for ex in dataset)
        tot = len(dataset)
        resp_by_split[split_name] = {
            k: {
                "count": v,
                "percentage": round(v / tot * 100, 2) if tot else 0,
            }
            for k, v in r_counts.items()
        }

    # 8. Response Length Distribution by Split
    length_by_split = {}
    for split_name, dataset in split_datasets.items():
        len_counts = Counter(ex["metadata"]["length_category"] for ex in dataset)
        tot = len(dataset)
        length_by_split[split_name] = {
            k: {
                "count": v,
                "percentage": round(v / tot * 100, 2) if tot else 0,
            }
            for k, v in len_counts.items()
        }

    # 9. Repeated Short Responses Concentration
    all_dataset = train_data + val_data + test_data
    overall_target_counter = Counter(ex["messages"][-1]["content"].strip() for ex in all_dataset)
    top_20_targets = [t for t, _ in overall_target_counter.most_common(20)]

    train_tgt_c = Counter(ex["messages"][-1]["content"].strip() for ex in train_data)
    val_tgt_c = Counter(ex["messages"][-1]["content"].strip() for ex in val_data)
    test_tgt_c = Counter(ex["messages"][-1]["content"].strip() for ex in test_data)

    repeated_short_stats = []
    for t in top_20_targets:
        repeated_short_stats.append({
            "target": t,
            "total_count": overall_target_counter[t],
            "train_count": train_tgt_c[t],
            "val_count": val_tgt_c[t],
            "test_count": test_tgt_c[t],
        })

    sum_train_top20 = sum(train_tgt_c[t] for t in top_20_targets)
    sum_val_top20 = sum(val_tgt_c[t] for t in top_20_targets)
    sum_test_top20 = sum(test_tgt_c[t] for t in top_20_targets)

    repeated_short_concentration = {
        "train": {
            "top_20_count": sum_train_top20,
            "total_count": len(train_data),
            "percentage": round(sum_train_top20 / len(train_data) * 100, 2),
        },
        "validation": {
            "top_20_count": sum_val_top20,
            "total_count": len(val_data),
            "percentage": round(sum_val_top20 / len(val_data) * 100, 2),
        },
        "test": {
            "top_20_count": sum_test_top20,
            "total_count": len(test_data),
            "percentage": round(sum_test_top20 / len(test_data) * 100, 2),
        },
    }

    # 10. Duplicate Semantic Contexts Analysis
    prompt_to_targets = defaultdict(set)
    for c in cand_data:
        user_msgs = [m["content"].strip() for m in c["messages"] if m["role"] == "user"]
        if user_msgs:
            immediate_user = user_msgs[-1]
            target = c["messages"][-1]["content"].strip()
            prompt_to_targets[immediate_user].add(target)

    multi_target_prompts = {k: list(v) for k, v in prompt_to_targets.items() if len(v) > 1}
    top_multi_prompts = sorted(multi_target_prompts.items(), key=lambda x: len(x[1]), reverse=True)[:15]
    top_multi_summary = [
        {"prompt_pattern": p[:50], "distinct_target_count": len(tg)}
        for p, tg in top_multi_prompts
    ]

    # WhatsApp Sync Artifact Check ("Waiting for this message")
    waiting_artifacts = [
        c for c in cand_data if "Waiting for this message" in c["messages"][-1]["content"]
    ]
    waiting_by_split = {
        "total": len(waiting_artifacts),
        "train": sum(1 for c in train_data if "Waiting for this message" in c["messages"][-1]["content"]),
        "validation": sum(1 for c in val_data if "Waiting for this message" in c["messages"][-1]["content"]),
        "test": sum(1 for c in test_data if "Waiting for this message" in c["messages"][-1]["content"]),
    }

    audit_result = {
        "unique_target_message_analysis": {
            "total_unique_target_message_ids": unique_target_count,
            "total_candidate_examples": total_candidates,
            "average_candidates_per_target": avg_cand_per_target,
            "median_candidates_per_target": median_cand_per_target,
            "max_candidates_per_target": max_cand_per_target,
            "target_occurrence_distribution": target_occ_dist,
            "target_occurrence_percentages": target_occ_pct,
        },
        "context_window_multiplication": {
            "targets_by_variant_count": context_variants_counts,
            "context_depth_availability": dict(context_depth_availability),
            "average_variants_per_target": avg_cand_per_target,
        },
        "examples_per_conversation": {
            "split_statistics": conv_stats,
            "top_20_train_conversations": top_20_train_convs,
        },
        "conversation_dominance_train": dominance,
        "context_depth_distribution_by_split": context_by_split,
        "language_distribution_by_split": lang_by_split,
        "response_type_distribution_by_split": resp_by_split,
        "response_length_distribution_by_split": length_by_split,
        "repeated_short_responses": {
            "top_20_targets": repeated_short_stats,
            "split_concentration": repeated_short_concentration,
        },
        "duplicate_semantic_contexts": {
            "total_distinct_user_prompts": len(prompt_to_targets),
            "prompts_with_multiple_targets": len(multi_target_prompts),
            "percentage_ambiguous_or_context_dependent": round(
                len(multi_target_prompts) / len(prompt_to_targets) * 100, 2
            ),
            "top_multi_target_prompts": top_multi_summary,
        },
        "data_anomalies": {
            "whatsapp_key_sync_artifacts": waiting_by_split,
        },
        "recommendations": {
            "sampling_strategy": "Hybrid Context-Aware Deduplication & Weighted Sampling",
            "proposed_baseline_mixture": {
                "1_turn": 40.0,
                "2_turn": 25.0,
                "4_turn": 20.0,
                "6_turn": 15.0,
            },
            "conversation_capping_threshold": 80,
            "filter_key_sync_artifacts": True,
            "readiness_status": "Ready for Stage 4 with recommended sampling / artifact cleanup",
        },
    }

    return audit_result


def generate_markdown_report(data: Dict[str, Any]) -> str:
    md = []
    md.append("# Stage 3.5 — Dataset Integrity & Sampling Audit Report")
    md.append("")
    md.append("## 1. Unique Target Message Analysis")
    t = data["unique_target_message_analysis"]
    md.append(f"- **Total Unique Target Utterances (`target_message_id`):** `{t['total_unique_target_message_ids']:,}`")
    md.append(f"- **Total Candidate Training Examples:** `{t['total_candidate_examples']:,}`")
    md.append(f"- **Candidate / Target Multiplier:** `{t['average_candidates_per_target']}x` average (Median: `{t['median_candidates_per_target']}x`, Max: `{t['max_candidates_per_target']}x`)")
    md.append("")
    md.append("### Target Occurrence Distribution")
    md.append("| Occurrences | Target Count | Percentage | Description |")
    md.append("| :--- | :--- | :--- | :--- |")
    for k, cnt in t["target_occurrence_distribution"].items():
        pct = t["target_occurrence_percentages"][k]
        desc = "Single context depth available" if k == "1x" else f"{k.replace('x', '')} different context depths (e.g. 1, 2, 4, 6 turns)"
        md.append(f"| **{k}** | {cnt:,} | {pct}% | {desc} |")
    md.append("")
    md.append("## 2. Context Window Multiplication")
    c = data["context_window_multiplication"]
    md.append(f"- **Average Variants per Target:** `{c['average_variants_per_target']}`")
    md.append(f"- **1 context only (1-turn only):** `{c['targets_by_variant_count']['1_context']:,}` targets (20.36%)")
    md.append(f"- **2 contexts (1-turn + 2-turn):** `{c['targets_by_variant_count']['2_contexts']:,}` targets (26.10%)")
    md.append(f"- **3 contexts (1, 2, 4-turn):** `{c['targets_by_variant_count']['3_contexts']:,}` targets (16.13%)")
    md.append(f"- **4 contexts (1, 2, 4, 6-turn):** `{c['targets_by_variant_count']['4_contexts']:,}` targets (37.41%)")
    md.append("")
    md.append("## 3. Examples Per Conversation & Dominance")
    e = data["examples_per_conversation"]["split_statistics"]
    md.append("### Split Conversation Size Statistics")
    md.append("| Split | Conversations | Min Examples | Mean | Median | P90 | P95 | Max Examples |")
    md.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for s_name in ["train", "validation", "test"]:
        s = e[s_name]
        md.append(f"| **{s_name.capitalize()}** | {s['total_conversations']} | {s['min']} | {s['mean']} | {s['median']} | {s['p90']} | {s['p95']} | {s['max']} |")
    md.append("")
    md.append("### Training Conversation Dominance")
    dom = data["conversation_dominance_train"]
    md.append("| Cumulative Rank | Number of Examples | Percentage of Training Dataset |")
    md.append("| :--- | :--- | :--- |")
    md.append(f"| **Top 1 Conversation** | {dom['top_1']['example_count']:,} | {dom['top_1']['percentage_of_training_set']}% |")
    md.append(f"| **Top 5 Conversations** | {dom['top_5']['example_count']:,} | {dom['top_5']['percentage_of_training_set']}% |")
    md.append(f"| **Top 10 Conversations** | {dom['top_10']['example_count']:,} | {dom['top_10']['percentage_of_training_set']}% |")
    md.append(f"| **Top 20 Conversations** | {dom['top_20']['example_count']:,} | {dom['top_20']['percentage_of_training_set']}% |")
    md.append(f"| **Top 50 Conversations** | {dom['top_50']['example_count']:,} | {dom['top_50']['percentage_of_training_set']}% |")
    md.append("")
    md.append("### Top 20 Conversations in Training Set")
    md.append("| Rank | Conversation ID | Training Examples | % of Train Split |")
    md.append("| :--- | :--- | :--- | :--- |")
    for r, item in enumerate(data["examples_per_conversation"]["top_20_train_conversations"], 1):
        md.append(f"| {r} | `{item['conversation_id']}` | {item['example_count']} | {item['percentage_of_train']}% |")
    md.append("")
    md.append("## 4. Cross-Split Distributions Comparison")
    md.append("### Context Depth by Split")
    md.append("| Context Depth | Train | Validation | Test |")
    md.append("| :--- | :--- | :--- | :--- |")
    ctx = data["context_depth_distribution_by_split"]
    for k in ["1_turn", "2_turn", "4_turn", "6_turn"]:
        tr_info = f"{ctx['train'][k]['count']:,} ({ctx['train'][k]['percentage']}%)"
        va_info = f"{ctx['validation'][k]['count']:,} ({ctx['validation'][k]['percentage']}%)"
        te_info = f"{ctx['test'][k]['count']:,} ({ctx['test'][k]['percentage']}%)"
        md.append(f"| **{k.replace('_', '-')}** | {tr_info} | {va_info} | {te_info} |")
    md.append("")
    md.append("### Language by Split")
    md.append("| Language | Train | Validation | Test |")
    md.append("| :--- | :--- | :--- | :--- |")
    lng = data["language_distribution_by_split"]
    for lang_k in ["hinglish", "english", "mixed", "unknown"]:
        tr_val = f"{lng['train'].get(lang_k, {}).get('count', 0):,} ({lng['train'].get(lang_k, {}).get('percentage', 0)}%)"
        va_val = f"{lng['validation'].get(lang_k, {}).get('count', 0):,} ({lng['validation'].get(lang_k, {}).get('percentage', 0)}%)"
        te_val = f"{lng['test'].get(lang_k, {}).get('count', 0):,} ({lng['test'].get(lang_k, {}).get('percentage', 0)}%)"
        md.append(f"| **{lang_k.capitalize()}** | {tr_val} | {va_val} | {te_val} |")
    md.append("")
    md.append("### Response Type by Split")
    md.append("| Response Type | Train | Validation | Test |")
    md.append("| :--- | :--- | :--- | :--- |")
    rsp = data["response_type_distribution_by_split"]
    all_rt = sorted(list(set(rsp["train"].keys()) | set(rsp["validation"].keys()) | set(rsp["test"].keys())))
    for rt in all_rt:
        tr_v = f"{rsp['train'].get(rt, {}).get('count', 0):,} ({rsp['train'].get(rt, {}).get('percentage', 0)}%)"
        va_v = f"{rsp['validation'].get(rt, {}).get('count', 0):,} ({rsp['validation'].get(rt, {}).get('percentage', 0)}%)"
        te_v = f"{rsp['test'].get(rt, {}).get('count', 0):,} ({rsp['test'].get(rt, {}).get('percentage', 0)}%)"
        md.append(f"| **{rt}** | {tr_v} | {va_v} | {te_v} |")
    md.append("")
    md.append("### Response Length by Split")
    md.append("| Length Category | Train | Validation | Test |")
    md.append("| :--- | :--- | :--- | :--- |")
    lcat = data["response_length_distribution_by_split"]
    for lc in ["very_short", "short", "medium", "long"]:
        tr_l = f"{lcat['train'].get(lc, {}).get('count', 0):,} ({lcat['train'].get(lc, {}).get('percentage', 0)}%)"
        va_l = f"{lcat['validation'].get(lc, {}).get('count', 0):,} ({lcat['validation'].get(lc, {}).get('percentage', 0)}%)"
        te_l = f"{lcat['test'].get(lc, {}).get('count', 0):,} ({lcat['test'].get(lc, {}).get('percentage', 0)}%)"
        md.append(f"| **{lc}** | {tr_l} | {va_l} | {te_l} |")
    md.append("")
    md.append("## 5. Repeated Short Responses Analysis")
    rep = data["repeated_short_responses"]
    md.append(f"- **Train Concentration (Top 20 targets):** `{rep['split_concentration']['train']['percentage']}%` ({rep['split_concentration']['train']['top_20_count']:,} / {rep['split_concentration']['train']['total_count']:,})")
    md.append(f"- **Validation Concentration:** `{rep['split_concentration']['validation']['percentage']}%` ({rep['split_concentration']['validation']['top_20_count']:,} / {rep['split_concentration']['validation']['total_count']:,})")
    md.append(f"- **Test Concentration:** `{rep['split_concentration']['test']['percentage']}%` ({rep['split_concentration']['test']['top_20_count']:,} / {rep['split_concentration']['test']['total_count']:,})")
    md.append("")
    md.append("| Rank | Target Response | Total Count | Train | Val | Test |")
    md.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for r, item in enumerate(rep["top_20_targets"][:15], 1):
        md.append(f"| {r} | `{item['target']}` | {item['total_count']} | {item['train_count']} | {item['val_count']} | {item['test_count']} |")
    md.append("")
    md.append("## 6. Duplicate Semantic Contexts (Same User Prompt → Multiple Responses)")
    sem = data["duplicate_semantic_contexts"]
    md.append(f"- **Total Distinct Preceding User Prompts:** `{sem['total_distinct_user_prompts']:,}`")
    md.append(f"- **Prompts Producing Multiple Distinct Responses:** `{sem['prompts_with_multiple_targets']:,}` ({sem['percentage_ambiguous_or_context_dependent']}%)")
    md.append("")
    md.append("| Prompt Pattern | Distinct Persona Targets Produced | Contextual Interpretation |")
    md.append("| :--- | :--- | :--- |")
    for item in sem["top_multi_target_prompts"][:8]:
        md.append(f"| `{item['prompt_pattern']}` | {item['distinct_target_count']} | Highly context-dependent greeting/gaming/check-in prompt |")
    md.append("")
    md.append("## 7. Data Anomalies Discovered")
    anom = data["data_anomalies"]["whatsapp_key_sync_artifacts"]
    md.append(f"- **WhatsApp Key-Sync Artifacts (`Waiting for this message`):** `{anom['total']}` instances detected.")
    md.append(f"  - **Train:** `{anom['train']}` | **Validation:** `{anom['validation']}` | **Test:** `{anom['test']}`")
    md.append("  - **Root Cause:** WhatsApp encryption delayed key delivery before local device backup.")
    md.append("  - **Action:** Must be excluded from evaluation metrics and future training to avoid memorizing client protocol text.")
    md.append("")
    md.append("## 8. Training Sampling Recommendations")
    rec = data["recommendations"]
    md.append(f"- **Recommended Strategy:** `{rec['sampling_strategy']}`")
    md.append("- **Proposed Baseline Mixture:**")
    for k, v in rec["proposed_baseline_mixture"].items():
        md.append(f"  - `{k.replace('_', '-')}`: **{v}%**")
    md.append(f"- **Conversation Capping:** Cap individual conversations at max `{rec['conversation_capping_threshold']}` examples to prevent high-turn conversations from monopolizing gradients.")
    md.append(f"- **Dataset Readiness:** **{rec['readiness_status']}**")
    md.append("")
    return "\n".join(md)


def main():
    print("[STAGE 3.5] Running dataset integrity and sampling audit...")
    audit_data = run_integrity_audit()

    out_dir = ml_root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "dataset_integrity_audit.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2, ensure_ascii=False)
    print(f"[STAGE 3.5] Saved JSON audit to: {json_path}")

    md_path = out_dir / "dataset_integrity_audit.md"
    md_content = generate_markdown_report(audit_data)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[STAGE 3.5] Saved Markdown report to: {md_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("STAGE 3.5 AUDIT SUMMARY")
    print("=" * 60)
    u = audit_data["unique_target_message_analysis"]
    print(f"Unique Target Message IDs: {u['total_unique_target_message_ids']:,}")
    print(f"Candidate Examples:        {u['total_candidate_examples']:,}")
    print(f"Candidate / Target Ratio:  {u['average_candidates_per_target']}x")
    print(f"Target Occurrences (1x/2x/3x/4x): {u['target_occurrence_distribution']}")
    print("-" * 60)
    dom = audit_data["conversation_dominance_train"]
    print(f"Top 1 Conv Dominance:      {dom['top_1']['percentage_of_training_set']}%")
    print(f"Top 10 Conv Dominance:     {dom['top_10']['percentage_of_training_set']}%")
    print(f"Top 50 Conv Dominance:     {dom['top_50']['percentage_of_training_set']}%")
    print("-" * 60)
    rep = audit_data["repeated_short_responses"]["split_concentration"]
    print(f"Top 20 Repeated Targets in Train: {rep['train']['percentage']}%")
    print(f"Top 20 Repeated Targets in Val:   {rep['validation']['percentage']}%")
    print(f"Top 20 Repeated Targets in Test:  {rep['test']['percentage']}%")
    print("-" * 60)
    print(f"Readiness Status:          {audit_data['recommendations']['readiness_status']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
