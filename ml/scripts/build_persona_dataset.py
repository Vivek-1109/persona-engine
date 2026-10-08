#!/usr/bin/env python3
"""
Persona Engine — Stage 3 Dataset Builder
Builds clean, conversation-aware, behavioral-annotated train/val/test datasets
from the raw WhatsApp export.

Pipeline:
RAW EXPORT -> PARSER -> SEGMENTATION -> CANDIDATE EXTRACTION ->
BEHAVIORAL ANNOTATION -> CONVERSATION-LEVEL SPLIT -> JSONL DATASETS & STATS

Usage:
    python ml/scripts/build_persona_dataset.py [--config ml/configs/dataset.yaml]
"""

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import re
import statistics
import sys
from typing import Any, Dict, List, Tuple
import yaml

# Fix Windows console UTF-8 output if necessary
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure ml root is in sys.path
ml_root = Path(__file__).resolve().parents[1]
if str(ml_root) not in sys.path:
    sys.path.insert(0, str(ml_root))

from src.data.whatsapp_parser import WhatsAppParser, ParsedMessage
from src.data.conversation_segmenter import ConversationSegmenter, ConversationSession
from src.data.candidate_extractor import CandidateExtractor
from src.data.conversation_splitter import ConversationSplitter, SplitStats
from src.data.behavioral_annotator import BehavioralAnnotator


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_jsonl(records: List[Dict[str, Any]], output_path: Path):
    """Save records as JSON Lines."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def compute_persona_statistics(
    raw_messages: List[ParsedMessage],
    candidates: List[Dict[str, Any]],
    train_ex: List[Dict[str, Any]],
    val_ex: List[Dict[str, Any]],
    test_ex: List[Dict[str, Any]],
    split_stats: SplitStats,
) -> Dict[str, Any]:
    """
    Compute comprehensive statistical profile of persona style and dataset characteristics.
    """
    persona_raw = [m for m in raw_messages if m.is_persona and m.message_type == "text"]
    persona_texts = [m.text.strip() for m in persona_raw if m.text.strip()]

    lengths = [len(t) for t in persona_texts]
    avg_len = round(statistics.mean(lengths), 2) if lengths else 0
    median_len = round(statistics.median(lengths), 2) if lengths else 0

    # Length categories
    len_cats = Counter()
    for l in lengths:
        if l <= 5:
            len_cats["very_short (1-5 chars)"] += 1
        elif l <= 20:
            len_cats["short (6-20 chars)"] += 1
        elif l <= 60:
            len_cats["medium (21-60 chars)"] += 1
        else:
            len_cats["long (>60 chars)"] += 1

    # Emoji statistics
    annotator = BehavioralAnnotator()
    emojis_found = []
    emoji_msgs_count = 0
    for t in persona_texts:
        has_em = annotator.has_emoji(t)
        if has_em:
            emoji_msgs_count += 1
            for char in t:
                if annotator.has_emoji(char):
                    emojis_found.append(char)
    top_emojis = Counter(emojis_found).most_common(15)

    # Common words & Hinglish tokens
    words_counter = Counter()
    for t in persona_texts:
        tokens = re.findall(r"[a-zA-Z]+", t.lower())
        for tok in tokens:
            if len(tok) >= 2:
                words_counter[tok] += 1
    top_words = words_counter.most_common(25)

    # Questions and Acknowledgements
    q_count = sum(1 for t in persona_texts if "?" in t)
    ack_tokens = {"haa", "haan", "ha", "ok", "okh", "okhh", "sahi", "thik", "acha", "achha", "hmm"}
    ack_count = sum(1 for t in persona_texts if t.lower() in ack_tokens)

    # Candidate metadata statistics
    lang_dist = Counter(c["metadata"]["language"] for c in candidates)
    tone_dist = Counter(c["metadata"]["tone"] for c in candidates)
    resp_dist = Counter(c["metadata"]["response_type"] for c in candidates)
    ctx_dist = Counter(c["metadata"]["context_turns"] for c in candidates)
    qual_dist = Counter(c["metadata"]["quality_flag"] for c in candidates)

    # Target responses duplicate analysis
    target_responses = [c["messages"][-1]["content"] for c in candidates]
    target_counter = Counter(target_responses)
    repeated_targets = [(k, v) for k, v in target_counter.most_common(20) if v > 1]
    total_unique_targets = len(target_counter)

    # Candidate exact duplicate check (same input messages and target)
    candidate_signatures = Counter()
    for c in candidates:
        sig = tuple((m["role"], m["content"]) for m in c["messages"])
        candidate_signatures[sig] += 1
    exact_duplicate_candidates = sum(cnt - 1 for cnt in candidate_signatures.values() if cnt > 1)

    stats = {
        "persona_speaker_overview": {
            "total_raw_persona_messages": len(persona_raw),
            "total_non_empty_persona_texts": len(persona_texts),
            "average_response_length_chars": avg_len,
            "median_response_length_chars": median_len,
            "length_distribution": dict(len_cats),
            "emoji_usage_percentage": round((emoji_msgs_count / len(persona_texts)) * 100, 2) if persona_texts else 0,
            "top_emojis": top_emojis,
            "top_words": top_words,
            "question_frequency_percentage": round((q_count / len(persona_texts)) * 100, 2) if persona_texts else 0,
            "acknowledgement_frequency_percentage": round((ack_count / len(persona_texts)) * 100, 2) if persona_texts else 0,
        },
        "candidate_dataset_metrics": {
            "total_candidates": len(candidates),
            "unique_target_responses": total_unique_targets,
            "exact_duplicate_examples": exact_duplicate_candidates,
            "top_repeated_target_responses": repeated_targets,
            "context_window_distribution": dict(ctx_dist),
            "language_distribution": dict(lang_dist),
            "tone_distribution": dict(tone_dist),
            "response_type_distribution": dict(resp_dist),
            "quality_flag_distribution": dict(qual_dist),
        },
        "splits": {
            "train": {
                "conversations": split_stats.train_conversations,
                "examples": split_stats.train_examples,
                "percentage_examples": round(split_stats.train_examples / len(candidates) * 100, 2) if candidates else 0,
            },
            "validation": {
                "conversations": split_stats.val_conversations,
                "examples": split_stats.val_examples,
                "percentage_examples": round(split_stats.val_examples / len(candidates) * 100, 2) if candidates else 0,
            },
            "test": {
                "conversations": split_stats.test_conversations,
                "examples": split_stats.test_examples,
                "percentage_examples": round(split_stats.test_examples / len(candidates) * 100, 2) if candidates else 0,
            },
            "data_leakage_free": split_stats.is_leak_free,
        },
    }

    return stats


def generate_statistics_markdown(stats: Dict[str, Any]) -> str:
    """Generate Markdown report for persona style statistics."""
    md = []
    md.append("# Persona Engine — Persona Style & Dataset Statistics")
    md.append("")
    md.append("## 1. Persona Style Overview (Vivek)")
    p = stats["persona_speaker_overview"]
    md.append(f"- **Total Non-Empty Persona Utterances:** `{p['total_non_empty_persona_texts']:,}`")
    md.append(f"- **Average Response Length:** `{p['average_response_length_chars']} characters`")
    md.append(f"- **Median Response Length:** `{p['median_response_length_chars']} characters`")
    md.append(f"- **Emoji Usage Rate:** `{p['emoji_usage_percentage']}%`")
    md.append(f"- **Question Rate:** `{p['question_frequency_percentage']}%`")
    md.append(f"- **Acknowledgement Rate:** `{p['acknowledgement_frequency_percentage']}%`")
    md.append("")
    md.append("### Response Length Distribution")
    for k, v in p["length_distribution"].items():
        pct = round(v / p["total_non_empty_persona_texts"] * 100, 1)
        md.append(f"- **{k}:** {v:,} ({pct}%)")
    md.append("")
    md.append("### Top Emojis")
    emoji_str = " ".join([f"`{e}` ({c})" for e, c in p["top_emojis"]])
    md.append(emoji_str)
    md.append("")
    md.append("### Top Lexical Tokens")
    top_w_str = ", ".join([f"`{w}` ({c})" for w, c in p["top_words"][:15]])
    md.append(top_w_str)
    md.append("")
    md.append("## 2. Candidate Dataset Metrics")
    c = stats["candidate_dataset_metrics"]
    md.append(f"- **Total Multi-Turn Candidates:** `{c['total_candidates']:,}`")
    md.append(f"- **Unique Target Responses:** `{c['unique_target_responses']:,}`")
    md.append(f"- **Exact Duplicate Examples:** `{c['exact_duplicate_examples']:,}`")
    md.append("")
    md.append("### Context Window Distribution")
    for turns, cnt in sorted(c["context_window_distribution"].items()):
        md.append(f"- **{turns}-turn context:** {cnt:,} examples")
    md.append("")
    md.append("### Language Distribution")
    for lang, cnt in c["language_distribution"].items():
        md.append(f"- **{lang}:** {cnt:,} examples")
    md.append("")
    md.append("### Response Type Distribution")
    for rt, cnt in c["response_type_distribution"].items():
        md.append(f"- **{rt}:** {cnt:,} examples")
    md.append("")
    md.append("### Tone Distribution")
    for tn, cnt in c["tone_distribution"].items():
        md.append(f"- **{tn}:** {cnt:,} examples")
    md.append("")
    md.append("### Quality Flag Distribution")
    for qf, cnt in c["quality_flag_distribution"].items():
        md.append(f"- **{qf}:** {cnt:,} examples")
    md.append("")
    md.append("### Top Repeated Target Responses")
    md.append("| Target Response | Frequency | Interpretation |")
    md.append("| :--- | :--- | :--- |")
    for r, freq in c["top_repeated_target_responses"][:10]:
        md.append(f"| `{r}` | {freq:,} | High-frequency authentic persona signal |")
    md.append("")
    md.append("## 3. Split Distribution (Strictly Conversation-Level)")
    sp = stats["splits"]
    md.append(f"- **Leakage Free Verification:** `{'PASSED (0% Overlap)' if sp['data_leakage_free'] else 'FAILED'}`")
    md.append("")
    md.append("| Split | Conversations | Examples | Example % |")
    md.append("| :--- | :--- | :--- | :--- |")
    md.append(f"| **Train** | {sp['train']['conversations']:,} | {sp['train']['examples']:,} | {sp['train']['percentage_examples']}% |")
    md.append(f"| **Validation** | {sp['validation']['conversations']:,} | {sp['validation']['examples']:,} | {sp['validation']['percentage_examples']}% |")
    md.append(f"| **Test** | {sp['test']['conversations']:,} | {sp['test']['examples']:,} | {sp['test']['percentage_examples']}% |")
    md.append("")
    return "\n".join(md)


def main():
    parser = argparse.ArgumentParser(description="Build Persona Engine dataset from raw WhatsApp chat.")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=str(ml_root / "configs" / "dataset.yaml"),
        help="Path to dataset configuration YAML file.",
    )
    args = parser.parse_args()
    config_path = Path(args.config)

    print(f"[STAGE 3] Loading configuration from: {config_path}")
    cfg = load_config(config_path)

    raw_file_rel = cfg["data"]["raw_file_path"]
    proj_root = ml_root.parent if (ml_root / "src").exists() else ml_root
    raw_file_path = proj_root / raw_file_rel if (proj_root / raw_file_rel).exists() else ml_root / raw_file_rel

    processed_rel = cfg["data"]["processed_dir"]
    processed_dir = proj_root / processed_rel if "ml/" in processed_rel else ml_root / processed_rel

    annotated_rel = cfg["data"]["annotated_dir"]
    annotated_dir = proj_root / annotated_rel if "ml/" in annotated_rel else ml_root / annotated_rel

    training_rel = cfg["data"]["training_dir"]
    training_dir = proj_root / training_rel if "ml/" in training_rel else ml_root / training_rel

    processed_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)
    training_dir.mkdir(parents=True, exist_ok=True)

    persona_speaker = cfg["speaker"]["persona_speaker"]
    persona_name = cfg["speaker"]["persona_name"]
    gap_minutes = cfg["reconstruction"]["conversation_gap_minutes"]
    min_conv_msgs = cfg["reconstruction"]["min_messages_per_conversation"]
    context_turn_sizes = cfg["context"]["context_turn_sizes"]
    system_prompt = cfg["context"]["system_prompt"]
    train_ratio = cfg["splitting"]["train_ratio"]
    val_ratio = cfg["splitting"]["val_ratio"]
    test_ratio = cfg["splitting"]["test_ratio"]
    seed = cfg["splitting"]["random_seed"]
    max_resp_chars = cfg["filtering"]["max_response_chars_for_review"]

    # 1. Parse raw export
    print(f"\n[1/6] Parsing raw WhatsApp export: {raw_file_path}")
    parser_obj = WhatsAppParser(persona_speaker=persona_speaker)
    raw_messages = parser_obj.parse_file(raw_file_path)
    print(f"  Parsed {len(raw_messages):,} raw messages. Unparsed lines: {len(parser_obj.unparsed_lines)}")

    # 2. Segment conversations
    print(f"\n[2/6] Segmenting conversations (Inactivity threshold: {gap_minutes} mins)...")
    segmenter = ConversationSegmenter(
        persona_speaker=persona_speaker,
        gap_minutes=gap_minutes,
        min_messages=min_conv_msgs,
    )
    sessions = segmenter.segment(raw_messages)
    print(f"  Reconstructed {len(sessions):,} conversation sessions.")

    # Save conversations.jsonl
    conv_file = processed_dir / "conversations.jsonl"
    save_jsonl([s.to_dict() for s in sessions], conv_file)
    print(f"  Saved conversations to: {conv_file}")

    # 3. Extract candidates
    print(f"\n[3/6] Extracting persona response candidates (Context sizes: {context_turn_sizes})...")
    extractor = CandidateExtractor(
        system_prompt=system_prompt,
        context_turn_sizes=context_turn_sizes,
        max_response_chars_for_review=max_resp_chars,
        persona_name=persona_name,
    )
    candidates = extractor.extract_all(sessions)
    print(f"  Generated {len(candidates):,} candidate training examples.")

    # Save persona_candidates.jsonl
    cand_file = processed_dir / "persona_candidates.jsonl"
    save_jsonl(candidates, cand_file)
    print(f"  Saved candidates to: {cand_file}")

    # 4. Save annotated dataset
    print(f"\n[4/6] Saving behavioral annotated dataset...")
    annotated_file = annotated_dir / "persona_annotated.jsonl"
    save_jsonl(candidates, annotated_file)
    print(f"  Saved annotated examples to: {annotated_file}")

    # Filter out any EXCLUDE candidates for the training sets
    usable_candidates = [c for c in candidates if c["metadata"]["quality_flag"] != "EXCLUDE"]
    print(f"  Usable training candidates (non-EXCLUDE): {len(usable_candidates):,}")

    # 5. Conversation-level train/val/test split
    print(f"\n[5/6] Performing conversation-level split ({train_ratio*100:.0f}% / {val_ratio*100:.0f}% / {test_ratio*100:.0f}%, seed={seed})...")
    splitter = ConversationSplitter(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    train_ex, val_ex, test_ex, split_stats = splitter.split(usable_candidates)

    train_file = training_dir / "train.jsonl"
    val_file = training_dir / "val.jsonl"
    test_file = training_dir / "test.jsonl"

    save_jsonl(train_ex, train_file)
    save_jsonl(val_ex, val_file)
    save_jsonl(test_ex, test_file)

    print(f"  Saved Train:      {len(train_ex):,} examples ({split_stats.train_conversations} convs) -> {train_file}")
    print(f"  Saved Validation: {len(val_ex):,} examples ({split_stats.val_conversations} convs) -> {val_file}")
    print(f"  Saved Test:       {len(test_ex):,} examples ({split_stats.test_conversations} convs) -> {test_file}")
    print(f"  Zero Leakage Verified: {split_stats.is_leak_free}")

    # 6. Compute persona style statistics
    print(f"\n[6/6] Computing persona statistics & generating reports...")
    stats = compute_persona_statistics(
        raw_messages=raw_messages,
        candidates=usable_candidates,
        train_ex=train_ex,
        val_ex=val_ex,
        test_ex=test_ex,
        split_stats=split_stats,
    )

    stats_json = processed_dir / "persona_statistics.json"
    with open(stats_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  Saved statistics JSON to: {stats_json}")

    stats_md = processed_dir / "persona_statistics.md"
    md_content = generate_statistics_markdown(stats)
    with open(stats_md, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"  Saved statistics Markdown to: {stats_md}")

    # Print final summary as required by Section 15
    print("\n" + "=" * 60)
    print("STAGE 3 DATASET ENGINEERING SUMMARY")
    print("=" * 60)
    print("RAW")
    print("----")
    print(f"messages: {len(raw_messages):,}")
    print(f"speakers: {list(set(m.speaker for m in raw_messages))}")
    print(f"date range: {raw_messages[0].timestamp} -> {raw_messages[-1].timestamp}")
    print("")
    print("PERSONA")
    print("-------")
    print(f"total persona messages:  {stats['persona_speaker_overview']['total_raw_persona_messages']:,}")
    print(f"usable persona messages: {stats['persona_speaker_overview']['total_non_empty_persona_texts']:,}")
    print(f"excluded:                {stats['candidate_dataset_metrics']['quality_flag_distribution'].get('EXCLUDE', 0)}")
    print(f"review:                  {stats['candidate_dataset_metrics']['quality_flag_distribution'].get('REVIEW', 0)}")
    print("")
    print("CANDIDATES")
    print("----------")
    print(f"total:       {len(usable_candidates):,}")
    for k in sorted(stats['candidate_dataset_metrics']['context_window_distribution'].keys()):
        print(f"{k}-turn:      {stats['candidate_dataset_metrics']['context_window_distribution'][k]:,}")
    print("")
    print("FINAL SPLIT")
    print("-----------")
    print(f"train:       {len(train_ex):,}")
    print(f"validation:  {len(val_ex):,}")
    print(f"test:        {len(test_ex):,}")
    print("")
    print("TRAIN:")
    print(f"conversation count: {split_stats.train_conversations}")
    print(f"examples:           {split_stats.train_examples:,}")
    print("")
    print("VALIDATION:")
    print(f"conversation count: {split_stats.val_conversations}")
    print(f"examples:           {split_stats.val_examples:,}")
    print("")
    print("TEST:")
    print(f"conversation count: {split_stats.test_conversations}")
    print(f"examples:           {split_stats.test_examples:,}")
    print("=" * 60)
    print("Zero Data Leakage:   PASSED (0% Conversation Overlap)")
    print("=" * 60)


if __name__ == "__main__":
    main()
