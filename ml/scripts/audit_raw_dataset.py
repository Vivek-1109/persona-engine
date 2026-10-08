#!/usr/bin/env python3
"""
Persona Engine — Raw WhatsApp Export Audit Script
Audits the raw WhatsApp export file and generates comprehensive statistics:
- Message counts, line counts, parsing errors
- Speaker distributions
- Date ranges and conversation periods
- Media, deleted, system, URL, and empty messages
- Linguistic distributions (Hindi, Hinglish, English, Emojis)
- Message length statistics (avg, median, max)
- Outputs results to raw_audit.json and raw_audit.md
"""

from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import re
import statistics
import sys
from typing import Any, Dict, List

# Fix Windows console UTF-8 output if necessary
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure ml root is in sys.path
ml_root = Path(__file__).resolve().parents[1]
if str(ml_root) not in sys.path:
    sys.path.insert(0, str(ml_root))

from src.data.whatsapp_parser import WhatsAppParser


# Common Hinglish lexical markers
HINGLISH_KEYWORDS = {
    "kya", "nhi", "ni", "nahi", "hai", "h", "kar", "kr", "karna", "raha", "rahe",
    "bhai", "bhaiya", "bc", "bsdk", "bencho", "saale", "yaar", "yr", "acha", "achha",
    "thik", "sahi", "chal", "dekh", "dekha", "bata", "bol", "kaise", "kyu", "kyun",
    "mai", "main", "tu", "tere", "mera", "meri", "mere", "tera", "teri", "ho", "ye",
    "yeh", "wo", "woh", "ab", "kuch", "pe", "par", "se", "ko", "ka", "ki", "ke",
    "tha", "thi", "the", "aaja", "jaa", "aaya", "gaya", "gyi", "wala", "wali",
    "wale", "kab", "kaha", "kahan", "bhi", "to", "toh", "mat", "matt", "aur", "orr",
    "hum", "apna", "apni", "le", "de", "kisko", "usse", "isse", "unhe", "hota", "hoti",
    "hote", "bhi", "sirf", "bass", "sab", "sabhi", "waise", "wahi", "yahi", "karega"
}

ENGLISH_WORDS = {
    "the", "to", "and", "a", "in", "is", "it", "you", "that", "he", "was", "for",
    "on", "are", "as", "with", "his", "they", "at", "be", "this", "have", "from",
    "or", "one", "had", "by", "word", "but", "not", "what", "all", "were", "we",
    "when", "your", "can", "said", "there", "use", "an", "each", "which", "she",
    "do", "how", "their", "if", "will", "up", "other", "about", "out", "many", "then",
    "them", "these", "so", "some", "her", "would", "make", "like", "him", "into",
    "time", "has", "look", "two", "more", "write", "go", "see", "number", "no", "way",
    "could", "people", "my", "than", "first", "water", "been", "call", "who", "oil",
    "its", "now", "find", "game", "movie", "phone", "check", "bro", "download", "link"
}


def audit_raw_dataset(
    raw_file_path: Path,
    persona_speaker: str = "Vivek Jha😎",
    conversation_gap_minutes: int = 120,
) -> Dict[str, Any]:
    """
    Perform deep audit of the raw WhatsApp export file.
    """
    if not raw_file_path.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_file_path}")

    with open(raw_file_path, "r", encoding="utf-8", errors="replace") as f:
        raw_lines = f.readlines()
    total_lines = len(raw_lines)

    parser = WhatsAppParser(persona_speaker=persona_speaker)
    messages = parser.parse_lines(raw_lines)
    total_parsed_messages = len(messages)
    unparsed_lines_count = len(parser.unparsed_lines)

    # Speakers distribution
    speaker_counts: Counter = Counter()
    for m in messages:
        speaker_counts[m.speaker] += 1

    # Date range and conversation periods
    timestamps: List[datetime] = []
    for m in messages:
        try:
            dt = datetime.strptime(m.timestamp, "%Y-%m-%d %H:%M:%S")
            timestamps.append(dt)
        except ValueError:
            pass

    date_range_start = timestamps[0].strftime("%Y-%m-%d %H:%M:%S") if timestamps else "N/A"
    date_range_end = timestamps[-1].strftime("%Y-%m-%d %H:%M:%S") if timestamps else "N/A"

    # Compute conversation sessions based on inactivity gap
    conversation_periods = 0
    if timestamps:
        conversation_periods = 1
        for i in range(1, len(timestamps)):
            gap = (timestamps[i] - timestamps[i - 1]).total_seconds() / 60.0
            if gap > conversation_gap_minutes:
                conversation_periods += 1

    # Message types
    type_counts = Counter(m.message_type for m in messages)
    media_count = type_counts.get("media", 0)
    deleted_count = type_counts.get("deleted", 0)
    system_count = type_counts.get("system", 0)
    url_count = type_counts.get("url", 0)
    empty_count = type_counts.get("empty", 0)
    text_count = type_counts.get("text", 0)

    # Content statistics for text messages
    emoji_count = 0
    hindi_devanagari_count = 0
    hinglish_count = 0
    english_count = 0
    text_lengths = []
    duplicate_counter = Counter()

    for m in messages:
        if m.message_type == "text":
            text_str = m.text.strip()
            text_lengths.append(len(text_str))
            duplicate_counter[(m.speaker, text_str)] += 1

            if m.contains_emoji:
                emoji_count += 1
            if m.contains_devanagari:
                hindi_devanagari_count += 1

            # Linguistic checks
            words = set(re.findall(r"[a-zA-Z]+", text_str.lower()))
            if any(k in words for k in HINGLISH_KEYWORDS):
                hinglish_count += 1
            if any(w in words for w in ENGLISH_WORDS):
                english_count += 1

    duplicate_messages_count = sum(c - 1 for c in duplicate_counter.values() if c > 1)

    avg_len = round(statistics.mean(text_lengths), 2) if text_lengths else 0
    median_len = round(statistics.median(text_lengths), 2) if text_lengths else 0
    max_len = max(text_lengths) if text_lengths else 0

    audit_data = {
        "raw_file_path": str(raw_file_path),
        "total_lines": total_lines,
        "total_parsed_messages": total_parsed_messages,
        "failed_or_unparsed_lines": unparsed_lines_count,
        "parsing_errors": parser.parsing_errors,
        "number_of_unique_speakers": len(speaker_counts),
        "messages_per_speaker": dict(speaker_counts),
        "persona_speaker": persona_speaker,
        "detected_partner_speakers": [s for s in speaker_counts.keys() if s not in (persona_speaker, "SYSTEM")],
        "date_range": {
            "start": date_range_start,
            "end": date_range_end,
            "duration_days": (timestamps[-1] - timestamps[0]).days if timestamps else 0,
        },
        "total_conversation_periods": conversation_periods,
        "conversation_gap_minutes": conversation_gap_minutes,
        "message_type_breakdown": {
            "text": text_count,
            "media_only": media_count,
            "url_only": url_count,
            "deleted": deleted_count,
            "system": system_count,
            "empty": empty_count,
        },
        "duplicate_messages": duplicate_messages_count,
        "linguistic_distribution": {
            "messages_containing_emojis": emoji_count,
            "messages_containing_hindi_devanagari": hindi_devanagari_count,
            "messages_containing_hinglish": hinglish_count,
            "messages_containing_english": english_count,
        },
        "text_length_statistics": {
            "average_length_chars": avg_len,
            "median_length_chars": median_len,
            "maximum_length_chars": max_len,
        },
    }

    return audit_data


def generate_markdown_report(data: Dict[str, Any]) -> str:
    """Generate clean, publication-ready Markdown audit report."""
    md = []
    md.append("# Persona Engine — Raw WhatsApp Export Audit Report")
    md.append("")
    md.append("## 1. Overview & Source File")
    md.append(f"- **Raw File Path:** `{data['raw_file_path']}`")
    md.append(f"- **Total File Lines:** `{data['total_lines']:,}`")
    md.append(f"- **Total Parsed Messages:** `{data['total_parsed_messages']:,}`")
    md.append(f"- **Unparsed / Failed Lines:** `{data['failed_or_unparsed_lines']}`")
    md.append(f"- **Parsing Errors:** `{len(data['parsing_errors'])}`")
    md.append("")
    md.append("## 2. Temporal & Conversation Dynamics")
    md.append(f"- **Date Range:** `{data['date_range']['start']}` to `{data['date_range']['end']}`")
    md.append(f"- **Active Span:** `{data['date_range']['duration_days']} days`")
    md.append(f"- **Inactivity Threshold:** `{data['conversation_gap_minutes']} minutes`")
    md.append(f"- **Reconstructed Conversation Sessions:** `{data['total_conversation_periods']:,}`")
    md.append("")
    md.append("## 3. Speakers & Participation")
    md.append(f"- **Unique Speakers:** `{data['number_of_unique_speakers']}`")
    md.append(f"- **Identified Persona Speaker:** `{data['persona_speaker']}`")
    md.append(f"- **Identified Partner(s):** `{', '.join(data['detected_partner_speakers'])}`")
    md.append("")
    md.append("| Speaker | Message Count | Percentage |")
    md.append("| :--- | :--- | :--- |")
    total_m = data["total_parsed_messages"]
    for spk, cnt in data["messages_per_speaker"].items():
        pct = (cnt / total_m) * 100 if total_m else 0
        md.append(f"| `{spk}` | {cnt:,} | {pct:.1f}% |")
    md.append("")
    md.append("## 4. Message Type Breakdown")
    md.append("| Message Type | Count | Description |")
    md.append("| :--- | :--- | :--- |")
    b = data["message_type_breakdown"]
    md.append(f"| **Text Messages** | {b['text']:,} | Standard conversational utterances |")
    md.append(f"| **Media-Only** | {b['media_only']:,} | `<Media omitted>` images, audio, video |")
    md.append(f"| **URL-Only** | {b['url_only']:,} | Standalone web links |")
    md.append(f"| **Deleted** | {b['deleted']:,} | Message deleted notices |")
    md.append(f"| **Empty** | {b['empty']:,} | Whitespace/blank exports |")
    md.append(f"| **System** | {b['system']:,} | Encryption and status notifications |")
    md.append("")
    md.append("## 5. Linguistic & Stylistic Characteristics (Text Messages)")
    ling = data["linguistic_distribution"]
    text_total = b["text"]
    md.append(f"- **Hinglish (Latin-Script Hindi):** `{ling['messages_containing_hinglish']:,}` ({ling['messages_containing_hinglish']/text_total*100:.1f}%)")
    md.append(f"- **English:** `{ling['messages_containing_english']:,}` ({ling['messages_containing_english']/text_total*100:.1f}%)")
    md.append(f"- **Hindi (Devanagari Unicode):** `{ling['messages_containing_hindi_devanagari']:,}`")
    md.append(f"- **Emoji Usage:** `{ling['messages_containing_emojis']:,}` ({ling['messages_containing_emojis']/text_total*100:.1f}%)")
    md.append(f"- **Duplicate Messages:** `{data['duplicate_messages']:,}`")
    md.append("")
    md.append("## 6. Text Length Statistics")
    lens = data["text_length_statistics"]
    md.append(f"- **Average Length:** `{lens['average_length_chars']} chars`")
    md.append(f"- **Median Length:** `{lens['median_length_chars']} chars`")
    md.append(f"- **Maximum Length:** `{lens['maximum_length_chars']} chars`")
    md.append("")
    md.append("## 7. Parsing Error Log")
    if not data["parsing_errors"]:
        md.append("*(Zero parsing errors encountered. All lines cleanly parsed.)*")
    else:
        md.append(f"Total Errors: {len(data['parsing_errors'])}")
        for err in data["parsing_errors"][:10]:
            md.append(f"- Line {err.get('line_number')}: {err.get('error')}")
    md.append("")
    return "\n".join(md)


def main():
    raw_path = ml_root / "data" / "raw" / "WhatsApp Chat with Naata.txt"
    out_dir = ml_root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[AUDIT] Running audit on: {raw_path}")
    audit_data = audit_raw_dataset(
        raw_file_path=raw_path,
        persona_speaker="Vivek Jha😎",
        conversation_gap_minutes=120,
    )

    json_path = out_dir / "raw_audit.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2, ensure_ascii=False)
    print(f"[AUDIT] Saved JSON audit report to: {json_path}")

    md_report = generate_markdown_report(audit_data)
    md_path = out_dir / "raw_audit.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_report)
    print(f"[AUDIT] Saved Markdown audit report to: {md_path}")

    print("\n" + "=" * 60)
    print("AUDIT SUMMARY")
    print("=" * 60)
    print(f"Total Lines:          {audit_data['total_lines']:,}")
    print(f"Parsed Messages:      {audit_data['total_parsed_messages']:,}")
    print(f"Speakers:             {audit_data['messages_per_speaker']}")
    print(f"Date Range:           {audit_data['date_range']['start']} -> {audit_data['date_range']['end']}")
    print(f"Conversations (120m): {audit_data['total_conversation_periods']:,}")
    print(f"Text Messages:        {audit_data['message_type_breakdown']['text']:,}")
    print(f"Media Omitted:        {audit_data['message_type_breakdown']['media_only']:,}")
    print(f"Parsing Errors:       {len(audit_data['parsing_errors'])}")
    print("=" * 60)


if __name__ == "__main__":
    main()
