"""
Persona Engine — Dataset File Loader
Supports loading, streaming, and saving JSON and JSONL datasets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Generator, List, Union


def normalize_conversation(conv: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizes various conversation representations into the standard schema:
    {
        "conversation_id": str,
        "persona_id": str,
        "messages": [{"speaker": str, "text": str}],
        "metadata": dict
    }
    Supports:
    - Standard multi-turn format (has 'messages' with 'speaker' and 'text')
    - WhatsApp context-response pair format (has 'context' and 'response')
    - SFT / ChatML format (has 'messages' with 'role' and 'content', and optional 'target')
    """
    if not isinstance(conv, dict):
        return conv

    # 1. Format: context + response pair (e.g. WhatsApp export samples)
    if "context" in conv and "response" in conv:
        conv_id = str(conv.get("conversation_id") or conv.get("example_id") or "conv-unknown")
        persona_spk = str(conv.get("persona_speaker") or conv.get("persona_id") or conv.get("target_speaker") or "persona")
        messages = []
        context = conv.get("context", [])
        if isinstance(context, list):
            for c in context:
                if isinstance(c, dict):
                    spk = str(c.get("speaker") or c.get("role") or "user")
                    txt = str(c.get("text") if "text" in c else c.get("content", ""))
                    messages.append({"speaker": spk, "text": txt})
        messages.append({"speaker": persona_spk, "text": str(conv.get("response", ""))})
        return {
            "conversation_id": conv_id,
            "persona_id": persona_spk,
            "messages": messages,
            "metadata": conv.get("metadata", {}),
        }

    # 2. Format: messages list where items use role / content (SFT format)
    messages = conv.get("messages")
    if isinstance(messages, list) and messages:
        first = messages[0]
        if isinstance(first, dict) and "role" in first and "speaker" not in first:
            conv_id = str(conv.get("conversation_id") or conv.get("example_id") or "conv-unknown")
            persona_spk = str(conv.get("persona_id") or "persona")
            norm_messages = []
            for m in messages:
                if isinstance(m, dict):
                    role = str(m.get("role", "user"))
                    content = str(m.get("content") if "content" in m else m.get("text", ""))
                    spk = "user" if role == "user" else ("persona" if role == "assistant" else role)
                    norm_messages.append({"speaker": spk, "text": content})
            if "target" in conv:
                norm_messages.append({"speaker": persona_spk, "text": str(conv.get("target", ""))})
            return {
                "conversation_id": conv_id,
                "persona_id": persona_spk,
                "messages": norm_messages,
                "metadata": conv.get("metadata", {}),
            }

    # 3. Already standard format or missing messages
    return conv


def load_jsonl(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Loads all records from a JSON Lines (.jsonl) file."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                records.append(json.loads(line_str))
            except json.JSONDecodeError as err:
                raise ValueError(
                    f"Invalid JSON on line {line_num} in {path.name}: {err}"
                ) from err
    return records


def iter_jsonl(file_path: Union[str, Path]) -> Generator[Dict[str, Any], None, None]:
    """Memory-efficient streaming generator for reading large JSONL files."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                yield json.loads(line_str)
            except json.JSONDecodeError as err:
                raise ValueError(
                    f"Invalid JSON on line {line_num} in {path.name}: {err}"
                ) from err


def save_jsonl(
    data: List[Dict[str, Any]],
    file_path: Union[str, Path],
    ensure_ascii: bool = False,
) -> Path:
    """Saves a list of dictionaries to a JSON Lines (.jsonl) file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=ensure_ascii) + "\n")
    return path


def load_json(file_path: Union[str, Path]) -> Any:
    """Loads a standard JSON file."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(
    data: Any,
    file_path: Union[str, Path],
    indent: int = 2,
    ensure_ascii: bool = False,
) -> Path:
    """Saves data to a standard JSON file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
    return path
