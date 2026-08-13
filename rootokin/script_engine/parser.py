"""
Rootokin Script Engine – turns free-form text into structured beats
with characters, camera cues, emotional arcs and temporal hints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import re

CHARACTER_PATTERN = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",
    re.UNICODE,
)

CAMERA_KEYWORDS = {
    "close-up": "close-up",
    "closeup": "close-up",
    "wide shot": "wide",
    "wide": "wide",
    "tracking": "tracking shot",
    "dolly": "dolly",
    "pan": "pan",
    "zoom": "zoom",
    "overhead": "overhead",
    "aerial": "aerial",
    "pov": "pov",
    "over-the-shoulder": "over-the-shoulder",
    "ots": "over-the-shoulder",
}

EMOTION_KEYWORDS = {
    "angry": "anger",
    "furious": "anger",
    "sad": "sadness",
    "cry": "sadness",
    "happy": "joy",
    "smile": "joy",
    "laugh": "joy",
    "fear": "fear",
    "scared": "fear",
    "tense": "tension",
    "calm": "calm",
    "surprised": "surprise",
    "determined": "determination",
    "loving": "affection",
    "romantic": "affection",
}

STOP_WORDS = {"the", "a", "an", "and", "then", "with", "from", "into", "scene"}


def _extract_characters(text: str, known: Optional[List[str]] = None) -> List[str]:
    known = known or []
    found = set()
    for name in known:
        if name.lower() in text.lower():
            found.add(name)
    for match in CHARACTER_PATTERN.finditer(text):
        word = match.group(1)
        if word.lower() not in STOP_WORDS:
            found.add(word)
    return sorted(found) if found else ["main"]


def _extract_camera(text: str) -> str:
    lower = text.lower()
    for key, value in CAMERA_KEYWORDS.items():
        if key in lower:
            return value
    return ""


def _extract_emotion(text: str) -> str:
    lower = text.lower()
    for key, value in EMOTION_KEYWORDS.items():
        if key in lower:
            return value
    return ""


def _normalize_json_beats(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return [beat if isinstance(beat, dict) else {"text": str(beat), "characters": []} for beat in data]
    if isinstance(data, dict) and isinstance(data.get("beats"), list):
        return _normalize_json_beats(data["beats"])
    raise ValueError("Unsupported script JSON format")


def _parse_script_text(raw: str, known_characters: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    raw = raw.strip()
    if not raw:
        return [{"text": "Empty script", "characters": ["main"], "camera": "", "emotion": ""}]

    blocks = re.split(r"\n\s*\n|Scene\s+\d+[:.]?\s*", raw, flags=re.IGNORECASE)
    blocks = [block.strip() for block in blocks if block.strip()]

    beats: List[Dict[str, Any]] = []
    for block in blocks:
        sentences = re.split(r"(?<=[.!?])\s+", block)
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 8:
                continue
            beats.append(
                {
                    "text": sent,
                    "characters": _extract_characters(sent, known_characters),
                    "camera": _extract_camera(sent),
                    "emotion": _extract_emotion(sent),
                    "notes": "",
                }
            )

    if not beats:
        beats.append(
            {
                "text": raw[:300],
                "characters": _extract_characters(raw, known_characters),
                "camera": "",
                "emotion": "",
                "notes": "fallback",
            }
        )

    return beats


def parse_script(
    script_path: Path,
    known_characters: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Parse a plain-text script into structured beats.
    Also preserves the existing JSON beat format used by tests and examples.
    """
    raw = script_path.read_text(encoding="utf-8").strip()
    if script_path.suffix.lower() == ".json":
        return _normalize_json_beats(json.loads(raw or "[]"))
    return _parse_script_text(raw, known_characters)


def parse_script_text(text: str, known_characters: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Convenience for in-memory strings."""
    return _parse_script_text(text, known_characters)
