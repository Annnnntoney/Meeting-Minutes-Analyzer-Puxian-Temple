from __future__ import annotations

import string
from typing import Dict, Iterable, List, Optional, Tuple

from .transcriber import TranscriptChunk


class ConversationFormatter:
    """Render diarized segments as readable dialogue."""

    SPEAKER_PREFIX = "Speaker "

    def __init__(self) -> None:
        self._alphabet = list(string.ascii_uppercase)

    def _label(self, ordinal: int) -> str:
        if ordinal < len(self._alphabet):
            suffix = self._alphabet[ordinal]
        else:
            suffix = f"X{ordinal}"
        return f"{self.SPEAKER_PREFIX}{suffix}"

    def label_speakers(self, chunks: Iterable[TranscriptChunk]) -> Tuple[List[TranscriptChunk], Dict[str, str]]:
        mapping: Dict[str, str] = {}
        relabelled: List[TranscriptChunk] = []
        for chunk in chunks:
            speaker_key = chunk.speaker or "speaker_0"
            if speaker_key not in mapping:
                mapping[speaker_key] = self._label(len(mapping))
            relabelled.append(
                TranscriptChunk(
                    start=chunk.start,
                    end=chunk.end,
                    text=chunk.text,
                    speaker=mapping[speaker_key],
                )
            )
        return relabelled, mapping

    def merge_runs(
        self,
        chunks: Iterable[TranscriptChunk],
        translations: Optional[Iterable[str]] = None,
    ) -> List[Dict[str, Optional[str]]]:
        chunks_list = list(chunks)
        translations_list = list(translations) if translations is not None else None

        dialogue: List[Dict[str, Optional[str]]] = []
        for idx, chunk in enumerate(chunks_list):
            translated_text = (
                translations_list[idx] if translations_list and idx < len(translations_list) else None
            )
            if dialogue and dialogue[-1]["speaker"] == chunk.speaker:
                dialogue[-1]["text"] = f"{dialogue[-1]['text']} {chunk.text}".strip()
                if translated_text:
                    existing = dialogue[-1].get("translated_text") or ""
                    dialogue[-1]["translated_text"] = (
                        f"{existing} {translated_text}".strip()
                    )
            else:
                dialogue.append(
                    {
                        "speaker": chunk.speaker or "Speaker A",
                        "text": chunk.text,
                        "translated_text": translated_text,
                    }
                )
        return dialogue
