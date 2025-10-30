from __future__ import annotations

import logging
from typing import List

from textrank4zh import TextRank4Keyword, TextRank4Sentence
from jieba import analyse

logger = logging.getLogger(__name__)


class SummaryService:
    """Generate lightweight extractive summaries without external LLMs."""

    def __init__(self, sentences: int = 4, keywords: int = 6) -> None:
        self.target_sentences = max(1, sentences)
        self.target_keywords = max(1, keywords)

    @staticmethod
    def _normalise(text: str) -> str:
        return text.replace("\n", " ").strip()

    def summarise(self, text: str) -> dict:
        cleaned = self._normalise(text)
        if not cleaned:
            return {"key_points": [], "keywords": []}

        try:
            sentence_ranker = TextRank4Sentence()
            sentence_ranker.analyze(cleaned, lower=False)
            sentences = [
                item.sentence.strip()
                for item in sentence_ranker.get_key_sentences(
                    num=self.target_sentences
                )
            ]

            keyword_ranker = TextRank4Keyword()
            keyword_ranker.analyze(cleaned, lower=True)
            keywords = [
                item.word for item in keyword_ranker.get_keywords(self.target_keywords)
            ]
        except Exception as exc:  # pragma: no cover - third-party failure path
            logger.warning("Summarisation failed: %s", exc)
            sentences = self._fallback_sentences(cleaned)
            keywords = self._fallback_keywords(cleaned)

        return {"key_points": sentences, "keywords": keywords}

    def _fallback_sentences(self, text: str) -> List[str]:
        delimiters = ("。", "！", "？", ".", "!", "?")
        segments: List[str] = []
        current = []
        for char in text:
            current.append(char)
            if char in delimiters and current:
                segments.append("".join(current).strip())
                current = []
        if current:
            segments.append("".join(current).strip())
        if not segments:
            segments = [text[:80]]
        return segments[: self.target_sentences]

    def _fallback_keywords(self, text: str) -> List[str]:
        try:
            return analyse.extract_tags(
                text, topK=self.target_keywords, allowPOS=("n", "ns", "nt", "nz")
            )
        except Exception:  # pragma: no cover - jieba fallback safety
            return []
