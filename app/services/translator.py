from __future__ import annotations

import logging
from functools import lru_cache
from typing import Iterable, List, Optional

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logger = logging.getLogger(__name__)


LANGUAGE_CODE_MAP = {
    "zh": "zho_Hant",
    "zh-cn": "zho_Hans",
    "zh_tw": "zho_Hant",
    "yue": "yue_Hant",
    "en": "eng_Latn",
    "nan": "nan_Latn",
    "nan-tw": "nan_Latn",
}


class TranslationService:
    """Wrapper around an open-source NMT model (default NLLB-200)."""

    def __init__(self, model_name: str, target_lang: str = "zho_Hant") -> None:
        self.model_name = model_name
        self.target_lang = target_lang
        self._tokenizer = None
        self._model = None

    def _load(self):
        if self._model is None or self._tokenizer is None:
            logger.info("Loading translation model '%s'", self.model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        return self._tokenizer, self._model

    def _resolve_lang(self, detected: Optional[str]) -> Optional[str]:
        if not detected:
            return None
        normalised = detected.lower()
        return LANGUAGE_CODE_MAP.get(normalised, normalised)

    def translate_text(self, text: str, source_lang: Optional[str]) -> str:
        if not text.strip():
            return ""

        src_lang = self._resolve_lang(source_lang)
        tokenizer, model = self._load()

        if src_lang and src_lang not in tokenizer.lang_code_to_id:
            logger.warning("Unsupported source language '%s'; skipping translation", src_lang)
            return text

        try:
            if src_lang:
                tokenizer.src_lang = src_lang
            forced_bos = tokenizer.lang_code_to_id.get(self.target_lang)
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            generated = model.generate(
                **inputs, forced_bos_token_id=forced_bos, max_length=768
            )
            return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        except Exception as exc:  # pragma: no cover - runtime failure handling
            logger.warning("Translation failed; returning original text: %s", exc)
            return text

    def translate_segments(
        self, segments: Iterable[str], source_lang: Optional[str]
    ) -> List[str]:
        translated: List[str] = []
        for text in segments:
            translated.append(self.translate_text(text, source_lang))
        return translated

