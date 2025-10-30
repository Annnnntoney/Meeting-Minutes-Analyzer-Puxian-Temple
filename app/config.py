from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Runtime configuration sourced from environment variables."""

    whisper_model_size: str = Field(
        default="medium",
        description="Whisper model size to load through whisperx/faster-whisper.",
    )
    whisper_compute_type: str = Field(
        default="int8",
        description="Quantization strategy for faster-whisper (int8/int8_float16/int16).",
    )
    max_speakers: Optional[int] = Field(
        default=None,
        description="Optional upper bound for number of speakers when diarizing.",
    )
    translator_model: str = Field(
        default="facebook/nllb-200-distilled-600M",
        description="Seq2Seq model used for neural machine translation.",
    )
    translator_target_lang: str = Field(
        default="zho_Hant",
        description="Target language code for translation (NLLB language codes).",
    )
    summary_sentences: int = Field(
        default=4, description="Number of key sentences to keep in the summary."
    )
    summary_keywords: int = Field(
        default=6, description="Number of keywords to surface from the transcript."
    )
    allowed_extensions: tuple[str, ...] = Field(
        default=("wav", "mp3", "m4a", "flac", "ogg"),
        description="File extensions accepted by the API.",
    )

    class Config:
        env_prefix = "APP_"
        env_file = ".env"
        case_sensitive = False


@lru_cache
def get_settings() -> Settings:
    """Expose cached settings instance for use across the app."""
    return Settings()

