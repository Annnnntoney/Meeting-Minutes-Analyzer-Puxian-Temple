from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")


def _pick_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # pragma: no cover - torch might be missing during type checks
        logger.debug("Torch not available; defaulting to CPU.")
    return "cpu"


@dataclass
class TranscriptChunk:
    start: float
    end: float
    text: str
    speaker: Optional[str]


class WhisperXTranscriber:
    """Thin wrapper around WhisperX with diarization support."""

    def __init__(
        self,
        model_size: str = "medium",
        compute_type: str = "int8",
        hf_token: Optional[str] = None,
        max_speakers: Optional[int] = None,
    ) -> None:
        self.model_size = model_size
        self.compute_type = compute_type
        self.device = _pick_device()
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.max_speakers = max_speakers

        self._model = None
        self._align_model = None
        self._align_metadata = None
        self._align_lang = None
        self._diarize_pipeline = None

    def _load_model(self):
        if self._model is None:
            import whisperx

            logger.info(
                "Loading WhisperX model size='%s' on %s (compute=%s)",
                self.model_size,
                self.device,
                self.compute_type,
            )
            self._model = whisperx.load_model(
                self.model_size, device=self.device, compute_type=self.compute_type
            )
        return self._model

    def _load_alignment(self, language: Optional[str]):
        if not language or language == self._align_lang:
            return

        import whisperx

        try:
            logger.info("Loading alignment model for language '%s'", language)
            self._align_model, self._align_metadata = whisperx.load_align_model(
                language_code=language, device=self.device
            )
            self._align_lang = language
        except Exception as exc:  # pragma: no cover - runtime failure handling
            logger.warning("Alignment model unavailable for %s: %s", language, exc)
            self._align_model = self._align_metadata = self._align_lang = None

    def _load_diarization(self):
        if self._diarize_pipeline is not None:
            return self._diarize_pipeline
        if not self.hf_token:
            raise RuntimeError(
                "Speaker diarization requires a free Hugging Face token. "
                "Set HF_TOKEN in the environment."
            )
        import whisperx

        diarization_kwargs: Dict[str, int] = {}
        if self.max_speakers:
            diarization_kwargs["min_speakers"] = 1
            diarization_kwargs["max_speakers"] = self.max_speakers

        logger.info("Initialising diarization pipeline (pyannote.audio)")
        self._diarize_pipeline = whisperx.DiarizationPipeline(
            use_auth_token=self.hf_token,
            device=self.device,
            **diarization_kwargs,
        )
        return self._diarize_pipeline

    @staticmethod
    def _iter_segments(result: Dict) -> Iterable[TranscriptChunk]:
        segments = result.get("segments", []) or []
        for segment in segments:
            yield TranscriptChunk(
                start=float(segment.get("start", 0.0)),
                end=float(segment.get("end", 0.0)),
                text=segment.get("text", "").strip(),
                speaker=segment.get("speaker"),
            )

    def transcribe(self, audio_path: Path) -> Tuple[str, List[TranscriptChunk]]:
        """Run ASR + diarization + (optional) alignment."""
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)

        model = self._load_model()

        import whisperx

        logger.info("Transcribing %s", audio_path)
        audio = whisperx.load_audio(str(audio_path))
        result = model.transcribe(audio, batch_size=16, vad_filter=True)
        language = result.get("language", "unknown")

        self._load_alignment(language)
        if self._align_model:
            try:
                logger.info("Running alignment for %s", audio_path.name)
                result = whisperx.align(
                    result["segments"],
                    self._align_model,
                    self._align_metadata,
                    audio,
                    device=self.device,
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("Alignment failed: %s", exc)

        try:
            diarization = self._load_diarization()
            diarize_segments = diarization(str(audio_path))
            result = whisperx.assign_word_speakers(diarize_segments, result)
        except Exception as exc:
            logger.warning("Diarization unavailable, falling back to single speaker: %s", exc)
            # Ensure every chunk has a speaker for downstream formatting.
            for segment in result.get("segments", []):
                segment.setdefault("speaker", "speaker_0")

        chunks = [chunk for chunk in self._iter_segments(result) if chunk.text]
        return language, chunks
