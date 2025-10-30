from __future__ import annotations

import logging
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import Settings, get_settings
from .schemas import ConversationTurn, TranscriptSegment, TranscriptionResponse
from .services import (
    ConversationFormatter,
    SummaryService,
    TranslationService,
    WhisperXTranscriber,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Speech-to-Text Translator",
    description="Offline-first speech transcription API with diarization, translation, and summarisation.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="ui")


@lru_cache
def _transcriber_cache(
    model_size: str, compute_type: str, max_speakers: Optional[int]
) -> WhisperXTranscriber:
    return WhisperXTranscriber(
        model_size=model_size, compute_type=compute_type, max_speakers=max_speakers
    )


@lru_cache
def _formatter_cache() -> ConversationFormatter:
    return ConversationFormatter()


@lru_cache
def _summariser_cache(sentences: int, keywords: int) -> SummaryService:
    return SummaryService(sentences=sentences, keywords=keywords)


@lru_cache
def _translator_cache(model_name: str, target_lang: str) -> TranslationService:
    return TranslationService(model_name=model_name, target_lang=target_lang)


def _validate_extension(filename: str, settings: Settings) -> None:
    if "." not in filename:
        raise HTTPException(status_code=400, detail="File lacks an extension.")
    ext = filename.rsplit(".", 1)[1].lower()
    if ext not in settings.allowed_extensions:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file format '.{ext}'. Allowed: {settings.allowed_extensions}",
        )


def _persist_temp_file(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "").suffix or ".tmp"
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    with tmp_file as buffer:
        upload.file.seek(0)
        shutil.copyfileobj(upload.file, buffer)
    return Path(tmp_file.name)


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    translate: bool = Form(True),
    target_language: Optional[str] = Form(None),
    settings: Settings = Depends(get_settings),
) -> TranscriptionResponse:
    _validate_extension(file.filename or "upload", settings)
    temp_path = _persist_temp_file(file)

    try:
        transcriber = _transcriber_cache(
            settings.whisper_model_size,
            settings.whisper_compute_type,
            settings.max_speakers,
        )
        language, chunks = transcriber.transcribe(temp_path)

        formatter = _formatter_cache()
        labelled_chunks, _ = formatter.label_speakers(chunks)
        raw_text = " ".join(chunk.text for chunk in labelled_chunks)

        translated_segments = None
        if translate:
            target = target_language or settings.translator_target_lang
            translator = _translator_cache(settings.translator_model, target)
            translated_segments = translator.translate_segments(
                [chunk.text for chunk in labelled_chunks], language
            )
        else:
            translated_segments = [None for _ in labelled_chunks]

        summariser = _summariser_cache(
            settings.summary_sentences, settings.summary_keywords
        )
        summary_text_source = (
            " ".join(translated_segments) if translate else raw_text
        ).strip()
        summary_payload = summariser.summarise(summary_text_source)

        transcript_payload = [
            TranscriptSegment(
                speaker=chunk.speaker or "Speaker A",
                start=chunk.start,
                end=chunk.end,
                text=chunk.text,
                translated_text=translated_segments[idx]
                if translated_segments
                else None,
            )
            for idx, chunk in enumerate(labelled_chunks)
        ]

        dialogue = formatter.merge_runs(labelled_chunks, translated_segments)
        conversation_payload = [
            ConversationTurn(
                speaker=item["speaker"],
                text=item["text"],
                translated_text=item.get("translated_text"),
            )
            for item in dialogue
        ]

        return TranscriptionResponse(
            language=language,
            transcript=transcript_payload,
            conversation=conversation_payload,
            summary=summary_payload,
        )
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except FileNotFoundError:  # pragma: no cover - best-effort cleanup
            pass


@app.get("/healthz")
async def healthcheck() -> dict:
    return {"status": "ok"}


@app.get("/")
async def index() -> FileResponse | dict:
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"status": "ok"}
