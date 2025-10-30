from typing import List, Optional

from pydantic import BaseModel, Field


class TranscriptSegment(BaseModel):
    speaker: str = Field(description="Canonical speaker label (Speaker A/B/...).")
    start: float = Field(description="Segment start time in seconds.")
    end: float = Field(description="Segment end time in seconds.")
    text: str = Field(description="Recognized text for the segment.")
    translated_text: Optional[str] = Field(
        default=None, description="Translated variant of the segment text."
    )


class ConversationTurn(BaseModel):
    speaker: str = Field(description="Speaker label for the merged utterance.")
    text: str = Field(description="Utterance rendered as natural dialogue.")
    translated_text: Optional[str] = None


class SummaryPayload(BaseModel):
    key_points: List[str] = Field(description="High-level bullet points.")
    keywords: List[str] = Field(description="Keyword list extracted via TextRank.")


class TranscriptionResponse(BaseModel):
    language: str = Field(description="Language detected by Whisper.")
    transcript: List[TranscriptSegment] = Field(
        description="Per-segment transcript with diarization."
    )
    conversation: List[ConversationTurn] = Field(
        description="Dialogue-style representation of the conversation."
    )
    summary: SummaryPayload = Field(description="Derived summary artefacts.")

