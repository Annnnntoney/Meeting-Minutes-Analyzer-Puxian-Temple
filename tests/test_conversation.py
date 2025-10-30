from app.services.conversation import ConversationFormatter
from app.services.transcriber import TranscriptChunk


def test_label_speakers_assigns_incremental_labels():
    chunks = [
        TranscriptChunk(start=0.0, end=1.0, text="你好", speaker="speaker_0"),
        TranscriptChunk(start=1.0, end=2.0, text="大家好", speaker="speaker_1"),
        TranscriptChunk(start=2.0, end=3.0, text="很高興見到你", speaker="speaker_0"),
    ]

    formatter = ConversationFormatter()
    relabelled, mapping = formatter.label_speakers(chunks)

    assert mapping == {"speaker_0": "Speaker A", "speaker_1": "Speaker B"}
    assert relabelled[0].speaker == "Speaker A"
    assert relabelled[1].speaker == "Speaker B"
    assert relabelled[2].speaker == "Speaker A"


def test_merge_runs_concatenates_text_and_translations():
    chunks = [
        TranscriptChunk(start=0.0, end=1.0, text="你好", speaker="Speaker A"),
        TranscriptChunk(start=1.0, end=2.0, text="嗎", speaker="Speaker A"),
        TranscriptChunk(start=2.0, end=3.0, text="我很好", speaker="Speaker B"),
    ]
    translations = ["hello", "there", "I'm fine"]

    formatter = ConversationFormatter()
    dialogue = formatter.merge_runs(chunks, translations)

    assert len(dialogue) == 2
    assert dialogue[0]["speaker"] == "Speaker A"
    assert dialogue[0]["text"] == "你好 嗎"
    assert dialogue[0]["translated_text"] == "hello there"
    assert dialogue[1]["speaker"] == "Speaker B"
    assert dialogue[1]["translated_text"] == "I'm fine"
