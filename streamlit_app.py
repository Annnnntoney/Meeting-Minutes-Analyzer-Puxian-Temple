from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
from openai import OpenAI


@dataclass
class MeetingAnalysis:
    language: str
    transcript: str
    summary_points: List[str]
    keywords: List[str]
    conversation: List[Dict[str, str]]
    action_items: List[str]


def _load_openai_client() -> OpenAI:
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("請於環境變數或 `.streamlit/secrets.toml` 設定 OPENAI_API_KEY。")
        st.stop()
    return OpenAI(api_key=api_key)


def _transcribe_audio(client: OpenAI, file_path: Path, model: str) -> Dict[str, Any]:
    with file_path.open("rb") as audio_file:
        result = client.audio.transcriptions.create(
            model=model,
            file=audio_file,
            response_format="verbose_json",
        )
    return result.model_dump()


def _build_analysis_schema() -> Dict[str, Any]:
    return {
        "name": "MeetingAnalysis",
        "schema": {
            "type": "object",
            "properties": {
                "language": {"type": "string"},
                "summary": {
                    "type": "object",
                    "properties": {
                        "key_points": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "keywords": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "action_items": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["key_points", "keywords", "action_items"],
                },
                "conversation": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "speaker": {"type": "string"},
                            "original_text": {"type": "string"},
                            "traditional_chinese": {"type": "string"},
                            "notes": {"type": "string"},
                        },
                        "required": ["speaker", "original_text", "traditional_chinese"],
                    },
                },
            },
            "required": ["language", "summary", "conversation"],
        },
        "strict": True,
    }


def _analyse_meeting(
    client: OpenAI,
    transcript_text: str,
    target_language: str,
    model: str,
) -> MeetingAnalysis:
    schema = _build_analysis_schema()
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are an assistant that rewrites meeting notes. "
                            "Return structured Traditional Chinese output describing the conversation. "
                            "Keep faithful to the original speech, infer speaker turns when possible, "
                            "and include concise action items."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Transcript:\n"
                            f"{transcript_text}\n\n"
                            f"請將重點摘要、關鍵字與待辦事項整理成繁體中文。"
                            f"對話段落中，請標註 Speaker A/B/... 並提供原始語句與繁體中文翻譯。"
                            f"如果僅有一位發言者，仍使用 Speaker A 表示。"
                            f"翻譯語言請使用 {target_language}。"
                            "避免新增未出現的資訊，必要時可以整合語句。"
                        ),
                    }
                ],
            },
        ],
        response_format={"type": "json_schema", "json_schema": schema},
    )

    payload = json.loads(response.output[0].content[0].text)
    summary = payload["summary"]
    return MeetingAnalysis(
        language=payload.get("language", "unknown"),
        transcript=transcript_text,
        summary_points=summary.get("key_points", []),
        keywords=summary.get("keywords", []),
        action_items=summary.get("action_items", []),
        conversation=payload.get("conversation", []),
    )


def _render_summary(analysis: MeetingAnalysis) -> None:
    st.subheader("摘要重點")
    if analysis.summary_points:
        for point in analysis.summary_points:
            st.markdown(f"- {point}")
    else:
        st.info("無摘要內容。")

    if analysis.action_items:
        st.subheader("待辦事項")
        for item in analysis.action_items:
            st.markdown(f"- ✅ {item}")

    if analysis.keywords:
        st.subheader("關鍵字")
        st.write("、".join(analysis.keywords))


def _render_conversation(analysis: MeetingAnalysis) -> None:
    st.subheader("對話紀錄")
    if not analysis.conversation:
        st.info("無對話資料。")
        return

    for turn in analysis.conversation:
        speaker = turn.get("speaker", "Speaker")
        original = turn.get("original_text", "")
        translated = turn.get("traditional_chinese", "")
        notes = turn.get("notes")

        st.markdown(f"**{speaker}**")
        if original:
            st.markdown(f"- 原文：{original}")
        if translated and translated != original:
            st.markdown(f"- 翻譯：{translated}")
        if notes:
            st.markdown(f"- 備註：{notes}")
        st.divider()


def _render_downloads(analysis: MeetingAnalysis, raw_transcript: Dict[str, Any]) -> None:
    export_payload = {
        "language": analysis.language,
        "summary_points": analysis.summary_points,
        "keywords": analysis.keywords,
        "action_items": analysis.action_items,
        "conversation": analysis.conversation,
        "raw_transcription": raw_transcript,
    }
    st.download_button(
        label="下載 JSON",
        data=json.dumps(export_payload, ensure_ascii=False, indent=2),
        mime="application/json",
        file_name="meeting_analysis.json",
    )


def main() -> None:
    st.set_page_config("會議記錄分析器・普賢宮", layout="wide")
    st.title("會議記錄分析器・普賢宮")
    st.caption("上傳台語或國語錄音，透過 OpenAI API 進行轉錄、翻譯與摘要。")

    client = _load_openai_client()

    with st.sidebar:
        st.header("設定")
        transcription_model = st.selectbox(
            "轉錄模型",
            options=[
                "gpt-4o-mini-transcribe",
                "gpt-4o-transcribe",
                "whisper-1",
            ],
            index=0,
        )
        analysis_model = st.selectbox(
            "分析模型",
            options=[
                "gpt-4o-mini",
                "gpt-4o",
                "o4-mini",
            ],
            index=0,
        )
        target_language = st.selectbox(
            "翻譯輸出語言",
            options=["繁體中文", "English", "日本語"],
            index=0,
        )

    uploaded_file = st.file_uploader(
        "上傳音檔", type=["mp3", "wav", "m4a", "aac", "flac", "ogg"]
    )

    if not uploaded_file:
        st.info("請先選擇音檔。")
        return

    if st.button("開始分析", type="primary"):
        with st.spinner("轉錄與分析中，請稍候..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                tmp.write(uploaded_file.read())
                temp_path = Path(tmp.name)

            try:
                transcription = _transcribe_audio(client, temp_path, transcription_model)
                transcript_text = transcription.get("text", "")
                if not transcript_text:
                    st.error("未取得轉錄結果，請稍後再試。")
                    return

                analysis = _analyse_meeting(
                    client=client,
                    transcript_text=transcript_text,
                    target_language=target_language,
                    model=analysis_model,
                )

            except Exception as exc:
                st.error(f"處理過程發生錯誤：{exc}")
                return
            finally:
                temp_path.unlink(missing_ok=True)

        st.success("分析完成！")
        _render_summary(analysis)
        _render_conversation(analysis)
        with st.expander("查看原始逐字稿"):
            st.write(analysis.transcript)
        _render_downloads(analysis, transcription)


if __name__ == "__main__":
    main()

