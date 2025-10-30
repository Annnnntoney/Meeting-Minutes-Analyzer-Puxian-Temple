#!/usr/bin/env python3
"""簡易腳本：對本地 FastAPI 服務呼叫 /transcribe 以預先下載模型。"""

import argparse
import sys
from pathlib import Path
from typing import Iterable, List

import requests


def iter_audio_files(inputs: Iterable[str]) -> Iterable[Path]:
    for item in inputs:
        path = Path(item).expanduser().resolve()
        if path.is_dir():
            yield from sorted(path.glob("**/*.wav"))
            yield from sorted(path.glob("**/*.mp3"))
            yield from sorted(path.glob("**/*.m4a"))
            yield from sorted(path.glob("**/*.flac"))
            yield from sorted(path.glob("**/*.ogg"))
        elif path.is_file():
            yield path
        else:
            print(f"[warm-cache] 找不到檔案/資料夾：{item}", file=sys.stderr)


def send_request(
    endpoint: str, audio_path: Path, translate: bool, target_language: str | None
) -> None:
    with audio_path.open("rb") as handle:
        files = {"file": (audio_path.name, handle, "application/octet-stream")}
        data = {"translate": str(translate).lower()}
        if target_language:
            data["target_language"] = target_language

        print(f"[warm-cache] POST {endpoint} ← {audio_path.name}")
        resp = requests.post(endpoint, files=files, data=data, timeout=600)
        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            print(f"[warm-cache] 發送失敗：{exc} - {resp.text}", file=sys.stderr)
            return
        payload = resp.json()
        lang = payload.get("language")
        speakers = {seg["speaker"] for seg in payload.get("transcript", [])}
        print(
            f"[warm-cache] ✔ 完成，偵測語言：{lang}，"
            f"說話者數：{len(speakers)}，摘要句數：{len(payload.get('summary', {}).get('key_points', []))}"
        )


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        help="單一音檔或資料夾，可傳多個。資料夾會自動搜尋常見音訊副檔名。",
    )
    parser.add_argument(
        "--endpoint",
        default="http://localhost:8000/transcribe",
        help="FastAPI 服務的 /transcribe URL。",
    )
    parser.add_argument(
        "--no-translate",
        action="store_true",
        help="僅做語音辨識（不翻譯），適合在翻譯模型尚未下載時先測試。",
    )
    parser.add_argument(
        "--target-language",
        default=None,
        help="翻譯目標語言（NLLB 語言碼，例如 zho_Hant、eng_Latn、nan_Latn）。",
    )

    args = parser.parse_args(argv)
    audio_files = list(iter_audio_files(args.inputs))
    if not audio_files:
        print("[warm-cache] 找不到任何音訊檔。", file=sys.stderr)
        return 1

    for audio_path in audio_files:
        send_request(
            endpoint=args.endpoint,
            audio_path=audio_path,
            translate=not args.no_translate,
            target_language=args.target_language,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

