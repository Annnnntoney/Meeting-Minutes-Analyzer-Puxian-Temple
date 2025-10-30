# Speech-to-Text Translator Service

## 概述
這是一個可本地運行與雲端部署的語音轉文字服務。流程包含：
- 使用開源的 **WhisperX / Faster-Whisper** 進行語音辨識與語言自動偵測。
- 透過 **pyannote.audio** 做說話者分離，輸出對話者標籤（A/B/...）。
- 使用 **NLLB-200**（開源的序列到序列翻譯模型）進行目標語言翻譯，預設輸出繁體中文。
- 利用 **TextRank** 產生摘要重點與關鍵字，不依賴任何付費大型語言模型。
- 回傳完整逐字稿、對話格式（A/B 誰說了什麼）、以及摘要資訊的 JSON。

整個堆疊皆為免費開源模型，可在 CPU 上運行（建議使用具備 AVX2 指令集的 x86-64 或 Apple Silicon）。若有 GPU 可加速推論。

## 主要功能
- 多語系語音辨識，支援台語 (`nan`)、國語 (`zh`) 等 Whisper 支援的語言。
- 自動說話者分離，輸出 `Speaker A/B/...` 對應的對話。
- 每段語句可選擇翻譯成指定語言（預設 `zho_Hant`，繁體中文）。
- 摘要功能：輸出重點句子與關鍵字列表。
- REST API 介面，可透過 curl 或任意 HTTP client 送出音訊檔案。
- 可 Docker 化，方便部署至 Render、Fly.io、Railway、Google Cloud Run 等免費/低成本平台。

## 系統需求
1. **Python 3.10** 或以上版本。
2. **FFmpeg**：音訊轉碼與處理必備。
3. **Hugging Face Token**：免費註冊即可取得，供 `pyannote.audio` 下載 diarization 模型。
4. 硬體：
   - CPU：建議 4 核心以上，記憶體 8 GB 以上。
   - GPU（選用）：若有 CUDA GPU，可將 `HF_TOKEN` 設定好後自動使用 GPU 加速。
5. 儲存空間：初次啟動會下載 Whisper 模型、NLLB 翻譯模型與 pyannote 模型，約 6~8 GB。

## 安裝步驟
```bash
git clone <repo-url>
cd 普賢宮翻譯器

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

複製環境設定檔並填入 Hugging Face token：
```bash
cp .env.example .env
```
填寫 `.env` 中的 `HF_TOKEN=hf_xxx`。

## 本地啟動
```bash
uvicorn app.main:app --reload --port 8000
```

### 健康檢查
```
GET http://localhost:8000/healthz
```

### 上傳音訊範例
```bash
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@sample.wav" \
  -F "translate=true" \
  -F "target_language=zho_Hant"
```

回傳 JSON 包含：
- `language`：Whisper 偵測的語言（如 `zh`、`nan`）。
- `transcript`：每段帶有 `start`、`end`、`speaker`、`text` 與 `translated_text`（若啟用翻譯）。
- `conversation`：將相同說話者連續句子合併後的對話列表。
- `summary`：`key_points`（摘要句）與 `keywords`。

## 前端介面
- API 啟動後可直接瀏覽 `http://localhost:8000/` 使用內建上傳頁面（同時掛載在 `/ui`）。
- 介面提供音檔上傳、翻譯開關、目標語言選擇，並顯示摘要、對話以及完整 JSON。
- 若要自訂介面，可編輯 `frontend/index.html`；FastAPI 會自動提供最新版本。

## 模型快取與驗證流程
1. 於 `.env` 設定 `HF_TOKEN`，並依需要調整 `APP_MAX_SPEAKERS`。
2. 準備測試音檔（參考 `samples/README.md`，建議各一段國語、台語）。
3. 啟動 API 後執行：
   ```bash
   python scripts/warm_cache.py samples/mandarin.wav samples/taiwanese.wav
   ```
   此腳本會對 `/transcribe` 發送請求，第一次呼叫會自動下載 Whisper、NLLB、pyannote 等模型。
4. 觀察腳本輸出或 API 回傳的 `language`、`conversation`、`summary`。若分離出的說話者不足，可在 `.env` 將 `APP_MAX_SPEAKERS` 調大（例如 4），再重新啟動服務測試。

## 組態說明
可透過環境變數調整模型與輸出：
- `APP_WHISPER_MODEL_SIZE`：Whisper 模型，預設 `medium`。可改 `small`/`large-v2` 等。
- `APP_WHISPER_COMPUTE_TYPE`：量化方式 `int8`/`int8_float16` 等，影響 CPU 記憶體占用。
- `APP_MAX_SPEAKERS`：預期最大說話者數，用於 diarization。
- `APP_TRANSLATOR_TARGET_LANG`：翻譯輸出語言（NLLB 語言碼，預設繁中 `zho_Hant`）。
- `APP_SUMMARY_SENTENCES`、`APP_SUMMARY_KEYWORDS`：摘要句數與關鍵字數。

若想關閉翻譯，可在呼叫 API 時將 `translate=false`。

## 雲端部署
### Docker 建置
```bash
docker build -t speech-translator .
docker run --env-file .env -p 8000:8000 speech-translator
```

### 部署建議
1. **Render / Railway / Fly.io**：提供免費層，可直接使用 Docker 映像或 `uvicorn` 启动命令。
2. **Google Cloud Run / AWS App Runner**：支援容器化部署，按需計費。
3. **Heroku（含容器）**：需付費層才能長期運行，但設定簡易。

> 注意：第一次啟動容器時會於 `/root/.cache` 下載模型，若平台提供暫存磁碟，請確保大小足夠（>8 GB）。

### 建議的資料永續化
- Whisper / NLLB 模型快取預設存在 `~/.cache/huggingface`。在雲端環境可使用掛載磁碟或啟動時預先下載。
- 若要記錄處理結果，可在程式中擴充（如寫入資料庫、雲端儲存）。

## 模組化架構
- `app/services/transcriber.py`：WhisperX + diarization，產生說話者分段。
- `app/services/translator.py`：NLLB 翻譯器，支援自訂目標語言。
- `app/services/summarizer.py`：TextRank 摘要與關鍵字萃取。
- `app/services/conversation.py`：對話格式化，輸出 `Speaker A/B/...`。
- `app/main.py`：FastAPI 定義 /transcribe 與 /healthz。
- `app/schemas.py`：Pydantic 模型，確保 API 回傳格式穩定。

## 擴充思路
1. **自動語言目標對應**：可依偵測語言自動選擇翻譯 `source_lang` 與 `target_lang`。
2. **支援多種摘要方法**：可整合 Qwen、LLaMA 等開源 LLM（可透過量化模型搭配 `llama-cpp-python`）。
3. **加入任務列隊**：大量音檔可配合 Celery、RQ 或任務佇列，避免同步請求超時。
4. **前端 UI**：可加上 React / Vue 前端或使用 Streamlit/BentoML。

## 測試與品質建議
- 單元測試位於 `tests/`，使用 `pytest`：
  ```bash
  pip install -r requirements-dev.txt
  pytest
  ```
- 可額外建立整合測試，上傳一小段音檔檢查 JSON 結構。
- 由於模型較大，請務必在部署環境先行下載，並設定健康檢查避免冷啟動超時。

## 許可與授權
- Faster-Whisper / WhisperX：MIT License。
- pyannote.audio：AGPL-3.0（推論可使用，若修改程式需遵守開源條款）。
- NLLB-200：Meta Research License (MIT-like，允許商業使用)。
- TextRank4ZH / jieba：MIT。
- FastAPI / Uvicorn：MIT。

> 使用者須確保遵循各模型授權條款，特別是若對 pyannote 做修改或重新散布。

---

如需協助整合前端或額外功能，歡迎隨時提出！祝開發順利。 👍
