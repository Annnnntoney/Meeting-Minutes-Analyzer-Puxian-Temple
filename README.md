# 會議記錄分析器・普賢宮

使用 Streamlit 與 OpenAI API 建置的語音轉文字翻譯器。上傳台語或國語錄音，系統會：

1. 透過 OpenAI 轉錄模型（`gpt-4o-mini-transcribe` / `whisper-1` 等）取得逐字稿。
2. 使用 GPT 模型將內容整理成摘要重點、待辦事項與關鍵字。
3. 自動生成「Speaker A/B/...」對話紀錄，並提供繁體中文翻譯。
4. 匯出完整 JSON 以便後續存檔或分析。

## 安裝與環境設定

1. 建立虛擬環境並安裝套件：
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate

   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. 設定 OpenAI API Key（需具備 Audio/Responses 權限）：
   ```bash
   cp .env.example .env
   # 編輯 .env，將 OPENAI_API_KEY 替換為自己的金鑰
   ```

   或在 `~/.streamlit/secrets.toml` 內設定：
   ```toml
   OPENAI_API_KEY = "sk-xxxxx"
   ```

## 執行

```bash
streamlit run streamlit_app.py
```

啟動後於瀏覽器開啟 `http://localhost:8501`。介面提供：
- 音檔上傳（mp3/wav/m4a/flac/ogg...）
- 轉錄與分析模型選擇
- 目標翻譯語言（預設繁體中文）
- 摘要、待辦、關鍵字與對話視圖
- 原始逐字稿檢視與 JSON 下載

## 部署建議

- **Streamlit Cloud**：直接匯入 GitHub 專案，於 Secrets 設定 `OPENAI_API_KEY`。
- **Render / Railway / Fly.io**：使用 `streamlit_app.py` 直接啟動，部署命令 `streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0`。
- **Docker**：可額外建立 Dockerfile，於容器內設定環境變數 `OPENAI_API_KEY` 後啟動 Streamlit。

## 注意事項

- OpenAI 轉錄可能無法提供真正的聲紋分離，對話欄位的 Speaker A/B 為模型推測，遇多位講者時建議人工複核。
- 上傳大型音檔會消耗較多 Token 與費用，建議事先剪輯至 30 分鐘以內。
- 若需要更精確的說話者分離，可整合第三方 diarization 服務或自行擴充程式碼。

## 授權

此專案程式碼以 MIT License 釋出，但使用 OpenAI API 需遵循官方服務條款並自行承擔對應費用。***
