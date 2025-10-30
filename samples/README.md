# Samples 說明

此資料夾用來放測試音檔，建議準備：

- `mandarin.wav`：5~10 秒的中文（國語）對話或敘述。
- `taiwanese.wav`：5~10 秒的台語（閩南語）對話。

## 快速錄製方式

### 使用手機
1. 開啟手機錄音工具，各自錄製一段國語與台語。
2. 將檔案命名為 `mandarin.m4a`、`taiwanese.m4a`。
3. 複製到此資料夾。

### 使用電腦 + ffmpeg
```bash
ffmpeg -f avfoundation -i ":0" -t 8 -ar 16000 -ac 1 samples/mandarin.wav
ffmpeg -f avfoundation -i ":0" -t 8 -ar 16000 -ac 1 samples/taiwanese.wav
```
> 若為 Windows / Linux，請將 `":0"` 改成對應的輸入裝置代碼。

### 轉換格式
若原始檔是 MP3/M4A，可先轉成 Whisper 效率較好的 wav：
```bash
ffmpeg -i samples/mandarin.m4a -ar 16000 -ac 1 samples/mandarin.wav
ffmpeg -i samples/taiwanese.m4a -ar 16000 -ac 1 samples/taiwanese.wav
```

## 驗證翻譯品質
錄製後可透過 `scripts/warm_cache.py` 或手動 `curl` 對 `/transcribe` 發送請求，檢查：
- `language` 是否為 `zh`（國語）、`nan`（台語）。
- `conversation` 中 `Speaker A/B` 的內容是否正確。
- `summary` 的重點與關鍵字是否符合預期。

如需調整說話者數量，在 `.env` 修改 `APP_MAX_SPEAKERS` 並重新啟動服務，再次上傳音檔比較效果。
