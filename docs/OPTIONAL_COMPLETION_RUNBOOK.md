# Optional completion runbook

這份文件只處理兩個無法由 repository 單方面完成的選配項目：

1. 5–10 張真實繁中手寫／印章收據，需要資料擁有者提供並人工確認 gold。
2. LlamaParse 商業 comparator，需要資料擁有者自行建立 API key、同意第三方上傳與承擔費用。

兩項都不影響本機可重現的 receipt benchmark、complex-document benchmark 或既有
NO-GO 結論。不要把 key、真實收據、逐張 label 或本機 manifest 加入 Git。

## 真實繁中收據 add-on

### 1. 準備 5–10 張本機影像

- 至少 2 張含真實手寫內容。
- 至少 2 張有印章遮擋文字。
- 每張長邊至少 1600 px；JPG、PNG 或 WebP。
- 一張照片只放一份文件，且不可裁掉要標註的欄位。
- 先移除或實心遮罩姓名、電話、會員編號、載具條碼及可識別 QR code。
- 店名與商家統編是商業欄位；若你仍不希望保留，也可一併遮罩並在 gold 填 `null`。

把已完成隱私檢查的副本放到 `data/raw/`。原始未遮罩版本不要放進專案目錄。

### 2. 建立本機 manifest

```powershell
Copy-Item data\real_receipts_manifest.example.json data\real_receipts_manifest.json
```

依實際檔名與挑戰類型修改 `data/real_receipts_manifest.json`。這個檔案已被 `.gitignore`
排除。只有在逐張看過、確認沒有個資或可用條碼後，才保留：

```json
"privacy_reviewed": true,
"contains_personal_data": false
```

### 3. 人工標註

```powershell
.venv\Scripts\python -m uvicorn annotator.main:app --port 8010
```

開啟 <http://localhost:8010>，逐張校正 OCR 預填結果並存檔。Gold 必須人工核對，不能直接把
OCR 或 VLM 輸出當答案。完成後停止 server。

### 4. 凍結前驗證

```powershell
.venv\Scripts\python scripts\verify_real_receipt_dataset.py
```

驗證器會檢查影像／label 配對、解析度、JSON Schema、canonical normalization、隱私聲明，
以及 handwriting／stamp_occlusion 各至少 2 張。詳細 SHA-256 報告只寫到受忽略的
`tmp/real_receipt_verification.json`。

### 5. 評估

這 5–10 張是小型外部驗證集，不用來調 prompt、規則或 threshold。先分開跑每個 factor，
避免模型 keep-alive 影響延遲：

```powershell
$out = "results\eval_real_zh_receipts"

# 傳統管線；PaddleOCR 主要用 CPU，但 qwen3:4b 補漏可能使用 GPU
.venv\Scripts\python scripts\run_eval.py --images-dir data\raw --labels-dir data\labels --backends ollama --only a_pre --out $out
.venv\Scripts\python scripts\run_eval.py --images-dir data\raw --labels-dir data\labels --backends ollama --only a_nopre --out $out

# 本機 Qwen3-VL；只在 GPU 空閒時執行
.venv\Scripts\python scripts\run_eval.py --images-dir data\raw --labels-dir data\labels --backends ollama --only ollama --out $out
.venv\Scripts\python scripts\run_eval.py --images-dir data\raw --labels-dir data\labels --backends ollama --only ollama_hint --out $out

.venv\Scripts\python scripts\make_report.py "$out\summary.json"
```

以上只使用本機 Ollama。不要對私人收據使用 `openai` 或 `gemini` backend，除非你另行同意該
服務的資料處理條款。逐張結果仍留在 ignored `results/`；公開時只提交去識別化聚合指標與
challenge counts。

## LlamaParse optional comparator

LlamaParse 不需要本機 GPU，但會把 manifest 內的三份完整公開 PDF 上傳到第三方雲端，並可能
產生費用。`page_ranges` 只限制解析頁，不代表只上傳 PDF 的部分位元組。

### 1. 安裝隔離的 optional extra

```powershell
.venv\Scripts\python -m pip install -e ".[llamaparse]"
```

### 2. 在目前 PowerShell 行程安全輸入 key

不要把 key 貼到聊天、command history、`.env` 或 repository：

```powershell
$secureLlamaKey = Read-Host "LLAMA_CLOUD_API_KEY" -AsSecureString
$llamaKeyPtr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureLlamaKey)
try {
    $env:LLAMA_CLOUD_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($llamaKeyPtr)
} finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($llamaKeyPtr)
    Remove-Variable secureLlamaKey, llamaKeyPtr
}
```

### 3. 執行 24 頁／39 題 comparator

```powershell
.venv\Scripts\python scripts\run_llamaparse_comparator.py --allow-cloud
```

固定比較 current PyMuPDF + fixed、LlamaParse + fixed、LlamaParse + structure-aware。Backend
版本預設鎖定 `agentic / 2026-07-15`，避免 `latest` 漂移。報告中的 API cost 保留 `null`；
權威費用以 LlamaCloud project billing dashboard 為準。結果是 descriptive-only，不可取代
本機 parser promotion gate。

重驗已保存 IR 時，不再呼叫雲端：

```powershell
.venv\Scripts\python scripts\run_llamaparse_comparator.py --reuse-ir
```

最後移除目前行程的 credential：

```powershell
Remove-Item Env:LLAMA_CLOUD_API_KEY
```

沒有 key、沒有 optional SDK，或沒加 `--allow-cloud` 時都會記錄為 `skipped`，不會讓測試失敗。
