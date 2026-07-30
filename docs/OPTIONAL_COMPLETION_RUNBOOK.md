# Optional completion runbook

這份文件記錄兩個附加項目：

1. 5 張真實臺灣繁中收據已改用 Wikimedia Commons 明確 CC 授權影像完成；repository
   保存來源、授權、checksum、隱私檢查與人工 gold，不保存原始影像。
2. LlamaParse 商業 comparator 已由專案擁有者決定暫不執行。參考流程保留，但不是目前
   完成條件，也不會要求 API key。

## 真實繁中收據 add-on

### 1. 重建公開資料

```powershell
.venv\Scripts\python scripts\download_public_receipts.py
```

`data/public_receipts_manifest.json` 固定 5 個 Wikimedia Commons file pages、下載 URL、
作者、CC-BY-SA 版本、SHA-256、像素尺寸、挑戰標籤與逐張隱私判定。下載器只接受
`upload.wikimedia.org` HTTPS URL，checksum 或尺寸不符即停止，也不會靜默覆寫既有檔案。

目前 coverage：

- 2 張真實手寫二聯式發票。
- 3 張印章遮擋。
- 2 張熱感紙 QR／條碼發票。
- 1 張長收據與 1 張正反面同框案例。

逐張人工 gold 在 `data/public_receipt_labels/`；不是 OCR 或 VLM 自動輸出。

### 2. 驗證

```powershell
.venv\Scripts\python scripts\verify_real_receipt_dataset.py `
  --manifest data\public_receipts_manifest.json `
  --raw-dir data\raw `
  --labels-dir data\public_receipt_labels `
  --output tmp\public_receipt_verification.json
```

驗證器會檢查影像／label 配對、解析度、JSON Schema、canonical normalization、隱私聲明，
以及 handwriting／stamp_occlusion 各至少 2 張。

### 3. 評估

這 5 張是小型外部 failure probe，不用來調 prompt、規則或 threshold。先分開跑每個 factor，
避免模型 keep-alive 影響延遲：

```powershell
$out = "results\eval_real_zh_receipts"

# 傳統管線；PaddleOCR 主要用 CPU
.venv\Scripts\python scripts\run_eval.py --images-dir data\raw --labels-dir data\public_receipt_labels --backends ollama --only a_pre --out $out
.venv\Scripts\python scripts\run_eval.py --images-dir data\raw --labels-dir data\public_receipt_labels --backends ollama --only a_nopre --out $out

# 本機 Qwen3-VL；只在 GPU 空閒時執行
.venv\Scripts\python scripts\run_eval.py --images-dir data\raw --labels-dir data\public_receipt_labels --backends ollama --only ollama --out $out
.venv\Scripts\python scripts\run_eval.py --images-dir data\raw --labels-dir data\public_receipt_labels --backends ollama --only ollama_hint --out $out

.venv\Scripts\python scripts\make_report.py "$out\summary.json"
.venv\Scripts\python scripts\export_official_results.py --check
```

逐張結果仍留在 ignored `results/`；公開時只提交
`results/official/public_zh_receipts_5_summary.json` 的聚合指標。

2026-07-30 實測結果：

| 配置 | avg exact | JSON validity | E2E warm p50 |
|---|---:|---:|---:|
| Pipeline A－有前處理 | 0.486 | — | 115.25s |
| Pipeline A－無前處理 | 0.400 | — | 99.06s |
| Qwen3-VL 8B | 0.714 | 0.800 | 26.77s |
| Qwen3-VL 8B + OCR hint | 0.914 | 1.000 | 125.17s |

小樣本結論為 GO-to-validate，不是 production promotion GO。
`items F1` 只衡量品名是否成功配對，不包含配對後金額正確性；且 5 張中有 2 張空品項 gold。
Hint 組的 latency 已包含每張 CPU OCR 與 VLM call；warm p50 約為純 VLM 的 4.7 倍。

### 4. 私人影像替代路徑

若未來要加入自行拍攝影像，才使用 `data/real_receipts_manifest.example.json`、
`data/labels/` 與 annotator。未遮罩原圖、私人 manifest 和私人 labels 都維持 gitignored；
姓名、電話、會員／載具與付款資訊必須先實心遮罩。

## LlamaParse optional comparator

**狀態：2026-07-30 由專案擁有者決定暫不執行。** 以下只保留未來若改變決定時的安全參考。

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
