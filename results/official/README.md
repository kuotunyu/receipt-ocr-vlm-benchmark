# 正式評估摘要

這個資料夾收錄可公開查核的 aggregate evaluation artifacts：

- [`synthetic_45_summary.json`](synthetic_45_summary.json)：固定種子 42 生成的 45 張合成繁體中文發票／收據。
- [`sroie_45_summary.json`](sroie_45_summary.json)：從 `rth/sroie-2019-v2` `test` split 以固定種子 42 抽出的 45 張收據。
- [`public_zh_receipts_5_summary.json`](public_zh_receipts_5_summary.json)：5 張 Wikimedia Commons
  公開臺灣繁中真實收據的小型 external failure probe；不是第三組正式 45 張 benchmark。

每份 artifact 都包含資料集識別、該資料集實際執行的配置、README 使用的 headline metrics、
完整聚合指標，以及來源 `summary.json` 的 SHA-256。兩個 45 張正式 benchmark 各有六個配置；
5 張 add-on 有四個本機配置。逐張預測、OCR 文字、圖片與執行期錯誤訊息不在公開 artifact 中。

## 產生與驗證

以下操作只讀取既有 JSON，不載入模型、不讀取 `.env`，也不發出網路請求：

```powershell
# 從本機正式結果重新匯出
.venv\Scripts\python scripts\export_official_results.py

# 不寫檔，只確認公開 artifact 與本機來源完全一致
.venv\Scripts\python scripts\export_official_results.py --check
```

預設來源為：

- `results/eval_synthetic_45/summary.json`
- `results/eval_sroie_45/summary.json`
- `results/eval_real_zh_receipts/summary.json`

這三個原始實驗資料夾刻意不進版控；保留評估工作區的人可以用 `--check` 重建查核。
Exporter 採欄位白名單，遇到未知欄位、配置缺漏、樣本數不符或非有限數字會直接停止，
輸出也不包含時間戳，因此相同來源會得到 byte-for-byte 相同的檔案。

SROIE 的來源、授權標示與論文引用見 [`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md)。
