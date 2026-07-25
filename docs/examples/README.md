# 範例截圖

給 README/EVAL_REPORT 用的示意圖，非正式資料集本體（正式資料集在 `data/synthetic/`、
`data/sroie/`，由對應腳本重新產生/下載，不進 repo）。

| 檔案 | 來源 | 說明 |
|---|---|---|
| `synthetic_clean.jpg` | 合成繁中 | 清晰基準 |
| `synthetic_rotate.jpg` | 合成繁中 | 旋轉（deskew 消融實驗的主角） |
| `synthetic_stamp.jpg` | 合成繁中 | 印章遮擋 |
| `synthetic_fade.jpg` | 合成繁中 | 熱感紙褪色模擬 |
| `sroie_001_masked.jpg` | SROIE（真實照片） | 已遮罩 |
| `sroie_003_masked.jpg` | SROIE（真實照片） | 已遮罩 |

## 遮罩政策

- **合成繁中圖片全為電腦生成的虛構內容**（店名/品項/金額皆隨機生成），不含任何真實個資，
  未做任何遮罩。
- **兩張 SROIE 圖片衍生自公開學術基準資料集**。本專案透過
  [`rth/sroie-2019-v2`](https://huggingface.co/datasets/rth/sroie-2019-v2) 取得資料；該 dataset card
  標示 [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/)，並指向 ICDAR 2019 SROIE 挑戰。
- **本專案做過修改**：電話號碼與員工／收銀員代碼已加上不透明遮罩；店名、地址、GST ID
  （統編等價物）等商家層級資訊予以保留。GST ID 是 `EVAL_REPORT.md` 中「Pipeline A 誤把 GST ID
  當統編」發現的示範素材。

根目錄的 MIT License 僅適用於本專案原創程式碼，不涵蓋這兩張 SROIE 衍生圖片。完整來源、
論文引用、修改標示與上游條款連結見 [`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md)；該聲明
是出處紀錄，不構成法律意見。
