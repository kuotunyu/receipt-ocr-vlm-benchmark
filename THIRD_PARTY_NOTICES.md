# 第三方資料與圖片聲明

本專案根目錄的 [MIT License](LICENSE) 僅適用於作者原創的程式碼，不會把第三方資料集、圖片或其衍生素材重新授權為 MIT。以下內容是為了清楚記錄來源、標示修改及方便使用者查核；它不是法律意見，也不替代上游條款。

## SROIE 2019

- 本專案的評估下載腳本使用 Hugging Face 上的 [`rth/sroie-2019-v2`](https://huggingface.co/datasets/rth/sroie-2019-v2) `test` split。該 dataset card 在本聲明撰寫時標示授權為 [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/)，並說明資料取自 ICDAR 2019 Robust Reading Competition 的 SROIE 挑戰。
- 原始挑戰頁面：[ICDAR 2019 Robust Reading Challenge on Scanned Receipts OCR and Information Extraction](https://rrc.cvc.uab.es/?ch=13)。
- 建議引用：Zheng Huang, Kai Chen, Jianhua He, Xiang Bai, Dimosthenis Karatzas, Shijian Lu, and C. V. Jawahar. “ICDAR2019 Competition on Scanned Receipt OCR and Information Extraction.” *2019 International Conference on Document Analysis and Recognition (ICDAR)*, pp. 1516–1520. [doi:10.1109/ICDAR.2019.00244](https://doi.org/10.1109/ICDAR.2019.00244).

### 本專案收錄的衍生圖片

以下兩個檔案衍生自上述資料集，不屬於本專案 MIT 程式碼授權的範圍：

- `docs/examples/sroie_001_masked.jpg`
- `docs/examples/sroie_003_masked.jpg`

修改內容：為降低範例展示中的識別風險，圖片中的電話號碼與員工／收銀員代碼已加上不透明遮罩；店名、地址與 GST ID 等商家層級資訊則保留，以呈現專案分析的實際錯誤案例。這些修改由本專案作者完成，並不表示原資料集作者或散布者為本專案背書。

若要重用 SROIE 圖片或衍生圖片，請自行查閱並遵守上游 dataset card 與 CC BY 2.0 的歸屬、連結及修改標示要求。

## Complex-document benchmark

新 track 使用 5 份臺灣官方網站公開下載的繁體中文 PDF。由於各發布機關的再散布條款並不一致，
repository **不收錄原始 PDF、完整頁面 screenshot 或 crop**；只收錄下載 URL、SHA-256、檔名、
挑戰標籤與人工 annotation。逐份來源與保守的授權備註在
[`data/complex_document/manifest.json`](data/complex_document/manifest.json)。

使用者執行 `scripts/download_complex_documents.py` 時，是直接向原發布機關下載，仍須自行遵守各
來源網站的最新條款。特別是中央銀行來源頁面有著作權限制，因此本專案明確採 download-only。

v0.4 external holdout 另外使用兩份未參與開發的官方文件：

- [中華民國113年桃園市警政統計年報](https://www.typd.gov.tw/index.php?action=view&catid=325&cid=0&id=1&pg=1)；
  [桃園市政府警察局政府網站資料開放宣告](https://www.typd.gov.tw/index.php?catid=277&id=3)
  允許在註明出處等條件下重製、改作與公開傳輸。
- [內政部移民署113年年報](https://www.immigration.gov.tw/5385/7353/7359/401712/cp_news)；
  [移民署政府網站資料開放宣告](https://www.immigration.gov.tw/6614/6673/15856/)
  對著作權保護範圍採政府資料開放授權條款第 1 版，並列明出處與例外條件。

這兩份 PDF 同樣不進 repository；精確 URL、SHA-256、頁數與頁面選擇記錄在
[`data/complex_document/holdout/manifest.json`](data/complex_document/holdout/manifest.json)。

## Complex-document parser dependencies

- [LiteParse](https://github.com/run-llama/liteparse)：Apache-2.0；作為 optional local parser dependency。
- [PyMuPDF](https://pymupdf.readthedocs.io/)：AGPL / commercial dual license；使用者須依自己的散布
  與部署方式確認合規。本專案只在 `complex-document` optional extra 固定測試版本。
- LlamaParse / LlamaCloud：optional commercial service comparator，不是可重現 benchmark 的必要路徑；
  未設定 key 時不呼叫服務。
