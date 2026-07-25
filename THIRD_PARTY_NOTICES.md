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
