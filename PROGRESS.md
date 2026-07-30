# Project progress

## Receipt benchmark（保留）

- 150 個原始測試在擴充前全數通過。
- `scripts/export_official_results.py --check` 通過。
- 12 組正式 raw artifacts 重算後與 README / official summary 的 avg exact 一致。
- repository name、remote、`results/official/` 均未修改。

## Complex-document track v0.1（2026-07-29）

- [x] 5 份 download-only 繁中公開文件 manifest + checksum。
- [x] 26 個精選難頁；37 個人工 hard cases；12 個 downstream questions。
- [x] Spatial Document IR + JSON Schema + 分層 artifacts。
- [x] PyMuPDF、PaddleOCR layout、LiteParse、Qwen3-VL adapters。
- [x] LlamaParse optional extra；無 key skip。
- [x] Parser structure metrics、fixed/structure-aware chunks、CPU retrieval/QA/error attribution。
- [x] caption-index 四模式與 original-pixel synthesis guard。
- [x] 自動 failure bbox visualization。
- [x] 178 個測試、integration smoke、summary-from-IR verification。
- [x] 首輪 factor table 與 NO-GO 決策。

## Complex-document track v0.2（2026-07-29）

- [x] LiteParse native text + local grid-table reconstruction adapter。
- [x] Caption-aware table detection 與跨頁 continuity linker。
- [x] 跨頁表格合成單一 atomic structure chunk，保留多頁 bbox provenance。
- [x] parser-native → Spatial IR normalization audit 與 result verification。
- [x] 新增 3b factor；表格結構 0.519 → 0.947，answer/citation 0.750 → 0.833。
- [x] 初步條件式結論：限定 grid-table routing 值得實作與 audit。
- [x] 原始缺失與 bbox 過寬 partial-recovery 圖可從 artifacts 重建。

## Complex-document track v0.3（2026-07-30）

- [x] 26 頁人工 routing gold：13 個 vector-grid positives / 13 個 fallback negatives。
- [x] 高信心 page router；同批資料 precision/recall 1.000。
- [x] Nested-table oversized tail split；人工 table bbox IoU 0.423 → 0.682。
- [x] Region-level hybrid：表格區用 reconstruction，其餘元素保留 PyMuPDF。
- [x] 新增 3c factor；parser mean 0.609 → 0.720，Recall@5 0.917 → 1.000。
- [x] Answer/citation 0.750 → 0.917；MRR 0.715 → 0.681。
- [x] 凍結 gate 結論：全域替換與 routing promotion 均 NO-GO；保留研究分支。

## Complex-document track v0.4（2026-07-30）

- [x] 凍結 `vector-grid-router-1` 與 threshold 0.62；holdout 禁止事後調參。
- [x] 新增桃園市警政統計年報與內政部移民署年報，皆為官方公開下載且與 v0.3 開發文件不重疊。
- [x] 12 頁人工先標後測 routing gold：6 個 vector-grid positives / 6 個 fallback negatives。
- [x] 外部 blind 結果：6 TP / 6 TN，precision、recall、F1、accuracy 均 1.000。
- [x] 0.50–0.74 threshold sensitivity 結果不變；只作描述，不用來選門檻。
- [x] 12 頁 CPU parsing 1.700 秒；無 GPU、無 API、無 secrets。
- [x] checksum → parser-native → Spatial IR → result 的獨立 runner 與 verifier。
- [x] 邊界不變：router 泛化 PASS，但 downstream MRR 0.681 < 0.715，promotion 仍 NO-GO。

## Complex-document track v0.5（2026-07-30）

- [x] 安裝指定的 local `qwen3-vl:8b`（Q4_K_M，6.1 GB model blob）。
- [x] 原始設定單頁 parser smoke 通過：10 elements、100.755 秒、約 13.2 GB 顯存峰值。
- [x] 找到 10,700 output/thinking tokens 的延遲來源；改為 `think=false` + JSON Schema。
- [x] VLM table 的 `text` 空白時由 Markdown 回填，避免 retrieval 把成功解析的表格當空白。
- [x] normalization audit 可解析 VLM 原始 response，不再把 native item count 錯算成 0。
- [x] caption artifact 保存 generic / structured / original-crop synthesis 的 latency、tokens 與原始像素答案。
- [x] 每頁／每次呼叫前檢查 competing Ollama model；忙碌時可恢復地 skip，不中斷對方。
- [x] `think=false` 同頁 A/B：100.755 秒降為 11.337 秒，約快 8.9 倍。
- [x] Ollama/Qwen 將 JSON 放在 `thinking` channel 的相容性 fallback；generic、structured、pixel answer 均以 schema 取得最終欄位。
- [x] 5 份文件、26 頁完整 Qwen parser：374.372 秒，parser mean 0.571、Recall@5 0.750、MRR 0.394、answer/citation 0.583。
- [x] 2 張圖、6 次 caption/crop 呼叫：8.605 秒；structured caption Recall@5 1.000，原 crop answer/citation 0.500。
- [x] 自動 decision：Qwen parser 與 caption-and-index 均為 NO-GO，不升級成預設路徑。
- [x] 191 tests 通過；12 個 complex result blocks、router holdout 與 receipt official results 均驗證通過。

## Complex-document track v0.6 CPU stage（2026-07-30）

- [x] PaddleOCR + layout 正式 26 頁 row：1,289.296 秒、parser mean 0.400、table structure 0.000。
- [x] Paddle fixed downstream：Recall@5 0.917、MRR 0.736、answer/citation 0.917；全域 parser replacement 仍為 NO-GO。
- [x] 2 份開發集外官方文件、7 頁、15 題人工先標後測 end-to-end QA holdout。
- [x] 外部 current parser：Recall@5 0.867、MRR 0.817、answer/citation 0.800。
- [x] 外部 Hybrid：Recall@5 0.867、MRR 0.706、answer 0.800、citation 0.733，因此 promotion NO-GO。
- [x] 保留 atomic table 的 char-bigram late-max：開發 MRR 0.681 → 0.861；外部 Hybrid MRR 0.706 → 0.833。
- [x] Late-max ranker 對 Hybrid research branch 為 GO，但完整 Hybrid 相較 current parser 仍因 citation 0.733 < 0.800 而 NO-GO。
- [x] 197 tests；14 個主 benchmark blocks、5 個外部 QA factors、MRR recovery、router holdout 與 receipt results 全部可重驗。
- [x] 外部 QA 的 Qwen factor、擴充 chart gold 與 targeted VLM routing 已於 v0.7 夜間 GPU window 完成。

## Complex-document track v0.7 GPU stage（2026-07-30）

- [x] 外部 full Qwen 7 頁正式 run：model-call latency 106.303 秒；Recall@5 0.733、MRR 0.469、answer 0.667、citation 0.600，維持 NO-GO。
- [x] 新增共同 interface 的 `TargetedVLMRouterAdapter` 與凍結 `native-visual-router-1`；只用 native text、raster area、table presence 選頁，不看問題、答案或 VLM output。
- [x] 開發集路由 3/26 頁，targeted latency 30.396 秒、parser mean 與 PyMuPDF 同為 0.609；structure downstream 仍退步，正式 gate 為 NO-GO。
- [x] 外部路由 1/7 頁，targeted + structure 救回 45,219 圖表值，answer 0.933、citation 0.867，但 MRR 0.411，正式 gate仍為 NO-GO。
- [x] 為拆開 parser/chunker confound 補做 fixed post-hoc diagnostic：Recall@5 0.933、MRR 0.883、answer/citation 0.867；明確標註不是 frozen promotion evidence，下一步需新 untouched holdout。
- [x] caption gold 擴為 3 張圖／流程圖、4 題人工問題；11 次呼叫（含一次可稽核 JSON retry）共 21.550 秒。
- [x] structured caption Recall@5 1.000；original-crop synthesis answer/citation 0.750，未達 0.800 gate，caption promotion 維持 NO-GO。
- [x] GPU 完成後卸載 Qwen、確認 Ollama 無載入模型，將 RTX 4090 交棒給「④ RAG Attribution」。
- [x] 201 tests；17 個主 benchmark blocks、8 個外部 QA factors、MRR recovery、router holdout 與 receipt official results 全部可重驗。

## Complex-document track v0.8 untouched promotion holdout（2026-07-30）

- [x] 新增勞動力發展署 113 年年報與財政部 113 年財政統計年報；官方來源、授權頁、URL、bytes 與 SHA-256 固定，原 PDF 不進 repository。
- [x] 與既有 5 份開發文件及 2 份 v0.6 holdout 文件完全不重疊。
- [x] 在任何新 parser／retriever／VLM prediction 前凍結 14 頁、26 題與 evidence。
- [x] promotion protocol 鎖定 `PyMuPDF + fixed` 對 `targeted-vlm + fixed`、`native-visual-router-1`、Recall@5 / MRR / answer / citation 與 no-regression + one-improvement gate。
- [x] 跨頁題新增 `evidence_mode=all`，允許證據分散於不同 chunks，但 Recall 與 citation 必須完整覆蓋所有 evidence。
- [x] CPU baseline 14 頁 3.416 秒；Recall@5 0.769、MRR 0.665、answer/citation 0.731。
- [x] 新增 5 個 source-pixel chart/infographic crops、7 題；加上 v0.7 後累計 8 個 visual targets、11 題。
- [x] GPU frozen targeted-VLM candidate：14 頁 20.455 秒、0.684 pages/s、本地成本 $0；Recall@5 0.769、MRR 0.626、answer/citation 0.731。
- [x] Candidate 的 Recall、answer、citation 與 baseline 持平，但 MRR -0.038；未達 no-regression 與 one-improvement gate，正式判定 NO-GO。
- [x] GPU 產生 5 個 generic / structured captions 與 7 個 original-crop answers；17 次呼叫共 19.573 秒、GPU 13.595 秒、19,240 prompt tokens、1,414 output tokens。
- [x] Generic caption 的 Recall/answer/citation 為 0.857；structured caption 為 0.714，structured + original crop 的 answer/citation 為 0.857、crop Recall 0.857，caption promotion 判定 NO-GO。
- [x] GPU 完成後卸載 Qwen3-VL，Ollama 無載入模型；RTX 4090 交棒給「④ RAG Attribution」。
- [x] 210 tests；既有 17 個主 benchmark blocks、三組舊 verifier、receipt results 與 v0.8 promotion/caption results 全部通過。

## Complex-document track v0.9 scale-validation（2026-07-30）

- [x] 新增中央氣象署、數位發展部、國家海洋研究院三份 OGDL v1 官方文件；24 個 layout-stratified pages、39 題，與既有資料完全不重疊。
- [x] Annotation 使用 source pixels + embedded text layer 輔助 exact transcription，protocol 明確標為 source-assisted，永遠不可覆蓋 v0.8 untouched promotion。
- [x] PyMuPDF + fixed：Recall@5 0.744、MRR 0.513、answer 0.744、citation 0.718；error attribution parsing/retrieval/generation = 7/2/1。
- [x] Targeted VLM + fixed：Recall@5 0.821、MRR 0.603、answer 0.846、citation 0.795；error attribution = 1/4/1。
- [x] Router 只送 6/24 頁進 Qwen；fresh observed wall 115.814 秒，artifact-reconstructed latency 110.282 秒，PyMuPDF CPU baseline 11.055 秒，本地 API cost $0；四項全改善，scale finding 為 SUPPORTS-CANDIDATE，但 recommendation 固定 NOT-PROMOTION-EVIDENCE。
- [x] 新增 8 個圖表／政策圖 crop、9 題。Structured caption 未提升 Recall（0.333）；original crop answer/citation/crop Recall 0.667，未通過 0.8/0.8/0.9 gates，DOES-NOT-SUPPORT-CAPTION-AND-INDEX。
- [x] Caption generator 新增 durable per-target checkpoint；一次 1,024-token length-stop 的 52.325 秒 discarded batch 明確揭露。保存的 28 calls 為 73.105 秒、GPU 63.067 秒。
- [x] LlamaParse adapter 依 SDK 2.13.0 native page items 正規化 bbox／table／heading／figure／caption，新增 `--allow-cloud` comparator；無 key、SDK 或明確授權均自動 skip。
- [x] 真實繁中收據 add-on 改採 5 張 Wikimedia Commons 公開臺灣收據：逐張固定來源、CC 授權、SHA-256、尺寸與隱私判定；2 張手寫、3 張印章遮擋，5 份人工 gold 全數通過 schema／canonical verifier。原圖仍只下載到 ignored `data/raw/`。
- [x] GPU 完成後卸載 Qwen3-VL，確認 `ollama ps` 空白並通知 RAG Attribution task 恢復。
- [x] 226 tests；receipt official summaries、17 個主 benchmark blocks、router/QA/MRR、v0.8 與 v0.9 parser/caption verifiers 全部通過。

## 明確限制

- LlamaParse 未呼叫；沒有讀取 API key 或其他 secrets。專案擁有者已決定暫不執行商業 comparator；程式與 no-key skip 保留，但不是目前完成條件。
- PaddleOCR 只做一頁 CPU smoke，不把慢速 smoke 當 25 頁正式效能數字。
- v0.4 holdout 只驗證 vector-grid page routing，不代表 borderless/raster table、cell extraction 或 downstream QA 已泛化。
- Caption QA 已擴為 3 張圖／4 題人工 gold，仍不足以宣稱圖表 QA 已泛化。
- v0.6 外部 QA 的 15 題與 evidence 在 GPU run 前凍結且未修改。targeted-fixed 是看過 structure MRR 退步後才補的 post-hoc diagnostic，不得冒充 untouched holdout GO。
- v0.8 已完成，但只有 2 份文件、26 題與 5 個 visual targets；兩項 promotion 均為 NO-GO，不可外推成所有繁中複雜文件或圖表 QA 的一般結論。
- v0.9 擴到 3 份文件、39 題與 8 個 visual targets，但 source-assisted annotation 不是 blind；只能支持研究方向，不能當新 promotion GO。
- 公開繁中收據目前只有 5 張，足以做小型 failure probe，不足以取代既有 45+45 正式 benchmark，亦不能宣稱已完整解決繁中 external validity。
