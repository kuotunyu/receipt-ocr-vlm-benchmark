# Complex-document benchmark results

`benchmark_summary.json` 是從本機下載 PDF → parser-native artifact → Spatial Document IR →
parser metrics / chunks / downstream evaluation 自動產生的 compact summary。原 PDF、逐頁 screenshot、
crop、native output 與 normalized IR 都受 `.gitignore` 保護。

目前固定變因結果（v0.7，14 題；parser-level 37 cases 未修改）：

| Factor | Recall@5 | MRR | Answer | Citation | 狀態 |
|---|---:|---:|---:|---:|---|
| current PyMuPDF + fixed | 0.929 | 0.756 | 0.786 | 0.786 | complete |
| PaddleOCR + layout + fixed | 0.929 | 0.774 | 0.929 | 0.929 | complete；global parser NO-GO |
| LiteParse + fixed | 0.929 | 0.732 | 0.786 | 0.786 | complete |
| LiteParse + structure-aware | 0.929 | 0.786 | 0.786 | 0.786 | complete |
| LiteParse + table reconstruction + structure | 0.929 | 0.619 | 0.857 | 0.857 | complete |
| PyMuPDF fallback + routed table regions | **1.000** | 0.726 | **0.929** | **0.929** | complete |
| Qwen3-VL + structure-aware | 0.786 | 0.386 | 0.643 | 0.643 | complete；NO-GO |
| targeted VLM + fixed diagnostic | 0.929 | 0.756 | 0.786 | 0.786 | complete；diagnostic |
| targeted VLM + structure-aware | 0.786 | 0.548 | 0.714 | 0.714 | complete；NO-GO |
| structured caption + original crop synthesis（4 chart questions） | 1.000 | 0.750 | 0.750 | 0.750 | complete；NO-GO |

Parser mean score：PyMuPDF 0.609；LiteParse 0.398；LiteParse + table reconstruction 0.635。
Hybrid table router 為 0.720。表格結構由 0.519 升到 0.947，人工 routing labels 的
precision/recall 為 1.000，人工 table bbox IoU 由 0.423 升到 0.682。

Qwen3-VL parser mean 為 0.571，低於 PyMuPDF 0.609；26 頁實測 374.372 秒（$0 API cost）。
相較 current parser + fixed，Recall@5、MRR、answer 與 citation 全數下降，因此 VLM parser
promotion 為 **NO-GO**。Structured caption 將四題圖表 Recall@5 從 0.750 提高到 1.000，
但原始 crop synthesis 只答對 3/4，answer/citation 皆為 0.750，因此 caption promotion
也是 **NO-GO**。

PaddleOCR + layout parser mean 為 0.400，table structure 為 0.000；但文字 OCR 使 fixed downstream
answer/citation 達 0.917、MRR 0.736。26 頁 CPU OCR 需 1,289.296 秒（49.6 秒/頁），因此
**全域 parser replacement NO-GO**；後續只評估掃描頁或低 native-text 頁面的 targeted fallback。

凍結 promotion gate 的結論是 **全域替換 NO-GO**：parser mean 只增加 0.027，未達 +0.050。
Hybrid 的 answer/citation 升到 0.917，但 MRR 0.681 仍未回到 baseline 0.715，因此
**限定 table-region routing promotion 也是 NO-GO**；保留研究分支，不升為預設。

## v0.4 external blind router holdout

`table_router_holdout.json` 使用兩份未參與 v0.3 開發的官方繁中年報。12 個頁面先由人工看
render 標註，再首次執行已凍結的 `vector-grid-router-1` / threshold 0.62：

| Pages | Positive / Negative | TP / FP / FN / TN | Precision | Recall | F1 | Accuracy | CPU time |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 6 / 6 | 6 / 0 / 0 / 6 | 1.000 | 1.000 | 1.000 | 1.000 | 1.700s |

0.50–0.74 的 descriptive sensitivity 分類均相同。這個結果讓 router generalization gate
得到 **PASS**，但只涵蓋 vector-ruled table page selection；不量測 cell extraction 或 QA。
由於 factor 3c 的 MRR 仍是 0.681，低於 current baseline 0.715，整體 promotion 結論維持
**NO-GO**。

## v0.6 external end-to-end QA holdout

兩份未參與 parser/chunker/QA 開發的官方文件，先以 120 DPI render 人工標註 7 頁、15 題，
再執行相同 deterministic answerer；無 LLM judge。

| Factor | Recall@5 | MRR | Answer | Citation |
|---|---:|---:|---:|---:|
| external current PyMuPDF + fixed | 0.867 | 0.817 | 0.800 | 0.800 |
| external LiteParse + structure | 0.800 | 0.600 | **0.933** | 0.800 |
| external Hybrid table router | 0.867 | 0.706 | 0.800 | 0.733 |
| Hybrid + selected late-max ranker | 0.867 | **0.833** | 0.800 | 0.733 |
| external full Qwen + structure | 0.733 | 0.469 | 0.667 | 0.600 |
| external targeted VLM + structure | 0.867 | 0.411 | **0.933** | 0.867 |
| external targeted VLM + fixed（post-hoc） | **0.933** | **0.883** | 0.867 | **0.867** |

Late-max 使用相同 char-bigram features，在長 atomic table 內以 240-char window、80-char overlap、
50% query coverage 做 local scoring。設定只由開發集五個預先列出的 variants 選出；外部題目不允許
重調。它讓 Hybrid 外部 MRR 0.706 → 0.833，且沒有再降低 answer/citation，因此
**ranker 對 Hybrid research branch 為 GO**。但 citation 0.733 仍低於 current parser 0.800，
所以 **完整 Hybrid promotion 維持 NO-GO**。

Full Qwen 7 頁 model-call latency 為 106.303 秒；targeted router 只送 1/7 頁進 Qwen，
包含 PyMuPDF baseline 約 10.384 秒。Targeted + structure 雖救回圖表答案，但 MRR 大幅下降，
正式結論仍為 **NO-GO**。Fixed row 是看到 structure MRR 下降後才補的 parser/chunker
confound 診斷，因此只標為 **promising, not validated**，不可當 frozen holdout promotion。

## v0.8 untouched targeted-VLM promotion holdout

`promotion_holdout_summary.json` 使用兩份全新官方年報，與開發集及 v0.6 holdout 共 7 份文件
完全不重疊。14 個 selected pages、26 題、兩組 factor 與 gate 均在新 parser prediction 前
凍結。比較只允許：

| Factor | Recall@5 | MRR | Answer | Citation | 狀態 |
|---|---:|---:|---:|---:|---|
| PyMuPDF + fixed | 0.769 | 0.665 | 0.731 | 0.731 | CPU baseline complete |
| targeted VLM + fixed | 0.769 | 0.626 | 0.731 | 0.731 | complete；NO-GO |

Candidate 必須四項 primary metrics 都不低於 baseline，且至少一項嚴格改善，才可 GO。
Candidate 的 MRR 下降 0.038，且沒有任一 primary metric 嚴格改善，因此自動 decision 是
**NO-GO**。Targeted parser 的 14 頁 wall-clock latency 為 20.455 秒（0.684 pages/s），
本地 API 成本 $0。跨頁題 `ph24` 使用
`evidence_mode=all`，分散在第 45、47 頁的兩份 evidence 都必須被取回；不再以「任一頁命中」
冒充 cross-page Recall。

`promotion_caption_summary.json` 的 5 個新 source-pixel crops 與 7 個 question links 也已
在 caption generation 前凍結。17 次呼叫共 19.573 秒（GPU 13.595 秒），四種模式結果如下：

| Caption mode | Recall@5 | MRR | Answer | Citation | Crop Recall@5 |
|---|---:|---:|---:|---:|---:|
| no image indexing | 0.714 | 0.643 | 0.714 | 0.714 | - |
| generic caption | **0.857** | 0.619 | **0.857** | **0.857** | - |
| structured caption | 0.714 | 0.607 | 0.714 | 0.714 | - |
| structured + original crop | 0.714 | 0.607 | **0.857** | **0.857** | 0.857 |

Generic caption 有實際增益，但 structured caption 沒提升 retrieval，原圖 synthesis 也因 crop
Recall 未達事前 1.0 gate 而判定 **NO-GO**。Caption 仍只供檢索，答案來自原始 crop pixels。

## v0.9 source-assisted scale-validation

`scale_validation_summary.json` 使用三份全新 OGDL v1 官方文件、24 頁與 39 題。Annotation
曾參考 PDF embedded text layer，所以這是 scale stability evidence，不是 untouched promotion
evidence。

| Factor | Recall@5 | MRR | Answer | Citation | Parsing / retrieval / generation errors |
|---|---:|---:|---:|---:|---|
| PyMuPDF + fixed | 0.744 | 0.513 | 0.744 | 0.718 | 7 / 2 / 1 |
| targeted VLM + fixed | **0.821** | **0.603** | **0.846** | **0.795** | 1 / 4 / 1 |

Targeted router 路由 6/24 頁，fresh observed wall-clock 115.814 秒；保存 artifact 重建的
parser + baseline latency 為 110.282 秒，另重測 PyMuPDF CPU baseline 11.055 秒，本地 API
cost $0。四項皆改善，因此
`scale_finding=SUPPORTS-CANDIDATE`，但 recommendation 是 `NOT-PROMOTION-EVIDENCE`，不可改寫
v0.8 的 NO-GO。

`scale_validation_caption_summary.json` 使用 8 crops／9 題。No-image、generic 與 structured
Recall@5 都是 0.333；structured + original crop 的 answer/citation/crop Recall 為 0.667，
未達 frozen gates，scale finding 為 `DOES-NOT-SUPPORT-CAPTION-AND-INDEX`。保存成功／重試
artifact 的 28 calls 為 73.105 秒、GPU 63.067 秒；另誠實揭露 checkpoint 上線前一次
52.325 秒的 JSON length-stop batch。

2026-07-30 已在無競爭 RTX 4090 window 完成 production `think=false` A/B、26 頁 parser、
外部 full/targeted Qwen 與三張圖／流程圖的 caption/original-crop synthesis。`benchmark_summary.json` 的 factor 4/5
由 raw artifacts 與 normalized IR 自動產生；不是手動補寫。

重建與驗證：

```powershell
.venv\Scripts\python scripts\download_complex_documents.py
.venv\Scripts\python scripts\run_complex_benchmark.py
.venv\Scripts\python scripts\verify_complex_results.py

.venv\Scripts\python scripts\download_complex_documents.py --manifest data/complex_document/holdout/manifest.json --output-dir data/complex_document/holdout/raw
.venv\Scripts\python scripts\run_table_router_holdout.py
.venv\Scripts\python scripts\verify_table_router_holdout.py

.venv\Scripts\python scripts\download_complex_documents.py --manifest data/complex_document/promotion_holdout/manifest.json --output-dir data/complex_document/promotion_holdout/raw
.venv\Scripts\python scripts\run_promotion_holdout.py
.venv\Scripts\python scripts\run_promotion_holdout.py --reuse-ir --include-candidate
.venv\Scripts\python scripts\verify_promotion_holdout.py
.venv\Scripts\python scripts\run_promotion_caption_eval.py
.venv\Scripts\python scripts\verify_promotion_caption_eval.py
```

逐欄定義、限制與 failure visualization 見
[`docs/COMPLEX_DOCUMENT_BENCHMARK.md`](../../docs/COMPLEX_DOCUMENT_BENCHMARK.md)。
