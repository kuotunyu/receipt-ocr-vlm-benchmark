# 繁中複雜文件 Structure-aware Parsing Benchmark

## 研究問題與邊界

這是原 receipt benchmark 旁邊的獨立 track，不覆寫收據 schema、pipeline 或正式結果，也不是
ChatPDF 產品。首輪只回答一個問題：**parser 的結構錯誤是否會透過 chunking / retrieval 傳到答案，
而且新 parser 的改善是否大到值得繼續投資？**

方法參考 [Beyond RAG 工作坊文章](https://blog.aihao.tw/2026/07/26/beyond-rag-llamaindex-workshop/)、
[LiteParse](https://github.com/run-llama/liteparse) 與
[ParseBench](https://arxiv.org/abs/2604.08538)，但 gold、題目、繁中規則與分數均由本專案定義。

## 開始前盤點

| 項目 | 原 receipt track 現況 | 複雜文件缺口 |
|---|---|---|
| 資料型別 | 45 張合成繁中收據 + 45 張 SROIE 英文收據 | 無多欄、跨頁表格、圖表、腳註或標題階層 benchmark |
| 輸出 | invoice-specific JSON；OCRBox 只在記憶體 | 無持久化 page/bbox/reading-order/section provenance |
| 指標 | scalar exact/fuzzy、items P/R/F1、validity、stability、latency、cost | 無 table/chart/order/grounding，也無 downstream propagation |
| parser | OpenCV + PaddleOCR + y 分行；invoice VLM JSON | 無通用文件 adapter / IR |
| 已知限制 | 規則換語言後崩跌；合成手寫/印章不等於真實；SROIE 無 items | 不足以判斷 Markdown 變漂亮是否改善 RAG |

既有 receipt artifacts 經 `export_official_results.py --check` 與 12 組 raw result 重算一致；
新增程式不修改 `results/official/`。

## 資料與人工 gold

- 5 份繁體中文官方公開下載文件、26 個精選頁，涵蓋純視覺掃描封面、legacy font、多欄、密集/跨頁表格、圖表、
  流程圖、腳註、頁首頁尾與標題階層。
- 37 個人工目視驗證 hard cases；不是 LLM judge。
- 26 個逐頁人工 routing labels，其中 13 頁應進 vector-grid reconstruction、13 頁應保留 fallback。
- v0.4 另有 2 份未參與開發的官方年報、12 個先標後測的 external holdout labels（6 positive / 6 negative）。
- 14 個 downstream 問題，涵蓋文字事實、表格單格、表格聚合、4 題圖表／流程圖讀值、跨頁與不可回答。
- 原 PDF 不進 repository。`data/complex_document/manifest.json` 保存 URL、SHA-256、頁數、挑戰與
  授權備註；下載程式在落盤前驗 checksum。
- `data/complex_document/gold/hard_cases.json` 保存期望文字/結構、閱讀順序、部分 normalized bbox。

## Spatial Document IR

所有 adapter 實作 `DocumentParserAdapter.parse(ParseRequest) -> SpatialDocument`，共同 IR 保存：

- document ID、SHA-256、source URI；
- parser 名稱、版本與完整 config；
- 1-based page number、頁面尺寸與座標空間；
- `heading / paragraph / table / figure / caption / footnote / list`；
- text、Markdown、bbox、reading order、parent section path、confidence；
- source screenshot/crop reference 與 UTC parsing timestamp。

JSON Schema 在 `schema/spatial_document_ir.schema.json`。artifact 分三層且可獨立重跑：

```text
data/complex_document/raw/                 # 原始 PDF，忽略
artifacts/complex_document/parser_raw/     # parser-native output，忽略
artifacts/complex_document/ir/             # normalized IR，忽略
artifacts/complex_document/screenshots/    # 頁面圖，忽略
artifacts/complex_document/crops/          # 圖表 crop，忽略
results/complex_document/benchmark_summary.json
```

`normalization_audit.py` 另外把 parser-native → IR 當成獨立邊界，量測字元 multiset
recall/precision、native item/IR element 數與被重建表格取代的 native elements。這使錯誤鏈可以拆成：
source → parser-native → normalized IR → chunk → retrieval → answer，而不是把正規化損失算到 parser。

## Parser adapters

| Adapter | 路徑 | 狀態 |
|---|---|---|
| PyMuPDF | spatial text/image + `find_tables` baseline | 26 頁完成 |
| PaddleOCR + layout | 既有 PP-OCRv6 + y-coordinate 分行；固定 `device=cpu` | v0.6 已完成 5 份文件、26 頁正式 CPU run |
| LiteParse local | v2.10.0 spatial text items；OCR 可配置 | 26 頁完成 |
| LiteParse + table reconstruction | LiteParse spatial text + PyMuPDF grid geometry/cells + caption/跨頁 linker | 26 頁完成 |
| Hybrid table router | PyMuPDF default；高信心表格頁只替換 table region，其餘元素保留 | 26 頁完成 |
| Qwen3-VL | 頁面像素 → 通用 element JSON，不沿用 invoice prompt | v0.5 已完成 5 份文件、26 頁正式 GPU run |
| Targeted VLM router | PyMuPDF default；native-signal 選頁後，掃描頁替換、raster visual 頁只補 figure/caption/table | v0.7 開發 3/26 頁、外部 1/7 頁 |
| LlamaParse | optional commercial comparator | optional extra；無 key 自動 skip |

商業 parser 不是唯一可重現路徑。Qwen row 不會偷偷改用另一個本機模型。

v0.5 的 Qwen production config 固定 `think=false`、temperature 0、JSON Schema output 與
`num_predict` 上限。第一次原始單頁 smoke 雖成功解析 10 elements，但因預設 thinking 產生
10,700 tokens，耗時 100.755 秒、顯存約 13.2 GB。相同頁面的 production config 為
11.337 秒、1,352 output tokens，約快 8.9 倍。Ollama 0.6.2 仍將 schema JSON 放入
`thinking` 欄位，因此 adapter 只在 `response` 空白時採用可驗證的 `thinking_fallback`，
並把實際 output channel 保存到 raw artifact。

為避免干擾同機其他工作，adapter 在每頁、caption generator 在每次呼叫前讀取 Ollama process
list；若存在非 `qwen3-vl:8b` 模型就以 `ParserUnavailable` / `SKIP` 停下。已實測
`taide-gemma3-12b` 使用 GPU 時不會載入 Qwen，也不會停止或卸載對方模型。

## Parser-level metrics

`parser_metrics.py` 使用人工規則計分，不以 CER/WER 取代結構：

- table row/column/cell preservation；
- chart axis/value/series textual spot checks；
- text completeness；
- pairwise reading order；
- heading/list style；
- footnote-anchor presence/order/type；
- bbox IoU grounding；
- header/footer contamination；
- cross-page continuity。

v0.3 結果（5 docs / 26 pages / 37 parser cases + 26 routing labels）：

| Parser | 平均分 | pass rate | table | text | order | bbox | 跨頁連續 | 26 頁時間 | API 成本 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PyMuPDF 1.26.3 | 0.609 | 0.486 | 0.519 | 0.850 | 0.800 | 0.347 | 0.250 | 3.99s | $0 |
| PaddleOCR 3.7.0 + layout | 0.400 | 0.270 | **0.000** | **0.950** | 0.800 | 0.334 | 0.250 | 1289.30s | $0 |
| LiteParse 2.10.0 | 0.398 | 0.270 | **0.000** | 0.850 | 0.800 | 0.269 | 0.250 | 3.14s | $0 |
| LiteParse + table reconstruction | 0.635 | 0.432 | **0.947** | 0.850 | 0.800 | 0.269 | **0.500** | 4.02s | $0 |
| Hybrid table-region router | **0.720** | **0.568** | **0.947** | 0.850 | 0.800 | **0.347** | **0.500** | 6.03s | $0 |
| Qwen3-VL parser | 0.571 | 0.405 | 0.440 | **1.000** | 0.775 | 0.297 | 0.000 | 374.37s | $0 |

LiteParse 此版本的 local JSON 在這批頁面提供 text items/bboxes，但沒有可用的 table element，因此表格結構
不是「字看得到就算對」，而是明確得 0。local reconstruction 將 ruled/grid table 轉為
Markdown table，並以頁緣、x 對齊、欄數與 caption 規則建立跨頁 continuity。它大幅補回表格結構，
但沒有改善原 LiteParse 的 heading、chart、bbox 等弱點。

v0.3 先用 PyMuPDF vector-grid 訊號選頁，再只將該頁 table region 換成 reconstructed table，
保留 PyMuPDF 的標題、圖表與一般文字。26 頁人工 routing gold 上 precision/recall 都是 1.000；
這是同批小樣本結果，不能當作外部文件的泛化保證。

## v0.4 外部 blind router holdout

為檢查 v0.3 的 perfect routing 是否只是同批資料過擬合，v0.4 在不改 router 的前提下加入兩份
不同機關文件：[桃園市警政統計年報](https://www.typd.gov.tw/index.php?action=view&catid=325&cid=0&id=1&pg=1)
與[內政部移民署 113 年年報](https://www.immigration.gov.tw/5385/7353/7359/401712/cp_news)。
兩個機關都有網站資料開放宣告；repository 仍採 download-only，只提交 URL、checksum 與人工 annotation。

Blind protocol：

1. 先凍結 `vector-grid-router-1`、primary threshold `0.62` 與禁止 retuning；
2. 將 12 個候選頁 render 成圖片，由人工逐頁判定 vector-grid reconstruction 是否適用；
3. gold 寫入檔案後才首次執行 router；
4. `0.50 / 0.56 / 0.62 / 0.68 / 0.74` 只作 sensitivity 描述，不選新門檻。

| Holdout | 頁數 | Positive / Negative | TP / FP / FN / TN | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| 桃園市警政統計年報 | 6 | 3 / 3 | 3 / 0 / 0 / 3 | 1.000 | 1.000 | 1.000 | 1.000 |
| 內政部移民署 113 年年報 | 6 | 3 / 3 | 3 / 0 / 0 / 3 | 1.000 | 1.000 | 1.000 | 1.000 |
| 合計 | 12 | 6 / 6 | 6 / 0 / 0 / 6 | **1.000** | **1.000** | **1.000** | **1.000** |

12 頁 PyMuPDF CPU parsing 共 1.700 秒，API 成本為 $0；正例最大 confidence 為
0.817–0.917，負例皆為 0.000，因此上述五個 sensitivity thresholds 的分類相同。
這使「page router 能在額外 vector-ruled 年報頁面泛化」通過預先設定的 0.90 gate，但樣本仍是
目的性選取的兩份官方年報，且不含 borderless/raster table。此 holdout 沒有評 cell correctness、
table bbox 或 QA，也不能取代原 factor table 的 downstream promotion gate。

定義、結果與重算鏈分別在
`data/complex_document/holdout/manifest.json`、
`data/complex_document/holdout/gold/table_routing_pages.json`、
`results/complex_document/table_router_holdout.json` 與
`scripts/verify_table_router_holdout.py`。

`arc-05` 的 nested-table tail split 將人工 table bbox IoU 從 PyMuPDF 0.423 提高到 0.682；
它偵測「巨大單格尾列包含另一個獨立 table candidate」才裁切，不會一般化地刪除 merged cells。

正規化 audit 的 mean character recall 為 PyMuPDF 1.000、LiteParse 0.998、reconstruction 1.000。
reconstruction precision 為 0.697，原因是 IR 同時保留 native element 與新增的 table Markdown；
1081 個表格區 native elements 被標為 shadowed，structure chunker 不會重複索引它們。

## Chunking 與最小 downstream RAG

固定變因為 `char-bigram-cosine-v1` CPU retriever、K=5 與同一個 deterministic extractive answerer；
只替換 parser / chunker。answer regex 是人工 gold 的可重現 reader，不是拿 LLM judge 當真值。

- baseline：800-char fixed/recursive chunk，120-char overlap；
- proposed：section 是主 context、table atomic、figure+caption 合併、footnote 附 anchor、空視覺頁
  page fallback；
- sentence nodes 只作 citation node，不進主要 synthesis context；
- 所有 chunk 保留 pages、bboxes、section path、parser name/version、element IDs。

| Factor | Recall@5 | MRR | Answer correctness | Citation validity | 錯誤歸因 |
|---|---:|---:|---:|---:|---|
| 1. PyMuPDF + fixed | 0.929 | 0.756 | 0.786 | 0.786 | retrieval 1 / generation 2 |
| 1b. PaddleOCR + layout + fixed | 0.929 | 0.774 | 0.929 | 0.929 | parsing 1 |
| 2. LiteParse + fixed | 0.929 | 0.732 | 0.786 | 0.786 | parsing 1 / generation 2 |
| 3. LiteParse + structure | 0.929 | **0.786** | 0.786 | 0.786 | parsing 1 / generation 2 |
| 3b. LiteParse + table reconstruction + structure | 0.929 | 0.619 | 0.857 | 0.857 | parsing 1 / generation 1 |
| 3c. PyMuPDF fallback + routed table regions | **1.000** | 0.726 | **0.929** | **0.929** | generation 1 |
| 4. Qwen3-VL + structure | 0.786 | 0.386 | 0.643 | 0.643 | parsing 2 / retrieval 1 / generation 2 |
| 4a. targeted VLM + fixed（diagnostic） | 0.929 | 0.756 | 0.786 | 0.786 | retrieval 1 / generation 2 |
| 4b. targeted VLM + structure | 0.786 | 0.548 | 0.714 | 0.714 | retrieval 3 / generation 1 |
| 5. structured caption + original crop（4 chart questions） | **1.000** | 0.750 | 0.750 | 0.750 | generation 1 |

純 structure-aware chunks 提高 LiteParse MRR（0.732 → 0.786），但答案與 citation 沒增加。
加入表格重建後，表格單格與跨頁題各修正一題。Hybrid 進一步保留 PyMuPDF 的非表格區域，
Recall@5 升到 1.000、answer/citation 升到 0.929，並把 full enrichment 的 MRR 0.619 拉回
0.726；但仍低於現行 baseline 0.756。剩餘錯誤只有一題圖表 generation。

## Caption-and-index 保護欄

`generate_chart_captions.py` 對人工 bbox crop 的**原始像素**呼叫 Qwen3-VL，要求 axis、unit、series、
values、trend 與 structured caption。評估保留四種模式：

1. no image indexing；
2. generic caption；
3. structured chart caption；
4. structured caption retrieval + original crop synthesis。

caption chunk 帶有 `caption_is_retrieval_only=true`；`answer_chart_from_original_crop()` 沒有 crop
就直接失敗，不能用 caption 冒充圖表證據。integration test 也驗證 vision answerer 實際收到原始
image bytes。v0.7 正式 run 擴為 3 張圖／流程圖、4 題，11 次呼叫（包含一次完整記錄的 bounded
JSON retry）共 21.550 秒。Structured caption 讓 Recall@5 從 0.750 升到 1.000；原始 crop
答對 3/4，answer/citation 為 0.750。這證明 caption 有檢索價值，但仍未達事前 0.800 gate，
不足以升級成可靠的圖表回答路徑。

## v0.6 外部 QA 與 MRR recovery

`qa_holdout/manifest.json` 與 `questions.json` 保存兩份外部官方文件的 checksum、7 個 selected
pages 與 15 題人工 gold。題型包含單一文字事實、表格單格、表格聚合、圖表讀值、跨頁與不可回答；
annotation 在執行 downstream 前凍結，原 PDF 不進 repository。

外部 current parser 的 Recall@5 / MRR / answer / citation 為
0.867 / 0.817 / 0.800 / 0.800；Hybrid 為
0.867 / 0.706 / 0.800 / 0.733，因此原 routing promotion 仍未泛化。

`late-max-table-retrieval-v1` 不拆 atomic table，也不換 embedding family；仍使用 char-bigram，
但在長 chunk 內以 240-char windows 與 query-feature coverage 計算 local max。五個 variants
事先列出，只在開發集依「answer/citation 不降、MRR 最大」選設定，外部集禁止調參。結果：

- 開發 Hybrid MRR：0.681 → 0.861；
- 外部 Hybrid MRR：0.706 → 0.833；
- 外部 answer/citation 維持 0.800 / 0.733。

因此 ranker 可進 Hybrid research branch（GO），但完整 Hybrid 仍未達 current parser 的 citation
0.800，promotion 維持 NO-GO。

## v0.7 外部 Qwen 與 targeted VLM

外部 15 題沒有因 GPU 結果修改。Full Qwen 對 7 頁的 model-call latency 為 106.303 秒，
Recall@5 / MRR / answer / citation 為 0.733 / 0.469 / 0.667 / 0.600，全面低於 current
parser，因此全頁 VLM 仍是 NO-GO。

`native-visual-router-1` 只看 PyMuPDF 的 native text characters、最大 raster area 與 table
presence；不看題目、答案、gold 或 Qwen output。開發集路由 3/26 頁，targeted parser latency
30.396 秒、parser mean 0.609，與 PyMuPDF 相同；但 structure factor 的 Recall/MRR/answer/citation
為 0.786 / 0.548 / 0.714 / 0.714，正式 gate 失敗。

外部集只路由桃園警政年報圖表頁 1/7 頁，約 10.384 秒。Structure factor 成功讀出 45,219，
answer 0.933、citation 0.867，但 MRR 降至 0.411，仍為 NO-GO。為拆開「parser 改善」與
「structure chunking 排名下降」的 confound，事後補做相同 fixed chunks 的診斷：
Recall 0.933、MRR 0.883、answer/citation 0.867，四項都高於 current baseline。
因這個 factor 是看過 structure MRR 後才新增，它只能標為 **promising, not validated**；
必須在新文件與新問題的 untouched holdout 重驗，不能回頭宣稱本次 frozen holdout GO。

## v0.8 untouched promotion protocol

為正式檢驗 v0.7 的 post-hoc 訊號，v0.8 另選勞動力發展署 113 年年報與財政部 113 年
財政統計年報。兩份文件不在既有開發集或外部 QA，repository 仍只保存官方 URL、授權頁、
bytes、SHA-256、selected pages 與 annotation，不提交 PDF。

在任何新 parser prediction 前已凍結：

- 14 個 selected pages、26 題，包含文字事實、table cell、table aggregation、chart value、
  cross-page 與 unanswerable；
- baseline `PyMuPDF + fixed`；
- candidate `targeted-vlm + fixed`，router 固定 `native-visual-router-1`；
- Recall@5、MRR、answer correctness、citation validity，K=5；
- 四項不可退步且至少一項嚴格改善，否則 NO-GO。

CPU baseline 為 Recall@5 0.769、MRR 0.665、answer/citation 0.731。Targeted candidate 尚未
執行，故狀態是 PENDING。`ph24` 的 evidence 分散於第 45、47 頁，使用
`evidence_mode=all`：retrieval 必須取回兩頁的獨立 evidence，MRR 以收齊最後一份 evidence
的 rank 計算。這避免舊的 any-evidence 邏輯高估跨頁檢索。

同一批 source pages 另凍結 5 個 chart/infographic crops、7 題，搭配既有 v0.7 targets
累計 8 個 visual targets、11 題。Caption 仍只供 retrieval；答案必須重新讀取原始 crop pixels。

## Failure visualization

`visualize_parsing_failure.py` 可重現 `arc-05`：綠框是人工標註表格；原 LiteParse IR 沒有任何
`table` element，因此沒有紅框。`--parser liteparse-table` 後出現重建紅框；v0.3 會辨識
PyMuPDF 吞入第二張表的巨大尾列並裁切，IoU 由 0.423 升至 0.682。輸出與 metadata 位於
`artifacts/complex_document/failures/arc-05-liteparse*.{png,json}`，不提交來源頁面衍生圖。

## 決策

**全域替換、Paddle global replacement、限定 table-region routing、Qwen VLM parser、
targeted VLM + structure、caption-and-index：全部 NO-GO。Late-max ranker 僅對 Hybrid
research branch 為 GO；targeted + fixed 只列 promising post-hoc diagnostic。**

全域 promotion 使用事前凍結的五個 gates。table structure +0.429、answer +0.083、citation +0.083，
且 table-cell/cross-page 兩種題型改善；但 parser mean 只 +0.027，未達 +0.050，因此不把現有
PyMuPDF 預設替換成 LiteParse reconstruction，也不因結果出來後改門檻。

Hybrid 已通過 routing precision/recall、answer 與 citation gates，也讓 parser mean 相對 baseline
增加 0.111；但 MRR 仍低 0.035，未通過「至少回到現行 baseline」的凍結門檻。因此不升為預設，
也不因 v0.4 的 12 頁外部 router PASS 就忽略 downstream regression。下一步應擴大到
borderless/raster table 與更多發布機關，並研究不拆 table atomicity 的 ranking/reranking 方法。
Qwen3-VL parser mean 0.571 低於 PyMuPDF 0.609；v0.7 的 14 題 factor 中，Recall@5、MRR、
answer/citation 仍全面下降。Caption-and-index 將 4 題圖表 retrieval recall 提高到 1.000，
但 answer/citation 只有 0.750，未達 0.800 gate。兩者均保留為研究分支，不升為預設。

完整逐 case / 逐 question 結果在 `results/complex_document/benchmark_summary.json`；外部 router
與 QA 分別在 `table_router_holdout.json`、`qa_holdout_summary.json`。執行
`verify_complex_results.py`、`verify_table_router_holdout.py` 與
`verify_external_qa_holdout.py` 會從 normalized IR 重算並比對。
