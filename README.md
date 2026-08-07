# receipt-ocr-vlm-benchmark

[![CI](https://github.com/kuotunyu/receipt-ocr-vlm-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/kuotunyu/receipt-ocr-vlm-benchmark/actions/workflows/ci.yml)
![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-passing-success)
[![Release](https://img.shields.io/badge/Release-v1.0.0-blue.svg)](https://github.com/kuotunyu/receipt-ocr-vlm-benchmark/releases/tag/v1.0.0)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

本專案提供繁體中文發票與收據結構化欄位擷取之全方位基準測試 (Benchmark Suite)：針對傳統多步驟組合管線 (OpenCV ➔ PaddleOCR ➔ Layout Split ➔ LLM Refine) 與端到端視覺語言模型管線 (VLM + OCR Hint + Schema Validator) 在欄位精確度 (Exact Match)、推理延遲、權重資源與 API 成本進行實測與量化對比。

---

## 系統設計與關鍵特性

1. **雙管線架構評測 (Pipeline A vs B)**：
   對比「傳統多步驟套件組合管線」與「端到端 VLM 視覺管線」，評估非標準格式下的泛化能力。
2. **OCR Hint 前置文字引導**：
   在 VLM Prompt 中引入輕量 OCR 文字座標作為提示，顯著提升微小字體與印章覆蓋字元之欄位擷取精確度。
3. **自動 Reask 與 JSON Schema 驗證**：
   內建 Pydantic / Output Parser 驗證器，當 VLM 輸出違背 Schema 格式時自動觸發二次修正 Reask 機制。
4. **多維度量化報告與重現腳本**：
   自動計算總金額 (`total_amount`)、日期 (`date`)、賣方統一編號 (`seller_tax_id`) 等欄位之 Exact Match，並自動匯出 CSV/JSON 報告。

---

## 系統架構與 pipeline

### 1. Pipeline A：傳統多步驟 OCR 組合管線

```mermaid
%%{init: {'themeVariables': {'fontSize': '18px'}}}%%
flowchart TD
    subgraph PipeA ["Pipeline A：傳統多步驟組合管線"]
        direction LR
        A1[("1. 影像輸入<br/>(發票/收據圖檔)")] --> A2["2. OpenCV 前處理<br/>(Deskew/Denoise)"] --> A3["3. PaddleOCR 辨識<br/>(PP-OCRv6)"] --> A4["4. Layout 特徵抽取<br/>(動態分行排版)"] --> A5["5. LLM 補銷校正<br/>(Qwen3-4B)"] --> A6[("6. JSON 結構化輸出<br/>(欄位與金額)")]
    end

    classDef normStyle fill:#e7f5ff,stroke:#1971c2,stroke-width:2px,color:#212529
    classDef outStyle fill:#e6fcf5,stroke:#0ca678,stroke-width:2px,color:#212529

    class A1,A2,A3,A4,A5 normStyle
    class A6 outStyle

    style PipeA fill:#f8f9fa,stroke:#1971c2,stroke-width:2px,stroke-dasharray: 4 4
```

### 2. Pipeline B：端到端 VLM 視覺語言管線

```mermaid
%%{init: {'themeVariables': {'fontSize': '18px'}}}%%
flowchart TD
    subgraph PipeB ["Pipeline B：端到端 VLM 視覺管線"]
        direction LR
        B1[("1. 影像輸入<br/>(發票/收據圖檔)")] --> B2["2. Prompt 組裝<br/>(含 OCR Hint 選用)"] --> B3["3. VLM 視覺推理<br/>(Qwen3-VL/GPT-5.4)"] --> B4["4. Schema 驗證<br/>(Reask 自動重試)"] --> B5[("5. JSON 結構化輸出<br/>(欄位與金額)")]
    end

    classDef vlmStyle fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#212529
    classDef outStyle fill:#e6fcf5,stroke:#0ca678,stroke-width:2px,color:#212529

    class B1,B2,B3,B4 vlmStyle
    class B5 outStyle

    style PipeB fill:#faf5ff,stroke:#7b1fa2,stroke-width:2px,stroke-dasharray: 4 4
```

### 3. OCR Hint 輔助與 Reask 錯誤重試時序 (Sequence Diagram)

```mermaid
%%{init: {'themeVariables': {'fontSize': '18px'}}}%%
sequenceDiagram
    autonumber
    actor User as 評測系統 / 使用者
    participant Hint as PaddleOCR Engine<br/>(--with-ocr-hint)
    participant Prompt as Prompt Assembler
    participant VLM as VLM Backend<br/>(Qwen3-VL / GPT-5.4)
    participant Val as JSON Schema Validator

    User->>Prompt: 輸入發票影像
    opt 啟用 OCR Hint 輔助
        User->>Hint: 前置文字掃描
        Hint-->>Prompt: 傳送原始辨識文字與座標
    end
    Prompt->>VLM: 傳送視覺 Prompt (影像 + 構造 Schema + Hint)
    VLM-->>Val: 回傳原始文字輸出

    alt 格式符合 JSON Schema
        Val-->>User: 200 OK 傳送結構化 JSON 成果
    else 格式非法 (Schema Violation)
        Val->>Prompt: 發起自動校正 Reask (夾帶錯誤訊息, ≤2次)
        Prompt->>VLM: 二次修正推理
        VLM-->>Val: 回傳校正後 JSON
        Val-->>User: 傳送結構化成果與用量統計
    end
```

---

## 評測矩陣與綜合結果

數據採合成繁中發票 (45 張) 與 SROIE 真實收據 (45 張) 測試集。

| 管線機制 | 合成繁中 Avg Exact | SROIE Avg Exact | p50 Latency (Warm) | 單位成本 (每百張) | 選型特性說明 |
|---|---:|---:|---:|---:|---|
| **Qwen3-VL-8B (地端 VLM)** | **0.971** | 0.829 | 23.3s / 55.2s | GPU 時間換算 | 高準確度且隱私安全，適合有地端 GPU 之生產環境 |
| **Qwen3-VL-8B + OCR Hint** | **0.981** | 0.803 | 26.5s / 52.7s | GPU 時間換算 | 複雜背景干擾與弱格式收據表現最佳 |
| **GPT-5.4-nano** | 0.660 | **0.857** | **1.8s / 3.6s** | **$0.04 / $0.10** | 超低延遲與極低 API 成本，但純 Prompt 在繁中表頭易受干擾 |
| **GPT-5.4-nano + OCR Hint** | 0.908 | **0.857** | **2.0s / 3.6s** | **$0.04 / $0.11** | API 方案首選，結合 OCR Hint 大幅降低繁中錯誤率 |

- **Exact Match 標準**：總金額、發票號碼與賣方統編必須 100% 完全相同。
- 完整詳細對比數據見 [docs/eval.md](docs/eval.md)。

---

## 快速開始

需求：Python 3.11+、`uv`、可用 GPU (地端 Qwen3-VL) 或 OpenAI API Key (GPT-5.4)。

### 1. 環境設定與套件安裝

```powershell
uv sync
copy .env.example .env
```

### 2. 執行評測基準測試

```powershell
# 執行 Pipeline A (傳統 OCR 組合) 評測
uv run python run_eval.py --pipeline traditional --dataset synthetic_zh

# 執行 Pipeline B (VLM + OCR Hint) 評測
uv run python run_eval.py --pipeline vlm --model qwen3-vl-8b --with-ocr-hint

# 執行單元測試
uv run pytest -q tests
```

---

## 專案結構

| 檔案 / 目錄 | 功能說明與職責 |
|---|---|
| `run_eval.py` | 基準評測執行主程式入口 |
| `src/pipelines/traditional.py` | 傳統多步驟組合管線 (OpenCV + PaddleOCR + Qwen) |
| `src/pipelines/vlm.py` | 端到端 VLM 視覺管線 (Qwen3-VL / GPT-5.4) |
| `src/evaluator.py` | Exact Match 評測與導出統計矩陣 |
| `data/` | 測試發票與收據標註資料集 |
| `reports/` | 評測數據 JSON/CSV 輸出 |

---

## 授權與聲明

本專案採 [MIT License](LICENSE)。標註資料集僅供學術研究與評測使用。
