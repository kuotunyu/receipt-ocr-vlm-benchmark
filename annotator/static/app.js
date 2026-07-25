"use strict";

const state = {
  images: [],       // [{filename, labeled}]
  index: -1,
  zoom: 1,
  rotation: 0,
  dirty: false,
};

const els = {
  progressFill: document.getElementById("progress-fill"),
  progressText: document.getElementById("progress-text"),
  imageSelect: document.getElementById("image-select"),
  btnPrev: document.getElementById("btn-prev"),
  btnNext: document.getElementById("btn-next"),
  viewerImage: document.getElementById("viewer-image"),
  viewerContainer: document.getElementById("viewer-container"),
  zoomLevel: document.getElementById("zoom-level"),
  btnZoomIn: document.getElementById("btn-zoom-in"),
  btnZoomOut: document.getElementById("btn-zoom-out"),
  btnZoomReset: document.getElementById("btn-zoom-reset"),
  btnRotate: document.getElementById("btn-rotate"),
  ocrStatus: document.getElementById("ocr-status"),
  ocrRawText: document.getElementById("ocr-raw-text"),
  btnOcrPrefill: document.getElementById("btn-ocr-prefill"),
  form: document.getElementById("label-form"),
  itemsList: document.getElementById("items-list"),
  btnAddItem: document.getElementById("btn-add-item"),
  itemRowTemplate: document.getElementById("item-row-template"),
  saveStatus: document.getElementById("save-status"),
};

const FIELD_IDS = [
  "doc_type", "seller_name", "date", "invoice_number",
  "seller_tax_id", "buyer_tax_id", "total_amount",
];

// ---------------------------------------------------------------------------
// 資料載入
// ---------------------------------------------------------------------------

async function init() {
  const res = await fetch("/api/images");
  state.images = await res.json();
  renderImageSelect();
  updateProgress();
  checkOcrStatus();

  if (state.images.length > 0) {
    await loadImageAt(0);
  } else {
    els.ocrStatus.textContent = "data/raw 內尚無圖片，請先放入照片";
  }
}

function renderImageSelect() {
  els.imageSelect.innerHTML = "";
  state.images.forEach((img, i) => {
    const opt = document.createElement("option");
    opt.value = String(i);
    opt.textContent = `${img.labeled ? "✓ " : "　"}${img.filename}`;
    els.imageSelect.appendChild(opt);
  });
}

async function updateProgress() {
  const res = await fetch("/api/progress");
  const { labeled, total } = await res.json();
  els.progressText.textContent = `${labeled} / ${total}`;
  els.progressFill.style.width = total ? `${(100 * labeled) / total}%` : "0%";
}

async function checkOcrStatus() {
  const res = await fetch("/api/ocr_status");
  const { available, reason } = await res.json();
  els.btnOcrPrefill.disabled = !available;
  els.ocrStatus.textContent = available ? "" : `OCR 尚不可用：${reason || "未安裝"}`;
}

async function loadImageAt(i) {
  if (state.dirty && !confirm("目前標註尚未儲存，確定要離開嗎？")) return;
  state.index = i;
  const img = state.images[i];
  els.imageSelect.value = String(i);
  els.viewerImage.src = `/images/${encodeURIComponent(img.filename)}`;
  resetView();

  const res = await fetch(`/api/label/${encodeURIComponent(img.filename)}`);
  const record = await res.json();
  populateForm(record);
  els.ocrRawText.textContent = "";
  els.saveStatus.textContent = "";
  els.saveStatus.className = "";
  clearDirty();

  els.btnPrev.disabled = i <= 0;
  els.btnNext.disabled = i >= state.images.length - 1;
}

function currentFilename() {
  return state.images[state.index]?.filename;
}

// ---------------------------------------------------------------------------
// 表單
// ---------------------------------------------------------------------------

function populateForm(record) {
  document.getElementById("f-doc_type").value = record.doc_type || "e_invoice";
  document.getElementById("f-seller_name").value = record.seller_name || "";
  document.getElementById("f-date").value = record.date || "";
  document.getElementById("f-invoice_number").value = record.invoice_number || "";
  document.getElementById("f-seller_tax_id").value = record.seller_tax_id || "";
  document.getElementById("f-buyer_tax_id").value = record.buyer_tax_id || "";
  document.getElementById("f-total_amount").value =
    record.total_amount === null || record.total_amount === undefined
      ? "" : record.total_amount;

  els.itemsList.innerHTML = "";
  (record.items || []).forEach((item) => addItemRow(item.name, item.amount));

  FIELD_IDS.forEach((id) => document.getElementById(`f-${id}`).classList.remove("suggested"));
}

function addItemRow(name = "", amount = "") {
  const frag = els.itemRowTemplate.content.cloneNode(true);
  const row = frag.querySelector(".item-row");
  row.querySelector(".item-name").value = name;
  row.querySelector(".item-amount").value = amount === null || amount === undefined ? "" : amount;
  row.querySelector(".btn-remove-item").addEventListener("click", () => {
    row.remove();
    markDirty();
  });
  row.querySelectorAll("input").forEach((inp) => inp.addEventListener("input", markDirty));
  els.itemsList.appendChild(row);
}

function collectFormRecord() {
  const num = (v) => (v === "" || v === null ? null : Number(v));
  const str = (v) => (v === "" ? null : v);

  const items = Array.from(els.itemsList.querySelectorAll(".item-row"))
    .map((row) => ({
      name: row.querySelector(".item-name").value.trim(),
      amount: num(row.querySelector(".item-amount").value),
    }))
    .filter((it) => it.name);

  return {
    doc_type: document.getElementById("f-doc_type").value,
    seller_name: str(document.getElementById("f-seller_name").value.trim()),
    date: str(document.getElementById("f-date").value),
    invoice_number: str(document.getElementById("f-invoice_number").value.trim()),
    seller_tax_id: str(document.getElementById("f-seller_tax_id").value.trim()),
    buyer_tax_id: str(document.getElementById("f-buyer_tax_id").value.trim()),
    total_amount: num(document.getElementById("f-total_amount").value),
    items,
  };
}

function markDirty() {
  state.dirty = true;
}
function clearDirty() {
  state.dirty = false;
}

// ---------------------------------------------------------------------------
// 檢視器（縮放 / 旋轉）——僅影響顯示，不影響原圖
// ---------------------------------------------------------------------------

function applyTransform() {
  els.viewerImage.style.transform = `scale(${state.zoom}) rotate(${state.rotation}deg)`;
  els.zoomLevel.textContent = `${Math.round(state.zoom * 100)}%`;
}

function resetView() {
  state.zoom = 1;
  state.rotation = 0;
  applyTransform();
}

els.btnZoomIn.addEventListener("click", () => {
  state.zoom = Math.min(state.zoom * 1.25, 8);
  applyTransform();
});
els.btnZoomOut.addEventListener("click", () => {
  state.zoom = Math.max(state.zoom / 1.25, 0.1);
  applyTransform();
});
els.btnZoomReset.addEventListener("click", resetView);
els.btnRotate.addEventListener("click", () => {
  state.rotation = (state.rotation + 90) % 360;
  applyTransform();
});
els.viewerContainer.addEventListener(
  "wheel",
  (e) => {
    e.preventDefault();
    state.zoom = Math.min(Math.max(state.zoom * (e.deltaY < 0 ? 1.1 : 0.9), 0.1), 8);
    applyTransform();
  },
  { passive: false }
);

// ---------------------------------------------------------------------------
// OCR 預填
// ---------------------------------------------------------------------------

els.btnOcrPrefill.addEventListener("click", async () => {
  const filename = currentFilename();
  if (!filename) return;
  els.btnOcrPrefill.disabled = true;
  els.ocrStatus.textContent = "辨識中…";
  try {
    const res = await fetch(`/api/ocr_prefill/${encodeURIComponent(filename)}`, { method: "POST" });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      els.ocrStatus.textContent = body.detail || "OCR 預填失敗";
      return;
    }
    const { raw_lines, suggestions } = await res.json();
    els.ocrRawText.textContent = raw_lines.join("\n");
    document.getElementById("ocr-raw-details").open = true;

    Object.entries(suggestions).forEach(([field, value]) => {
      const el = document.getElementById(`f-${field}`);
      if (!el || value === null || value === undefined) return;
      if (el.value.trim() !== "") return; // 不覆蓋已填的值
      el.value = value;
      el.classList.add("suggested");
    });
    els.ocrStatus.textContent = "已套用建議，請逐欄確認後再儲存";
    markDirty();
  } catch (err) {
    els.ocrStatus.textContent = `OCR 預填發生錯誤：${err}`;
  } finally {
    els.btnOcrPrefill.disabled = false;
  }
});

// ---------------------------------------------------------------------------
// 儲存
// ---------------------------------------------------------------------------

els.form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const filename = currentFilename();
  if (!filename) return;

  const record = collectFormRecord();
  const res = await fetch(`/api/label/${encodeURIComponent(filename)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(record),
  });
  const body = await res.json();

  if (res.ok && body.ok) {
    els.saveStatus.textContent = "已儲存 ✓";
    els.saveStatus.className = "ok";
    clearDirty();
    state.images[state.index].labeled = true;
    renderImageSelect();
    els.imageSelect.value = String(state.index);
    updateProgress();
  } else {
    els.saveStatus.textContent = `儲存失敗：\n${(body.errors || []).join("\n")}`;
    els.saveStatus.className = "err";
  }
});

// ---------------------------------------------------------------------------
// 導覽
// ---------------------------------------------------------------------------

els.btnPrev.addEventListener("click", () => {
  if (state.index > 0) loadImageAt(state.index - 1);
});
els.btnNext.addEventListener("click", () => {
  if (state.index < state.images.length - 1) loadImageAt(state.index + 1);
});
els.imageSelect.addEventListener("change", () => {
  loadImageAt(Number(els.imageSelect.value));
});
els.btnAddItem.addEventListener("click", () => {
  addItemRow();
  markDirty();
});

document.querySelectorAll("#label-form input, #label-form select").forEach((el) => {
  el.addEventListener("input", () => {
    el.classList.remove("suggested");
    markDirty();
  });
});

document.addEventListener("keydown", (e) => {
  const tag = document.activeElement?.tagName;
  const typing = tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA";

  if (e.ctrlKey && e.key.toLowerCase() === "s") {
    e.preventDefault();
    els.form.requestSubmit();
    return;
  }
  if (typing) return;
  if (e.key === "ArrowLeft") els.btnPrev.click();
  if (e.key === "ArrowRight") els.btnNext.click();
});

init();
