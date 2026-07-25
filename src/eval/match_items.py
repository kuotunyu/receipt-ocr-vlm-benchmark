"""品項清單沒有固定順序/數量對應——預測多寫或少寫一項，若直接按索引比對，
後面所有品項都會被誤判成錯位。這裡先用名稱相似度做貪婪對齊，再算 P/R/F1。
"""

from __future__ import annotations

from dataclasses import dataclass

from rapidfuzz.distance import Levenshtein


@dataclass
class ItemMatch:
    gt_index: int | None
    pred_index: int | None
    name_similarity: float
    amount_exact: bool


def align_items(
    gt_items: list[dict], pred_items: list[dict], name_threshold: float = 0.6
) -> list[ItemMatch]:
    """貪婪對齊：依名稱相似度由高到低配對尚未配對的 (gt, pred)，低於門檻不強配。"""
    candidates = []
    for gi, g in enumerate(gt_items):
        for pi, p in enumerate(pred_items):
            sim = Levenshtein.normalized_similarity(g.get("name") or "", p.get("name") or "")
            if sim >= name_threshold:
                candidates.append((sim, gi, pi))
    candidates.sort(key=lambda x: -x[0])

    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    matches: list[ItemMatch] = []

    for sim, gi, pi in candidates:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        amount_exact = gt_items[gi].get("amount") == pred_items[pi].get("amount")
        matches.append(ItemMatch(gt_index=gi, pred_index=pi, name_similarity=sim, amount_exact=amount_exact))

    matches += [
        ItemMatch(gt_index=gi, pred_index=None, name_similarity=0.0, amount_exact=False)
        for gi in range(len(gt_items)) if gi not in matched_gt
    ]
    matches += [
        ItemMatch(gt_index=None, pred_index=pi, name_similarity=0.0, amount_exact=False)
        for pi in range(len(pred_items)) if pi not in matched_pred
    ]
    return matches


def score_items(gt_items: list[dict], pred_items: list[dict], name_threshold: float = 0.6) -> dict:
    """precision/recall/f1 只衡量「有沒有配對到」（名稱相似度 ≥ threshold 就算配到）。
    這會讓「拿鐵咖啡」配到「拿鐵咖啡」跟配到「拿鐵咖琲」（單字錯字）算同一分——
    所以另外報 name_exact_rate/name_avg_similarity，衡量配對到的那些「名稱到底準不準」，
    避免 f1=1.0 掩蓋掉品項名稱本身還是有錯字的事實（曾在 qwen3-vl 的品項名稱上實測到這個情況）。
    """
    matches = align_items(gt_items, pred_items, name_threshold)
    true_positives = [m for m in matches if m.gt_index is not None and m.pred_index is not None]
    tp, fn = len(true_positives), sum(1 for m in matches if m.pred_index is None)
    fp = sum(1 for m in matches if m.gt_index is None)

    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    amount_accuracy = (sum(m.amount_exact for m in true_positives) / tp) if tp else None
    name_exact_rate = (sum(m.name_similarity == 1.0 for m in true_positives) / tp) if tp else None
    name_avg_similarity = (sum(m.name_similarity for m in true_positives) / tp) if tp else None

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "amount_accuracy": amount_accuracy,
        "name_exact_rate": name_exact_rate,
        "name_avg_similarity": name_avg_similarity,
        "n_gt": len(gt_items),
        "n_pred": len(pred_items),
        "n_matched": tp,
    }
