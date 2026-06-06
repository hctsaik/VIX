"""Human-readable YOLO weakness report (model-loop-v2).

Rolls the model-validated signals VIX already computes — per-class AP, confusion matrix,
localization gap, typed FP/FN — plus confidently-wrong mining (hardneg) and a per-weak-class
label queue into ONE durable Markdown artifact that answers the owner's two questions:
WHERE is YOLO weak, and WHICH unlabeled data to label next. Pure renderer (stdlib only); the
pipeline supplies the structured data.

Two modes: a GT block (needs a labelled val set via eval-ingest) is the centrepiece; a GT-free
block (embedding overturns) fires with no labels. HONESTY: with no retraining, every "go label
these" ranking is a PROXY (proximity to demonstrated errors / confident embedding-overturns),
never proof that labelling raises mAP — stamped throughout.
"""

from __future__ import annotations

_PROXY = "_(PROXY:未重訓,此為嫌疑/優先排序,非實測 mAP 增益)_"


def render_weakness_report(data: dict) -> str:
    mode = data.get("mode", "gt_free")
    L: list[str] = ["# YOLO 弱點報告\n"]
    L.append("模式:**%s** %s\n" % (
        mode, "(有標註 val set → 以 GT 區塊為主)" if mode == "gt" else "(GT-free → 靠嵌入翻盤/代理訊號)"))
    if data.get("mAP") is not None:
        loc = data.get("loc_gap")
        L.append("- **mAP@0.5 = %s**%s\n" % (data["mAP"], f"　定位尾巴 loc_gap = {loc}(框越鬆越大)" if loc else ""))

    pc = data.get("per_class") or []
    if pc:
        L.append("\n## 哪一類最弱(per-class AP,弱 → 強)\n")
        L.append("| 類別 | AP | n_gt | 主要漏報型態 | 最常混淆成 |")
        L.append("|---|---|---|---|---|")
        for r in pc:
            L.append(f"| {r['cls']} | {r['ap']} | {r['n_gt']} | {r.get('dom_fn_type') or '-'} | {r.get('top_confusion') or '-'} |")
        L.append("")

    conf = data.get("confusion") or []
    if conf:
        L.append("\n## 混淆(truth → pred,前 10)\n")
        L += [f"- {pair}: {n}" for pair, n in conf]
        L.append("")

    cw = data.get("confident_wrong") or []
    if cw:
        L.append("\n## 最「自信卻錯」的偵測(GT 證實的誤報,conf 高 → 低)\n")
        L.append("這些是 YOLO 最該優先修的盲點(高信心的真誤報)。")
        L.append("| 影像 | 類別 | conf | 型態 |")
        L.append("|---|---|---|---|")
        for r in cw:
            L.append(f"| {r['id']} | {r.get('pred_class')} | {r['conf']} | {r.get('fp_type', '-')} |")
        L.append("")

    ov = data.get("overturns") or []
    if ov:
        L.append("\n## 自信但嵌入翻盤(無 GT,適用未標註新資料)\n")
        L.append("YOLO 高信心、但 DINOv2 嵌入判定離該類太遠 → 疑似自信誤報。" + _PROXY)
        L.append("| 影像 | 類別 | conf | knn_dist | dist_thr | wrongness |")
        L.append("|---|---|---|---|---|---|")
        for r in ov:
            L.append(f"| {r['id']} | {r.get('pred_class')} | {r['conf']} | {r.get('knn_dist')} | {r.get('dist_thr')} | {r['wrongness']} |")
        L.append("")

    q = data.get("queue") or {}
    if q:
        L.append("\n## 該標哪些(逐弱類「去標這些」佇列)\n")
        L.append("每個弱類,列出未標註資料中最接近該類失敗處的候選。" + _PROXY)
        for c, cands in q.items():
            L.append(f"- **{c}**: " + ", ".join(f"{x['id']}({x['closeness']})" for x in cands))
        L.append("")

    if not (pc or cw or ov):
        L.append("\n_(無可用訊號:需先 `vix eval-ingest`(有標註 val set)或 `vix calibrate`(GT-free 翻盤)。)_\n")

    L.append("\n---\n> " + _PROXY.strip("_"))
    return "\n".join(L)
