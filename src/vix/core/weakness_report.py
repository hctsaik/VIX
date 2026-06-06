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

    cons = data.get("consistency") or []
    if cons:
        L.append("\n## 一致性歸因(GT × 嵌入:這個失敗是 taxonomy / model / label 問題?)\n")
        L.append("每個易混類別對:在 embedding 空間可不可分、模型混淆多少、成因。" + _PROXY)
        L.append("| 類別對 | 可分(embedding) | sep_err [CI] | O[i→j] | C[i→j] | 判定 | 支撐 | 建議 |")
        L.append("|---|---|---|---|---|---|---|---|")
        for f in cons:
            pair = f"{f['pair'][0]}↔{f['pair'][1]}"
            sep = f"{f['sep_err']} {f.get('sep_ci')}"
            o = f.get("O_ij"); c = f.get("C_ij")
            sup = f"g{f['support']['golden_i']}/{f['support']['golden_j']} ({f['tier']})"
            L.append(f"| {pair} | {f['separable_in_embedding']} | {sep} | {o} | {c if c is not None else '-'} "
                     f"| **{f['verdict']}** | {sup} | {f['action']} |")
        L.append("")

    if not (pc or cw or ov or cons):
        L.append("\n_(無可用訊號:需先 `vix eval-ingest`(有標註 val set)或 `vix calibrate`(GT-free 翻盤),或建立 golden(一致性歸因)。)_\n")

    L.append("\n---\n> " + _PROXY.strip("_"))
    return "\n".join(L)


def _esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def render_weakness_report_html(data: dict) -> str:
    """Browsable HTML render (same data dict). The consistency-attribution table is the headline
    surface (id='consistency') — this is what the Playwright test verifies renders."""
    mode = data.get("mode", "gt_free")
    h = ["<!doctype html><html lang='zh-Hant'><head><meta charset='utf-8'>",
         "<title>YOLO 弱點報告</title><style>",
         "body{font-family:system-ui,Segoe UI,'Microsoft JhengHei',sans-serif;margin:24px;color:#1a1a1a;max-width:1100px}",
         "h1{font-size:22px}h2{font-size:17px;margin-top:26px;border-bottom:2px solid #eee;padding-bottom:4px}",
         "table{border-collapse:collapse;width:100%;margin:8px 0;font-size:13px}",
         "th,td{border:1px solid #ddd;padding:6px 8px;text-align:left;vertical-align:top}",
         "th{background:#f5f5f7}.v{font-weight:700}.proxy{color:#a15c00;background:#fff8e6;padding:8px;border-radius:6px;font-size:12px}",
         ".tax{color:#b00020}.model{color:#0064b0}.label_noise{color:#7a00b0}.clean{color:#2e7d32}",
         "</style></head><body>"]
    h.append(f"<h1>YOLO 弱點報告</h1><p>模式:<b>{_esc(mode)}</b>"
             + (f" ｜ mAP@0.5 = <b>{_esc(data['mAP'])}</b>" if data.get("mAP") is not None else "")
             + (f" ｜ loc_gap = {_esc(data.get('loc_gap'))}" if data.get("loc_gap") else "") + "</p>")
    h.append(f"<p class='proxy'>{_esc(_PROXY.strip('_'))}(未重訓 → 排序為嫌疑/優先,非實測 mAP;可分性綁定目前 embedding 空間。)</p>")

    pc = data.get("per_class") or []
    if pc:
        h.append("<h2 id='per-class'>哪一類最弱(per-class AP,弱→強)</h2>")
        h.append("<table><tr><th>類別</th><th>AP</th><th>n_gt</th><th>主要漏報型態</th><th>最常混淆成</th></tr>")
        for r in pc:
            h.append(f"<tr><td>{_esc(r['cls'])}</td><td>{_esc(r['ap'])}</td><td>{_esc(r['n_gt'])}</td>"
                     f"<td>{_esc(r.get('dom_fn_type') or '-')}</td><td>{_esc(r.get('top_confusion') or '-')}</td></tr>")
        h.append("</table>")

    cons = data.get("consistency") or []
    h.append("<h2 id='consistency'>一致性歸因(GT × 嵌入:taxonomy / model / label?)</h2>")
    if cons:
        h.append("<table id='consistency-table'><tr><th>類別對</th><th>可分(embedding)</th><th>sep_err [CI]</th>"
                 "<th>O[i→j]</th><th>C[i→j]</th><th>判定</th><th>支撐</th><th>建議</th></tr>")
        for f in cons:
            pair = f"{f['pair'][0]}↔{f['pair'][1]}"
            sup = f"g{f['support']['golden_i']}/{f['support']['golden_j']} ({f['tier']})"
            cls = {"taxonomy": "tax", "model": "model", "label_noise": "label_noise", "clean": "clean"}.get(f["verdict"], "")
            h.append(f"<tr><td>{_esc(pair)}</td><td>{_esc(f['separable_in_embedding'])}</td>"
                     f"<td>{_esc(f['sep_err'])} {_esc(f.get('sep_ci'))}</td><td>{_esc(f.get('O_ij'))}</td>"
                     f"<td>{_esc(f.get('C_ij') if f.get('C_ij') is not None else '-')}</td>"
                     f"<td class='v {cls}'>{_esc(f['verdict'])}</td><td>{_esc(sup)}</td><td>{_esc(f['action'])}</td></tr>")
        h.append("</table>")
    else:
        h.append("<p>(無一致性發現:需 ≥2 類 golden;接 eval-ingest 才能歸因 taxonomy/model/label。)</p>")

    cw = data.get("confident_wrong") or []
    if cw:
        h.append("<h2 id='confident-wrong'>最「自信卻錯」(GT 證實的誤報)</h2><table>"
                 "<tr><th>影像</th><th>類別</th><th>conf</th><th>型態</th></tr>")
        for r in cw:
            h.append(f"<tr><td>{_esc(r['id'])}</td><td>{_esc(r.get('pred_class'))}</td>"
                     f"<td>{_esc(r['conf'])}</td><td>{_esc(r.get('fp_type','-'))}</td></tr>")
        h.append("</table>")

    q = data.get("queue") or {}
    if q:
        h.append("<h2 id='queue'>該標哪些(逐弱類佇列,PROXY)</h2><ul>")
        for c, cands in q.items():
            h.append(f"<li><b>{_esc(c)}</b>: " + ", ".join(f"{_esc(x['id'])}({_esc(x['closeness'])})" for x in cands) + "</li>")
        h.append("</ul>")

    h.append("</body></html>")
    return "".join(h)
