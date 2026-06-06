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
_CLOSENESS_LEGEND = "closeness = 對該類失敗區的 cosine 鄰近度(0–1,僅排序用,非機率);已解決的候選已標記,不需重做。"
_WRONGNESS_LEGEND = "wrongness = conf × 超出該類嵌入門檻的程度(排序用,非機率);knn_dist>dist_thr 即翻盤依據。"
# a 'label' queue's "hit" == "a human acted on it", so its hit-rate is identically its coverage —
# it measures effort, not suggestion quality. Shown as coverage so it isn't misread as precision.
_LABEL_HITRATE_NOTE = "'label' 佇列(error-mine/weakness)的「命中」=有人處理,故命中率≡覆蓋率,只表行動量、非品質。"


def _provenance_lines(data: dict) -> list[str]:
    """Provenance stamp (L1) + eval-set comparability banner (L3) — Markdown."""
    prov = data.get("provenance") or {}
    if not prov:
        return []
    out: list[str] = []
    parts = []
    if prov.get("eval_set_hash"):
        parts.append(f"eval_set={prov['eval_set_hash'][:8]}")
    if prov.get("pool_hash"):
        parts.append(f"pool={prov['pool_hash'][:8]}")
    parts.append(f"上份報告 {prov['prev_report_ts']}" if prov.get("prev_report_ts") else "首份報告")
    out.append(f"_出處:{' ｜ '.join(parts)}_\n")
    if prov.get("comparable") is False:
        out.append("> ⚠ **本期 eval set 與上期不同 → mAP/AP 不可與上期直接比較**(可能只是 val 變簡單)\n")
    elif prov.get("comparable") and prov.get("prev_mAP") is not None and data.get("mAP") is not None:
        out.append(f"> 與上期同一 eval set,可比較:mAP {prov['prev_mAP']} → {data['mAP']}\n")
    return out


def render_weakness_report(data: dict) -> str:
    mode = data.get("mode", "gt_free")
    L: list[str] = ["# YOLO 弱點報告\n"]
    s = data.get("summary") or {}
    if s:  # TL;DR: 10-second "where is it weak / do this now"
        L.append(f"> **健康度:{s.get('health', '?')}** ｜ 最弱:{s.get('weakest') or '-'}")
        if s.get("todo"):
            L.append(">\n> **現在做這個:** " + " ｜ ".join(s["todo"]))
        L.append("")
    L.append(f"_{_PROXY.strip('_')};可分性綁定目前 embedding 空間。_\n")
    bscope = f"　範圍:**batch {data['batch']}**(佇列/翻盤僅看這批)" if data.get("batch") else ""
    L.append("模式:**%s** %s%s\n" % (
        mode, "(有標註 val set → 以 GT 區塊為主)" if mode == "gt" else "(GT-free → 靠嵌入翻盤/代理訊號)", bscope))
    if data.get("mAP") is not None:
        loc, mbi = data.get("loc_gap"), data.get("map_by_iou")
        if loc is not None:
            extra = f"　定位尾巴 loc_gap = {loc}" + (f"(mAP@0.5={mbi.get('0.5')} vs @0.75={mbi.get('0.75')};越大=框越鬆)" if mbi else "")
        else:
            extra = "　(loc_gap N/A:eval 為單一 IoU,未評估定位)"
        L.append(f"- **mAP@0.5 = {data['mAP']}**{extra}\n")
    L += _provenance_lines(data)

    pc = data.get("per_class") or []
    if pc:
        L.append("\n## 哪一類最弱(per-class AP,弱 → 強)\n")
        L.append("| 類別 | AP | n_gt | 漏報型態(分佈) | 最常混淆成 |")
        L.append("|---|---|---|---|---|")
        for r in pc:
            fnt = r.get("fn_types") or {}
            fn_s = " / ".join(f"{n} {t}" for t, n in sorted(fnt.items(), key=lambda kv: -kv[1])) or "-"
            L.append(f"| {r['cls']} | {r['ap']} | {r['n_gt']} | {fn_s} | {r.get('top_confusion') or '-'} |")
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
        L.append("YOLO 高信心、但 DINOv2 嵌入判定離該類太遠 → 疑似自信誤報。")
        L.append(f"_{_WRONGNESS_LEGEND}_")
        L.append("| 影像 | 類別 | conf | knn_dist | dist_thr | wrongness |")
        L.append("|---|---|---|---|---|---|")
        for r in ov:
            L.append(f"| {r['id']} | {r.get('pred_class')} | {r['conf']} | {r.get('knn_dist')} | {r.get('dist_thr')} | {round(r['wrongness'], 2)} |")
        L.append("")

    q = data.get("queue") or {}
    if q:
        L.append("\n## 該標哪些(逐弱類「去標這些」佇列)\n")
        L.append("每個弱類,列出未標註資料中最接近該類失敗處的候選(完整清單見 `weakness_worklist.csv`)。")
        L.append(f"_{_CLOSENESS_LEGEND}_")
        hr_wq = next((h for h in (data.get("hit_rate") or []) if h["queue"] == "weakness_queue"), None)
        if hr_wq:  # show THIS queue's track record where it's used — as coverage (see _LABEL_HITRATE_NOTE)
            cov = "-" if hr_wq.get("coverage") is None else hr_wq["coverage"]
            note = "(樣本不足)" if hr_wq.get("insufficient") else ""
            L.append(f"_此佇列已處理率(coverage):{cov}(已解決 {hr_wq['resolved']}/{hr_wq['emitted']}){note};{_LABEL_HITRATE_NOTE}_")
        for c, cands in q.items():
            n_res = sum(1 for x in cands if x.get("resolved"))
            head = f"- **{c}**" + (f"(待辦 {len(cands) - n_res} / 已解決 {n_res})" if n_res else f"({len(cands)} 個)") + ": "
            L.append(head + ", ".join(
                (f"~~{x['id']}~~" if x.get("resolved") else f"{x['id']}") + f"({round(x['closeness'], 2)})" for x in cands))
        L.append("")

    cons = data.get("consistency") or []
    if cons:
        L.append("\n## 一致性歸因(GT × 嵌入:這個失敗是 taxonomy / model / label 問題?)\n")
        L.append("每個易混類別對:在 embedding 空間可不可分、模型混淆多少、成因。")
        L.append("| 類別對 | 可分 | sep_err [CI] | O[i→j] [CI] | C[i→j] [CI] (n_gt) | Δ=O−C [CI] | 判定 | 支撐 | 建議 |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for f in cons:
            pair = f"{f['pair'][0]}↔{f['pair'][1]}"
            o = f"{f.get('O_ij')} {f.get('O_ci')}"
            c = (f"{f['C_ij']} {f.get('C_ci')} (n={f['support'].get('n_gt_i')})" if f.get("C_ij") is not None else "-")
            dlt = f"{f['delta']} {f.get('delta_ci')}" if f.get("delta") is not None else "-"
            sup = f"g{f['support']['golden_i']}/{f['support']['golden_j']} ({f['tier']})"
            # representation_fixable rows are NOT a taxonomy dead-end (a learned projection separates
            # them) — render that, never `taxonomy(可修)`, so the verdict can't contradict its action.
            v = ("**representation-fixable**(非 taxonomy 死路)" if f.get("representation_fixable")
                 else f"**{f['verdict']}**")
            L.append(f"| {pair} | {f['separable_in_embedding']} | {f['sep_err']} {f.get('sep_ci')} | {o} | {c} "
                     f"| {dlt} | {v} | {sup} | {f['action']} |")
        L.append("")

    hr = data.get("hit_rate") or []
    if hr:
        L.append("\n## 佇列命中率(VIX 的建議到底準不準?自我校準)\n")
        L.append("把過去的建議佇列 join 後來的人工裁決:準度越高、趨勢越上 = 這個佇列越值得跟。")
        L.append(f"_{_LABEL_HITRATE_NOTE}以覆蓋率呈現。_")
        L.append("| 佇列 | 預測 | 已解決/發出 | 命中率/覆蓋率 | 趨勢 | 註 |")
        L.append("|---|---|---|---|---|---|")
        for q in hr:
            note = "樣本不足僅供參考" if q.get("insufficient") else ""
            if q.get("predict") == "label":
                cell = ("-" if q.get("coverage") is None else f"{q['coverage']}(覆蓋)")
            else:
                cell = "-" if q.get("precision") is None else str(q["precision"])
            L.append(f"| {q['queue']} | {q['predict']} | {q['resolved']}/{q['emitted']} | {cell} | {q.get('trend')} | {note} |")
        L.append("")

    if not (pc or cw or ov or cons or hr):
        L.append("\n_(無可用訊號:需先 `vix eval-ingest`(有標註 val set)或 `vix calibrate`(GT-free 翻盤),或建立 golden(一致性歸因)。)_\n")

    L.append("\n---\n> " + _PROXY.strip("_"))
    return "\n".join(L)


def _esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _provenance_html(data: dict) -> str:
    prov = data.get("provenance") or {}
    if not prov:
        return ""
    parts = []
    if prov.get("eval_set_hash"):
        parts.append(f"eval_set={_esc(prov['eval_set_hash'][:8])}")
    if prov.get("pool_hash"):
        parts.append(f"pool={_esc(prov['pool_hash'][:8])}")
    parts.append(f"上份報告 {_esc(prov['prev_report_ts'])}" if prov.get("prev_report_ts") else "首份報告")
    h = [f"<p style='color:#666;font-size:12px'>出處:{' ｜ '.join(parts)}</p>"]
    if prov.get("comparable") is False:
        h.append("<div style='border-left:5px solid #a15c00;padding:8px 12px;margin:8px 0;background:#fff8e6'>"
                 "⚠ <b>本期 eval set 與上期不同 → mAP/AP 不可與上期直接比較</b>(可能只是 val 變簡單)</div>")
    elif prov.get("comparable") and prov.get("prev_mAP") is not None and data.get("mAP") is not None:
        h.append(f"<p style='color:#2e7d32;font-size:13px'>與上期同一 eval set,可比較:mAP "
                 f"{_esc(prov['prev_mAP'])} → {_esc(data['mAP'])}</p>")
    return "".join(h)


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
         ".legend{color:#666;font-size:12px;margin:2px 0}.done{color:#999;text-decoration:line-through}",
         ".tax{color:#b00020}.model{color:#0064b0}.label_noise{color:#7a00b0}.clean{color:#2e7d32}.fixable{color:#0a7a3a}",
         "</style></head><body>"]
    loc, mbi = data.get("loc_gap"), data.get("map_by_iou")
    loc_html = ""
    if data.get("mAP") is not None:
        if loc is not None:
            loc_html = f" ｜ loc_gap = {_esc(loc)}" + (f" (mAP@0.5={_esc(mbi.get('0.5'))} vs @0.75={_esc(mbi.get('0.75'))})" if mbi else "")
        else:
            loc_html = " ｜ loc_gap N/A(單一 IoU,未評估定位)"
    bscope = f" ｜ 範圍:<b>batch {_esc(data['batch'])}</b>(佇列/翻盤僅看這批)" if data.get("batch") else ""
    h.append(f"<h1>YOLO 弱點報告</h1><p>模式:<b>{_esc(mode)}</b>"
             + (f" ｜ mAP@0.5 = <b>{_esc(data['mAP'])}</b>" if data.get("mAP") is not None else "") + loc_html + bscope + "</p>")
    s = data.get("summary") or {}
    if s:  # TL;DR health banner
        color = {"RED": "#b00020", "AMBER": "#a15c00", "GREEN": "#2e7d32"}.get(s.get("health"), "#555")
        todo = " ｜ ".join(s.get("todo") or []) or "—"
        h.append(f"<div id='tldr' style='border-left:5px solid {color};padding:8px 12px;margin:10px 0;background:#fafafa'>"
                 f"<b style='color:{color}'>健康度:{_esc(s.get('health'))}</b> ｜ 最弱:{_esc(s.get('weakest') or '-')}"
                 f"<br><b>現在做這個:</b> {_esc(todo)}</div>")
    h.append(f"<p class='proxy'>{_esc(_PROXY.strip('_'))};可分性綁定目前 embedding 空間。</p>")
    h.append(_provenance_html(data))

    pc = data.get("per_class") or []
    if pc:
        h.append("<h2 id='per-class'>哪一類最弱(per-class AP,弱→強)</h2>")
        h.append("<table><tr><th>類別</th><th>AP</th><th>n_gt</th><th>漏報型態(分佈)</th><th>最常混淆成</th></tr>")
        for r in pc:
            fnt = r.get("fn_types") or {}
            fn_s = " / ".join(f"{n} {t}" for t, n in sorted(fnt.items(), key=lambda kv: -kv[1])) or "-"
            h.append(f"<tr><td>{_esc(r['cls'])}</td><td>{_esc(r['ap'])}</td><td>{_esc(r['n_gt'])}</td>"
                     f"<td>{_esc(fn_s)}</td><td>{_esc(r.get('top_confusion') or '-')}</td></tr>")
        h.append("</table>")

    cons = data.get("consistency") or []
    h.append("<h2 id='consistency'>一致性歸因(GT × 嵌入:taxonomy / model / label?)</h2>")
    if cons:
        h.append("<table id='consistency-table'><tr><th>類別對</th><th>可分</th><th>sep_err [CI]</th>"
                 "<th>O[i→j] [CI]</th><th>C[i→j] [CI] (n_gt)</th><th>Δ=O−C [CI]</th><th>判定</th><th>支撐</th><th>建議</th></tr>")
        for f in cons:
            pair = f"{f['pair'][0]}↔{f['pair'][1]}"
            sup = f"g{f['support']['golden_i']}/{f['support']['golden_j']} ({f['tier']})"
            o = f"{f.get('O_ij')} {f.get('O_ci')}"
            c = (f"{f['C_ij']} {f.get('C_ci')} (n={f['support'].get('n_gt_i')})" if f.get("C_ij") is not None else "-")
            dlt = f"{f['delta']} {f.get('delta_ci')}" if f.get("delta") is not None else "-"
            # rescued -> render as representation-fixable (not taxonomy(可修)); verdict must not contradict action
            if f.get("representation_fixable"):
                cls, vtxt = "fixable", "representation-fixable（非 taxonomy 死路）"
            else:
                cls = {"taxonomy": "tax", "model": "model", "label_noise": "label_noise", "clean": "clean"}.get(f["verdict"], "")
                vtxt = f["verdict"]
            h.append(f"<tr><td>{_esc(pair)}</td><td>{_esc(f['separable_in_embedding'])}</td>"
                     f"<td>{_esc(f['sep_err'])} {_esc(f.get('sep_ci'))}</td><td>{_esc(o)}</td><td>{_esc(c)}</td>"
                     f"<td>{_esc(dlt)}</td><td class='v {cls}'>{_esc(vtxt)}</td><td>{_esc(sup)}</td><td>{_esc(f['action'])}</td></tr>")
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

    ov = data.get("overturns") or []
    if ov:
        h.append("<h2 id='overturns'>自信但嵌入翻盤(無 GT)</h2>")
        h.append(f"<p class='legend'>{_esc(_WRONGNESS_LEGEND)}</p><table>"
                 "<tr><th>影像</th><th>類別</th><th>conf</th><th>knn_dist</th><th>dist_thr</th><th>wrongness</th></tr>")
        for r in ov:
            h.append(f"<tr><td>{_esc(r['id'])}</td><td>{_esc(r.get('pred_class'))}</td><td>{_esc(r['conf'])}</td>"
                     f"<td>{_esc(r.get('knn_dist'))}</td><td>{_esc(r.get('dist_thr'))}</td><td>{_esc(round(r['wrongness'], 2))}</td></tr>")
        h.append("</table>")

    q = data.get("queue") or {}
    if q:
        hr_wq = next((hh for hh in (data.get("hit_rate") or []) if hh["queue"] == "weakness_queue"), None)
        wq_note = ""
        if hr_wq:
            cov = "-" if hr_wq.get("coverage") is None else hr_wq["coverage"]
            wq_note = f"｜已處理率(coverage) {cov}(已解決 {hr_wq['resolved']}/{hr_wq['emitted']}{',樣本不足' if hr_wq.get('insufficient') else ''})"
        h.append(f"<h2 id='queue'>該標哪些(逐弱類佇列,PROXY)</h2><p class='legend'>{_esc(_CLOSENESS_LEGEND)}</p>"
                 f"<p>完整清單見 weakness_worklist.csv {_esc(wq_note)}</p><ul>")
        for c, cands in q.items():
            n_res = sum(1 for x in cands if x.get("resolved"))
            head = f"<b>{_esc(c)}</b>" + (f"(待辦 {len(cands) - n_res} / 已解決 {n_res})" if n_res else f"({len(cands)} 個)") + ": "
            items = ", ".join(
                (f"<span class='done'>{_esc(x['id'])}</span>" if x.get("resolved") else _esc(x['id']))
                + f"({_esc(round(x['closeness'], 2))})" for x in cands)
            h.append(f"<li>{head}{items}</li>")
        h.append("</ul>")

    hr = data.get("hit_rate") or []
    if hr:
        h.append("<h2 id='hit-rate'>佇列命中率(VIX 建議準不準?自我校準)</h2>"
                 f"<p class='legend'>{_esc(_LABEL_HITRATE_NOTE)}以覆蓋率呈現。</p><table id='hit-rate-table'>"
                 "<tr><th>佇列</th><th>預測</th><th>已解決/發出</th><th>命中率/覆蓋率</th><th>趨勢</th><th>註</th></tr>")
        for qq in hr:
            note = "樣本不足僅供參考" if qq.get("insufficient") else ""
            if qq.get("predict") == "label":
                cell = ("-" if qq.get("coverage") is None else f"{qq['coverage']}(覆蓋)")
            else:
                cell = "-" if qq.get("precision") is None else str(qq["precision"])
            h.append(f"<tr><td>{_esc(qq['queue'])}</td><td>{_esc(qq['predict'])}</td>"
                     f"<td>{_esc(qq['resolved'])}/{_esc(qq['emitted'])}</td><td>{_esc(cell)}</td>"
                     f"<td>{_esc(qq.get('trend'))}</td><td>{_esc(note)}</td></tr>")
        h.append("</table>")

    h.append("</body></html>")
    return "".join(h)
