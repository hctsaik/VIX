"""Build a self-contained HTML report of the VIX GUI walkthrough (screenshots base64-embedded, so the
single .html opens/shares anywhere). Run after dogfood_walkthrough.py has produced the screenshots:
    python docs/examples/gen_walkthrough_html.py   ->   docs/guide/GUI_WALKTHROUGH.html
"""
from __future__ import annotations

import base64
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SHOTS = ROOT / "docs" / "guide" / "walkthrough"
OUT = ROOT / "docs" / "guide" / "GUI_WALKTHROUGH.html"
EMBED = "--embed" in sys.argv  # --embed -> base64 self-contained single file (portable); default -> relative links

STEPS = [
    ("01_grid-real-pothole-data.png", "你的資料", "真 patHole 影像 + GT/模型預測框,載入 live FiftyOne App。"),
    ("02_operator-browser-open.png", "動作入口", "按 <code>`</code> 打開 operator browser —— 所有 VIX 動作都從這裡叫。"),
    ("03_generate-report-pick-eval.png", "目標一 · 用「選的」選 eval",
     "選 <b>VIX: 產生模型弱點報告(選 eval)</b>,下拉自動列出 workspace 裡的 eval JSONL(每行 {vix_hash, gt, pred}),也可填自訂路徑。等同 CLI 的 <code>vix eval-ingest</code> + <code>vix weakness-report</code>。"),
    ("04_weakness-report-panel.png", "目標一 · 報告直接在 App 內呈現",
     "<b>VIX: 弱點/一致性報告</b> 面板:per-class AP、最「自信卻錯」(GT 證實的高信心誤報)、健康度、PROXY 誠實標記。本次真實 <b>mAP@0.5 = 0.7234</b>。"),
    ("05_flag-label-issues-operator.png", "目標二 · 標出疑似不準的標註",
     "選 <b>VIX: 標出疑似不準的標註</b> —— 一鍵跑 <code>audit_labels</code>(疑似標錯類別)+ <code>box_qa</code>(框幾何:退化/截邊/面積·長寬離群)。"),
    ("06_inaccurate-label-worklist.png", "目標二 · 不準標註工作清單",
     "頂端 view bar 已套 <code>MatchTags vixq:box_issue</code> —— 格狀只剩被標記的影像。本次種了 4 個壞框,VIX 共標出 <b>28 張</b>(4 個種的 + 24 個真實面積/長寬離群)。"),
    ("07_flagged-sample-bad-box.png", "點進一張,直接看到不準的框", "從清單點開,直接檢視那個有問題的標註框。"),
    ("08_review-queue-panel.png", "一站式 · 可點的覆核佇列",
     "<b>VIX: 覆核佇列</b> 面板:按風險排序的 風險 / vix_hash / 原因(proxy) 表,列動作 看圖 / 確認→golden / 誤報排除 —— 整個覆核迴圈在一個面板完成。"),
]


def _img(name: str) -> str:
    if EMBED:  # portable single file
        return "data:image/png;base64," + base64.b64encode((SHOTS / name).read_bytes()).decode("ascii")
    return f"walkthrough/{name}"  # small file; loads the adjacent committed screenshots


def main():
    cards = []
    for i, (fn, title, desc) in enumerate(STEPS, 1):
        cards.append(
            f"<section class='step'><div class='hd'><span class='num'>{i}</span><h2>{title}</h2></div>"
            f"<p class='desc'>{desc}</p><img loading='lazy' src='{_img(fn)}' alt='{title}'></section>")
    html = f"""<!doctype html><html lang='zh-Hant'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>VIX GUI 操作走查</title><style>
body{{font-family:system-ui,'Microsoft JhengHei',sans-serif;margin:0;background:#0f1115;color:#e6e6e6}}
.wrap{{max-width:1040px;margin:0 auto;padding:28px 20px 60px}}
h1{{font-size:26px;margin:0 0 6px}} .sub{{color:#9aa4b2;font-size:14px;margin:0 0 18px}}
.note{{background:#1b2433;border-left:4px solid #d9a441;border-radius:8px;padding:12px 16px;font-size:13px;color:#cdd6e2;margin:14px 0 26px}}
.goals{{display:flex;gap:12px;flex-wrap:wrap;margin:0 0 24px}}
.goal{{flex:1 1 320px;background:#16202e;border:1px solid #243245;border-radius:10px;padding:12px 16px}}
.goal b{{color:#5fd0a0}}
.step{{background:#141b26;border:1px solid #222e3f;border-radius:12px;padding:16px 18px;margin:18px 0}}
.hd{{display:flex;align-items:center;gap:12px;margin:0 0 6px}}
.num{{background:#3a78d6;color:#fff;font-weight:700;border-radius:50%;width:30px;height:30px;display:inline-flex;align-items:center;justify-content:center;flex:0 0 30px}}
h2{{font-size:18px;margin:0}} .desc{{color:#b9c2cf;font-size:14px;margin:4px 0 12px;line-height:1.6}}
code{{background:#0c1118;color:#7fd0ff;padding:1px 6px;border-radius:5px;font-size:13px}}
img{{width:100%;border:1px solid #2a3a50;border-radius:8px;display:block}}
footer{{color:#7e8a99;font-size:12px;margin-top:28px;border-top:1px solid #222e3f;padding-top:14px}}
</style></head><body><div class='wrap'>
<h1>VIX 在 FiftyOne App 的操作走查</h1>
<p class='sub'>Playwright 實機截圖 · 真 patHole 資料 + 真訓練的 YOLOv8n(mAP@0.5 0.7234)</p>
<div class='goals'>
<div class='goal'><b>目標一</b>:在 GUI 用「選的」產生模型弱點報告(步驟 3–4)</div>
<div class='goal'><b>目標二</b>:容易看出哪些標註不準、要調整(步驟 5–7)</div></div>
<div class='note'><b>誠實說明:</b>有表單的 operator(產生報告)截「表單畫面」證明可選,效果用它<b>同一支 pipeline.*</b> 套用;<code>flag_label_issues</code> 同理。為示範「抓不準標註」<b>故意種了 4 個壞框</b>,其餘為真標註。每個關鍵步驟另有非視覺交叉驗證(eval_results.json 寫出、vixq tag 數、hash 鏈完整)。重現:<code>python docs/examples/dogfood_walkthrough.py</code>。</div>
{''.join(cards)}
<footer>侷限:<code>box_qa</code> 抓「框幾何不對」、<code>audit_labels</code> 抓「類別標錯」(單類別 pothole 下後者通常為 0);抓「框畫得鬆但形狀正常(像素級不貼合)」需 opt-in 的 SAM —— 尚未納入。<br>由 <code>docs/examples/gen_walkthrough_html.py</code> 產生(截圖 base64 內嵌,單檔可攜)。</footer>
</div></body></html>"""
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size/1_000_000:.1f} MB, {len(STEPS)} steps embedded)")


if __name__ == "__main__":
    main()
