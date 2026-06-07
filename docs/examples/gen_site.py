# -*- coding: utf-8 -*-
"""Generate the VIX beginner documentation SITE (multi-page, shared sidebar, dark zh-Hant theme).

One HTML file per page under docs/guide/site/, sharing assets/site.css and an identical nav so a
first-time engineer can read straight through. Reuses existing screenshots in place (App walkthrough,
DINO label-audit, embeddings) and the freshly generated `vix diagnose` report shots in site/img/.

Regenerate:  python docs/examples/gen_site.py
(Report screenshots: python docs/examples/gen_beginner_report.py && python docs/examples/shoot_beginner_report.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

SITE = Path(__file__).resolve().parent.parent / "guide" / "site"

# ---- shared nav (ordered; (href, label, right-tag)) grouped by section ----
NAV = [
    ("入門", [
        ("index.html", "首頁 / 5 分鐘上手", ""),
        ("install.html", "安裝", ""),
    ]),
    ("核心流程", [
        ("diagnose.html", "診斷你的模型", "A"),
        ("report.html", "讀懂弱點報告", "A"),
        ("formats.html", "輸入格式 (YOLO/VOC/COCO)", "A"),
        ("loop.html", "修了有沒有幫助?", "A"),
    ]),
    ("進階 (需 DINOv2 / App)", [
        ("audit.html", "稽核標籤本身", "B"),
        ("app.html", "在 App 裡覆核", "B"),
        ("similarity.html", "找相似的物件 (DINO)", "B"),
    ]),
    ("觀念 / 參考", [
        ("honesty.html", "誠實邊界與限制", ""),
        ("reference.html", "參考與完整手冊", ""),
    ]),
]


def nav_html(active: str) -> str:
    out = ['<nav class="toc">',
           '<a class="brand" href="index.html">VIX 文件</a>',
           '<p class="tagline">yolo val 給你 mAP;VIX 告訴你「該修什麼」</p>']
    for section, items in NAV:
        out.append(f"<h2>{section}</h2>")
        for href, label, tag in items:
            cls = "active" if href == active else ""
            r = f' <span class="r">{tag}</span>' if tag else ""
            out.append(f'<a class="{cls}" href="{href}">{label}{r}</a>')
    out.append("</nav>")
    return "\n".join(out)


def page(fname: str, title: str, sub: str, body: str, cta: str = "") -> None:
    body = body.replace("[[TAGS]]", TAGS)
    html = f"""<!DOCTYPE html><html lang="zh-Hant"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} · VIX 文件</title>
<link rel="stylesheet" href="assets/site.css">
</head><body><div class="layout">
{nav_html(fname)}
<main>
<h1>{title}</h1>
<p class="sub">{sub}</p>
{body}
{cta}
<footer>VIX 新手文件 · 本頁可能落後於程式碼,CLI 一律以 <code>vix --help</code> 為準 ·
由 <code>docs/examples/gen_site.py</code> 產生。</footer>
</main></div></body></html>"""
    (SITE / fname).write_text(html, encoding="utf-8")
    print("wrote", fname)


def cta(href: str, label: str, desc: str) -> str:
    return f'<div class="cta"><h3>下一步</h3><p>{desc}</p><a href="{href}">→ {label}</a></div>'


TAGS = ('<span class="tag a">Tier A = 離線、免 FiftyOne/DINOv2</span>'
        '<span class="tag b">Tier B = 需 DINOv2 / FiftyOne App</span>')

# ============================ PAGES ============================

index_body = """
<div class="note">這份文件給<b>第一次使用 VIX 的電腦視覺工程師</b>。照著讀、照著貼上指令,十分鐘內就能對「你自己的模型 + 你自己的資料」產出一份「該修什麼」的報告。[[TAGS]]</div>

<h2 class="sec">VIX 是什麼?</h2>
<p><code>yolo val</code> 會告訴你 mAP=0.72,但<b>不會</b>告訴你:哪一類最弱、為什麼弱(漏報?混淆?框太鬆?)、該先看哪些圖。
<b>VIX 把單一個 mAP 變成一張可行動的待辦清單</b> —— 離線、一行指令、不用重訓、不用架服務。</p>

<h2 class="sec">5 分鐘上手(最短路徑)</h2>
<div class="ref">前提:你有 (1) 一個影像資料夾、(2) 對應的標籤(YOLO <code>labels/*.txt</code> / Pascal VOC <code>annotations/*.xml</code> / COCO <code>instances.json</code>)、(3) 你訓練好的模型 <code>best.pt</code>。沒有 FiftyOne、沒有 GPU 也能跑(Tier A)。</div>
<pre><code># 安裝核心(任何機器都能跑 Tier A)
git clone &lt;repo&gt; &amp;&amp; cd VIX
pip install -e .
# 要實際跑模型才需要(若你已有 yolo val 環境就略過):
pip install ultralytics

# 一行:匯入你的標籤 + 跑你的模型 → 弱點報告
vix diagnose ./dataset --labels yolo --weights best.pt --data-yaml data.yaml</code></pre>
<div class="ref"><code>data.yaml</code> 就是你訓練時用的那個(內含 <code>names:</code> 類名)。沒有的話,改用 <code>--names car,person,bike</code> 直接給類名(類別是數字 0/1/2 時必填其一)。</div>
<div class="tip">報告會寫到目前目錄下自動建立的 <code>vix_workspace/weakness_report.html</code>(可用 <code>--out</code> 改路徑)。VIX <b>只讀不寫</b>你的影像與標籤,不會就地改動你的資料。</div>
<p>終端會印出像這樣的摘要,並寫出一份可瀏覽的 HTML 報告:</p>
<pre><code>匯入 133 影像 / 412 框 / 1 類(參照=未覆核標籤)
Tier A 模型評估:mAP@0.5=0.7234  loc_gap=0.04  per-class AP={'pothole': 0.7234}
健康度:AMBER ｜ 最弱:pothole AP=0.72 (n_gt=412)
  → 覆核 12 個自信誤報
報告:vix_workspace/weakness_report.md(同名 .html 可瀏覽)
下一步(閉環):這是本輪基準… 凍結這個 eval set…</code></pre>

<h2 class="sec">這份報告長什麼樣?</h2>
<figure><a href="report.html"><img src="img/report_full.png" alt="VIX 弱點報告範例"></a>
<figcaption>真實的 <code>vix diagnose</code> 報告(多類別範例)。最上方是健康度與「現在做這個」,接著是 per-class AP(弱→強)、與上次同一 eval set 的 Δ、混淆、最自信卻錯的影像。點圖看「讀懂報告」。</figcaption></figure>

<h2 class="sec">接下來看哪裡</h2>
<div class="cards">
<a href="install.html"><b>安裝</b><span>核心(Tier A)與 Tier-2(App/DINOv2)兩條路</span></a>
<a href="diagnose.html"><b>診斷你的模型 ★</b><span>on-ramp 的完整說明與選項</span></a>
<a href="report.html"><b>讀懂弱點報告</b><span>每個欄位的意思 + 誠實 hedge</span></a>
<a href="loop.html"><b>修了有沒有幫助?</b><span>用凍結 eval set 看 per-class Δ</span></a>
<a href="audit.html"><b>稽核標籤本身</b><span>用 DINOv2 嵌入找疑似標錯</span></a>
<a href="app.html"><b>在 App 裡覆核</b><span>FiftyOne 視覺化點選工作流</span></a>
<a href="similarity.html"><b>找相似的物件</b><span>選一個框→放大鏡,DINO 物件級相似(免 Enterprise)</span></a>
</div>
<div class="warn"><b>一句話誠實聲明:</b>你<b>匯入的標籤是「未覆核的參照」</b>,不是真理。報告裡的「誤報」有可能其實是<b>你的標籤漏框</b>而模型是對的。詳見 <a href="honesty.html">誠實邊界</a>。</div>
"""

install_body = """
<div class="note">VIX 有兩條安裝路線。<b>絕大多數新手只需要第一條(Tier A)</b> —— 它能跑 <code>vix diagnose</code> 的全部 headline 功能,無需 FiftyOne、無需 GPU、可離線。</div>

<h2 class="sec">路線 1:核心(Tier A,推薦先裝這個)</h2>
<pre><code>git clone &lt;repo&gt; &amp;&amp; cd VIX
pip install -e .          # 純 Python:numpy + pyyaml + pillow
vix --help               # 應看到 ~16 個常用指令(其餘進階指令已隱藏)
vix                      # 印出 5 分鐘上手指引</code></pre>
<p>要用 <code>vix diagnose --weights</code> 實際跑你的模型,需要你原本的推論環境。若你<b>還沒有</b> <code>yolo val</code> 的環境:</p>
<pre><code>pip install ultralytics      # 只有在你需要 VIX 替你跑模型時才需要</code></pre>
<p>若你已經有推論環境(<span class="pill">ultralytics</span> / <span class="pill">torch</span>),通常什麼都不用再裝。</p>
<div class="ref">速度:Tier A 的瓶頸是「你的模型推論」那一步,和你平常 <code>yolo val</code> 同量級(CPU 上每張數十毫秒~數百毫秒)。VIX 自己的比對是純 Python、很快。大資料集建議先在 val 子集上試跑。</div>
<div class="tip">沒有模型、只有 predictions+GT 的 JSONL?用 <code>vix eval-ingest results.jsonl</code> —— 純標準函式庫,真正離線,連 torch 都不用。見 <a href="formats.html">輸入格式</a>。</div>

<h2 class="sec">路線 2:Tier-2(App + DINOv2 嵌入稽核)</h2>
<p>只有在你要用 <b>FiftyOne 視覺化 App</b>(<a href="app.html">在 App 裡覆核</a>)或 <b>嵌入式標籤稽核</b>(<code>--audit</code>,見 <a href="audit.html">稽核標籤</a>)時才需要。</p>
<div class="warn"><b>Python 版本很重要:</b>FiftyOne 目前需要 <b>Python 3.11</b>(別用 3.13/3.14)。本專案用一個獨立的 <code>.venv311</code> 虛擬環境跑 Tier-2。</div>
<pre><code># 在 Python 3.11 的虛擬環境裡
pip install -r requirements-tier2.txt   # fiftyone + torch + playwright(版本已釘選)
# DINOv2 權重第一次使用會自動下載一次(需網路);air-gapped 請先預先放好快取
vix diagnose ./dataset --labels voc --weights best.pt --audit
vix app                                  # 開啟 FiftyOne + @vix/review 外掛</code></pre>
<div class="ref">完整跨機器安裝、CIFAR demo、疑難排解表:見 <code>docs/SETUP_OTHER_MACHINE.md</code>。</div>

<h2 class="sec">裝好了嗎?自我檢查</h2>
<div class="check"><b>Tier A OK:</b> <code>vix --help</code> 列出 <code>diagnose / import-labels / eval-run …</code>;<code>vix status</code> 不報錯。<br>
<b>Tier B OK:</b> <code>python -c "import fiftyone"</code> 成功;<code>vix app</code> 能開瀏覽器。</div>
"""

diagnose_body = """
<div class="note">這是 VIX 最重要的一個指令,也是新手唯一需要先學會的東西。它把<b>你既有的標籤</b>和<b>你自己的模型</b>串起來,一行就產出弱點報告。[[TAGS]]</div>

<h2 class="sec">一行指令</h2>
<pre><code>vix diagnose ./dataset --labels yolo --weights best.pt --data-yaml data.yaml</code></pre>
<p>它依序做了四件事(你不用一步步下指令):</p>
<ol class="steps">
<li><b>匯入你的標籤</b>(<code>import-labels</code>):用<b>內容雜湊</b>把標籤對到影像(不是用檔名,避免重名出錯)。匯入的標籤標記為「未覆核參照」,<b>永遠不會</b>被當成 golden。</li>
<li><b>跑你的模型</b>(<code>eval-run</code>):用你的 <code>best.pt</code> 在這些影像上推論。</li>
<li><b>比對</b>:算出 per-class AP、typed 漏報/誤報、混淆矩陣、最自信卻錯的偵測。</li>
<li><b>寫報告</b>:<code>weakness_report.md</code> 與可瀏覽的 <code>.html</code>。</li>
</ol>

<h2 class="sec">常用選項</h2>
<table>
<tr><th>選項</th><th>用途</th></tr>
<tr><td><code>--labels yolo|voc|coco</code></td><td>你的標籤格式(見 <a href="formats.html">輸入格式</a>)</td></tr>
<tr><td><code>--weights best.pt</code></td><td>你的模型(Tier A:算 FP/FN/AP/混淆)。省略則只匯入標籤</td></tr>
<tr><td><code>--data-yaml data.yaml</code> 或 <code>--names a,b,c</code></td><td>YOLO 數字類別 → 可讀類名</td></tr>
<tr><td><code>--json instances.json</code></td><td>COCO 標註檔路徑</td></tr>
<tr><td><code>--audit</code></td><td><span class="tag b">Tier B</span> 加做嵌入式標籤稽核 + 失敗歸因(需 DINOv2)</td></tr>
<tr><td><code>--out 路徑</code></td><td>報告輸出位置(預設 <code>workspace/weakness_report.md</code>)</td></tr>
</table>

<h2 class="sec">你會在終端看到</h2>
<pre><code>匯入 133 影像 / 412 框 / 1 類(參照=未覆核標籤)
Tier A 模型評估:mAP@0.5=0.7234  loc_gap=0.04  per-class AP={'pothole': 0.7234}
健康度:AMBER ｜ 最弱:pothole AP=0.72 (n_gt=412)
  → 覆核 12 個自信誤報
報告:.../weakness_report.md(同名 .html 可瀏覽)
下一步(閉環):這是本輪基準(首份或與上次不可比)。**凍結這個 eval set**…</code></pre>
<figure><img src="img/report_tldr.png" alt="報告最上方的健康度橫幅與待辦"><figcaption>報告最上方:健康度(RED/AMBER/GREEN)+「現在做這個」待辦。</figcaption></figure>

<div class="warn"><b>找不到框?</b>若你指到錯的資料夾,VIX 會<b>大聲報錯而不是給你一份假的「全部健康」報告</b>,並告訴你各格式該放哪(yolo: <code>labels/</code>;voc: <code>annotations/</code>;coco: <code>--json</code>)。</div>
<div class="check"><b>你現在應該有:</b>一個 <code>weakness_report.html</code>,最弱的類別排在 per-class 表最上面。接著學怎麼讀它。</div>
"""

report_body = """
<div class="note">這頁教你讀 <code>vix diagnose</code> 產出的 <code>weakness_report.html</code>。每個區塊都附「怎麼用」與「怎麼<b>不要</b>被誤導」。</div>
<figure><img src="img/report_full.png" alt="完整弱點報告"><figcaption>完整報告(多類別範例)。下面逐區塊說明。</figcaption></figure>

<h2 class="sec">① 健康度橫幅 + 現在做這個</h2>
<p><b>RED / AMBER / GREEN</b> 一眼看出整體狀況,後面「現在做這個」給出最高優先的動作(覆核自信誤報、補某類樣本…)。最弱的類別與其 AP 也在這裡。</p>

<h2 class="sec">② 未覆核參照的誠實橫幅</h2>
<div class="ref">⚠ 參照 = 你匯入的標籤(未經 VIX 覆核)。下方「誤報/漏報」是「<b>模型 vs 你的標籤</b>」的不一致,<b>不是</b>「模型 vs 真實世界」。一個「誤報」可能其實是<b>你的標籤漏框/標錯</b>而模型是對的。要分辨,用 <code>--audit</code>。</div>
<p>這不是免責聲明客套話 —— 它是 VIX 的核心原則。報告不會把你的標籤當成真理。</p>

<h2 class="sec">③ per-class AP(弱 → 強)</h2>
<p>最該關注的表。<b>AP</b>(Average Precision)是該類偵測的綜合準確度,0~1 越高越好;<b>mAP</b> 是所有類別 AP 的平均。每一列:該類 <b>AP</b>、<b>n_gt</b>(這類有幾個 GT 框)、<b>漏報型態分佈</b>(classification 分類錯 / localization 框不準 / missed 完全沒偵到)、<b>最常被混淆成哪一類</b>。<b>弱的排在最上面</b>,所以你一眼知道先修哪類。</p>
<div class="note"><b>Δ(同 eval set)</b> 欄:當這次和上次是在<b>同一個 eval set</b> 上跑(見 <a href="loop.html">修了有沒有幫助</a>)才會出現,顯示每類 AP 的變化。<b>n_gt &lt; 20 的類別</b>會標 <b>「n少不穩」而不畫 ↑/↓ 箭頭</b> —— 因為 6 個框的 +0.5 多半是雜訊,不是真的進步(範例中的 traffic_light 就是這樣)。</div>

<h2 class="sec">④ 漏報型態 / 混淆</h2>
<p>VIX 會把每個錯誤<b>分類且不重複計數</b>:同一個「框太鬆」只算一次 localization 漏報,不會又算一次 background 誤報。混淆區塊列出「真實 A 被偵成 B」最多的配對 —— 例如 <code>bicycle → motorcycle (6)</code>,告訴你模型把腳踏車當機車。</p>

<h2 class="sec">⑤ 最自信卻錯(confident-wrong)</h2>
<p>模型高信心、但在你標籤裡找不到對應框的偵測。這些最該優先人工看。<b>但記得 ②:</b><code>type=background</code> 代表「你的標籤這裡沒有框」→ 可能是模型幻覺,<b>也可能是你漏標的 GT</b>。</p>
<div class="warn"><b>所有「該標這些」的排序都是 PROXY:</b>因為 VIX 不重訓,它給的是「嫌疑/優先順序」,<b>不是</b>「標了就會讓 mAP 上升多少」的保證。</div>
<div class="check"><b>你會讀報告了。</b>想知道「修了到底有沒有讓模型變好」?那需要正確的量法 —— 看下一頁。</div>
"""

formats_body = """
<div class="note">VIX 直接吃三種主流標註格式。重點是<b>資料夾擺對位置</b>,VIX 才找得到標籤。找不到時它會大聲報錯,不會假裝健康。</div>

<h2 class="sec">YOLO(sibling labels/)</h2>
<pre><code>dataset/
  images/  a.jpg  b.jpg ...
  labels/  a.txt  b.txt ...     # 每行: cls cx cy w h(都已正規化 0~1)
  data.yaml                      # names: [car, person, ...]
vix diagnose ./dataset --labels yolo --weights best.pt --data-yaml data.yaml</code></pre>
<p>類別是數字索引時(下載的資料集常見 <code>0 1 2</code>),用 <code>--data-yaml</code>(讀 <code>names:</code>)或 <code>--names</code> 給它可讀名稱:</p>
<pre><code># 標籤是 "0 0.5 0.5 0.2 0.2" 這種數字類別,且沒有 data.yaml:
vix diagnose ./dataset --labels yolo --weights best.pt --names car,person,bike
# 標籤檔不在 sibling labels/(例如放在 ann/)時:
vix diagnose ./dataset --labels yolo --weights best.pt --names car,person --label-dir ann</code></pre>
<div class="ref">沒給 <code>--names</code> 也沒 <code>--data-yaml</code> 而標籤是數字時,VIX 會大聲報錯「類別索引 N 不在 names 對照」,不會默默用數字當類名。</div>

<h2 class="sec">Pascal VOC(annotations/*.xml)</h2>
<pre><code>dataset/
  images/ ...               annotations/ a.xml b.xml ...   # 絕對座標 xmin/ymin/xmax/ymax
vix diagnose ./dataset --labels voc --weights best.pt</code></pre>

<h2 class="sec">COCO(單一 instances.json)</h2>
<pre><code>vix diagnose ./images --labels coco --json instances.json --weights best.pt</code></pre>

<h2 class="sec">我沒有要在 VIX 裡跑模型(只有 predictions + GT)</h2>
<p>若你已經自己跑過推論,手上有一份 JSONL,每行一張圖,就直接餵給 VIX(<b>純離線、免 torch</b>):</p>
<pre><code>{"vix_hash": "img001", "gt": [{"label": "car", "bbox": [0.5,0.5,0.2,0.2]}],
 "pred": [{"label": "car", "bbox": [0.51,0.5,0.2,0.2], "conf": 0.93}]}
# bbox 為正規化 (cx, cy, w, h)
vix eval-ingest results.jsonl
vix weakness-report</code></pre>

<div class="warn"><b>常見錯誤:</b>標籤檔用 Windows 編輯器存成帶 BOM 的檔 —— VIX 已能容忍。但若 <code>--labels yolo</code> 卻沒有 <code>labels/</code> 目錄,會看到「找到 N 張影像但 0 個標籤配對成功」並附上它找過哪些路徑,照提示修即可。</div>
<div class="ref">座標慣例:VIX 內部一律用正規化 <code>(cx, cy, w, h)</code>。VOC 的絕對 <code>xyxy</code> 與 COCO 的絕對 <code>xywh</code> 會自動換算;帶標註卻沒有有效影像尺寸的檔案會<b>大聲報錯</b>(而不是默默丟掉框)。</div>
"""

loop_body = """
<div class="note">這頁回答工程師最在意的問題:<b>「我花一週修標籤、重訓了,到底有沒有變好?」</b> VIX 能誠實地告訴你 —— 前提是你用對量法。</div>

<h2 class="sec">誠實的閉環</h2>
<ol class="steps">
<li><code>vix diagnose</code> → 看報告,知道哪類弱、哪些框可能錯。</li>
<li>在<b>你的標註工具</b>修「<b>訓練集</b>」標籤 → <b>外部重訓</b>你的模型(VIX 從不替你訓練)。</li>
<li>在<b>同一個「凍結」的 held-out eval set</b> 上再跑一次 <code>vix diagnose</code>。</li>
<li>報告的 per-class 表會出現 <b>Δ(同 eval set)</b> 欄,告訴你每類 AP 動了多少。(或用 <code>vix ap-trend</code> 看跨輪趨勢。)</li>
</ol>

<h2 class="sec">關鍵誠實規則:eval set 必須「凍結」</h2>
<div class="warn"><b>為什麼不能改 eval set 的標籤?</b> VIX 用 <code>eval_set_hash</code> 綁定一次評估的「那一組 GT」。<b>這個雜湊包含 GT 標籤與框</b> —— 你只要動了 eval set 的標籤,雜湊就變,VIX 會<b>拒絕</b>顯示 Δ 並標示「本期 eval set 與上期不同 → 不可比較」。<br><br>
所以正確做法是:<b>修「訓練集」的標籤,不要動「eval set」</b>;用一份不變的 held-out eval set 來量改善。否則你看到的 mAP 變化分不清是「模型變好」還是「考卷變簡單」。</div>

<h2 class="sec">VIX 不會替你宣稱因果</h2>
<p>Δ 欄只說「在這個固定 eval set 上,這類 AP 從 0.41 → 0.58」,並註明「<b>你於外部重訓;非 VIX 造成</b>」。它不會說「VIX 讓你的模型變好」或「標這些一定讓 mAP +X」。低支撐類別只標「n少不穩」不畫箭頭。這些都是刻意的:<b>給你決策所需的事實,但不誇大。</b></p>
<div class="check"><b>你現在能回答:</b>「這類 AP 在固定考卷上真的進步了嗎?」→ 看 Δ 欄。值不值得繼續修某類 → 你自己決定,VIX 只給誠實數字。</div>
"""

audit_body = """
<div class="note"><span class="tag b">Tier B</span> 前面幾頁靠「模型 vs 你的標籤」找問題。這頁反過來:用 <b>DINOv2 嵌入</b>直接檢查<b>標籤本身</b>對不對 —— 不需要模型。需要 Tier-2 安裝(見 <a href="install.html">安裝</a>)。</div>

<h2 class="sec">嵌入式標籤稽核</h2>
<pre><code>vix diagnose ./dataset --labels voc --audit          # 最簡單:一個指令把標籤匯入+嵌入+稽核做完
# 或分步(需先匯入並計算嵌入):
vix import-labels ./dataset --labels voc
vix embed
vix audit-labels --top 20                            # 列出最可疑的標籤</code></pre>
<div class="ref">先決條件:<code>audit-labels</code>/<code>near-dup-labels</code> 需要工作區裡已有「匯入的標籤 + DINOv2 嵌入」。直接用 <code>vix diagnose --audit</code> 會一次幫你做完,新手建議走這條。</div>
<p>原理:把每個框的影像嵌入到 DINOv2 空間,看它的<b>最近鄰多半是別類</b> → 疑似標錯(<code>given → suggested(disagree=…)</code>)。下圖是一個植入錯誤的示範資料集,離群點一眼可見:</p>
<div class="gallery">
<figure><img src="../dino_labelaudit/00.png" alt="DINO 標籤稽核 1"></figure>
<figure><img src="../dino_labelaudit/03.png" alt="DINO 標籤稽核 2"></figure>
<figure><img src="../dino_labelaudit/07.png" alt="DINO 標籤稽核 3"></figure>
</div>
<div class="ref">完整離群圖庫見 <a href="../DINO_LABELAUDIT.html">DINO_LABELAUDIT.html</a>。</div>

<h2 class="sec">因果確定的標錯:near-dup-labels</h2>
<pre><code>vix near-dup-labels</code></pre>
<p>找出<b>幾乎一模一樣的影像卻帶矛盾標籤</b>的群組。這是「至少有一個一定錯」的鐵證(不是 proxy),最適合優先處理。</p>

<h2 class="sec">誠實防火牆(重要)</h2>
<div class="warn">當參照是你<b>未覆核的匯入標籤</b>時,VIX <b>不會</b>直接判「label_noise(你的標籤是雜訊)」—— 那等於用受審的標籤定罪自己(循環論證)。它只會給 <b>label_audit_needed</b>:「先人工覆核這些標籤」。可分性這種「在嵌入空間分不分得開」的幾何陳述照常顯示(它不主張誰對誰錯)。</div>
<div class="warn"><b>離線品質提醒:</b><code>--adapter memory</code> 的像素 fallback 嵌入<b>不適合</b>稽核;標籤稽核請用真正的 DINOv2(Tier-2)。第一次使用 DINOv2 會下載一次權重。</div>
"""

app_body = """
<div class="note"><span class="tag b">Tier B</span> 喜歡用滑鼠點、用眼睛看?把 VIX 的訊號丟進 <b>FiftyOne App</b>,在 <code>@vix/review</code> 外掛裡點選覆核。需 Tier-2(Python 3.11 + FiftyOne)。下面的截圖都是在真實的 pothole 資料上實際操作。</div>

<h2 class="sec">開啟 App</h2>
<pre><code>vix diagnose ./dataset --labels voc --weights best.pt --audit
vix weakness-report --worklist        # 把待辦轉成可點的 saved views(vixq:* 標籤)
vix app                               # 開 FiftyOne + @vix/review 外掛</code></pre>
<div class="check"><b>成功的樣子:</b>瀏覽器自動開啟 <code>http://localhost:5151</code>,看到影像網格;左側出現 review/pass 與 <code>vixq:*</code> 工作清單的 saved views;右上可開 operator 瀏覽器(快捷鍵 <kbd>`</kbd>)。</div>
<div class="warn"><b>App 開不起來 / 空白?</b>九成是 Python 版本:FiftyOne 需要 <b>Python 3.11</b>。請在 <code>.venv311</code> 裡跑(見 <a href="install.html">安裝</a>),不要用基礎的 3.13/3.14。其次檢查 <code>python -c "import fiftyone"</code> 是否成功。</div>

<h2 class="sec">實際操作(逐步截圖)</h2>
<ol class="steps">
<li>用真實資料開啟 App 的網格檢視。
<figure><img src="../walkthrough/01_grid-real-pothole-data.png" alt="App 網格"></figure></li>
<li>打開 operator 瀏覽器,選 <code>generate_weakness_report</code> 並挑 eval。
<figure><img src="../walkthrough/03_generate-report-pick-eval.png" alt="產生報告 operator"></figure></li>
<li>在面板裡直接看弱點報告(mAP、per-class、混淆)。
<figure><img src="../walkthrough/04_weakness-report-panel.png" alt="弱點報告面板"></figure></li>
<li>用 <code>flag_label_issues</code> / <code>audit_label_errors</code> 標出疑似錯標,形成工作清單。
<figure><img src="../walkthrough/06_inaccurate-label-worklist.png" alt="不準標籤工作清單"></figure></li>
<li>點一張被標記的樣本,看到那個有問題的框。
<figure><img src="../walkthrough/07_flagged-sample-bad-box.png" alt="被標記的壞框"></figure></li>
<li>用 <code>flag_loose_boxes</code>(SAM)找出太鬆的框。
<figure><img src="../walkthrough/09_flag-loose-boxes-operator-sam.png" alt="SAM 框鬆稽核"></figure></li>
</ol>
<div class="ref">完整 10 步走查(含可點的覆核佇列):見 <a href="../GUI_WALKTHROUGH.html">GUI_WALKTHROUGH.html</a>。</div>

<h2 class="sec">(選用)看 Embeddings 分群</h2>
<p>在 App 的 Embeddings 面板可以框選一個群集、只看選取、檢視那些影像是什麼 —— 很適合找離群與重複。</p>
<div class="gallery">
<figure><img src="../img/raw_embeddings.png" alt="Embeddings 面板"></figure>
<figure><img src="../img/step5.png" alt="框選群集"></figure>
</div>
<div class="ref">圖解走查見 <a href="../EMBEDDINGS_HOWTO.html">EMBEDDINGS_HOWTO.html</a>。</div>
"""

similarity_body = """
<div class="note"><span class="tag b">Tier B</span> 在 App 裡點 <b>Similarity Search</b> 看到「<b>Upgrade to FiftyOne Enterprise</b>」?那個<b>面板</b>是付費功能 —— 但「按相似度搜尋」這個<b>能力</b>在開源版就有,而且 VIX 用你已經算好的 <b>DINOv2 物件嵌入</b>讓它變成<b>物件級</b>的(找長得像的<b>瑕疵</b>,不是只找像的整張背景)。離線、不需 Enterprise、不需外部向量資料庫。</div>

<h2 class="sec">關鍵:比的是「框裡的物件」,不是「整張圖」</h2>
<p>相似度準不準,取決於你<b>拿什麼去比</b>:</p>
<table>
<tr><th>比較對象</th><th>找到的是</th><th>對找瑕疵</th></tr>
<tr><td>整張圖(scene-level)</td><td>構圖 / 光線 / 路面像的<b>照片</b></td><td>會被背景主導 → 較不準</td></tr>
<tr><td><b>物件框(object-level,VIX 用這個)</b></td><td>那個<b>缺陷本身</b>長得像的</td><td>找相似瑕疵 / 找漏標 / 找重複 → 準</td></tr>
</table>
<p>VIX 對<b>每個框的裁切區</b>算 DINOv2 嵌入,索引就建在這些<b>物件向量</b>上(<code>patches_field</code>),所以「相似」= 物件相似。底層用 <b>sklearn 精確最近鄰</b>(exact NN),不是近似 —— 排序就是真正的 cosine 最近鄰。</p>

<h2 class="sec">① 建立索引(一次)</h2>
<p><b>App 裡(最省事):</b>格狀檢視工具列點 <b>🔎 VIX: 建立相似搜尋索引</b> 按鈕(或按 <kbd>`</kbd> 搜 <code>build_similarity</code>)。框還沒有 DINO 嵌入時,它會先離線算一次;之後重點按也只會更新索引。</p>
<div class="tip"><b>自動偵測加速硬體:</b>算嵌入前 VIX 會自動偵測並使用最快的裝置 —— <b>CUDA(NVIDIA)→ MPS(Apple)→ CPU</b>,並印出用了哪個(例:「偵測到 NVIDIA GPU → 使用 cuda 加速」)。有 GPU 時很快;純 CPU 整個資料集可能數分鐘(每框約零點幾秒)。要強制指定可設環境變數 <code>VIX_DINOV2_DEVICE=cuda|mps|cpu</code>。</div>
<p><b>CLI 等價:</b></p>
<pre><code>vix similarity      # 對每個框算 DINOv2 嵌入(若還沒) + 建立物件框相似索引</code></pre>
<figure><img src="img/sim_build.jpg" alt="建立相似搜尋索引按鈕與完成訊息"><figcaption>工具列的「建立相似搜尋索引」按鈕;完成後 App 會提示用放大鏡排相似。</figcaption></figure>

<h2 class="sec">② 用:選一個框 → 放大鏡 → 全資料集按相似排</h2>
<ol class="steps">
<li>到 <b>Samples</b> 分頁,<b>勾選一個框</b>(在格狀檢視選一張並選取它的 label,或在展開的單張圖裡點一個 label)。</li>
<li>點工具列的 <b>放大鏡(Sort by similarity)</b>。</li>
<li>整個資料集會<b>按該物件的相似度重新排序</b> —— 最像的排最前面。很適合:同一種瑕疵一次撈齊、找你可能漏標的同類、找重複。</li>
</ol>
<figure><img src="img/sim_sorted.jpg" alt="按物件相似度排序後的結果"><figcaption>選一個 pothole 框、按放大鏡後,資料集依「這個坑洞長得多像」重排 —— 全離線、用 VIX 的 DINOv2 物件嵌入,免 Enterprise。</figcaption></figure>

<h2 class="sec">誠實邊界</h2>
<div class="warn"><b>需要嵌入:</b>這是 <span class="tag b">Tier B</span> 功能,要算 DINOv2 物件嵌入(第一次會下載一次權重)。Tier-A 的 <code>vix diagnose</code> 不算嵌入,所以用這個前要先建索引。<br><br>
<b>像素 fallback 不適合:</b><code>--adapter memory</code> 的像素嵌入只夠測試;真要找相似請用真 DINOv2(同 <a href="audit.html">標籤稽核</a>)。<br><br>
<b>這不是分類器:</b>相似排序是「看起來多像」的<b>排序工具</b>,不對誰是同一類下結論 —— 由你看圖判斷。</div>
<div class="check"><b>你現在能:</b>在 App 裡選一個瑕疵框、一鍵把「長得像它的」全撈到最前面 —— 用的是你自己的 DINO 嵌入,完全離線。</div>
"""

honesty_body = """
<div class="note">VIX 的身分就是「<b>誠實邊界</b>」:它寧可說「我不知道」也不假裝。這頁把全站的誠實規則集中一處 —— 都和程式碼裡的行為一致。</div>

<table>
<tr><th>原則</th><th>意思</th><th>為什麼</th></tr>
<tr><td><b>H1 匯入標籤 = 未覆核參照</b></td><td>你的標籤標記為 provisional,<b>永不</b>當 golden,不會進 calibrate/route/gate/export。</td><td>VIX 不把你尚未確認的標籤當真理。</td></tr>
<tr><td><b>H2 「誤報」可能是你漏標</b></td><td><code>type=background</code> 的誤報附雙因 hedge:可能模型幻覺,也可能你的 GT 漏框。</td><td>對未覆核的標籤,方向性的對錯不能單方面斷定。</td></tr>
<tr><td><b>H3 可比性需凍結 eval set</b></td><td><code>eval_set_hash</code> 含 GT;改了 eval 標籤就「不可比較」,Δ 不顯示。</td><td>否則分不清是模型變好還是考卷變簡單。</td></tr>
<tr><td><b>H4 排序是 PROXY</b></td><td>「該標這些」是優先順序,<b>非</b>實測 mAP 增益(VIX 不重訓)。</td><td>不誇大「標了一定變好」。</td></tr>
<tr><td><b>H5 不用受審標籤定罪自己</b></td><td>未覆核參照下,<code>label_noise</code> 被抑制成 <code>label_audit_needed</code>;歸因不會阻擋你的重訓。</td><td>避免循環論證。</td></tr>
<tr><td><b>H6 低支撐不畫箭頭</b></td><td>n_gt &lt; 20 的 Δ 標「n少不穩」,不給 ↑/↓。</td><td>小樣本的大幅變化多半是雜訊。</td></tr>
<tr><td><b>H7 像素 fallback 不適合稽核</b></td><td><code>--adapter memory</code> 的像素嵌入只適合測試/離線示範;稽核請用真 DINOv2。</td><td>嵌入品質不足會給出誤導結論。</td></tr>
</table>
<div class="tip">看到這些 hedge 不是 VIX 沒用,而是它<b>值得信任</b>的原因:每個它敢說的結論,都是它真的能證明的。</div>
"""

reference_body = """
<div class="note">這份新手文件是「<b>入口</b>」;需要每個細節時,看下面的深入資料。CLI 一律以 <code>vix --help</code> 為準。</div>

<h2 class="sec">完整操作手冊</h2>
<p><a href="../VIX_SOP.html"><b>VIX_SOP.html</b></a> —— 一頁涵蓋原生 FiftyOne 與 VIX 自訂功能的完整 SOP:概念、設定、happy path、CLI 工作流、App 內 <code>@vix/review</code> operators、稽核帳本、疑難排解。</p>

<h2 class="sec">CLI 指令</h2>
<pre><code>vix --help          # ~16 個常用指令(其餘 ~60 個進階指令已隱藏,但仍可直接執行)
vix &lt;verb&gt; --help    # 單一指令的選項
vix quickstart      # 5 分鐘上手文字
vix status          # 我現在在哪、下一步該下什麼</code></pre>
<div class="ref">本站只內聯了 on-ramp 的少數指令(<code>diagnose / import-labels / eval-run / eval-ingest / app</code>)。完整指令清單會隨版本變動,請以 <code>vix --help</code> 為準,不要背這份文件。</div>

<h2 class="sec">主題式走查(既有 HTML)</h2>
<div class="cards">
<a href="../GUI_WALKTHROUGH.html"><b>GUI 走查</b><span>App 內 10 步覆核工作流(真實資料)</span></a>
<a href="../EMBEDDINGS_HOWTO.html"><b>Embeddings 圖解</b><span>0–7 步框選群集、檢視影像</span></a>
<a href="../DINO_LABELAUDIT.html"><b>DINO 標籤稽核</b><span>嵌入離群點圖庫</span></a>
<a href="../../SETUP_OTHER_MACHINE.md"><b>跨機器安裝</b><span>Tier-2 設定 + 疑難排解</span></a>
</div>

<h2 class="sec">疑難排解(常見第一次失敗)</h2>
<table>
<tr><th>症狀</th><th>原因 / 解法</th></tr>
<tr><td>「找到 N 張影像但 0 個標籤配對成功」</td><td>資料夾結構不對。YOLO 要有 sibling <code>labels/</code>(或用 <code>--label-dir</code>);VOC 要有 <code>annotations/</code>;COCO 要 <code>--json</code>。</td></tr>
<tr><td>「類別索引 N 不在 names 對照」</td><td>標籤是數字類別卻沒給類名。加 <code>--names a,b,c</code> 或 <code>--data-yaml</code>。</td></tr>
<tr><td><code>eval-run 需要 ultralytics</code></td><td><code>pip install ultralytics</code>;或改走 <code>vix eval-ingest results.jsonl</code>(免 torch)。</td></tr>
<tr><td><code>vix app</code> 開不起來 / 空白</td><td>用 Python 3.11(<code>.venv311</code>);確認 <code>import fiftyone</code> 成功。</td></tr>
<tr><td>第一次 <code>--audit</code> 卡住 / 報網路錯</td><td>DINOv2 權重首次需下載一次;air-gapped 請先預放快取(見 SETUP 文件)。</td></tr>
<tr><td><code>vix export</code> 說「尚無 golden」</td><td>diagnose 匯入的是未覆核參照(非 golden)。你本就擁有標籤檔;要透過 VIX 匯出需先 <code>vix resolve &lt;hash&gt; --confirm</code>。</td></tr>
<tr><td>報告找不到</td><td>預設在目前目錄的 <code>vix_workspace/weakness_report.html</code>;或用 <code>--out</code> 指定。</td></tr>
</table>

<h2 class="sec">設計緣由(進階讀者)</h2>
<p><a href="../../discussion/landable-system.md">docs/discussion/landable-system.md</a> —— 多代理討論如何讓 VIX「真的落地」:on-ramp、誠實防火牆、凍結 eval 的閉環。其餘設計與規格見 <code>docs/spec/</code>。</p>
"""

PAGES = [
    ("index.html", "VIX:給第一次使用的 CV 工程師", "yolo val 給你 mAP;VIX 告訴你「該修什麼」—— 離線、一行指令、不用重訓。", index_body, ""),
    ("install.html", "安裝", "兩條路線:核心(Tier A,推薦先裝)與 Tier-2(App / DINOv2)。", install_body, cta("diagnose.html", "診斷你的模型", "裝好核心後,用一行指令對你的模型 + 資料產出弱點報告。")),
    ("diagnose.html", "診斷你的模型", "vix diagnose —— VIX 最重要、也是新手唯一必學的指令。", diagnose_body,
     cta("report.html", "讀懂弱點報告", "報告每個區塊與欄位是什麼意思,以及哪些數字要小心解讀。")),
    ("report.html", "讀懂弱點報告", "報告每個區塊與欄位的意思,以及哪些數字要小心解讀。", report_body, cta("loop.html", "修了有沒有幫助?", "學會用凍結的 eval set 誠實量測改善。")),
    ("formats.html", "輸入格式 (YOLO / VOC / COCO)", "資料夾擺對位置,VIX 才找得到你的標籤。", formats_body, cta("report.html", "讀懂弱點報告", "產出報告後,逐區塊讀懂它。")),
    ("loop.html", "修了有沒有幫助?", "誠實的閉環:固定考卷,才能比較分數。", loop_body, cta("audit.html", "稽核標籤本身", "想直接檢查標籤對不對?用 DINOv2 嵌入稽核。")),
    ("audit.html", "稽核標籤本身", "用 DINOv2 嵌入找疑似標錯,不需要模型。", audit_body, cta("app.html", "在 App 裡覆核", "把這些訊號丟進 FiftyOne App,用點選的方式覆核。")),
    ("app.html", "在 FiftyOne App 裡覆核", "把 VIX 的訊號丟進視覺化 App,用滑鼠點選覆核(真實操作截圖)。", app_body, cta("similarity.html", "找相似的物件", "選一個瑕疵框,一鍵把長得像的全撈出來(DINO,離線,免 Enterprise)。")),
    ("similarity.html", "找相似的物件 (DINO)", "選一個框 → 放大鏡 → 全資料集按物件相似度重排。離線、免 FiftyOne Enterprise。", similarity_body, cta("honesty.html", "誠實邊界與限制", "用之前,先了解 VIX 對你誠實的每一條規則。")),
    ("honesty.html", "誠實邊界與限制", "VIX 的身分:寧可說「我不知道」也不假裝。", honesty_body, cta("reference.html", "參考與完整手冊", "需要每個細節?看完整 SOP 與 CLI。")),
    ("reference.html", "參考與完整手冊", "需要更深入時看這裡;CLI 一律以 vix --help 為準。", reference_body, ""),
]


def main():
    SITE.mkdir(parents=True, exist_ok=True)
    for fname, title, sub, body, c in PAGES:
        page(fname, title, sub, body, c)
    print("\nsite ->", SITE)


if __name__ == "__main__":
    main()
