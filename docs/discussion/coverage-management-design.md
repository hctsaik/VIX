# DINO 嵌入式「資料覆蓋管理」設計討論

> /goal:起 multi-agent 討論需求(≥3 輪,**先不用串流服務**),設計一個用高維 DINOv2 嵌入來「管理資料集」的能力——告訴 CV 工程師哪些資料**太少(該加收)**、哪些**夠了**、哪些**太多(要不要砍?)**;新進樣本若**填補空缺**就提示;UMAP 只做視覺化(含 `transform()` 求穩定)。
>
> 目標系統 = 現有 VIX(離線、零訓練、純 `core/` + `DatasetAdapter` 接縫、誠實邊界/PROXY)。
>
> 方法:每輪 3 個獨立代理從不同 lens grounded 讀碼;Round 1 發散、Round 2 辯論四個歧見、Round 3 紅隊壓力測試 + 定稿。本文是三輪的紀錄與**可建造的 v1 spec**。

---

## 實作狀態(2026-06-08:v1 已落地)

v1 spec 已實作 + 測試全綠(16 個新測試,308 純測試零回歸):
- **core**(`src/vix/core/analytics.py`):`coverage_regions()`、`coverage_gapfill()`、`_partition_regions`、`_region_representative`、`_MIN_SUPPORT=20`。
- **pipeline**(`src/vix/pipeline.py`):`coverage_map`(fail-closed + golden→provisional fallback + 時戳快照 + before/after Δ + encoder_fp 戳記 + 條件式 log)、`gap_fill`、`prune_candidates`/`prune`(四護欄 + 兩段式 + 可 restore)。
- **CLI**(`src/vix/cli.py`):`vix coverage-map` / `vix gap-fill` / `vix prune`。
- **測試**:`tests/test_coverage_regions.py`(core)、`tests/test_coverage_manager.py`(pipeline)。
- **一處合理偏離 spec**:`coverage_map` 改成 fail-closed 在「**無參照**(無 golden 也無 provisional)」,而非 spec 寫的「先過 `_coverage_verdict`(需 thresholds.json)」——因為覆蓋地圖是純幾何、不需校準,擋在「無校準」會誤傷只想看分布的使用者;encoder 漂移改由快照的 `encoder_fp` + before/after 的 `encoder_changed` 旗標誠實揭露(紅隊 #5)。
- **未做(依 spec 屬 v2/optional)**:App operator、HDBSCAN/intrinsic-dim、真 eval-AP 加權、營運重要度權重。串流仍 = v3 停車場。

---

## TL;DR(先看這段)

- **決策一律在高維 cosine 空間;UMAP 只給人看**(fit 一次 → freeze → `transform()` 投新點,週週可比)。2D 的「洞」可能是投影撕裂的假象,**永遠不拿 2D 下判斷**。t-SNE 無 out-of-sample transform,所以接 DINO 用 UMAP。
- **「稀疏 = 該收」是錯的**。純密度會送你去收「簡單但稀疏」的垃圾角落。稀疏必須被**模型弱點**加權——而 VIX 有**免 eval 的弱點代理**:`1 − confidence`(`active_learning_ranking`/`triage` 已在算)+ `OutlierScorer` 的 `low_support`/`knn_dist`。標成 PROXY,**絕不宣稱「實測 mAP 增益」**。
- **複用既有積木,真正新的極少**:`coverage_gaps`、`coverage_delta`(= 現成 `vix value`,就是 gap-fill 旗標)、`near_duplicate_groups`、`active_learning_ranking`、`weakness_report`。
- **物件層級**:偵測資料集要用 **per-detection crop embedding**(整張圖的 DINO 距離被背景主導)。
- **絕不自動刪除**:OVER 只出「候選冗餘 + 保留代表 + 門檻敏感度」清單,人來 `--confirm`,可 `restore`,上稽核鏈(複用 `harmful → harmful --remove` 兩段式)。
- **🔴 Round 3 紅隊抓到的 v1 阻斷點**:把 `near_duplicate_groups`(單鏈 union-find)在鬆閾值跑在「類別內密集資料」上會**傳遞鏈化**把整類併成一坨 → 必須改名為「密度群(density groups)」、印出群數/size 分布、**當最大群佔比過高就拒絕出 per-region verdict**(誠實 withhold,不引入 sklearn/HDBSCAN)。

---

## Round 1 — 三個 lens 的發散(grounded 讀碼)

三個獨立代理:**①ML 演算法嚴謹度**、**②VIX 整合 + 誠實邊界**、**③CV 工程師工作流/產品**。

### 三 lens 高度一致的共識
1. **高維決策 / UMAP 僅視覺化**。`compute_visualization`(`fiftyone_adapter.py:281`)目前每次重 fit;做覆蓋**監測**要改成 fit 一次後持久化、之後只 `transform()`,否則每週的圖不可比。scorer docstring 已立規:「距離一律 cosine,在原 DINOv2 空間算——絕不在 UMAP 上」。
2. **稀疏 ≠ 需要**。偵測要的是「模型會錯 / 罕見但重要」的地方,不是均勻覆蓋。均勻覆蓋會把預算燒在已解決的簡單稀疏角落。→ 收集要 **task-guided**(綁模型弱點),不是純密度。
3. **絕不自動刪除**;繼承 VIX 的 `_PROXY` 戳記、`Tag.PROVISIONAL` 防火牆、encoder-fingerprint 綁定。
4. **複用既有積木**——三 lens 各自獨立列出同一組:`coverage_gaps`(have/need + 沒被用到的 `sparse_ids`)、`coverage_delta`(`novel_fraction` = 填補空缺,已以 `vix value` 出貨)、`near_duplicate_groups`(冗餘)、`active_learning_ranking`(novelty+uncertainty)、`weakness_report`(per-class AP)。
5. **物件層級**:用 per-detection crop embedding;整張圖的 DINO 被背景主導(同物件+新背景會被誤判成「填補空缺」)。

### 各 lens 提出的「最小第一個交付物」(此處分歧 → 進 Round 2)
- **①ML**:先出**純幾何**的 per-class 稀疏排名,誠實地**不假裝**是收集計畫,等接上 eval 才加弱點加權。
- **②VIX**:`coverage_regions()` 新 core fn + `vix coverage-map` 一個 verb,複用既有 guard。
- **③CV**:`vix coverage --weak` 表(have/need join 模型 AP)——並嗆「純幾何表是個玩具,跟我的類別直方圖沒兩樣」。

### 四個真歧見(Round 2 要解)
| # | 歧見 |
|---|------|
| **T1** | 第一個交付物該是**純幾何**(ML)還是**弱點加權**(CV)? |
| **T2** | v1 需要**類別內分群**嗎?(ML:per-class only;CV/US2:「car 但缺夜間」必須有子群) |
| **T3** | **eval 缺席**時怎麼辦?有沒有不需 eval 的弱點代理? |
| **T4** | curse-of-dimensionality 防線 + intrinsic-dim 報告的力道? |

---

## Round 2 — 辯論四個歧見(每個代理逼出 DECISION)

### T1 — 第一個交付物:**(b) 幾何 + 免 eval 不確定度代理**(三方收斂)
- **關鍵發現(grounded)**:`active_learning_ranking`(`analytics.py:223`)**已經**在算 `uncertainty = 1 − confidence` 並與 novelty 混合;`triage.review_queue`(`triage.py:46`)也是 `unc = 1.0 − c.confidence`;`EmbItem.confidence` 來自**真實預測**(`pipeline.py:438`)。所以「免 eval 的弱點代理」機制**早就存在且接好了**。
- **ML 立場(修正)**:純幾何確實是直方圖雙胞胎,但**不要把 `scarcity × (1−conf)` 乘成單一 priority**(那是假裝量測)。改成**兩個並排的欄位**:`geometry`(稀疏)與 `model_stress`(平均 `1−conf`/margin),verdict 由 geometry 決定,`model_stress` 是同級的排序提示。
- **CV 立場**:要一個可排序的清單,否則超過 5 個類別就無法用 → 想要 `collect_priority = scarcity × (1−conf) × operational_importance`。
- **VIX-arbiter**:eval-free 弱點來自 `1−conf` + `low_support`,標 PROXY,**非** eval-set 弱點;真 eval-AP 加權是 `cfg.eval_results_path` 存在時的 opt-in 升級(`pipeline.py:896`),不是 v1 必需。
- **殘留分歧**:乘 vs 不乘(T1 的核心)→ 留給 Round 3 裁決。

### T2 — 類別內分群:**v1 = per-class + 類別內 region,但 region 用既有 `near_duplicate_groups` 鬆閾值**
- **VIX-arbiter(grounded 關鍵)**:core 裡**唯一**的分群積木就是 `near_duplicate_groups`(union-find,`analytics.py:105`);**沒有** k-means/HDBSCAN/silhouette。`suspected_new_classes` 已示範用**鬆閾值**(`cluster_distance=0.2` vs dedup 的 `0.05`)切子群(`analytics.py:308,326`)。**sklearn 不是 core 相依**(`pyproject.toml` core 只有 numpy/PyYAML/pillow;scikit-learn 只在 `requirements-tier2.txt`)。→ 加 HDBSCAN/k-means 會破壞「純 numpy、免 FiftyOne、可單機單測」契約。
- **此選擇順帶化解 ML 的擔憂**:ML 怕的是 k-means/HDBSCAN 的 k 選擇 + 隨機初始化不穩定;而**鬆閾值 union-find 是確定性的**(無 k、無隨機),反而比較穩。
- **CV 讓步**:不會去追「cluster 7」這種匿名群;v1 只把 region 當 `sparse_ids` + **範例 crop** 端出來,**由人看圖命名**,命名後才變成收集目標。「工具提議 region,人定義 gap」。

### T3 — 免 eval 代理品質 + 營運先驗
- **三方同意**:出 `1−conf`(+ `low_support`)代理,但**明確標「不確定度代理,非實測弱點」**(偵測器常過度自信,「高信心卻錯」抓不到)。margin 比 raw `1−conf` 略好(對絕對 miscalibration 較不敏感)。
- **CV 加碼(被採納)**:加一個**使用者自填的 per-class 營運重要度權重**(預設 1.0 = 均勻),持久化、每次沿用 → 收集清單反映「我的營運優先序」而非嵌入空間的幾何。對「夜間雨」這種尾巴:`night-rain 幾何稀疏且模型弱,但你把營運重要度設 0.2 → 排在 daytime-occluded 之下」。
- **誠實字串(zh-Hant,沿用 house 風格)**:`model_stress = 模型對該類平均不確定度(1−信心),非實測弱點;偵測器常過度自信,高信心卻錯不會被抓到 → 僅供同模型同資料內相對排序,不可當「標了會提升 mAP」。要實測弱點請接 vix eval-ingest(需凍結 val set)。`

### T4 — curse-of-dimensionality 防線
- **MUST-HAVE(v1)**:
  1. **一律 percentile-within-class**(`coverage_gaps` 已是這樣做);跨類別**絕不**比 raw cosine 距離。這是讓高維密度數字有意義的唯一防線。
  2. **support floor → 低於就 REFUSE verdict**。沿用既有 `_MIN_SUPPORT = 20`(`weakness_report.py:24`,gate/regression_check 同此值),低於則出「n少,不予判定」(沿用 `_delta_cell` 的 no-arrow pattern),count + proxy 仍可顯示但 **verdict 格留空**。(ML 主張 30 較穩,但讓步接受 20 以對齊既有常數。)
- **intrinsic-dimension 報告**:**v2**,nice-to-have(它是「信任刻度」,但當 v1 gate 太重且自帶估計器失敗模式)。
- **CV 的信任三要素(act 任何數字前必須出現)**:① **範例 crop / 最近鄰影像**(壓倒一切——一眼判斷是真 gap 還是垃圾群)② **PROXY 戳記 + 哪個軸驅動 + eval-free vs 真 AP** ③ **門檻敏感度計數**。其餘(encoder fingerprint 細節、NN 距離直方圖、跨run 穩定度)= v2。

---

## Round 3 — 紅隊壓力測試 + 定稿

兩個並行代理:**紅隊(假設已上線,找會燒到工程師的破口)** 與 **scope/mechanics(把共識變成可建造 spec + prune/gap-fill 機制 + roadmap)**。

### 🔴 紅隊的 5 大失敗模式(依「燒人程度」排序)

**#1(v1 阻斷點)單鏈 union-find 鏈化把 region 融成一坨。**
`near_duplicate_groups`(`analytics.py:105-148`)是教科書單鏈 union-find:`if dist[i,j] < max_distance: union(i,j)`,**無鏈長/直徑/size 上限**。在 tight `0.05` 安全(半徑小),但設計要在 **`~0.2`** 跑「類別內」——DINO 同類 crop 是**連續流形**(白天→黃昏→夜晚是平滑路徑,不是斷點),單鏈 0.2 極可能把整類併成**一個巨群**。`suspected_new_classes` 用 0.2 沒事是因為它跑在**已過濾的稀疏新尾巴**上(鏈化天然受限),設計把這個閾值搬到**密集類別內**,讓它安全的前提就沒了。
→ **燒點**:報「這類 count=N、ENOUGH/OVER」,工程師以為「這類夠了」,而夜間子群隱形在巨群裡——正是這功能要防的失敗。且一個橋接樣本就能在兩次 run 間合併兩個真群(region 數對插入不穩)。
→ **最便宜誠實修法(不加 dep)**:① **改名「密度群 density groups」**並揭露 linkage;② **印 region 數 + size 分布 + 鏈化警報**:最大群 > 類別 ~60% 就**拒絕 per-region verdict**,標「此類嵌入連成一片,無法切出可信子區(single-linkage 串接);只給整類 scarcity」——把沉默的謊言變成誠實 withhold(複用 `_MIN_SUPPORT`/`assess_coverage` pattern);③(可選)union 時加便宜的 centroid-diameter reject(仍純 numpy)。

**#2(融合形式才是阻斷點)`scarcity × (1−conf)` 乘成單一數字 = 假裝量測。**
裁決:**ML 贏數學、CV 贏人因,兩者可兼得**。`active_learning_ranking` 已回傳 `uncertainty/novelty/score` **各自獨立欄位**(`return_reasons`)——這就是對的先例。→ **出一個可排序清單,但是「公式透明的排序提示」不是融合度量**:各分量各自成欄;預設排序把**公式內聯印出**並標「排序提示,非實測增益」(沿用 `_closeness/_wrongness` legend 的「僅排序用,非機率」)。**絕不**把 composite 叫做 gain/priority「score」而不附公式。

**#3(可接受+硬護欄)crop 仍漏背景。**
`coverage_delta`(`analytics.py:205`)flag novel = `nearest_existing_distance > radius`。crop 仍含背景像素、DINO 對場景敏感 → 同物件+新背景可能 > radius 被誤標「填補空缺」;反之新姿態+熟悉背景被靜默丟棄。量級是 **false-novelty rate**(膨脹收集清單、浪費標註預算),非災難性錯標。→ **硬護欄**:每個 gap-fill flag **必附最近鄰 crop 的 id + 距離並顯示該影像**,人眼否決「這是新物件還是只是新天空」,絕不讓 `novel_fraction` 在無人否決下自動驅動「去收集」。

**#4(僅在四護欄齊備時可接受)OVER=prune candidate 即使 advisory 也會塌尾。**
最糟結果:工程師信任冗餘清單批次 reject。因 #1,OVER 群正是鏈化巨群,清單會把夜間樣本和白天主體一起標「冗餘」→ **塌尾**。兩個 code-confirmed 漏洞:(a) 保留代表落在 test/val 而砍掉 train 近重複 = **test 洩漏**(`cross_split_leakage` `analytics.py:334` 正為此存在,prune 清單必須過它);(b) 砍到 `_MIN_SUPPORT=20` 以下會毀掉該類校準能力。→ **四硬護欄**:① 絕不砍到類別低於 `_MIN_SUPPORT`;② protected/rare 類別永不可砍(鏡射 `regression_check` 的 `protected` fail-closed);③ OVER 群先過 `cross_split_leakage`,跨 split 就當洩漏而非「冗餘可刪」;④ 永遠顯示保留代表 id + 人工 `--confirm`(worklist 寫「建議人工複核」非「redundant」)。

**#5(可接受+收緊接線)encoder drift 靜默翻轉所有 verdict。**
每個距離都是 encoder-relative;CPU↔GPU 或重抓 torch.hub 權重 → region 重排、scarcity 翻轉、無警告。好消息:fingerprint guard 是**行為性**的(`probe_digest` `encoder_fingerprint.py:39`,固定探針→L2→round 3 位→hash),真改會 trip、no-op 點版本不會;`assess_coverage`(`calibration_coverage.py:56`)已把 `fp_mismatch` 變成大聲拒絕。**兩個仍會燒的縫**:(a) **任一邊缺 fp 時 fail-OPEN** → 新 coverage 產物**必須把自己的 `encoder_fp` 寫進輸出**(像 `thresholds.meta`)否則 guard 對它永不觸發;(b) `_PROBE_DECIMALS=3` rounding 之下的次門檻噪音仍會擾動 0.2 單鏈邊界(單鏈對門檻附近的微小距離變化最敏感)。→ **修法是文件 + 一個 provenance 欄位**:持久化並檢查自己的 `encoder_fp`(關掉 fail-open)+ 明說「region 身分跨 re-embed 不穩定,別把 region 數當量測逐 run diff」。

### 🔴 紅隊最終裁決
**目前的共識「尚不可原樣上線」——#1 強迫小幅重設計核心積木**(在出「region」前必須改名密度群 + 印群數/size 分布 + 最大群佔比過高就 withhold verdict)。**#2 僅融合形式是阻斷**(分量揭露的排序提示安全)。**#3/#4/#5 可上線**但各帶不可妥協的護欄。**修掉 #1 的 region 誠實問題 + #2 去融合後,其餘是 caveat-and-guard,設計即可建造。**

---

## ✅ 定稿:可建造的 v1 Spec

> 範圍:**離線、批次、人工確認**的覆蓋管理迴圈。**串流/線上 = v3 延後(使用者已明確 postpone)**。

### A. 新增 core 函式(`core/analytics.py`,純 numpy、可用 InMemoryAdapter 單測)

```python
_MIN_SUPPORT = 20  # 沿用 weakness_report._MIN_SUPPORT(weakness_report.py:24)

def coverage_regions(
    items: list[EmbItem],
    region_distance: float = 0.25,   # 鬆 near_duplicate_groups 閾值 → 類別內密度群
    min_support: int = _MIN_SUPPORT, # SCARCE/ENOUGH 樓地板
    over_factor: float = 3.0,        # count >= over_factor*min_support → OVER
    chain_frac: float = 0.6,         # 🔴#1:最大群佔比 > 此值 → withhold per-region verdict
    k: int = 5,
) -> dict[str, dict]:
    """每類別切成類別內「密度群」(density groups,非語意 cluster;單鏈揭露)。
    每群:{region_id, ids, count, status, weakness_proxy, representative_id}。
    status:'SCARCE'(count<min_support)|'OVER'(count>=over_factor*min_support)|'ENOUGH'。
    若最大群 ids 佔該類 > chain_frac → 該類 regions 標 'CHAINED',只回整類 scarcity,
      per-region verdict 留空(誠實 withhold,複用 _MIN_SUPPORT pattern)。
    weakness_proxy:PROXY 標記的群內 mean(1-confidence)(+ low_support 旗標)。
    回 {class: {count, n_regions, max_region_frac, regions:[...], scarce_regions, over_regions}}。
    誠實:region = 目前 embedding 空間的 cosine 密度群;weakness 是 PROXY 非實測 AP;
      決策在高維 cosine,UMAP 僅視覺化;低於 min_support 不予判定。
    """

def coverage_gapfill(
    new_items: list[EmbItem],
    existing_items: list[EmbItem],
    radius: float = 0.2,        # 同 coverage_delta 預設(analytics.py:205)
    dup_distance: float = 0.05, # 同 near_duplicate_groups 預設(analytics.py:105)
) -> list[dict]:
    """每個新樣本:fills_gap / redundant / duplicate。
    複用 coverage_delta(novel_ids) + 最近鄰查詢:
      novel(nearest>radius) → fills_gap;else nearest<dup_distance → duplicate;else redundant。
    每列 {id, verdict, nearest_id, nearest_distance, novelty};nearest_id = 要並排顯示的既有影像。
    """
```

`coverage_regions` 複用 `near_duplicate_groups`(`analytics.py:105`)分群、`coverage_gaps` 的 `dens`/sparse-tail(`:185-192`)。`coverage_gapfill` 複用 `coverage_delta` 的 `novel_ids`(`:210-220`,即現成 `vix value`),唯一新東西 = 每樣本的 `nearest_id/nearest_distance`(要顯示的鄰居影像)。

### B. 新增 pipeline stage(`pipeline.py`,鏡射 `coverage` `:488` / `coverage_value` `:497`)

```python
def coverage_map(adapter, cfg, region_distance=0.25):  # S5b;crop 級
    items = _detection_items(adapter, want_tags=[Tag.GOLDEN])
    verdict = _coverage_verdict(adapter, cfg)          # 🔴 先過 fail-closed guard(pipeline.py:633)
    if not verdict.ok: return {"ok": False, "reason": verdict.reason}  # _NO_GOLDEN_REASON
    regions = coverage_regions(items, region_distance)
    _snapshot_coverage(regions, encoder_fp=...)        # 🔴#5 寫自己的 encoder_fp + 時戳快照
    # DecisionLog.append 只在「emit 可行動 verdict」時(複用 new_classes 條件式 append,:792)
    return regions

def gap_fill(adapter, cfg, radius=0.2):  # S4b;鏡射 coverage_value(:497)
    new = _image_items(adapter, exclude_tags=[Tag.GOLDEN, Tag.ANCHOR, Tag.EVAL])
    existing = _image_items(adapter, want_tags=[Tag.GOLDEN])
    return coverage_gapfill(new, existing, radius)
```

- **logging 規則(grounded)**:純查詢**不記帳**(`coverage`/`coverage_value`/`dedup` 都不 append,`:481-503`);只有 emit「可行動 verdict / ALERT」才 `DecisionLog.append`(鏡射 `new_classes` `:792`、`drift_periods` `:529`)。
- **fail-closed**:讀 golden,所以**必須先過 `_coverage_verdict` → `assess_coverage`**(`:633-644`),無 golden 出 `_NO_GOLDEN_REASON`(同 `review_queue` `:663-673`)。

### C. CLI verb(`cli.py`,兩層 help)
```
vix coverage-map [--region-distance 0.25]   # core verb(進 _CORE_VERBS,:414)
vix gap-fill     [--radius 0.2]             # 長尾(註冊但隱藏,如 value/harmful)
vix prune        [--class C] [--confirm]    # 兩段式;預設 read-only worklist
```
`coverage-map` 印出範例(鏡射 `new-classes` print loop `cli.py:879`):
```
person   (n=412, 3 群)
  R0  n=287  ENOUGH  proxy 0.18
  R1  n=24   ENOUGH  proxy 0.41
  R2  n=8    SCARCE  proxy 0.63   ← 補這區優先(vix active-learn / error-mine)
forklift (n=31, 2 群)
  R1  n=3    SCARCE  proxy 0.55   ← 補這區優先
helmet   (n=190, 1 群, CHAINED)  ⚠ 嵌入連成一片,只給整類 scarcity(OVER, proxy 0.09)
注:region=目前 embedding 空間的密度群(single-linkage);proxy=平均(1-信心),僅排序非實測 AP(PROXY)。
```

### D. GAP-FILL 機制(步驟)
1. 新樣本取 mean detection embedding(`_image_items`,鏡射 `coverage_value` 的輸入),existing = golden `_image_items`。
2. `coverage_delta(new, existing, radius=0.2)` 的 `novel_ids` → **fills_gap**(= 現成 `vix value` 的計算,100% 複用)。
3. 非 novel 者算最近鄰距離:`< 0.05` → **duplicate**;否則 **redundant**。
4. **每個 verdict 並排顯示 `nearest_id` 的影像 + 距離**(🔴#3 人眼否決)。
5. 門檻 `radius=0.2`/`dup_distance=0.05` 皆既有預設,無新調參面。
6. **新 vs 複用**:`coverage_delta` 全複用;NEW = per-sample 最近鄰 id/距離 + 三分類。`active_learning_ranking` **不**用於 gap-fill(它答「給定預算挑 N 個標」是另一問題)。

### E. PRUNE 機制(步驟)— v1 **絕不**自動刪
複用既有兩段式:`harmful`(read-only,`:806`)→ `harmful --remove`(`cli.py:891`)→ `harmful_remove`(tag `Tag.REJECTED` + audit,`:2486`)→ 可 `restore-dismissed` 還原(`:2061`)。
1. **`vix prune`(預設 read-only)**:每個 OVER 群出 `{redundant_id, kept_representative_id, region, class, threshold_sensitivity}`。代表 = 群 medoid(最近群均值,鏡射 `cross_period_drift` representative `:294`);`threshold_sensitivity` = `region_distance ± δ` 重跑 `near_duplicate_groups` 後仍被標的數(看哪些 removal 穩健 vs 邊緣)。
2. **四硬護欄(emit 前強制,v1 不可 override)**:① 不砍到類別 < `_MIN_SUPPORT=20`;② protected 類別跳過(`set_eval_baseline` 寫 `protected` `:1486`,`pre_train_gate` 讀 `:916`);③ 先過 `cross_split_leakage`(`analytics.py:334`)不砍跨 split 的 train 端 keeper;④ 永遠顯示保留代表。
3. **`vix prune --confirm`**:tag `Tag.REJECTED` + append `prune` 稽核事件(`{kept, removed_ids, region}`),複用 `dismiss` pattern(`:2052`),可 `restore`。**無 `--confirm` 絕不刪任何一列**。

### F. Before/After 迴圈(US7:「我照建議收了那批,缺口補上了嗎?」)
複用 `health_report` 的 auto-diff-vs-last(`_latest_prior_report` `:576` → `prev=` `:617` → 寫時戳副本 `:622`)。
1. `coverage_map` 寫時戳快照 `workspace/coverage/coverage_<stamp>.json`(per-class/region 的 count+status+proxy)。
2. 下次 run `_latest_prior_coverage`(3 行 clone)載入前快照。
3. 輸出加 **每 region Δ 欄**:count then→now + 狀態轉移;**低於 `_MIN_SUPPORT` 不畫箭頭**(沿用 `_delta_cell` `:27`,+5 on n=8 不算「補上」)。
```
person  R2  SCARCE→ENOUGH  n 8 → 31  (+23)   ✓ 缺口已補
forklift R1 SCARCE          n 3 → 9   (+6, n少不穩)   仍缺 (< 20)
```
這是純 eval-free PROXY 迴圈(region count + 1-conf),明確**有別於** `weakness_report` 的 AP-Δ 迴圈(那需凍結 eval set)。v1 誠實說「你被告知補的 SCARCE 區現在多了 N / 跨過樓地板」,**不**說「AP 提升了」。

### G. 誠實字串(沿用既有,**不發明新詞**)
- PROXY 戳記:`weakness_report._PROXY`(`:17`)`_(PROXY:未重訓,此為嫌疑/優先排序,非實測 mAP 增益)_`
- 低支持 no-arrow:`_MIN_SUPPORT=20` + `_delta_cell` 的「n少不穩」(`:24,27`)
- 排序非機率:`_closeness/_wrongness` legend「(0–1,僅排序用,非機率)」(`:43`)
- 無 golden / backend 不符:`assess_coverage` reasons(`calibration_coverage.py:48`)+ `_NO_GOLDEN_REASON`(`pipeline.py:648`)
- PROVISIONAL 防火牆:跑在 diagnose 匯入標籤上時,`_UNVERIFIED_REF`(`weakness_report.py:20`)「你匯入的標籤(未經 VIX 覆核)」

---

## 三層 Roadmap

| 面向 | **v1(最小誠實、現可建)** | **v2(更豐富、仍離線)** | **v3(延後 — 串流,使用者已 postpone)** |
|---|---|---|---|
| Region | `near_duplicate_groups` 鬆閾值 per-class;鏈化 withhold | HDBSCAN(tier-2 opt-in)+ intrinsic-dim/群 + bootstrap 穩定度(Jaccard≥0.5) | — |
| 弱點 | `1−conf` + `low_support` PROXY | + 真 eval-AP 加權(join `eval_results.json` `:1433`)+ 營運重要度權重 | — |
| Status 樓地板 | `_MIN_SUPPORT=20` 平樓地板 | 隨 intrinsic-dim/營運重要度縮放 | — |
| Gap-fill | `coverage_delta` 於 sample-mean,radius 0.2,三分類 + 最近鄰影像 | crop 級 verdict(`_image_items`→`_detection_items`)+ 落在哪個 region | **線上/串流觸發 — OUT** |
| Prune | OVER → read-only worklist;`--confirm` → REJECTED + audit;四護欄 | 敏感度帶、營運重要度感知 keep-set、依覆蓋擴散挑代表 | 串流上自動 prune — **OUT(即使 v3 也絕不自動刪)** |
| Before/After | per-region count Δ + 狀態轉移快照;低於樓地板不畫箭頭 | + 有凍結 eval 時的 per-region AP-Δ | 連續 drift 觸發 re-eval — **OUT** |
| Viz | UMAP fit-once + `transform()`,僅視覺化;決策全高維 cosine | 同(UMAP 永不取得決策權) | — |

**明確排除 v1+v2**:任何串流/線上/連續觸發的 gap-fill。整個覆蓋管理是批次、離線、人工確認迴圈。v3 是串流觸發的停車場,而且**即使 v3,prune 永不自動刪**(`--confirm` 防火牆永久)。

---

## 給使用者問題的直接回答

- **「我想知道哪些資料少該加收、哪裡夠、哪裡太多(要不要砍?)」** → `vix coverage-map`:per-class + 類別內密度群,每群 SCARCE/ENOUGH/OVER,SCARCE 標「補這區優先」、OVER 出**冗餘候選清單**(但**絕不自動刪**,人 `--confirm`)。
- **「ML 判斷這筆有沒有填補空缺 → DINO 嵌入 → kNN 距離/局部密度 → 稀疏區=填補空缺→觸發」** → `vix gap-fill`:即現成 `coverage_delta`(`vix value`)的三分類版,落在 golden 覆蓋不到的區 = fills_gap,**但必附最近鄰影像給你否決**(防 crop 背景漏判)。
- **「想用高維 DINO + UMAP」** → 決策全在高維 DINO cosine;UMAP **fit 一次 + `transform()`** 只做穩定視覺化,2D 永不下判斷。
- **「太多一般會再砍掉嗎?」** → 會,但只砍**近重複 + 過密的簡單區**,且四護欄(不砍到 < 20、不砍 protected/罕見、過 `cross_split_leakage`、顯示保留代表)+ 人工兩段確認 + 可還原 + 上稽核鏈。**尾巴/罕見/難例永不砍**——那正是最有價值的資料。
- **「能更好管理資料嗎?」** → 能,但誠實邊界:這是**幾何 + 不確定度代理**(PROXY),不是實測 mAP;要實測弱點請接 `vix eval-ingest`(需凍結 val)。

---

*本文為設計討論記錄(read-only 分析,未改任何程式碼)。三輪 multi-agent,grounded 讀碼。串流服務依使用者指示排除,列為 v3 停車場。*
