# GT × 嵌入:一致性歸因(多代理結論 + 已實作 v1)

> 問題:除了 DINO embedding,若還有 labeling GT,能幫這系統什麼?對資料一致性如何設計?是否有價值?如何放大?
> 經三輪多代理辯論(實戰工程師 / data-centric 研究者 / 產品懷疑者 / 架構師 → 收斂)。

## 結論:**確實有價值(HIGH,邊界清楚)→ 走「放大」路線**
連最硬的懷疑者都從 MEDIUM 升到 HIGH(他原本的反對逐一被拆:不是獨立 pillar 而是整合進既有 weakness-report、不是 commodity 因為 separability 沒人做、不受「不重訓無法證明」限制因為 separability 是可證偽的幾何陳述)。

**核心洞見**:VIX 既有的標籤訊號都是**自我參照**(只能說「和鄰居不一樣」),所以在「兩類重疊處」結構性失明 —— 正是 SAFE pothole/crack 0.27 的死因。**GT 打破循環**,把「偵測」升級成「**歸因**」:這個失敗是 **taxonomy / model / label** 問題?三者在 AP 數字上長一樣,但正確處置完全相反(停止多標 / 沿邊界補資料 / 重新裁決),今天工程師只能用直覺選、錯約 ⅓ —— 這就是 painkiller。

額外:類別定義爛掉會污染**每個** embedding 驅動訊號(drift/coverage/error-mine/active-learn/bank-audit),所以一致性歸因是**驗證地基**的最高槓桿,而非 sidecar。

## 已實作 v1(整合進 weakness-report,非獨立 pillar;諮詢式 + CI + 支撐閘)
| 能力 | 做法 | 檔案 |
|---|---|---|
| **可分性 separability** | 逐類對 LOO-kNN 誤差(cosine, k=max(3,⌊√n_min⌋) 取奇);高=「**在目前 embedding 空間**不可分」(綁定編碼器,非「taxonomy 壞了」)+ Wilson CI | `core/consistency.py` |
| **重疊 × 混淆 2×2** | O[i→j](golden GT 點的 k-NN 落在 j 的比例,bootstrap CI)對 model 混淆 C[i→j](eval,Wilson CI);Δ=O−C 的保守 CI → **taxonomy / model / label_noise / clean** | `core/consistency.py` |
| **支撐階梯** | n_min<10 或 pair<25 → `insufficient_support`(不給判決);10–20 → `provisional`(禁建議 merge);≥20 → `supported` | `core/consistency.py` |
| **整合 + 介面** | findings 進 `weakness_report`(.md + **.html**);`vix consistency` 直接看;審計記判定 | `pipeline.consistency` / `weakness_report`、CLI |

**誠實契約**(對應研究者的小樣本疑慮 + 懷疑者的 encoder 疑慮):永遠附 support + CI;`0∈CI(Δ)` 不給方向性歸因;**merge 建議需 supported + taxonomy + CI 半寬足夠窄**;separability 措辭一律綁「目前 embedding 空間」;**絕不自動改標 / 自動 merge**。資料來源 = 人工確認的 **golden** 嵌入(=GT 標籤的真實例)+ eval 混淆矩陣;不需多標註者(κ 路線需 `Detection.source` schema 變更 + 多標註者資料,**延後**)。

## 測試
`tests/test_consistency.py`(估計子 + taxonomy/model/label_noise/insufficient 判定 + pipeline + HTML 寫出)、
`tests/test_weakness_report_gui.py`(**Playwright** 開 headless Chromium 載入真實 `weakness_report.html`,斷言一致性表格 + `taxonomy` 判定 + 樣式化判定格 + PROXY 標記在瀏覽器中渲染)。全套綠。

## 放大價值的下一步(優先序)
1. **佇列命中率回饋飛輪** —— ✅ **已實作 v1**:`vix queue-hit-rate` 把過去的建議佇列(error-mine/hardneg/weakness/bank-hardpos)join 後來的人工裁決 → 逐佇列**命中率 + 覆蓋 + 趨勢**,併進 weakness-report。誠實:只算「發出後才被裁決」且「已解決」的 id;命中定義依佇列預測(wrong→reject 算中、defect→confirm 算中、label→被採納)。檔案 `core/queue_metrics.py`、`pipeline.{_log_queue,queue_hit_rate}`、CLI;測試 `tests/test_queue_hit_rate.py`。**待:把命中率實際回授去重排下一輪佇列(低命中降權)** —— 目前是測量+趨勢+顯示。
   - 套用投影到**全 stack** 的另一半(routing/threshold/bank-audit 在投影空間重校準)仍待(見 #2 邊界)。
2. **領域自適應 embedding**(最大槓桿)—— ✅ **已實作 v1**:用 golden GT 在凍結 DINO 上學正規化 LDA 投影(PCA 預降維 + shrinkage,閉式、秒級、$0、**非訓練 YOLO**)。`vix adapt-embedding` 逐對報告**凍結→投影**可分性(**k-fold CV**,投影只在 train fold fit,防過擬合),標記 **rescued**(凍結不可分 → 投影後可分 = 表徵問題、可修,非 taxonomy 死路);`--save` 持久化 `embed_projection.npz`。檔案 `core/embed_adapt.py`、`pipeline.adapt_embedding`、CLI;測試 `tests/test_embed_adapt.py`(救回 noise-swamped 對、CV 不假性救回真不可分對、save/load、pipeline)。
   - **此 v1 的邊界**:投影目前是**診斷 + 已存 artifact**;把它**套用到全 stack**(routing/bank-audit/error-mine 在投影空間重校準閾值)是下一步——需經 gate/eval 驗證不退步才打開。DINO 768 維、golden 少時 PCA 可能丟低變異判別方向(shrinkage 緩解)。
3. **去風險檢查**:出強勢「停止標註」措辭前,對被判不可分的對做一次離線「真的訓個小偵測器看分不分得開」sanity check。註:`adapt-embedding` 的 CV rescued 旗標已是這個檢查的**離線、零標註版本**(LDA 投影代替小偵測器)。

## 報告強化(多代理批判 → Tier 0/1,已實作 + 測試)
三輪多代理評 `weakness_report`(操作員 / data-centric / 產品三視角)後修正:
- **Tier 0 正確性(修「說謊的」)**:① `label_noise` 守門 —— 沒有正向模型混淆(c_cnt>0)且 embedding 真難分(sep_err>0.35)就不歸因為標籤雜訊(原本會對「可分、從不混淆」的類別對誤判 label_noise,白導人力);C=0+可分 → `clean`。② 把已算的 **O/C/Δ 信賴區間 + n_gt 分母**渲染出來(原本只顯示點估計)。③ `loc_gap` 區分 **N/A(單一 IoU)vs 0**;附 `map_by_iou`。④ 漏報型態顯示**全分佈**(非 most_common(1))。
- **Tier 1 可用性 + 行動化**:TL;DR 健康度(RED/AMBER/GREEN)+ 最弱類別 + 「現在做這個」;**`weakness_worklist.csv` 匯出**(把佇列從「讀」變「可清的清單」)+ `--worklist` 打 `vixq:*` tag 供 App 篩選;佇列旁就近顯示**歷史命中率**;PROXY 標記去重(頂部一次)。
- 範例:`docs/examples/weakness_report.html`(+ `gen_weakness_report.py` 可重現)。

## Tier 2 應用(進行中)
- ✅ **一致性判定接進 gate**:`pre_train_gate` 在**受保護類別對**(來自 `set-eval-baseline --protect`)被判 **supported 的 taxonomy/label_noise** 且**非** representation_fixable 時 → **NO-GO**(類別定義疑有問題,別匯出/重訓)。opt-in、需 baseline 的 protected 集 + eval;報告從「諮詢」升級為**能擋退步**。
- ✅ **App 可點工作清單**:`weakness-report --worklist` 打 `vixq:*` tag → `vix app` 自動把它們建成 **saved views**(`pipeline.worklist_views`),清單從「讀 vix_hash」變成「在 App 內點開該類候選」。
- ✅ **批次範圍**:`weakness-report/error-mine/hardneg --batch w23` 把「該標/該清什麼」的佇列 + 翻盤**只看這批**(GT 模型弱點維持全域,因為那是模型本質、與批次無關)。答週用問題「**這一批**要清什麼」。
- ✅ **趨勢(以 audit log 為準,非 snapshot 膨脹)**:`vix ap-trend` 從 hash 鏈日誌讀 `eval_ingest`(mAP/per-class AP/`eval_set_hash`)+ `weakness_report`(health)→ 逐類 AP 軌跡 + Δ + 健康度軌跡;**eval set 變過會標記「不可直接比較」**(避免「val 變簡單」假漲)。答「過去幾輪策展有沒有讓 bubble 變好」,離線可查、可稽核。`core/trend.py`。
- 關於 offline 價值的取捨:**hit-rate 回授「重排」佇列暫不做** —— 對小型/離線/演進中的資料,跨輪累積慢 + 非平穩 + 回授放大噪音,邊際小於風險;測量+顯示(已做)才是 offline 的真價值。
- 仍待:完整 FiftyOne **App 面板**(把報告表格嵌進 App);snapshot 綁 content_hash↔mAP(嚴格 release registry)。
