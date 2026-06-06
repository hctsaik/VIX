# weakness & consistency report 還有改善空間嗎?(多代理三輪 → 已實作)

> 問題:這份報告(已含一致性 CI、TL;DR 健康度、worklist、佇列命中率、批次範圍、representation_fixable、consistency→gate)是否還有改善空間?經三輪多代理討論。

## 結論:有,但不是重做——是約 8 個小幅**誠實/工作流**修補(多為 S);最「高級」的統計再造**該否決**。

報告現況已相當成熟(R1 四視角打 **72/72/81/74 ≈ 75**)。真正的價值來自**對抗 + 讀碼**:R2/R3 推翻了 R1 看似最高級的提案——它們會讓**安全閘變鈍**、用更差的統計工具、或根本不可行。沒有這幾輪,我們會去「優化」而把工具改差。

## 三輪軌跡
- **R1(獨立四視角:落地工程師 / data-centric 研究者 / 產品懷疑者 / MLOps)** — 共識:「誠實、密度高,但**最後一哩**(該標哪些)+ 幾個**最受信任格子**有過度信任陷阱」。
- **R2(對抗三角色:建構者 / 紅隊 / 誠實仲裁者)** — 浮現真實分歧:研究者+建構者想加統計機制;紅隊主張那會讓 NO-GO 閘變鈍。
- **R3(主席收斂,以程式碼裁決)** — 分歧是「程式碼到底做了什麼」的事實問題,直接讀碼定案。

## 關鍵裁決(讀碼)
NO-GO 閘([pipeline.py](../../src/vix/pipeline.py) `pre_train_gate`)需**四把鎖**:`verdict ∈ {taxonomy, label_noise}` 且 `tier == "supported"`(n≥20)且 `not representation_fixable` 且 `pair ∩ protected`(人工指定)。由此:
- **「taxonomy 在小 n 預設觸發」是誤讀** — 小 n → `insufficient_support`;n10–19 → `taxonomy_watch`;硬 taxonomy 需 supported。
- **「taxonomy 違反 0∈CI(Δ)」也是誤讀** — taxonomy 是 `zero_in`(O 與 C **一致**)+ 兩者皆高的**非方向**判定;方向判定 `model`/`label_noise` 各自要求 CI 遠離 0。契約其實**被遵守**。
- **多重比較校正對閘門無關且有害** — 閘門的比較家族是 `protected`(人工挑的 1–3 對),不是全部 ≤20 對。對閘門做校正只會**降低靈敏度**,讓真 taxonomy 碰撞悄悄放行。

## ✅ 已實作(8 項;`core/weakness_report.py` + `pipeline.py`;測試 `tests/test_report_improvements.py` +8;全套 240 綠)
| 標籤 | 缺陷 → 修法 |
|---|---|
| **H1a** | rescued 那格曾渲染 `taxonomy(可修)` 但動作說「**非** taxonomy 死路」——自相矛盾的頭條格(且該格 gate merge/NO-GO)。改渲染 `representation-fixable(非 taxonomy 死路)`;**非** rescued 的 taxonomy 不變(`td.v.tax` 仍紅、仍進閘)。 |
| **H2** | label 佇列「命中率」結構上恆=1.0([queue_metrics.py:28](../../src/vix/core/queue_metrics.py#L28),docstring 自承),卻和 hardneg 0.8 並列。label 佇列改顯示**覆蓋率**並加註「命中≡覆蓋率,只表行動量非品質」。 |
| **H3** | `closeness`/`wrongness` 是無圖例的 4 位小數 cosine → 被讀成機率。加圖例(「cosine 鄰近度,非機率」)+ 顯示改 2 位小數。 |
| **H4** | banner 的 PROXY 句自身**重複**(像複製貼上 bug)。去重為一句。 |
| **A2** | 「現在做這個」按歸因排序:當有 `representation_fixable` 弱對時,**升級 adapt-embedding、降級「去標」**(表徵問題,標註不是這裡的槓桿)。 |
| **L1** | report 事件+產物蓋上 `eval_set_hash` + 前一份報告時間戳 → 報告能定位在自己的趨勢上、可證可比。 |
| **L3** | eval set 與上期不同時,在**報告內**(非只在 `ap-trend`)顯示「不可比較」橫幅(複用既有 eval_set_hash 比對);同 set 則顯示「上期 mAP → 本期 mAP」。 |
| **L4** | 佇列**已處理感知**:標示「待辦 N / 已解決 M」、已解決候選打刪節線,**順序不變**(非被否決的重排)。資料來自既有 review/dismiss 日誌。 |

## ❌ 否決(經對抗 + 讀碼確認會讓工具更糟或不可行)
- **TOST 降級 taxonomy / 閘門多重比較校正** — 會讓安全閘變鈍(`protected` 已是人工小家族);taxonomy 邏輯其實合契約。
- **bootstrap sep_err** — Wilson 才是比例在小 n / 近 0,1 的正確區間;bootstrap binomial 更差。
- **per-class AP 誤差帶** — VIX **不算 AP**,只吃外部 eval-ingest 摘要;要 bootstrap 得重建 COCO 級子系統,不成比例。
- **砍「混淆 truth→pred 前10」/ 把 knn_dist 移走** — 那是**證據層**:overturn 的 `knn_dist>dist_thr` 就是「為什麼翻盤」;移走等於「相信我就對了」。保留。
- **報告端塞檔名/縮圖** — 留給 App(`vixq:*` → saved-views 已是瀏覽面);報告是耐久決策紀錄,把可變檔名烤進不可變紀錄是反模式。
- **佇列「清乾淨」跨類候選** — 落在某類失敗邊界的他類影像正是值得標的 taxonomy 邊界樣本;改為**保留 + 標記**而非刪除。
- **hit-rate 自動重排佇列** — 先前已否決,維持。

## 三個最高槓桿(75 → ~90)
1. **H1a** — gate 治理的頭條格不再自相矛盾(信任)。
2. **L4** — 佇列跨週可清、不重發已處理項(行動)。
3. **L1+L3** — 蓋 hash + 報告內不可比較橫幅,把快照變成可比較的縱向工具,堵住「val 變簡單卻當成 +mAP 慶祝」的漏洞(延續性)。

範例:`docs/examples/weakness_report.html`(`gen_weakness_report.py` 可重現,已含 rescued 判定 + 已處理候選 + 出處)。
