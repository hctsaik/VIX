# 還有什麼值得做?(多代理兩輪)+ 已修兩個確認缺陷 + 真資料 DOGFOOD

> 問題:VIX 還有特別重要、值得往下開發的嗎?多代理討論 → 修確認的缺陷 → 在真資料驗證。

## 結論
VIX 對其 niche 大致 **feature-complete**;真正該做的是**修兩個讀碼確認的缺陷**(不是加功能),外加**證明在真資料有用**。R2 對抗 + 讀碼**推翻了看似最高級的兩個提案**(eval 覆蓋率 % / 把 adapt-embedding 套全 stack),**確認了兩個真實 bug**。

## 兩輪軌跡
- **R1(四視角:落地工程師 / data-centric / 產品懷疑者 / 架構)**:各自指出不同真缺口——框級稽核漏洞、eval 代表性、缺「證明有用」、ap-trend 誠實 bug。
- **R2(建構者 / 紅隊 / 誠實仲裁者,皆讀碼)**:三位一致。

## R2 裁決
| 項目 | 裁決 |
|---|---|
| **BOX 框級稽核漏洞** | ✅ **必修(最危險)**。讀碼確認:`content_hash`=hash(golden 的 vix_hash + 門檻),`vix_hash`=圖片位元組 SHA-256,`export` 讀**活的**框 → 在原生編輯器改框,匯出變、但 snapshot 身分 + DecisionLog **位元相同**。最小誠實修法:golden 框摘要折進 content_hash + export 蓋 boxes_hash。**不做**逐框 changelog。 |
| **TRENDFIX** | ✅ **必修**。`ap-trend` 即使 eval set 變過仍印 ↑進步/↓退步,而 gate 在同條件已 withhold。eval set 變過時隱藏方向箭頭(留序列 + 警告)。 |
| eval 覆蓋率「%」 | ❌ **否決**:幾百張上是任意半徑造的假精確、製造新過度自信;誠實版(per-class eval 支撐不足)已是 challenge-guard。 |
| adapt-embedding 套全 stack | ❌ **凍結**:在幾十個 golden 上 fit、全域套用會犧牲讓它誠實的 CV gate;維持診斷。 |
| CLI 整併 | ❌ 凍結(無行為變更的重構=純風險,真衝突再做)。 |
| DOGFOOD(真資料 FINDINGS) | ⭐ 策略最高,已做(見下)。 |
| PACKAGING/CIFIX、LOCQUEUE、CSVQUEUE、P4STATUS | DEFER。 |

## ✅ 已實作(commit 待填;全套綠)
- **BOX**:`core/snapshot._content_hash(golden, thr, box_digests=None)` 可把框幾何+標籤折進雜湊;`pipeline._box_digests(adapter, tags)` 算 golden 框指紋;`snapshot()` 與 `_training_pool_hash()`(batch-admit 用)都改成綁框內容;`export()` 在 DecisionLog `export` 事件 + 回傳蓋 `boxes_hash`(記錄「訓練了哪些框」非只有數量)。→ 原生改框不再對稽核/snapshot 隱形。
- **TRENDFIX**:`cli.py` ap-trend 在 `eval_set_changed` 時改印「Δ 不可比較:eval set 期間內變過」,不再印假箭頭。
- 測試 `tests/test_box_audit_trendfix.py`(5):content_hash 綁框幾何/標籤、訓練池雜湊隨改框變、export boxes_hash 隨改框變且入事件、ap-trend 變更時藏箭頭/不變時顯示。既有 snapshot/batch_gate/round6 雜湊測試不受影響(box_digests 預設 None = 舊行為)。

## 真資料 DOGFOOD(`C:\code\claude\patHole_Dataset`,可重現 `docs/examples/dogfood_pathole.py`)
誠實範圍:無 YOLO 權重/GPU、且 VIX 本就不訓練 → 用 VOC ground-truth 框當偵測、pixel_fallback 嵌入(離線);**不量 mAP**(無訓練工具的範圍外),驗的是策展/稽核機制 + 兩個修補在真資料成立。

- 載入 **665 張 / 1740 個 GT 框**(golden),10.6s,classes=['pothole']。
- **[BOX] 在真 pothole 框上驗證**:改一個真框 → 訓練池 content_hash **變(True)**、還原 **完全相同(True)**、export boxes_hash **變(True)**。漏洞在真資料確認關閉。
- **[FLAGS]** VIX 在這批標的:suspected_label_issues=0(單一類別 → 結構上無跨類混淆,誠實)、near_duplicate_groups=2、coverage=2 群。
- **[TRENDFIX]** 兩次不同 eval_set_hash → `eval_set_changed=True` → 每類方向箭頭withheld、印警告。

## DOGFOOD 進階:訓練一版真 YOLO 跑「模型弱點」那半(已完成)
VIX 本身**不訓練**;這是使用者會做的**外部**訓練,用來產生 VIX 消費的預測。
- `docs/examples/dogfood_train_yolo.py`:VOC→YOLO、deterministic 80/20(**532 train / 133 val**)、單類 pothole、YOLOv8n、CPU、12 epochs(~31 分)。結果 **mAP@0.5=0.754, P=0.80, R=0.64**(刻意不完美 → 有真 FP/FN 給 VIX 抓)。
- `docs/examples/dogfood_eval_yolo.py`:把 best.pt 在 133 張 val 推論(conf≥0.05 以抓 FP)→ 建 `{gt,pred}` JSONL → **`vix eval-ingest` + `vix weakness-report`**。真實輸出:
  - **mAP@0.5 = 0.7234**;`map_by_iou {0.5:0.7234, 0.75:0.5119}` → **loc_gap = 0.21**(框在高 IoU 變鬆=真定位弱點)。
  - **FP 型態:background 369**(低門檻下過度預測=真誤報);**FN 型態:missed 37 / localization 21**(真漏報)。
  - weakness-report:**health=AMBER**,最弱=pothole AP 0.7234;**15 個「自信卻錯」**(GT 證實的高信心誤報,最高 conf=0.83 的 background FP)——正是 VIX 要優先推給人覆核的盲點。
- **結論**:VIX 確實能消費**真實訓練模型**的 eval,產出可行動的弱點(mAP/typed FP-FN/loc_gap/自信誤報),不需自己訓練。VIX 的 all-point AP(0.7234)與 ultralytics 報的 mAP50(0.754)略有差異(AP 計法/匹配不同)——誠實揭露,非 bug。
- 侷限:單類別 → 一致性/混淆歸因(跨類)無用武之地(本就如此);pixel_fallback 非 DINOv2(離線無 GPU,不影響 eval 指標)。

## 下一步(可選)
你自己的真資料只要有 YOLO 權重 + held-out eval set,同一條流程即可:`vix infer --weights <yolo.pt>` → `vix eval-ingest <val.jsonl>` → `vix weakness-report`,得真實 per-class AP/弱點/一致性歸因。
