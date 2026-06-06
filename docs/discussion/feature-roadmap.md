# VIX 缺哪些重要功能?(多代理深度討論共識)

三位不同專長代理(YOLO 偵測訓練專家 / Data-Centric AI 研究者 / MLOps 資料治理主管)獨立討論
「還缺哪些重要功能,能幫我管好資料集 **並** 提升 YOLO 準度」,結論高度收斂。

## 核心診斷:VIX 目前「對模型是盲的」

VIX 是一個封閉的 embedding 世界 —— `route`/`gate`/`coverage`/`label-noise`/`harmful`/`active-learn`
全部由 **YOLO 信心度 + DINOv2 距離** 算出,這些都是「資料看起來如何」的**代理**,
**從未被告知「訓練後的模型在 held-out 驗證集上實際錯在哪(哪一類 AP 低、混淆哪兩類、哪些 FP/FN)」。**
→ 這就是 README 承諾「我加的資料讓模型變好還變壞?」目前**做不到**的根因。

---

## 第一優先(三位一致的 keystone)— ✅ 已實作(v0.1)

> `vix eval-ingest <results.json>` 回灌驗證集(GT+pred,IoU 配對)→ per-class AP / mAP / 混淆矩陣 / 逐張 FP-FN,
> 寫入 `eval_results.json` 並把 `eval_fp`/`eval_fn` 掛回樣本;`vix error-mine` 把 FP/FN 投影回 embedding,
> 排序「最接近模型實際失敗處」的未標註候選。核心在 [`src/vix/core/eval_ingest.py`](../../src/vix/core/eval_ingest.py),純函式、有單元測試。

### ★ #1 驗證結果回灌 + 失敗案例反查 `vix eval-ingest` / `vix error-mine`
匯入一次驗證集評估(GT + 預測,IoU 配對)→ 算出 **per-class AP、混淆矩陣、逐張 FP/FN(含框)**,
再把每個錯誤投影回 DINOv2 空間,用既有 kNN 索引找出「最像這些錯誤」的未標註候選。

- **→ mAP 機制**:把誤差拆成分類錯/定位錯/重複/背景FP/漏標(FN);mAP@0.5:0.95 主要被**定位誤差尾巴**拉低,VIX 現在完全看不到。反查 FN/低 IoU 的影像 → 餵進既有 `review-queue→golden` → 等於**用你已有的資料做硬例挖掘**,直接補模型正在失敗的地方。per-class AP 告訴你標註預算該花在哪一類(目前 `coverage` 只看數量,與 AP 相關性差)。
- **→ 資料管理價值**:健康報告從「12 個疑似標錯(embedding 猜的)」升級成「`bubble` 類 AP 掉 6 分,60% 的 FN 聚在某個你只有 4 張訓練圖的視覺區」。讓 `gate` 有**真實指標**可守。
- **效用關鍵**:它讓**現有每個啟發式訊號變成「模型驗證過」的訊號** —— 與 FN 群同位置的 label-noise 才是真的,落在模型已掌握區域的就是雜訊。是 #2/#4/#6 的前置。
- **成本**:中。核心是 COCO 式 IoU 配對器(純函式、可測、零重依賴);重用既有 embedding+kNN。需要你實際跑一次 val 評估把預測丟給 VIX(本來就該有的紀律,且維持 $0/離線)。

---

## 第二層:純資料、可立即出貨、直接拉 mAP(不需等 eval)

> **更新(model-loop v2,已實作 + 測試):** #1 的解析度已加深(逐錯誤分型 classification/localization/missed/background、IoU sweep `loc_gap`、error-mine 改用誤差框 region),
> **#2 box-qa**、**#7 challenge-guard**(把 mAP/受保護類別 AP 退步接進 gate 硬擋)均已落地,另修兩個帳本完整性 bug。
> 設計與審查見 [model-loop-v2-design.md](model-loop-v2-design.md),操作見 SOP §B8。下方 #2/#7 條目保留作背景脈絡。

### #2 逐框「框品質」QA `vix box-qa` ✅ 已實作
逐張、逐框靜態檢查:**過鬆/過大框、退化框(w或h≈0)、貼邊截斷框、長寬比超出該類包絡**。
- **→ mAP**:**標註框緊度是 mAP@0.5:0.95 的頭號天花板**。系統性鬆 10–15% 的框會教出鬆的先驗,在嚴格 IoU(0.75–0.95)大量丟分;退化框是 NaN/loss 爆衝來源;截斷框該設 ignore。
- 現有 `geometry` 只做兩期**聚合**均值漂移,**從不檢查單一框品質** → 這是最高「mAP/工」的純資料項。成本低。

### #3 尺度感知覆蓋 `vix coverage --by-scale`
依**物件面積**(small/medium/large,COCO 式)分層統計,給「還缺 X 張小物件」目標。
- **→ mAP**:小物件是公認的 mAP 黑洞(APsmall 常比 APlarge 低 2–4 倍),病因幾乎都是**資料**;某類「總數看起來夠」但小尺寸樣本被餓死。直接抬升 mAP 平均中最低那一項,也指引 `imgsz`/tiling/mosaic 決策。框面積已有,成本低。

### #4 train/val 分布匹配 + 精確分割洩漏 `vix split-audit` / `vix split-leakage`
稽核 train vs val 的**類別/尺度/長寬比分布是否匹配**,以及該模型實際用的 train/val 之間的近重複洩漏。
- **→ mAP(尤其「可信的」mAP)**:val 與 train 分布不匹配 → **回報的 mAP 本身在說謊**,你會朝錯的目標最佳化;val 洩漏會灌水 AP(模型等於看過 val)。修好分割讓 mAP 可信,是其他所有改善能不能「真的生效」的前提。重用既有 dedup/kNN,成本低。

---

## 第三層:模型驗證過的閉環(依賴 #1)

### #5 eval 導引的主動學習 `vix active-learn --guided-by eval`
把目前 uncertainty+novelty 換成**朝模型實測誤差面**取樣(靠近 FP/FN 群、且該類 AP 低者優先)。
- **→ mAP**:純不確定度會系統性漏掉「**有自信但答錯**」的最高價值樣本;誤差導引把固定標註預算花在真正能動指標的樣本。需保留 novelty 下限避免隧道視野。

> 另:**低信心 YOLO proposal 探勘 + 三銀行 DINO embedding 審查**(SMM 提案)亦已實作為 `vix bank-audit`,
> 設計與共識見 [bank-audit-design.md](bank-audit-design.md),操作見 SOP §B7。

### #6 標籤錯誤「修正工作流」(非只偵測)`vix fix-labels`
把 `label-noise` + eval FN/混淆的建議,做成 App 內 接受/編輯/拒絕 → 寫回 golden(全程稽核)→ 重匯出,並用 challenge set 確認修正真的有幫助。
- **→ mAP**:VIX 目前只「偵測」標錯就斷在一張清單;修正才是最高 ROI 的資料介入。錯框注入梯度雜訊直接壓 AP,修幾%的標錯常換來數個 mAP 點。

### #7 帶標籤的 held-out「挑戰集」/ 回歸守門 `vix challenge-guard` ✅ 已實作
一個凍結、永不訓練、**有標籤**的評估集(與 unlabeled `anchor` 互補),每次資料變更都對它打分;
mAP 掉(整體或受保護類別)超過門檻 → gate **硬擋**。
- **→ mAP**:這是讓上面所有功能能「大膽改」而不會偷偷掉準的安全閥。現在 gate 只看覆核積壓/洩漏/漂移/稽核,**沒有任何準度指標**。

### #8 模型/資料發布登記簿 + 快照差異 `vix release` / `vix snapshot-diff`
每個釋出的模型版本綁定:確切快照(content_hash)+ **相對上一版的資料 delta** + #1 量到的 mAP。
- **→ mAP**:把「資料變更 → mAP 變化」變成可查詢的歷史,學會哪些策展動作真的有用(加 reflection 硬負例有效;狂塞 scratch 沒用),把預算導向**已證實有效**的 delta。`snapshot-diff` 是算 delta 的前置。

### #9 逐樣本影響力 / 邊際價值 `vix influence`(最後做,風險最高)
用 challenge set 上的小規模消融 / 對誤差群的相似度近似,估「哪些**訓練**樣本真的幫了 vs 害了」某指標。
- **→ mAP**:找出**淨負面**樣本(移除反而漲 mAP),把 embedding-only 的 `harmful` 升級成模型驗證過的版本,零標註成本拉準度。

---

## 與既有驗證缺口的交集
- **逐類覆核政策(#2 of MLOps 視角 / 本路線圖未獨立列)**:正是 Round 13 的 AE6 缺口 —— 需要 `vix set-threshold/policy <class>` 作為一級設定(逐類門檻/目標/gate 動作)。
- **release 登記簿 / challenge-guard** 也回應 Round 13 的 AE3(「GO 但 mAP 掉」無法歸因)。

## 建議建置順序(共識)
1. **先出純資料、零依賴、立即拉 mAP 的**:#2 box-qa、#3 scale-coverage、#4 split-audit。
2. **建 keystone**:#1 eval-ingest(解鎖整個模型驗證層)。
3. 接著 #5 guided-AL、#7 challenge-guard、#8 release/diff,最後 #9 influence。

> **一句話**:VIX 現在能說「這筆資料看起來可疑」,但說不出「這筆資料**確實**害了 mAP」。
> 補上 **eval 回灌(#1)** 後,既有所有工具都從「embedding 看起來」升級為「模型證實」,
> README 承諾的「資料 → 模型」閉環才真正成立。要證明它有效就追蹤一個指標:
> **凍結挑戰集上、被誤差挖掘驅動去補的那幾類的 per-class AP,是否每次改版都比等量隨機標註漲得快。**
