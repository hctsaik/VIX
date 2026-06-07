# VIX v1 收尾與「宣告完成」(多代理三輪 × 五代理 結論)

> 問題:VIX 已很成熟,還有什麼功能需要加?經三輪、每輪五代理(共 15 個)、多位讀原始碼後的討論。

## 結論:**VIX 基本 feature-complete。做一個小而精的清單,然後宣告完成 —— 不是繼續加功能。**

剩下真正有價值的只有少數,其中**一個是真缺陷(不是功能)**。本次收尾**實作了共識的「下一個 PR」**(缺陷修復 + 一個因果確定的便宜功能),其餘列為採用里程碑/延後。

## ✅ 本次收尾已實作(commit 待填;全套 **303** 綠)

### 1. 編碼器指紋(encoder fingerprint)—— 真缺陷修復
讀碼確認:編碼器身份(`_vix_tag`)算出來卻**只丟進 log**;審計記的 `embedding_backend` 是**靜態設定字串**,權重/torch 版本/裝置/重抓的 cache 改了它都不變 → 兩個實質不同的編碼器蓋出**相同的審計身份**並通過既有後端檢查,而每個 PROXY 數字(kNN 距離、可分性、漂移、門檻)默默漂移。**同先前「框級稽核漏洞」那一類**(承諾默默失效)。
- `core/encoder_fingerprint.py`(純):`encoder_fingerprint(material)`→{fp,components};`probe_digest` 用一張固定探針影像取**行為指紋**(粗略四捨五入 → torch 小版本不誤報,真換編碼器才變)。
- `compute_embeddings`(vix embed)在 `dataset.info["vix_encoder_fp"]` 蓋上編碼器身份(行為探針 + 權重/版本/裝置/前處理)。
- `calibrate` 把它寫進 `policy.meta["encoder_fp"]` → 進 `thresholds.json` → **進 snapshot `content_hash` 與訓練池雜湊**(編碼器變,snapshot 身分就變)。
- `pre_train_gate`:資料目前的編碼器指紋 ≠ calibrate 時 → **NO-GO**(門檻不可比);無指紋則 fail-open(向後相容)。
- **誠實邊界**:身份/可重現性檢查,**非**安全簽章;不做 PKI、不做位元級權重雜湊、不做 torch 版本鎖(魔鬼代言人標為鍍金)。

### 2. 近重複標註一致性 —— 因果確定的標錯
`core/analytics.near_dup_label_conflicts`:近乎相同的影像(DINO,門檻收緊到 0.03)卻帶**矛盾標註** = 至少一個標錯(**因果確定,非 proxy**)。CLI `vix near-dup-labels`。**諮詢式**:只列出矛盾配對給人裁決,**絕不自動改標**(避免在 bubble-vs-reflection 類難例上亂報 → 高相似 + 標註矛盾才報)。

## 採用 / 里程碑(共識認可,但屬「宣告完成後」的下一步,非本次)
- **release registry(唯讀)**:把外部重訓得到的 mAP 綁回 snapshot content_hash,閉環。魔鬼代言人標為**最高風險滑坡**(別擁有重訓)→ 嚴格 append-only/唯讀「綁定收據」,只存數字+雜湊、無任何消費它去觸發動作的指令,只收綁得上 VIX 自產 snapshot 的數字。R3 已有完整 spec。
- **離線安裝 bundle + 驗證器**(獨立里程碑):「$0 離線 air-gapped」目前是有程序但未驗證;驗證器須**硬失敗**(known-answer 嵌入檢查)防靜默退化成 pixel_fallback。
- **真資料案例 + Day1→Week4 線性文件**:便宜的採用增益。
- 在 App 呈現**已存在**的策展狀態 + 「我這週」摘要(引擎已在 `_resolved_ids`/`throughput`)。

## 完成定義(Definition of Done)
1. honest gate 端到端(**編碼器指紋缺陷已修 + 回歸測試** ✅)。
2. 陌生人能離線安裝並跑(留待 bundle+verifier 里程碑)。
3. verb 集**凍結**、皆有測試與文件。
4. 一份真資料案例(dogfood + FINDINGS 已有雛形)。
5. 操作者看得到自己的狀態(CLI 已有;App 摘要待呈現)。
6. determinism / $0 / 單機 不變式鎖定 ✅。
→ 達成後 **VIX v1 凍結,之後只做維護**(bugfix / 相依鎖定 / 文件)。

## 凍結 / 否決(別再加;附解凍條件)
- DVC bridge(工作區已可 DVC 追蹤)— 除非真有用戶無法追蹤。
- COCO export(刻意只輸出 YOLO)— 除非有指名的下游消費者。
- rank-stability proxy 分數 — 只能當「自我不信任」訊號,且要有真實誤判案例才做。
- per-class calibration-at-threshold、更多 App 面板、P-series、scale、multi-user — 各需真實阻塞案例。
- **標準否決維持**:重訓、自動改標/merge、in-app admit、AP 趨勢線、hit-rate 重排、dashboards/sliders/TUI、eval 覆蓋率 %、PKI、PII 偵測。

**一句話的「完成」**:陌生人離線裝好 VIX、指向自己的資料,一週內就信任它的 admit/reject —— 因為閘門的每個決定都在帳本裡,且綁定了產生它的編碼器與資料,連編碼器悄悄換掉都會被擋下。
