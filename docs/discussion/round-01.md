# Round 1 — 發散:各角色獨立立場

> 階段:Divergent。四個角色在不知道彼此結論的情況下,各自獨立提出立場。本輪目的是「攤開所有觀點與分歧」,不求收斂。

## 一、本輪討論方向

「如何把 VIX 概念落地、且基於 FiftyOne 開發?」拆成四個視角各自獨立作答:
- **A 演算法**:embedding、距離度量、label error、cartography、門檻、漂移、資料歸因。
- **B 架構**:build-vs-reuse 邊界、資料模型、UI、部署。
- **C MLOps/治理**:凍結評估集、版控、閉環安全、ablation、漂移治理。
- **D 產品/MVP**:分期、最小可用範圍、刪減清單、採用指標。

## 二、各角色重點摘要

### A — Data-Centric ML 演算法
- embedding:FiftyOne Model Zoo 的 **DINOv2 (ViT-B/14)**,對 YOLO bbox 裁切後逐框抽向量(`compute_embeddings` + `to_patches()`)。
- 距離度量:選 **kNN(k≈10, cosine)在原始 embedding 空間**,反對 centroid cosine(假設球形分布)與 Mahalanobis(樣本少時協方差不穩、p≫n)。對應 `fob.compute_similarity(metric="cosine")`。
- label error:用 `fob.compute_mistakenness`(查證後確認 **FiftyOne 無原生 cleanlab API**,需自建橋接);golden set 也要定期自審。
- cartography:在 Ultralytics callback 記錄每樣本 confidence mean/std → 寫進 FiftyOne 自訂 float 欄位(無原生 API,需自建)。
- 門檻:**per-class calibrated threshold + temperature scaling 優先**,Conformal Prediction 延到 V2(需專屬校準集、類別不平衡時失效)。
- 漂移:centroid cosine drift、intra-class variance、`compute_hardness` 分布位移 + KS test。
- TRAK/TracIn:**延後**(需 checkpoint 梯度、封裝成本高)。

### B — 系統架構師(FiftyOne 整合)
- **廢除實體資料夾搬移狀態機**(判定為反模式):狀態改用單一 FiftyOne Dataset 的 **tags/fields + SavedView**。
- **FiftyOne App 取代 Streamlit**:Embeddings Panel(Lasso 選點→批次 tag)+ Sample Grid + Sidebar Filter + 自訂 **Operator**(確認/重標/刪除)。逐張覆核卡需自建 Python Panel(~300 行)。
- **FastAPI 縮為「外部觸發層」**:僅留 `/ingest`、`/run-inference`、`/export-golden`,可呼叫 delegated operator。
- UMAP 由 `compute_visualization` 產生,只在 App 呈現。
- 提出完整 sample schema(tags:golden/review/pass/rejected/drift_candidate;fields:yolo_detections、yolo_conf_max、dino_embedding、dino_dist_to_centroid、routing_decision、review_decision、hardness、uniqueness…)。

### C — MLOps / 資料治理
- **FiftyOne OSS 無原生 versioning/snapshot(Enterprise 限定)** → 版控用 **DVC + Git**;OSS 的 `clone()` 只複製 metadata 不複製媒體檔。
- 凍結測試集存成**獨立 dataset**(`vix_golden_test_FROZEN`),物理隔離優於邏輯隔離,CI 前置檢查 sample count。
- 閉環 guardrail:Frozen Anchor Set(5–10%,不進訓練)、**獨立 reference model**(KL 散度 >0.15 升人工)、**人工雙簽**、每季 re-audit(Cohen's κ<0.75 觸發定義評審)、抽樣盲測。
- ablation 協定:baseline → DVC tag 新批次 → 只加新資料重訓 → 同一凍結測試集量 delta → W&B compare。列出必記 metadata。
- 實驗追蹤:W&B Artifacts 對應 DVC tag。

### D — 產品 / 務實落地(MVP)
- 核心主張:**這是資料「可見性」問題,不是演算法問題**。其他角色合計 >3 個月工時,但今天痛點 2 週能解 80%。
- **v0.1(2 週)**:本機 CLI + YOLO 推論信心 tag + DINOv2 cosine 距離 tag + FiftyOne App 瀏覽勾選 + 匯出 `selected_images.txt`。無 API、無 DB 遷移、無儀表板。
- 大膽刪減:Conformal Prediction(砍)、TRAK(砍)、FastAPI(MVP 砍)、雙簽(MVP 砍,單人作業是官僚化)、廢除實體資料夾(延後,強推會讓既有訓練腳本爆掉)。
- 成功指標走「工具採用面」:連續採用率、決策時間 <30 分、主動排除張數、是否被繞過。

## 三、已浮現的共識點(四方無異議或多數一致)

1. **基於 FiftyOne 開發,最大化複用 Brain**:`compute_embeddings / compute_similarity / compute_visualization / compute_mistakenness / compute_hardness / compute_uniqueness` 能省掉大量自建。
2. **DINOv2 經 Model Zoo 取得**,不自建推論管線。
3. **UMAP 只用於「看」,路由判定一律在原始 embedding 空間做**(A、B 一致;修正原草案)。
4. **FiftyOne OSS 沒有原生版本控制**(C 查證確認),版控需外部工具補位。
5. **FiftyOne 無原生 cleanlab 整合**(A 查證),需自建橋接;MVP 先用 `compute_mistakenness`。
6. **凍結 golden test set 是「量模型有沒有變好」的唯一尺**,必須與閉環隔離(A、C 一致,D 認同但延後制度化)。
7. **閉環 review→golden→重訓 的正回饋偏差是真風險**(C 明列,A 也點出)。

## 四、主要衝突點(Round 2 要對撞解決)

| # | 衝突 | 一方 | 另一方 | 本質 |
|---|------|------|--------|------|
| C1 | **整體高度** | A/B/C 要嚴謹完整 | D 要 2 週 MVP、其餘都過度工程 | 範圍 vs 嚴謹 |
| C2 | **實體資料夾** | B:廢除,改 FiftyOne Dataset+tags | D:MVP 不可強推,既有訓練腳本依賴路徑 | 架構正確 vs 遷移成本 |
| C3 | **距離度量** | A:現在就上 kNN | D:MVP 用 cosine 就好 | (註:`compute_similarity` 的 kNN 幾乎免費,疑似偽衝突) |
| C4 | **路由門檻** | A:per-class calibrated;Conformal→V2 | D:Conformal 直接砍,門檻越簡單越好 | 統計正確 vs 認知負擔 |
| C5 | **閉環 guardrail** | C:雙簽 + reference model + KL gate | D:單人作業雙簽是官僚化 | 治理 vs 團隊規模現實 |
| C6 | **FastAPI** | B:留作觸發層 | D:MVP 砍,純 CLI | 自動化 vs 最小化 |
| C7 | **版控** | C:DVC 從一開始 | D:MVP 用 txt + git commit 湊合 | 可追溯 vs 上手速度 |

## 五、本輪決策(暫定,待 Round 2 確認)

- **D1**:採 FiftyOne 為底層平台、Brain 為演算法骨幹。(共識,鎖定)
- **D2**:UMAP 不參與任何路由判定。(共識,鎖定)
- **D3**:距離/相似度走 `compute_similarity` 的 kNN(cosine),不走 centroid。(傾向 A,待確認 C3 是否偽衝突)
- **D4**:版控不用 FiftyOne OSS(無此能力),走 DVC。(共識,鎖定;啟用時機待定)
- **D5**:採「分期路線圖」作為調和 C1 的框架——把 A/B/C 的高級項目分配到正確階段,而非二選一。(提案,Round 2 表決)

## 六、下一輪(Round 2)要澄清/對撞的方向

1. **用「分期」當作調和總框架**:請各角色針對自己的爭議項目,明確指認它該落在 v0.1 / v0.2 / v1.0 哪一期,以及「絕不可被錯放」的一個非協商底線。
2. **C2 資料夾**:能否折衷為「FiftyOne Dataset 作唯一事實來源,但提供 export 轉接器把 golden 匯回使用者既有的資料夾/YAML 格式」,使既有訓練腳本零改動?
3. **C3 距離**:確認 `compute_similarity` 的 kNN 是否真的不比 cosine-to-centroid 貴(若是,衝突消失,直接採 kNN)。
4. **C5 guardrail**:guardrail 強度是否應「依團隊人數動態調整」(單人→只留凍結測試集+anchor+reference 打分;多標註者→才啟用雙簽)?
5. **待使用者釐清(各角色重複追問,採合理預設先行,不阻塞)**:
   - 既有訓練腳本吃什麼資料格式(資料夾/YAML/自訂 Dataset)?→ 決定 export 格式。
   - 本機 GPU / VRAM?→ 決定 DINOv2 可行性。
   - 每批影像量級?→ 決定時間預算與是否需換向量後端(sklearn→LanceDB)。
   - 標註人數?→ 決定雙簽是否可行。
   - 有無 FiftyOne Enterprise 預算?→ 已預設「無,用 DVC」。
   - 漂移定義是「類內分散變大」還是「跨類別界線模糊」?

> **處理原則**:依目標指示不暫停等使用者回覆。Round 2 各角色將對上述未知採用**明確的工作假設**(預設 persona:單人工程師/標註者、本機單 GPU、每批數百至數千張、無 Enterprise 預算、既有訓練腳本吃資料夾+YOLO txt),並在假設上收斂。使用者可隨時插話覆寫假設。
