# Consensus — VIX 落地共識(已鎖定)

> 經 Round 1(發散)→ Round 2(收斂),四個角色(演算法 / 架構 / MLOps / 產品)達成**有條件共識**,所有條件已彼此相容並納入本文件為**binding 約束**。本文件是多代理人討論的最終產物。

---

## 0. 一句話共識

> **VIX 不是「XAI 工具」,而是一套建立在 FiftyOne 之上的「Data-Centric AI 資料守門員」**:用凍結的 DINOv2 embedding 空間 + YOLO 信心,把「哪批資料讓模型變好/變壞」與「類別定義有沒有漂掉」變成**可見、可追溯、可制度化**的閉環;以**分期**控制複雜度,以**凍結參照點**防止閉環放大模型自身偏差。

---

## 1. 問題重構(共識起點)

使用者原以為的痛點是「可解釋性」,但真正的痛點是兩個**資料層面**的問題,兩者解法不同:

| 痛點 | 本質 | 對應手段 |
|------|------|----------|
| 加了哪批資料讓模型變好/變壞 | **資料歸因 (Data Attribution)** | 凍結測試集 + 版控 + ablation(地基)→ Cartography → TRAK(進階) |
| 加的資料讓 bubble/reflection 等定義變不一致 | **標註品質 + 概念漂移** | label-error 偵測 + 凍結參照漂移監控 + (多人時)一致性 κ |

→ 兩者皆屬 **Data-Centric AI**,非傳統模型可解釋性(Grad-CAM/attention)。

---

## 2. 鎖定的架構原則(不分期,全程適用)

1. **FiftyOne 為底層平台,Brain 為演算法骨幹**,最大化複用 `compute_embeddings / compute_similarity / compute_visualization / compute_mistakenness / compute_hardness / compute_uniqueness`。
2. **單一 FiftyOne Dataset 為唯一事實來源**;狀態用 **tags/fields**,**不靠搬檔**。〔B 底線〕
3. **UMAP 只用於「看」**,所有路由/離群/漂移判定一律在**原始 DINOv2 embedding 空間**進行。〔A/B 共識〕
4. **kNN(cosine, `compute_similarity`)取代 centroid**——成本相近但語意更正確。〔C3 消解〕
5. **export 轉接器為單向**(讀 tags → 產生資料夾/YOLO txt/data.yaml),**禁止反向**寫回 Dataset 狀態;讓既有訓練腳本**零改動**。〔B 底線 + C2 折衷〕
6. **凍結 golden test set 物理隔離**(獨立 dataset),**第一天建立、永不參與訓練/調參**。〔C 底線〕
7. **閉環必須有不被訓練資料污染的外部參照點**,從 v0.1 起即存在。〔C 底線〕

---

## 3. 分期路線圖(總調和框架)

### v0.1 —「可見性」(目標 ~2–3 週,單人本機)
**主解痛點:**「這批新資料加進去後,模型哪些預測變奇怪了?」

- 從既有資料夾 import 進**單一 FiftyOne Dataset**(tags/fields 為狀態,非搬檔)。
- **DINOv2(Model Zoo)** 逐 bbox embedding;**`compute_similarity` kNN(cosine)** 算離群分數;YOLO 推論信心。
- 兩信號寫成 fields + tags:`low_conf`、`far_from_known`。
- **FiftyOne App 原生 UI** 覆核:Embeddings Panel + Lasso 選點 + 內建 tagging + SavedView 過濾(**不自建 Operator/Panel**)。
- **三項便宜閉環防護**:
  - 凍結 golden test set(獨立 dataset)。
  - frozen anchor set(從初始標註抽 5–10%,永不進訓練)。
  - **frozen pretrained DINOv2** 作 reference:比較新批次相對 anchor 的 embedding 分布位移 / anchor 類別 kNN 標籤一致率;超閾值(KL>0.15 或一致率↓>5%)→ **單人自我 gate:暫停該次更新 + 書面自審紀錄**。〔C 條件〕
  - 同時把這份 frozen DINOv2 embedding **snapshot 凍結保存**,作為未來 KS 漂移的基準。〔A 條件,一物兩用〕
- **export 轉接器**(單向)把 golden/selected 匯回既有資料夾 + YOLO txt + data.yaml。
- 門檻用 **per-class 百分位**(基於訓練集 embedding 分布),README 寫明「誰算/存哪/怎麼更新」。〔A/D 條件〕
- 版控用**結構化 txt manifest + git**(欄位:hash / 來源 / 日期 / 標註版本)。〔C 條件〕
- **紅線**:無 API/服務化/多人並發寫入。〔D 底線〕

### v0.2 —「可追溯」(~+4–6 週)
- **DVC + Git** 資料版控;**W&B** 實驗追蹤(run 綁資料版本指紋)。
- **完整 ablation 協定**:baseline → 只加新批次重訓 → 同一凍結測試集量 delta → W&B compare。
- 門檻升級 **per-class calibrated threshold + temperature scaling**。
- **cleanlab 橋接**做 label-error(取 pred + gt 送 cleanlab,issue 寫回 sample field)。
- **漂移 KS test**(此時才有穩定基準,比對 v0.1 凍結的 embedding snapshot);同時量類內分散↑與類間界線↓。
- **凍結 reference YOLO 任務模型**(C 原意)做預測分布 KL gate。
- **FastAPI 觸發層**:`/ingest`、`/run-inference`、`/export-golden`,接 CI/訓練 pipeline。
- v0.2 末:**Dataset Cartography 相容性 PoC** 檢查點(確認訓練 log 格式可用,不強制上線)。〔A 條件〕

### v1.0 —「可制度化」(~+8 週)
- 自建 **FiftyOne Operator/Panel**(確認/重標/刪除按鈕、逐張覆核卡)取代部分 CLI。
- **Dataset Cartography** 訓練動態(confidence mean/std → easy/ambiguous/hard 分群,寫回 fields/tags)。
- **Conformal Prediction** 路由(統計保證覆蓋率)。
- **多標註者治理**:雙簽狀態機(pending→approved_a→approved_b→golden)+ 每季 re-audit + Cohen's κ(<0.75 觸發定義評審會)+ 抽樣盲測。
- **Mahalanobis** 補強(僅樣本數 > 5×embedding dim 的類)。

### 研究 / 延後
- **TRAK / TracIn** 細粒度資料歸因(需 checkpoint 梯度、封裝成本高)。

---

## 4. Build-vs-Reuse(合併定稿)

| 功能 | FiftyOne 提供(複用) | VIX 自建 | 階段 |
|------|----------------------|----------|------|
| embedding | `compute_embeddings` + Model Zoo DINOv2 | bbox 裁切前處理 | v0.1 |
| 離群/相似度 | `compute_similarity`(kNN, cosine, sklearn 後端) | 離群分數→tag 規則 | v0.1 |
| UMAP 視覺化 | `compute_visualization` + App Embeddings Panel | 無 | v0.1 |
| 覆核 UI | App 原生 tagging/Lasso/SavedView | 無 | v0.1 |
| 狀態管理 | Dataset tags/fields(MongoDB) | export 轉接器(單向) | v0.1 |
| 凍結測試集/anchor | 獨立 Dataset + tag | 隔離與 CI 檢查 | v0.1 |
| reference 打分 | frozen DINOv2 embedding | 分布位移/一致率 + 自我 gate | v0.1 |
| label-error | `compute_mistakenness` | cleanlab 橋接(無原生整合) | v0.1 / v0.2 |
| 版控 | —(OSS 無原生 versioning) | txt manifest→DVC | v0.1 / v0.2 |
| 實驗追蹤 / ablation | — | W&B 整合 + 協定 | v0.2 |
| 外部觸發 | delegated operator | FastAPI 3 端點 | v0.2 |
| 覆核動作按鈕/逐張卡 | App Modal(部分) | Operator/Panel(~150–300 行) | v1.0 |
| Cartography | 任意自訂 field | 訓練動態 callback | v1.0 |
| 漂移告警 | `compute_hardness` | KS test + 凍結 YOLO KL + Slack | v0.2 / v1.0 |

---

## 5. Sample 資料模型 Schema(定稿)

**Tags**:`golden` / `review` / `pass` / `rejected` / `low_conf` / `far_from_known` / `drift_candidate` /(v1.0)`approved_a` `approved_b`

**Fields**:
`batch_id`(str)、`ingested_at`(datetime)、`yolo_detections`(`fo.Detections`)、`yolo_conf_max`(float)、`dino_embedding`(768-d)、`knn_outlier_score`(float)、`routing_decision`(str: auto_pass/auto_reject/send_review)、`review_decision`(str)、`reviewer_id`(str,可為 "auto")、`reviewed_at`(datetime)、`hardness`(float, v0.2+)、`uniqueness`(float, v0.2+)、`cartography_conf_mean`/`cartography_conf_std`(float, v1.0)。
UMAP 座標存 `brain_key`(不入 sample)。

---

## 6. 部署拓樸(分期)

- **v0.1**:純 Python 套件/CLI + `fiftyone app launch`(App UI)+ 本機 MongoDB(FiftyOne 內嵌)。無 API。
- **v0.2**:加 `uvicorn vix.api:app`(FastAPI 觸發層)+ DVC remote(S3/GCS)+ W&B。
- **v1.0**:加 `fiftyone delegated launch`(背景 operator)+ VIX Plugin(Operator/Panel)。

---

## 7. Binding 條件清單(來自四角色,全程約束)

1. 〔A〕kNN embedding 來源**鎖定 DINOv2 Model Zoo**,禁用 YOLO 中間層特徵。
2. 〔A〕v0.1 末**凍結 reference embedding snapshot** 作 KS 基準(比較邏輯可延 v0.2)。
3. 〔A〕v0.1 百分位門檻的分位基礎 = 訓練集 embedding 分布,寫明於 README。
4. 〔B〕export 轉接器**單向**;v0.1 ingest 禁止以搬檔作為狀態變更。
5. 〔C〕reference 超閾值→**暫停 + 書面自審紀錄**(單人自我 gate)。
6. 〔C〕凍結 golden test set 第一天建立、永不參與訓練/調參。
7. 〔C〕v0.1 txt manifest 須結構化強制(hash/來源/日期/標註版本)。
8. 〔D〕reference model = **frozen pretrained DINOv2,不另訓練**(v0.1)。
9. 〔D〕v0.1 紅線:無 API/服務化/多人並發寫入;不自建 Operator/Panel。

---

## 8. 系統成功指標(工具採用面,非模型指標)

- 連續 4 週每個訓練週期都用 VIX 篩資料(採用率)。
- 「收到新批次 → 決定加入/排除」時間從「不知道」縮到 < 30 分鐘。
- 主動透過 VIX 排除的張數 > 0(代表信任)。
- 負面訊號:使用者繞過 VIX 直接 cp 資料夾。

---

## 9. 開放問題(工作假設,使用者可覆寫)

見 [round-02.md §6]。預設:單人 persona、本機單 GPU、每批數百~數千張、既有腳本吃資料夾+YOLO txt+data.yaml、無 Enterprise 預算。**覆寫任一假設會觸發對應條款重評**(例:多標註者→雙簽提前)。

**已確認約束(2026-05-30,由假設升級為硬約束)**:工廠影像有**保密/法規限制**、**資料不能外流(air-gapped/地端)**、**無 Enterprise 預算**。→ 見 §12。

---

## 10. v0.1 交付清單(可直接開工)

- [x] `vix ingest <folder>`:import → Dataset(tags/fields),寫 `batch_id`/`ingested_at` + manifest。`pipeline.ingest` / `cli`
- [x] `vix embed`:DINOv2 ViT-B/14 逐 bbox embedding + LanceDB kNN index。`FiftyOneAdapter.compute_embeddings/build_knn_index`(⏳ 待離線環境跑)
- [x] `vix infer`:YOLO 推論 → detections。`detect.run_yolo`(⏳ 待真實權重)
- [x] `vix route`:per-class 百分位雙軸門檻 → tag `low_conf`/`far_from_known`/`review`/`pass`。`pipeline.route`(✅ 實測)
- [x] `vix guard`:凍結 anchor + frozen DINOv2 reference 打分 + 自我 gate 自審紀錄。`pipeline.guard`/`core.reference`(✅ 實測)
- [x] `vix export <dst>`:單向匯出 golden → YOLO txt + data.yaml。`core.exporter`(✅ 實測)
- [x] `vix app`:launch FiftyOne App + 預設 SavedViews。`FiftyOneAdapter.launch_app`(⏳ 待離線環境)
- [x] 結構化 manifest + append-only DecisionLog(hash-chain)+ README/TESTING。(✅ 實測)

> **v0.1 已實作並測試(25 passed,Tier 1 零 FiftyOne)**:程式碼 `src/vix/`、測試 `tests/`、實作狀態見 [../spec/v0.1-technical-spec.md §14](../spec/v0.1-technical-spec.md);測試分層見 [../spec/TESTING.md](../spec/TESTING.md)。FiftyOne/YOLO 依賴路徑(Tier 2)為 API-correct,待離線目標機驗證。

---

## 11. 附錄:直接用 FiftyOne vs 自建(Round 3 裁決)

**結論:直接用 FiftyOne,但當「函式庫/SDK」用,並從 v0.1 第一天就用一層薄 `FiftyOneAdapter` 把 VIX 領域邏輯與 FiftyOne 解耦。** 完整辯論見 [round-03-use-vs-build.md](round-03-use-vs-build.md)。

**查證後的關鍵事實**:`fiftyone`(core)與 `fiftyone-brain` **目前皆為 Apache 2.0**(商用自由)。注意:`fiftyone-brain` 早期版本曾為限制性授權,商用時應確認所 pin 版本的 LICENSE。

**為何不自建**:自建等效平台需 3–6 個月;真正難的(互動式 embedding 投影 + lasso、多模態瀏覽)不值得重造。VIX 真正該自己擁有的只是**業務邏輯**,不是基礎設施。

**鎖定怎麼防**(adapter 邊界):
- 直接委託 FiftyOne:`compute_embeddings/similarity`、`fo.Dataset` 工作空間、`launch_app()` 覆核、標註整合。
- 必須是 VIX 獨立 module(可單獨存活):`OutlierScorer`、`ThresholdPolicy`、`FrozenReference`(存 JSON 不靠 fo tag)、`DecisionLog`、`DatasetExporter`。
- `FiftyOneAdapter` 為唯一膠水層,抽換 FiftyOne 只改這一檔。

**何時才該偏向自建**(任一成立則重評):資料不得進第三方資料模型且需 air-gap、UI 必須深度內嵌既有 MES/QC portal、需求極簡到用不上 FiftyOne 9 成功能、團隊已有資料平台。

**Enterprise 付費牆**:OSS = 單人本地工具;多人協作 portal、Dataset versioning/snapshot、RBAC、SSO、審計、air-gap 部署為 Enterprise(無公開報價)。但 VIX 共識已把版控交給 **DVC**、v0.1 單人,故付費牆要到 v1.0 多標註者治理才咬;屆時再評估 Enterprise vs 自建 portal vs adapter-swap。

---

## 12. 附錄:保密/air-gap/無預算約束下的最終裁定(Round 4)

完整查證見 [round-04-airgap-licensing.md](round-04-airgap-licensing.md)。

- **「資料不能外流」≠「必須自建」**:FiftyOne OSS 100% 本機/離線運作(Python + bundled MongoDB + localhost App),預設組態下影像/標註無外流路徑 → **air-gap 可行**。
- **原始碼完整性(親查確認)**:`fiftyone`(含 App)與 `fiftyone-brain` 皆 Apache 2.0,且 repo 含**完整可建置/可稽核實作**(`similarity.py` 62KB、`visualization.py` 37KB、`uniqueness/hardness/mistakenness` 皆實質)。→ 拿得到完整原始碼,可稽核、可 fork。
- **費用 = $0**;Apache 2.0 對已發佈版本不可撤回 → 永久可用、可 fork,無鎖死。
- **air-gap 硬化清單**(v0.1 交付前置):`FIFTYONE_DO_NOT_TRACK=true` 關遙測;離線 pip + 離線模型權重(`FIFTYONE_MODEL_ZOO_DIR`);MongoDB 地端;標註僅用 App 原生 / 自架 CVAT / 自架 Label Studio,**禁用 Labelbox/V7/app.cvat.ai**(會上傳雲)。
- **合規缺口與零預算補法**:版控→DVC(✅);業務審計→`DecisionLog`(升級為**覆核者身分 + DB server 時間戳 + append-only**);存取控制→OS/網路層(Linux 多帳號、Nginx 認證、內網隔離)。
- **真正紅線**(法規若強制):①集中式 SSO/IdP;②法庭級不可竄改 audit log;③per-user 存取稽核 → 需自建 append-only audit store(排 v0.2)或買 Enterprise。多數內部工具不會踩到。
- **結論**:三約束**不推翻**「直接用 FiftyOne + adapter 解耦」共識,反而**強化**它(合規邏輯留 VIX service layer,FiftyOne 只當地端視覺化/查詢引擎)。
