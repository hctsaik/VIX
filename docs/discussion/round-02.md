# Round 2 — 收斂:衝突點對撞

> 階段:Convergent。主持人把 Round 1 的 7 個衝突整理成一份「分期調和提案」(以產品 D 的分期框架為總結構,把各角色的高級項目分配到正確階段),交給四個角色逐項表態。

## 一、本輪討論方向

驗證「分期調和提案」是否能讓四方同時收斂:
- 總框架:**v0.1 可見性 / v0.2 可追溯 / v1.0 可制度化**。
- 逐條解 C1–C7(見 round-01.md 衝突表)。

## 二、四角色的收斂表態

| 角色 | 結論 | 提出的條件(binding) |
|------|------|----------------------|
| A 演算法 | **有條件是** | (1) v0.1 末就要**凍結 reference embedding snapshot** 當未來 KS 漂移基準(比較邏輯可延 v0.2,但快照不能延);(2) kNN 的 embedding 來源**鎖定 DINOv2 Model Zoo,不可用 YOLO 中間層特徵**;(3) v0.1 百分位門檻的分位基礎要講清楚(訓練集 embedding 分布);(4) v0.2 末加一個 Cartography 相容性 PoC 檢查點 |
| B 架構 | **有條件是** | (1) export 轉接器必須是**單向**(讀 tags→產生資料夾/txt/yaml),**禁止反向**把資料夾結構寫回 Dataset 狀態;(2) v0.1 ingest **第一行就以 Dataset tags/fields 為唯一事實來源**,禁止「先搬檔、v0.2 再遷移」(沉沒成本陷阱) |
| C 治理 | **有條件是** | (1) reference 打分超閾值(KL>0.15 或一致率↓>5%)→ 當次模型更新**暫停並留書面自審紀錄**(單人自我 gate,非雙簽);(2) 凍結 golden test set 物理隔離**第一天建立、永不參與訓練/調參**;(3) v0.1 的 txt manifest 要**結構化強制**(hash/來源/日期/標註版本),為 v0.2 遷移 DVC 留乾淨歷史 |
| D 產品 | **有條件是** | (1) **reference model = frozen pretrained DINOv2 權重、不另外訓練**(否則 2 週變 6 週);(2) 紅線:v0.1 **不得有 API 層/服務化/多人並發寫入**;(3) per-class 百分位門檻的「誰算、存哪、怎麼更新」要寫進 README(三行字) |

**四方一致接受(無異議)**:FiftyOne 為事實來源、kNN 取代 centroid(C3 偽衝突消解)、FastAPI 延 v0.2、Conformal 延 v1.0、DVC 延 v0.2、雙簽延 v1.0、export 轉接器化解 C2。

## 三、唯一需要主持人裁決的張力

**「reference model」一詞,C 與 D 的理解不同:**
- C 原意:一個**凍結的 YOLO 任務模型**(用 anchor 初訓、永不更新),比較「現役 YOLO vs 凍結 YOLO」的預測信心分布(KL)。
- D 限制:v0.1 **不准訓練任何新模型**(會爆時程)。

**主持人裁決(納入共識)**:採「分期化的 reference」——
- **v0.1**:reference = **frozen pretrained DINOv2 embedding 空間**(零訓練)。閉環防護訊號改為**在凍結 DINOv2 空間中,比較新批次相對 frozen anchor 的 embedding 分布位移 / anchor 類別的 kNN 標籤一致率**。超閾值→單人自我 gate + 書面自審。此設計同時滿足:D(零訓練)、A(凍結 embedding snapshot 即此物,一物兩用當 KS 基準)、C(從第一天就有不被污染的外部參照點)。
- **v0.2**:當 DVC + 訓練基礎設施就緒,才加入 C 原意的**凍結 reference YOLO 任務模型**做預測分布 KL gate。

> 這個裁決把四個條件全部變成相容:A 的「凍結 embedding snapshot」與 D 的「frozen pretrained DINOv2」其實是同一個物件;C 的「外部參照」在 v0.1 由它承擔,完整版(任務模型 KL)順延 v0.2。

## 四、本輪決策(全部鎖定,寫入 consensus.md)

- **D-A**:採分期框架 v0.1/v0.2/v1.0 為總結構。
- **D-B**:v0.1 即以 FiftyOne Dataset tags/fields 為唯一事實來源;export 轉接器單向、零改動接回既有腳本。
- **D-C**:kNN(`compute_similarity`, cosine, 原始 DINOv2 空間)為路由相似度;UMAP 僅顯示。
- **D-D**:v0.1 保留三項便宜防護(凍結 test set / frozen anchor / frozen DINOv2 reference 打分 + 單人自我 gate)。
- **D-E**:門檻 v0.1 百分位 → v0.2 calibrated(temperature scaling) → v1.0 conformal。
- **D-F**:版控 v0.1 結構化 txt manifest+git → v0.2 DVC+W&B。
- **D-G**:reference 採分期化裁決(v0.1 embedding-based、v0.2 才加凍結 YOLO)。
- **D-H**:v0.1 紅線——無 API/服務化/多人並發寫入;無自訂 Operator/Panel(用 App 原生 UI)。

## 五、共識達成判定

四角色皆為**「有條件是」**,且所有條件**經第三節裁決後彼此相容、無互斥**。→ **判定:已達成多代理人共識**,進入 [consensus.md](consensus.md) 鎖定最終規格。所有 binding 條件原文納入共識文件作為附帶約束。

## 六、留給使用者的開放問題(不阻塞共識,採工作假設先行)

共識建立在以下**工作假設**上;使用者可隨時覆寫,覆寫會觸發對應條款重評:
1. persona:單人(同時是工程師與標註者)→ 若為多標註者團隊,雙簽/κ re-audit 提前至 v0.1–v0.2。
2. 本機單 GPU、每批數百~數千張 → 若 >5–10 萬 patch,向量後端由 sklearn 換 LanceDB(A 已備案)。
3. 既有訓練腳本吃「資料夾 + YOLO txt + data.yaml」→ 決定 export 轉接器的輸出格式。
4. 無 FiftyOne Enterprise 預算 → 版控走 DVC(已鎖定)。
5. 漂移定義含兩面:類內分散變大(intra-class variance↑)與跨類界線模糊(類間距離↓),v0.2 KS 監控兩者皆量。
