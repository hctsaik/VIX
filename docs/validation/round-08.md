# Round 8 — 情境 Z1–Z10:最終收斂驗證

## 一、評分結果(最終打磨後,80 pytest 綠)

| 情境 | Judge A | Judge B | 平均 | 主要缺口 |
|------|--------:|--------:|-----:|----------|
| Z1 新人一指令全流程 | 95 | 96 | 95.5 | YOLO infer 離線跳過 |
| Z2 合併前分佈預覽+洩漏 | 88 | 91 | 89.5 | merge-preview 需手填 JSON counts;leakage 無 split 時靜默 |
| Z3 每週漂移監控 | 90 | 90 | 90.0 | 幾何漂移僅比 mean;排程需外部 cron |
| Z4 有害一步清除+稽核 | 95 | 95 | 95.0 | harm 權重固定 |
| Z5 air-gap 凍結+還原 | 92 | 94 | 93.0 | snapshot 未凍結 anchor 向量 |
| Z6 active learning 選50+理由 | 93 | 93 | 93.0 | 理由為數值,非白話句 |
| Z7 覆核者一致性稽核 | 87 | 88 | 87.5 | **無 --class 過濾;reviewer_id 規範;CLI smoke 缺** |
| Z8 開放集+處置建議 | 90 | 92 | 91.0 | 建議固定字串;novelty_radius CLI 不可調 |
| Z9 增量冪等重跑 | 95 | 95 | 95.0 | routing prev 只留最後兩輪 |
| Z10 單張白話下鑽 | 94 | 93 | 93.5 | 未 calibrate 時 sensitivity 消失 |
| **平均** | **91.9** | **92.7** | **92.3** | — |

## 二、共識
- **生產級水準**:Z1/Z4/Z9 達 95,全部 ≥ 87.5。兩評審皆評為「生產就緒」。剩餘純為**介面參數暴露 + 輸出豐富度 + 少數整合細節**,無核心演算法缺口。
- 一致點名:**(1) reviewer-audit --class 過濾 + CLI 測試;(2) merge-preview 直接吃 dataset(免手填 JSON);(3) 幾何漂移加分佈比較;(4) new-classes CLI 參數 + 差異化建議;(5) active-learn 白話句;(6) explain 未校準提示;(7) snapshot 凍結 anchor。**

## 三、爭議
- 無方向性爭議,兩評審分數高度一致(91.9 vs 92.7)。

## 四、後續方向(針對最低分的精準補強,皆可本機 pytest)
| 優先 | 能力 | 解掉 |
|------|------|------|
| 1 | reviewer_consistency + --class 過濾 + CLI 測試 | Z7(最低) |
| 2 | merge-preview 直接從 adapter 兩個 tag 子集算 counts | Z2 |
| 3 | new-classes CLI 參數(--novelty-radius/--cluster-distance)+ 依群大小差異化建議 | Z8 |
| 4 | 幾何漂移加 p05/p95 分佈比較 | Z3 |
| 5 | active-learn 每筆附白話句 | Z6 |
| 6 | explain_image 未校準時給提示 | Z10 |
| 7 | snapshot 凍結 anchor_ref hash | Z5 |

> **決策**:平均 92.3 < 95 → 不停。實作精準補強(含測試),Round 9 產生**全新 10 情境**重評,目標平均 ≥ 95。
