# Round 7 — 情境 Y1–Y10:收斂驗證

## 一、評分結果(補強 Round 6 收尾批次後,77 pytest 綠)

| 情境 | Judge A | Judge B | 平均 | 主要缺口 |
|------|--------:|--------:|-----:|----------|
| Y1 新人一指令健檢 | 92 | 93 | 92.5 | explain 需多一步 |
| Y2 交付 go/no-go+稽核 | 93 | 95 | 94.0 | gate 未自動印 hash-chain verify |
| Y3 每週自動 diff+幾何漂移 | 88 | 91 | 89.5 | diff 僅類別層級;排程需外部 cron |
| Y4 label-map 合併+預覽 | 90 | 92 | 91.0 | merge-preview 需手動 JSON counts |
| Y5 快照 --apply 還原 | 91 | 94 | 92.5 | before/after diff 無獨立輸出 |
| Y6 active learning 選樣 | 89 | 90 | 89.5 | CLI 只印 id,無逐項理由 |
| Y7 有害一步清除+留痕 | 91 | 96 | 93.5 | 移除後未自動更新 report |
| Y8 air-gap 離線 | 94 | 88 | 91.0 | 無「偵測環境自動切換」 |
| Y9 開放集未知類別 | 88 | 91 | 89.5 | 未印處置建議 |
| Y10 季度一條龍全驗收 | 91 | 92 | 91.5 | run 未含 leakage/drift、未自動 build reference、結尾無 verify |
| **平均** | **90.7** | **92.2** | **91.5** | — |

## 二、共識
- 系統已「好用且能落地」(全部 10 情境 ≥ 88,Y2/Y7 達 93+)。剩餘純為**輸出豐富度與一條龍自動化**的最後打磨,無功能空白。
- 兩評審一致點名:**(1) active-learn 附理由;(2) run 自動 build reference + 含 leakage/drift + 結尾 hash-chain verify;(3) gate 印 verify 結果;(4) new-classes 處置建議;(5) auto adapter 缺 FiftyOne 時自動降級 memory。**

## 三、爭議
- 僅 Y8(A 94 vs B 88):B 要求「偵測環境自動切換」。採納,make_adapter auto 在缺 FiftyOne 時自動降級 memory + 標記。

## 四、後續方向(最終打磨,皆可本機 pytest)
| 優先 | 能力 | 解掉 |
|------|------|------|
| 1 | active_learning 回傳逐項理由(uncertainty/novelty)+ CLI 印出 | Y6 |
| 2 | run:自動 build reference + 加 leakage/drift 檢查 + 結尾 verify_chain | Y10/Y2 |
| 3 | gate CLI 印 hash-chain verify 結果 | Y2 |
| 4 | new-classes 每群附處置建議 | Y9 |
| 5 | make_adapter auto 缺 FiftyOne → 自動降級 memory + 後端標記 | Y1/Y8 |

> **決策**:平均 91.5 < 95 → 不停。實作最終打磨(含測試),Round 8 產生**全新 10 情境**重評,目標平均 ≥ 95。
