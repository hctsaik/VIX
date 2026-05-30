# Round 6 — 情境 X1–X10:最終驗證

## 一、本輪討論項目
補強 Round 5(自動快照 diff/guard→gate/embedding 後端標記/bbox 幾何漂移/合併預覽,73 pytest 綠)後,用**最終 10 情境** X1–X10 驗證系統是否真的「好用、能落地」。

## 二、評分結果

| 情境 | Judge A | Judge B | 平均 | 主要缺口 |
|------|--------:|--------:|-----:|----------|
| X1 一指令能否訓練 | 92 | 92 | 92.0 | 幾乎完備 |
| X2 一條龍清理+報告 | 88 | 90 | 89.0 | run 未自動含 ingest+infer |
| X3 可重現+未竄改 | 90 | 93 | 91.5 | 完備 |
| X4 最少人工收斂 | 85 | 88 | 86.5 | 無批次 dismiss 互動 |
| X5 離線全流程+標記 | 88 | 72 | 80.0 | 缺 pixel_fallback 的 run e2e 測試 |
| X6 合併分佈預覽 | 90 | 82 | 86.0 | preview 函式有但未掛 CLI |
| X7 框尺寸幾何漂移 | 92 | 90 | 91.0 | 完備 |
| X8 新人5分鐘掌握 | 78 | 68 | 73.0 | **無 QUICKSTART/概念說明/指令分群(最低分)** |
| X9 有害樣本+留痕 | 88 | 88 | 88.0 | harmful 與移除未一步 |
| X10 重現八週前版本 | 87 | 91 | 89.0 | restore 僅回傳結構,無 --apply 重播 |
| **平均** | **87.8** | **85.4** | **86.6** | — |

## 三、共識
- 系統已高度可用(7/10 達 88+,X1/X3/X7 達 91+)。剩餘純粹是**CLI 掛接 + 文件 + 少數一步化**,非功能缺口。
- 兩評審一致點名最該補:**X8 新人 QUICKSTART(73 最低)、X6 merge-preview 掛 CLI、X5 離線 run e2e 測試、X10 restore --apply、X2 run 含 ingest/infer、X9 harmful 一步移除**。

## 四、爭議
- 僅 X5(A 88 vs B 72)落差:B 嚴格要求「Tier-1 真實 pixel_fallback 的 run 端到端測試」才算離線路徑可驗收。採 B 標準,補該測試。

## 五、後續方向(收尾批次,皆可本機 pytest)
| 優先 | 能力 | 解掉 | 模組 |
|------|------|------|------|
| 1 | QUICKSTART.md + 概念詞彙 + `vix quickstart` 指令 + help 分群 | X8 | docs/cli |
| 2 | `merge-preview` CLI(讀兩份 counts) | X6 | cli |
| 3 | `run` 支援 --input/--batch/--weights 先 ingest+infer | X2 | pipeline/cli |
| 4 | `restore --apply` 依 composition 重播進 adapter | X10 | pipeline/cli |
| 5 | `harmful --remove --top N`(一步標記+留痕) | X9 | pipeline/cli |
| 6 | pixel_fallback 的 run 端到端 Tier-1 測試 | X5 | tests |

> **決策**:平均 86.6 < 95 → 不停。實作收尾批次(含測試),Round 7 產生**全新 10 情境**重評,目標平均 ≥ 95。
