# Round 4 — 情境 V1–V10:整合/穩健性重評

## 一、本輪討論項目
補強 Round 3 缺口(開放集/洩漏/有害/趨勢/覆核者/閘/驗證/下鑽,61 pytest 綠)後,用**整合與穩健性導向的全新 10 情境** V1–V10 重評。

## 二、10 情境(V1–V10)
V1 週循環一條龍|V2 空/單張/單類別韌性|V3 同圖跨三月冪等+歷史|V4 極端門檻可解釋+警示+重路由|V5 八週後重建放行依據|V6 誤報風暴收斂|V7 類別合併後重校+前後差異+rollback|V8 外部交接hash驗收+稽核對齊|V9 冷啟動到首份可訓練集|V10 PM 品質報告

## 三、評分結果

| 情境 | Judge A | Judge B | 平均 | 主要缺口 |
|------|--------:|--------:|-----:|----------|
| V1 一條龍 | 72 | 62 | 67.0 | 無單一 orchestrator,需手動串 8+ 指令 |
| V2 邊界韌性 | 68 | 72 | 70.0 | 多數 n<2 已 guard,缺統一結構化「跳過」訊息 |
| V3 跨月冪等+歷史 | 82 | 78 | 80.0 | 冪等強;但只留首次,無 per-image 多次送入歷史 |
| V4 極端門檻 | 63 | 58 | 60.5 | 可重路由不重推論,但無門檻警示+前後差異報告 |
| V5 八週重建 | 78 | 70 | 74.0 | snapshot 只存 meta 非完整門檻值;無 batch 範圍篩 |
| V6 誤報風暴 | 62 | 48 | 55.0 | 無「略過同類/誤報回饋」收斂機制+FP 率 |
| V7 合併後重校 | 70 | 75 | 72.5 | rollback 僅 pure function 無 CLI;無路由前後 diff |
| V8 外部交接 | 75 | 82 | 78.5 | 強;export_manifest 只含圖,未含 labels/data.yaml |
| V9 冷啟動可訓練 | 60 | 60 | 60.0 | 無「還需 X 張」量化;無嚮導 |
| V10 PM 報告 | 45 | 47 | 46.0 | 無單一品質分數;report/gate 分離;無圖表 |
| **平均** | **67.5** | **65.2** | **66.4** | — |

## 四、共識
- 單點能力幾乎完備(V3/V8 達 80);拖分的是**整合(orchestrator)、可用性(品質分數/報告一體化)、與少數回饋機制(誤報收斂、還需X張)**。
- 兩評審一致點名三大缺口:**(a) 一條龍 + 還需X張;(b) labels 納入驗證 + rollback CLI;(c) 單一品質分數 + report/gate 合併**。

## 五、爭議
- 兩評審分數接近(67.5 vs 65.2),無方向性爭議,缺口高度一致。

## 六、後續方向(改進清單,皆 pure-core、可本機 pytest)
| 優先 | 能力 | 解掉 | 模組 |
|------|------|------|------|
| 1 | 單一品質分數 0–100 + report 內嵌 gate 結論 | V10 | report |
| 2 | `run` 一條龍 orchestrator(fail-fast+每步留痕) | V1/V9 | pipeline/CLI |
| 3 | 覆蓋「還需 X 張」量化 + gate 可設每類最低 | V9 | analytics/gate |
| 4 | per-image 送入歷史(每次 ingest 記 log)+ history 查詢 | V3 | pipeline/CLI |
| 5 | 門檻極端警示(flag/pass 率)+ routing 前後 diff | V4 | pipeline |
| 6 | 誤報 dismiss + FP 率追蹤,review_queue 排除已 dismiss | V6 | pipeline |
| 7 | rollback CLI + 合併後重校重跑便利 | V7 | CLI |
| 8 | export_manifest 納入 labels/*.txt + data.yaml | V8 | verify/exporter |
| 9 | snapshot 凍結完整 thresholds 值 + batch 範圍 | V5 | snapshot |

> **決策**:平均 66.4 < 95 → 不停。實作上述 9 項(含測試、CLI),Round 5 產生**全新 10 情境**重評,目標平均 ≥ 95。
