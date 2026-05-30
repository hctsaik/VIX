# Round 3 — 情境 U1–U10:再進階重評

## 一、本輪討論項目
補強 Round 2 缺口(labelmap/errors/lsh/triage/explain,52 pytest 綠)後,用**再進階的全新 10 情境** U1–U10 重評,聚焦時間序列、split、整合閘、開放集等更深的資料治理面向。

## 二、10 情境(U1–U10)
U1 開放集未知類別預警|U2 覆核者自我一致性|U3 train/val 洩漏|U4 每週差異摘要|U5 有害樣本Top-N排行|U6 新成員導覽報告|U7 訓練前放行閘(go/no-go)|U8 匯出內容雜湊可驗證|U9 單張下鑽解釋+敏感度|U10 類別難度逐週漂移趨勢

## 三、評分結果

| 情境 | Judge A | Judge B | 平均 | 主要缺口 |
|------|--------:|--------:|-----:|----------|
| U1 開放集未知類別 | 45 | 28 | 36.5 | far_from_known 隱含,無「疑似新類別聚類+命名」輸出 |
| U2 覆核者自我一致性 | 15 | 5 | 10.0 | **未做**:同人對相似樣本相反決定偵測 |
| U3 train/val 洩漏 | 62 | 18 | 40.0 | 有 dedup 但無 split 維度、無跨 split 洩漏報告 |
| U4 每週差異摘要 | 72 | 30 | 51.0 | report diff 僅類別/總數,缺 dup/review/標錯 delta |
| U5 有害樣本排行 | 68 | 55 | 61.5 | review_queue 有,但未併入重複信號、非單一指令 |
| U6 新成員導覽 | 78 | 22 | 50.0 | health report 接近,缺 batch/版本/子集摘要+首步建議 |
| U7 訓練前放行閘 | 55 | 35 | 45.0 | **未整合**:回寫完整性+golden/train重疊+go/no-go |
| U8 內容雜湊驗證 | 70 | 62 | 66.0 | snapshot 有 content_hash,缺匯出附逐檔 manifest + verify 指令 |
| U9 單張下鑽+敏感度 | 72 | 42 | 57.0 | explain 有句子,缺各軸數值下鑽 + 敏感度 |
| U10 類別難度漂移 | 22 | 20 | 21.0 | cross_period_drift 是 embedding,缺逐週合格率/信心時序 |
| **平均** | **55.9** | **31.7** | **43.8** | — |

## 四、共識
- 這批情境比前兩輪更難(整合性、時間序列、split),系統雖有「原料」但缺「明確、整合、可驗收的單一指令 + 測試」。
- 評審 B 的核心要求很清楚:**每個情境要有一個明確函式/指令給出可稽核結論,並有 pytest**,而非讓使用者自行拼湊原料。
- 兩大共通缺口:**時間序列維度**(batch/週)、**整合閘/報告**(把零件組成一個 go/no-go 或一頁摘要)。

## 五、爭議
- 兩評審落差大(A 55.9 vs B 31.7):A 認可「原料齊可拼湊」,B 堅持「需單一可驗收指令+測試才算」。**採 B 的嚴格標準**作為改進方向(可驗收性是 P2/P3 原則本身),這樣收斂的系統更紮實。

## 六、後續方向(改進清單,皆 pure-core、可本機 pytest)
| 優先 | 能力 | 解掉 | 模組 |
|------|------|------|------|
| 1 | 時間/ split 維度:EmbItem 帶 batch/split(由 tags 解析) | U3/U4/U10 基礎 | analytics |
| 2 | 開放集 suspected_new_classes(離群子集聚類命名) | U1 | analytics |
| 3 | reviewer 自我一致性(相似樣本相反決定 + 一致率) | U2 | `core/quality.py` |
| 4 | cross_split_leakage(跨 split 重複報告) | U3 | analytics |
| 5 | harmful_ranking(標錯+重複+離群整合 Top-N) | U5 | analytics |
| 6 | 類別品質逐週趨勢 + 顯著下滑 | U10/U4 | `core/quality.py` |
| 7 | pre_train_gate(回寫完整性+golden/train重疊+分布健康→go/no-go) | U7 | `core/gate.py` |
| 8 | 匯出逐檔 hash manifest + verify | U8 | `core/verify.py` |
| 9 | explain_image(各軸數值下鑽 + 敏感度) | U9 | analytics/explain |
| 10 | onboarding/週報強化(batch 跨度+版本+子集+首步) | U6/U4 | report |

> **決策**:平均 43.8 ≪ 95 → 不停。實作上述 10 項(含測試、CLI),Round 4 產生**全新 10 情境**重評,目標平均 ≥ 95。
