# Round 20 — 情境 AL1–AL10:**88.9 < 95**(回升;正確性 92.8)

> 軌跡:…→ 89.6 → 87.8 → **88.9**。Judge C(資料完整性)給 **92.8**,確認保證多為「真的強制」
> (尾端截斷/eval 重疊/後端混用/匯出完整性都翻真實 verdict+exit code,非僅印警告)。
> Judge A/B(86/87.8)集中在 discoverability 小缺口。

## 一、評分結果

| 情境 | Judge A | Judge B | Judge C | 平均 |
|------|--------:|--------:|--------:|-----:|
| AL1 每週覆核 burndown | 90 | 90 | 93 | 91.0 |
| AL2 是否淘汰舊批次 | 88 | 84 | 88 | 86.7 |
| AL3 攝影機角度多樣性 | 85 | 90 | 95 | 90.0 |
| AL4 帶非 CLI 同事用 App | 86 | 88 | 94 | 89.3 |
| AL5 「漏小物件」資料佐證 | 84 | 93 | 92 | 89.7 |
| AL6 驗外包是否只標範圍內類別 | 87 | 88 | 90 | 88.3 |
| AL7(邊界)golden 是否過時 | 90 | 92 | 96 | 92.7 |
| AL8 golden 自上次發布的差異 | 82 | 82 | 95 | 86.3 |
| AL9(邊界)真重複 vs 有用難例 | 88 | 80 | 91 | 86.3 |
| AL10 大跑前 calibration 健檢 | 80 | 91 | 94 | 88.3 |
| **平均** | **86.0** | **87.8** | **92.8** | **88.9** ❌ |

## 二、共識(具體可修,多為 discoverability)

1. **【Judge A,影響 AL3/AL5/AL6】EVAL 洩漏進候選池**:`review_queue`/`active_learn`/`new_classes` 排除 GOLDEN/ANCHOR/REJECTED 但**沒排除 EVAL** → held-out eval 樣本一再以候選出現(route/gate 已正確視為 held-out)。→ 三處 exclude_tags 各加 `Tag.EVAL`。
2. **【Judge B,影響 AL2/AL3】`ingest --batch X` 不打 `batch:X` 樣本標籤**:`compare`/`drift-type`/`parity`/`geometry` 在沒手動加標籤時回全 0,且無「matched 0」提示。→ ingest 自動加 `batch:<id>` 標籤;這些指令對 0 命中印提示。
3. **【Judge B,AL1/AL8】audit 易用性**:`audit --event resolve` 回 0(resolve 記成 `review`);`--since <本地時間>` 因 log 是 UTC 而少匹配。→ 事件名別名(resolve→review、remove→harmful_remove);0 命中時印時區提示。
4. **【Judge C,AL2/AL9】"removed" 誇大了「只是打 REJECTED 標籤」且無還原指令**。→ 新增 `vix restore-dismissed <ids>`(去標+稽核 undismiss);輸出改「excluded N(可用 restore-dismissed 還原,已記稽核)」。
5. **【Judge A,AL10】calibrate-confidence 無最小樣本警告**(6 列也照算,ECE 可能變差誤導)。→ n<~50 印警告。
6. **【Judge C,AL7/AL10】`.hwm` 與帳本同位置(對手可一併改);backend_mixed 忽略未蓋章紀錄**。→(P2)把高水位也寫進不可變 snapshot;未蓋章 embedding 事件視為 unknown 後端。

## 三、後續方向(本輪落地,然後 Round 21 全新 AM1–AM10)

| 優先 | 補強 | 影響 |
|---|---|---|
| P0 | review_queue/active_learn/new_classes 排除 Tag.EVAL | AL3/AL5/AL6 |
| P0 | ingest 自動加 batch:<id> 標籤;drift/compare/parity 對 0 命中提示 | AL2/AL3 |
| P1 | audit 事件名別名 + 0 命中時區提示;新增 restore-dismissed + 改字 | AL1/AL8/AL2/AL9 |
| P1 | calibrate-confidence 最小樣本警告 | AL10 |
| P2 | 高水位寫入 snapshot;backend_mixed 計入未蓋章 | AL7/AL10 |

> **狀態:未達標(88.9),持續逼近。** 落地上述(含測試)→ AM1–AM10 重驗。
