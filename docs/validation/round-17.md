# Round 17 — 情境 AI1–AI10:**未達標(逐情境平均 ≈ 85.9 < 95)**

> 軌跡:78.1 → 70.0 → 76.2 → 81.6 → 87.8 → 84.7 → **85.9**。R16 修補經實機確認全部成立。
> (Judge A 本輪僅評 AI1–AI7 平均 91.4;Judge B 82.9、Judge C 86.1。下表逐情境取有評者平均。)

## 一、評分結果(逐情境)

| 情境 | Judge A | Judge B | Judge C | 平均 |
|------|--------:|--------:|--------:|-----:|
| AI1 門檻敏感度 | 93 | 86 | 89 | 89.3 |
| AI2 只靠 git 重建 workspace | 95 | 84 | 86 | 88.3 |
| AI3 同圖兩批次標籤衝突 | 90 | 90 | 90 | 90.0 |
| AI4 5 分鐘向總監示範 | 92 | 83 | 88 | 87.7 |
| AI5 rejected 影像保留政策 | 92 | 85 | 87 | 88.0 |
| AI6 半數資料用錯 embedding 模型 | 88 | 80 | 78 | **82.0** |
| AI7 匯出能否在 ultralytics 載入 | 90 | 82 | 91 | 87.7 |
| AI8 golden 是否都經人工確認 | — | 81 | 84 | 82.5 |
| AI9 規劃下季標註人力 | — | 80 | 85 | 82.5 |
| AI10 類別定義被悄悄改 | — | 78 | 83 | **80.5** |
| **平均** | (7項)91.4 | **82.9** | **86.1** | **85.9** ❌ |

## 二、共識(本輪主題:logged-but-not-enforced + 多指令摩擦)

1. **【整合 P0,AI6 最低之一】後端一致性「有記錄但未強制」**:`embedding_backend` 已寫進 route/export 稽核,但 `thresholds.json` meta 沒存校準時的後端 → 用 DINOv2 校準、卻用 pixel_fallback route 不會被擋。→ calibrate 把後端寫入 policy.meta;route/gate 比對 cfg 後端不符就警告 + NO-GO;gate 偵測稽核中混用後端→NO-GO。
2. **【AI8/AI2】reviewer 身分與帳本尾端未強制**:`reviewer_id` 預設通用字串、無身分綁定;鏈尾截斷仍偵測不到。→(較大)snapshot 寫入 chain 長度+tip 作錨點,verify 可比對偵測截斷;review 事件要求非預設 reviewer_id。
3. **【Judge B,AI1/AI9/AI2】期望「單一動詞」卻要串多指令**:敏感度掃描、人力估算、重建都要手動串。→ 加薄殼 `vix sweep`、`vix capacity`(只是串既有階段+印表)。
4. **【AI4】cp950 主控台中文亂碼影響示範**:report 檔本身 UTF-8 正確,但終端 screen-share 看起來壞。→ run/report 末行印 ASCII-safe 一行摘要。
5. **【AI3/AI6】真相只在文件、未在動作當下提示**:重複雜湊但不同標籤被丟棄、混用後端 —— 沒在 ingest/gate 當下印。→ ingest 跳過時提示「標籤未採用」。
6. **【Judge C】細節**:`routing-diff` 只比兩次都有的 id(漏新增/消失);concept drift 對「保持類別計數的重定義」盲(label-marginal JS);throughput p90 無最小樣本守門。

## 三、後續方向(本輪落地,然後 Round 18 全新 AJ1–AJ10)

| 優先 | 補強 | 影響 |
|---|---|---|
| P0 | calibrate 記後端入 policy.meta;route/gate 後端不符→警告+NO-GO;gate 偵測混用後端 | AI6 |
| P1 | ingest 跳過時印「標籤未採用,用 resolve --label 更正」;run/report 印 ASCII 末行 | AI3/AI4 |
| P1 | `routing-diff` 納入新增/消失的 id | AI1 |
| P1 | 薄殼 `vix sweep`、`vix capacity`(串既有階段) | AI1/AI9 |
| P2 | snapshot chain 錨點偵測尾端截斷;concept drift 加 kNN 純度檢查;throughput p90 最小樣本 | AI2/AI8/AI10 |

> **狀態:未達標(85.9)。** 落地上述(含測試)→ AJ1–AJ10 重驗。
