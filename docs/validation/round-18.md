# Round 18 — 情境 AJ1–AJ10:**89.6 < 95(最高分,逼近)**

> 軌跡:78.1 → 70.0 → 76.2 → 81.6 → 87.8 → 84.7 → 85.9 → **89.6**。本輪為「公平、貼近 CV 工程師真實日常」套組
> (8 個例行任務 + 2 個誠實邊界)。三評審實機跑 CLI 確認:gate exit code、稽核鏈竄改偵測、export 完整性、
> 後端混用 NO-GO、誠實空狀態,皆成立。

## 一、評分結果

| 情境 | Judge A | Judge B | Judge C | 平均 |
|------|--------:|--------:|--------:|-----:|
| AJ1 週一「能不能合併」把關 | 94 | 94 | 92 | 93.3 |
| AJ2 找修最糟 10 張 | 90 | 90 | 84 | 88.0 |
| AJ3 乾淨 train/val 切分 | 90 | 93 | 90 | 91.0 |
| AJ4 本月策展有無改善品質 | 88 | 92 | 86 | 88.7 |
| AJ5 匯入公開瑕疵資料集 | 86 | 90 | 88 | 88.0 |
| AJ6 60 秒站立會健康概況 | 93 | 95 | 80 | 89.3 |
| AJ7 五批先處理哪一批 | 85 | 88 | 85 | **86.0** |
| AJ8(邊界)修壞掉的 routing | 95 | 90 | 90 | 91.7 |
| AJ9 重現同事結果 | 92 | 93 | 91 | 92.0 |
| AJ10(邊界)relabel campaign 檢查 | 90 | 91 | 82 | 87.7 |
| **平均** | **90.3** | **91.6** | **86.8** | **89.6** ❌ |

## 二、共識(剩餘的小缺口,多為具體可修)

1. **【真缺陷 Judge C,AJ2/AJ6】統一 review_queue 漏接 label-error 訊號**:`pipeline.review_queue` 呼叫 `_review_queue` 時沒傳 `label_issue_ids` → 文件宣稱的標籤錯誤風險權重(0.2)與 `suspected_label_error` 理由在正式佇列中從不觸發(harmful() 有傳)。→ 算一次 label_issue_imgs 傳進去。
2. **【整合 Judge C,跨情境】鏈尾截斷仍只揭露未監控**:`audit_chain_intact` 把「截斷後較短但合法」當完好→GO。→ snapshot 寫入單調遞增的 chain 長度/last-hash 高水位 sidecar,gate 比對、回退即 NO-GO(關掉這個自 R13 以來反覆被點名的殘留漏洞)。
3. **【Judge C,AJ10】relabel 未在輸出表面化「一致≠語意正確」**:caveat 只在註解。→ relabel CLI 印一行註記。
4. **【Judge A,AJ4】report 週差缺品質分數 delta 行**。→ 加「品質分數變化: prev → cur (Δ)」。
5. **【Judge B】ergonomics**:`report <dst>` 把 .md 當目錄;`merge-preview --counts-a '<json字串>'` 在 PowerShell 被當檔名。→ counts 參數先試 json.loads 再當檔案;report 印 wrote→ 路徑。
6. **【Judge A/B,AJ7/AJ5】缺 `vix triage` 一指令排序多批;pixel_fallback 乾跑 ranking 空洞**(需真 DINOv2 才有料,屬已知後端取捨)。

## 三、後續方向(本輪落地,然後 Round 19 全新 AK1–AK10)

| 優先 | 補強 | 影響 |
|---|---|---|
| P0 | review_queue 傳入 label_issue_ids(讓標籤錯誤風險真的生效) | AJ2/AJ6 |
| P0 | snapshot 寫 chain 高水位;gate 偵測尾端截斷→NO-GO | AJ1/AJ9 跨情境 |
| P1 | relabel CLI 印「一致≠語意正確」註記;report 加品質 delta 行 | AJ10/AJ4 |
| P1 | merge-preview counts 參數容忍 JSON 字串或檔案路徑 | ergonomics |
| P2 | `vix triage --tags a,b,c` 一指令排序多批 | AJ7 |

> **狀態:未達標(89.6),但逼近。** 落地上述(含測試)→ AK1–AK10 重驗。
