# Round 16 — 情境 AH1–AH10:**未達標(平均 84.7 < 95)**,本輪偏誠實邊界探測

> 軌跡:R11 78.1 → R12 70.0 → R13 76.2 → R14 81.6 → R15 87.8 → **R16 84.7**(小幅回落)。
> 本輪 10 情境有 6 個刻意探測「誠實限制」(無 model-eval、無 SLA、無 3-way 合併…),
> Judge B(易用性)對「誠實但需多指令手動拼」扣分較重。R15 修補經實機確認成立。

## 一、評分結果

| 情境 | Judge A | Judge B | Judge C | 平均 |
|------|--------:|--------:|--------:|-----:|
| AH1 證明重訓勝舊模型 | 92 | 84 | 91 | 89.0 |
| AH2 隔離 eval set 不入訓練 | 80 | 82 | 78 | **80.0** |
| AH3 本週最高 ROI 行動 | 85 | 80 | 86 | 83.7 |
| AH4 零偵測:真背景 vs 漏報 | 90 | 83 | 80 | 84.3 |
| AH5 單圖標籤編輯史 | 88 | 88 | 90 | 88.7 |
| AH6 退役一個類別 | 93 | 79 | 88 | 86.7 |
| AH7 審外部協作者 PR | 90 | 81 | 90 | 87.0 |
| AH8 估覆核積壓工時 | 90 | 74 | 84 | 82.7 |
| AH9 GPU 壞→信任 pixel_fallback | 91 | 85 | 76 | 84.0 |
| AH10 合併三個 workspace | 86 | 73 | 85 | 81.3 |
| **平均** | **88.5** | **80.9** | **84.8** | **84.7** ❌ |

## 二、共識(可落地的具體缺口)

1. **【真 bug】Windows BOM JSON 讀取失敗**:`merge`/`merge-preview` 用 `read_text("utf-8")`,PowerShell `Set-Content -Encoding utf8` 產生的 BOM JSON 會報錯(雖被乾淨攔截 exit 2,但合法檔失敗)。→ 改 `utf-8-sig`。
2. **【整合缺口 AH2,三評審】無一級 `eval` 標籤,與 `anchor` 混用**:anchor 是「漂移凍結參考」,被當 held-out eval set 是概念硬湊;且無機制阻止 eval 樣本後來被標 golden 而洩漏。→ 新增 `Tag.EVAL`,於 calibrate/route/export/build_reference 一律硬排除;ingest 時 golden∧eval 互斥;gate 對 eval∩golden 重疊回 NO-GO。
3. **【AH9】後端未寫入逐筆稽核**:`pixel_fallback` 只進 report,未進每筆 route/export 的 decision log → 事後逐筆稽核查不到是哪個後端。→ 把 `embedding_backend` 寫入 route/export 的 extra。
4. **【AH4】零偵測「真背景 vs 漏報無法區分」未在 CLI 印出**:route 已 fail-safe 送 review,但該邊界未表面化。→ route/reasons 印註記。
5. **【AH3】report 建議未依量級排序、未點名首要行動**。→ 依 count 排序 + 標「本週首要」。
6. **【AH8 最弱之一】無工時/週轉估算**:counts 都有但要自己算。→ 新增 `vix throughput`(配對 route→resolve、印 median/p90 週轉、用開放佇列數×速率給「約 N 人時」估計,明示為估計)。
7. **【AH6/AH10 友善度】退役類別、合併 workspace 為多指令手動串**:可選加薄殼 orchestrator(本輪先補 SOP 條目)。

## 三、後續方向(本輪落地,然後 Round 17 全新 AI1–AI10)

| 優先 | 補強 | 影響 |
|---|---|---|
| P0 | merge/merge-preview JSON 改 utf-8-sig(Windows BOM) | 真 bug |
| P0 | `Tag.EVAL` 一級隔離 + 互斥 + gate 重疊 NO-GO | AH2 |
| P1 | route/export 稽核寫入 embedding_backend | AH9 |
| P1 | route/reasons 印 no_detection 邊界註記 | AH4 |
| P1 | report 建議依量級排序+點名首要;新增 `vix throughput` | AH3/AH8 |
| P2 | SOP 補:eval≠anchor、三條 hash-chain 不可串接 | AH2/AH10 |

> **狀態:未達標(84.7)。** 落地上述(含測試)→ AI1–AI10 重驗。
