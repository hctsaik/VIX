# Round 19 — 情境 AK1–AK10:**87.8 < 95**(被一個真實 bug 拖累)

> 軌跡:…→ 85.9 → 89.6 → **87.8**。本輪同為公平日常套組。多數情境 88–92;
> 主要拖累是 AK4 的一個真實缺陷(coverage --target 對均衡但低於絕對目標的類別不報缺額)。

## 一、評分結果

| 情境 | Judge A | Judge B | Judge C | 平均 |
|------|--------:|--------:|--------:|-----:|
| AK1 夜間自動策展 cron | 92 | 90 | 89 | 90.3 |
| AK2 這張為何一直被攔 | 95 | 86 | 90 | 90.3 |
| AK3 當天單標籤熱修+重匯出 | 90 | 91 | 92 | 91.0 |
| AK4 本季類別平衡 vs 目標 | 55 | 80 | 88 | **74.3** |
| AK5 信任前先驗權重 | 85 | 84 | 87 | 85.3 |
| AK6 單類別子集匯出 | 95 | 93 | 88 | 92.0 |
| AK7(邊界)新領域預設參數 | 90 | 88 | 90 | 89.3 |
| AK8 兩人 golden 是否一致 | 88 | 89 | 86 | 87.7 |
| AK9 從 flag 萃取標註準則 | 90 | 85 | 88 | 87.7 |
| AK10(邊界)專案長期封存 | 93 | 92 | 84 | 89.7 |
| **平均** | **87.3** | **87.8** | **88.2** | **87.8** ❌ |

## 二、共識(具體可修)

1. **【真 bug,AK4 最低】`coverage --target` 對均衡但低於目標的類別不報缺額**:`coverage_gaps` 用 `under = count < 0.5×median`,類別均衡時即使全部低於絕對 target 也回 need:0。→ 傳了 target 時改 `under = count < target`、`need = max(0, ceil(target)-count)`,median 只當 target=None 的預設。
2. **【新引入的小整合,Judge C,AK10】`.hwm` 尾端截斷錨點可被竄改**:刪掉 `.hwm` 或偽造 count 即可讓 `is_truncated()` 回 False。→ fail-closed:非空帳本但 `.hwm` 缺失/不可解析→視為可疑(truncated);docstring 註明 `.hwm` 防誤刪非防有寫入權限的對手;並把高水位也寫進不可變 snapshot。
3. **【顯示,Judge A/B,AK10】gate 在截斷時仍印「audit hash-chain verified: True」**,與 NO-GO 並列像矛盾。→ 改印「鏈結完整;尾端錨點: FAIL(較 .hwm 少 N 筆)」。
4. **【Judge B,AK2】`resolve` 記為 event `review`,`audit --event resolve` 查無**。→ 別名/提示。
5. **【Judge C,AK8】merge-preview 的 <5% 是慣例非強制**。→ 回傳 flagged 清單(超過 max_delta 的類別)。
6. **【Judge B,AK4/AK5/AK9】離線 pixel_fallback 乾跑無偵測→ranking 空洞**(屬已知後端取捨,需 Tier-2 真權重)。

## 三、後續方向(本輪落地,然後 Round 20 全新 AL1–AL10)

| 優先 | 補強 | 影響 |
|---|---|---|
| P0 | coverage_gaps:傳 target 時用絕對目標算 under/need | AK4 |
| P0 | `.hwm` fail-closed(缺失/壞→可疑)+ docstring 註記 | AK10 |
| P1 | gate 截斷時改印「尾端錨點 FAIL」而非 verified True;resolve audit 別名/提示 | AK10/AK2 |
| P1 | merge_preview 回傳 flagged(超過 max_delta) | AK8 |

> **狀態:未達標(87.8)。** 落地上述(含測試)→ AL1–AL10 重驗。
