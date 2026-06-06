# Round 21 — 情境 AM1–AM10:**91.6 < 95(新高,逼近)**

> 軌跡:…→ 85.9 → 89.6 → 87.8 → 88.9 → **91.6**。R20 的 discoverability 修補奏效:
> Judge A 89.8、B 91.9、C 93.0。多數情境 89–95;剩餘缺口已很小且具體。

## 一、評分結果

| 情境 | A | B | C | 平均 |
|------|--:|--:|--:|-----:|
| AM1 自動產 model card Data 段 | 90 | 93 | 90 | 91.0 |
| AM2 標註者對調兩類名 | 86 | 90 | 93 | 89.7 |
| AM3 凍結成 benchmark | 92 | 92 | 95 | 93.0 |
| AM4 本週覆核率突升 | 84 | 89 | 92 | **88.3** |
| AM5 給標註團隊的逐類清單 | 90 | 94 | 94 | 92.7 |
| AM6 data.yaml 類別順序 | 95 | 95 | 95 | **95.0** |
| AM7(邊界)併入第二家供應商 | 88 | 90 | 90 | 89.3 |
| AM8 從稽核寫事故 postmortem | 91 | 93 | 96 | 93.3 |
| AM9 重 ingest 確認不重複 | 93 | 95 | 95 | 94.3 |
| AM10(邊界)用工時定 AL 預算 | 89 | 88 | 90 | 89.0 |
| **平均** | **89.8** | **91.9** | **93.0** | **91.6** ❌ |

## 二、共識(剩餘小項,具體)

1. **【三評審,AM2】兩類名對調的 relabel 需 `__tmp` 中介**:`relabel --map a=b --map b=a` 若逐序套用會把兩類collapse。→ 改為「以原始標籤一次算出目標標籤」的原子排列(或偵測循環映射時警告)。
2. **【Judge C,跨情境整合】`.hwm` 與帳本同位置可被一併改**:對手有寫權限可同時改 log 與 .hwm。→ 把高水位 count+tip 寫進不可變 snapshot 的 content_hash payload,gate 可跨 snapshot 比對。
3. **【Judge C,AM1】`quality_score` 看起來像指標其實是啟發式**(權重 30/30/20/5 硬編)。→ MD 明確標註「非校準啟發式」+ 列出各分項扣分;強調只有 gate 是強制 go/no-go。
4. **【Judge B,AM1/AM2】`dedup` 把 100+ 雜湊印成一行**,難貼進 card/ticket。→ 預設印「N 群 + 前幾筆」,`--full`/`--json` 給完整。
5. **【Judge B,AM4】離線 dry-run 無 YOLO→無偵測→pipeline 空**(屬已知;可加 `infer --synthetic`)。
6. **【Judge C,AM4】SPC 的 in-control 基線由操作者給**,可能用已漂移視窗估參→誤certify。→(P2)baseline 不穩警告。

## 三、後續方向(本輪落地,然後 Round 22 全新 AN1–AN10)

| 優先 | 補強 | 影響 |
|---|---|---|
| P0 | relabel 改原子排列(swap 一指令即正確) | AM2 |
| P1 | dedup 預設精簡輸出(N 群+前幾筆)+ `--full` | AM1/AM2 |
| P1 | report 標註 quality_score 為非校準啟發式 + 分項 | AM1 |
| P2 | 高水位寫入 snapshot content_hash;gate 跨 snapshot 比對 | AM8 |

> **狀態:未達標(91.6),非常逼近。** 落地上述(含測試)→ AN1–AN10 重驗。
