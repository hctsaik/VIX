# Round 13 — 情境 AE1–AE10:**未達標(平均 76.2 < 95)**,壓力測試操作韌性

> 本輪情境刻意壓力測試「規模、當機復原、並行、機器遷移、golden 腐化」等營運面。
> R11/R12 的修補經實機驗證**全部成立**(export 排除 rejected、跨指令持久化、稽核鏈損毀→NO-GO、verify 偵測注入檔、compare),分數由 R12 的 70.0 回升到 76.2。

## 一、評分結果

| 情境 | Judge A | Judge B | Judge C | 平均 |
|------|--------:|--------:|--------:|-----:|
| AE1 大資料集相信數字 | 78 | 72 | 84 | 78.0 |
| AE2 三個月後回來接手 | 90 | 84 | 86 | 86.7 |
| AE3 GO 但下游 mAP 掉 | 88 | 78 | 88 | 84.7 |
| AE4 ingest 半途當機 | 88 | 80 | 58 | 75.3 |
| AE5 兩人共用 workspace | 70 | 58 | 47 | **58.3** |
| AE6 逐類覆核政策 | 82 | 55 | 70 | **69.0** |
| AE7 給管理層的敘事報告 | 90 | 62 | 82 | 78.0 |
| AE8 遷移到新機器 | 86 | 80 | 64 | 76.7 |
| AE9 稀有但關鍵類別 | 80 | 76 | 73 | 76.3 |
| AE10 golden 本身腐化 | 82 | 74 | 80 | 78.7 |
| **平均** | **83.4** | **71.9** | **73.2** | **76.2** ❌ |

## 二、共識(實機收斂的真實缺口)

1. **【P0】稽核帳本 append 非原子、無鎖、torn-line 會崩潰(AE4/AE5)**:`DecisionLog`/`Manifest` 用裸 append,無 fsync、無鎖。當機或並行寫入留下半行 → `read_all()` 直接 `JSONDecodeError`,使 `verify_chain`/`audit`/`gate` **崩潰**而非優雅降級;尾端截斷則靜默無感。
2. **【P1】逐類門檻覆寫非一級公民(AE6)**:`thresholds.json` 可看可手改,但無 `vix set-threshold <class>` 指令、手改也不進稽核。安全關鍵類別要更嚴,只能手編 JSON。
3. **【P1】給管理層的「依理由分類」彙總缺失(AE7)**:`flag_reason` 有存,但沒有彙總成「低信心 N / 漂移 N / 重複 N」的一頁表,要自己 jq。
4. **【P1】小樣本誠實度(AE9)**:`parity` 無最小樣本門檻,1 筆也照樣判;`calibrate` 對 n_support=2 仍輸出門檻卻無 low_confidence 旗標。
5. **【P1】快照 content_hash 不跨機器(AE8)**:hash 摻入絕對路徑 `ref_snapshot` → 同內容在不同機器得到不同 hash,「同 hash 證明相等」失效。
6. **【P2】誠實限制揭露不足**:單寫者要求、鏈尾截斷弱點、規模下 LSH 近似、golden 是自我參照的信任根需定期人工再驗證 —— 都該明寫進限制章節。

## 三、後續方向(本輪要落地,然後 Round 14 全新 AF1–AF10)

| 優先 | 補強 | 影響 |
|---|---|---|
| P0 | `DecisionLog.read_all` 容忍 torn line + utf-8-sig;append 加檔案鎖 + fsync | AE4/AE5 |
| P1 | 新增 `vix set-threshold <class>`(改 JSON + 記稽核) | AE6 |
| P1 | 新增 `vix reasons`(依 flag_reason 彙總)+ report 區塊 | AE7 |
| P1 | `parity` 加 min_samples→low_confidence;threshold 加 low_confidence 旗標 + calibrate 警告 | AE9 |
| P1 | 快照 content_hash 只雜湊 {golden hashes, 門檻值, anchor_ref_sha256},絕對路徑移為 sidecar | AE8 |
| P2 | SOP 限制章節補:單寫者/鏈尾截斷/LSH 近似/golden 信任根 | AE4/5/8/10 |

> **狀態:未達標(76.2)。** 歷程:R11 78.1 → R12 70.0 → R13 76.2。落地上述補強後產生 AF1–AF10 重驗。
