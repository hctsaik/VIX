# Round 15 — 情境 AG1–AG10:**未達標(平均 87.8 < 95),但已逼近**

> 軌跡:R11 78.1 → R12 70.0 → R13 76.2 → R14 81.6 → **R15 87.8**。R14 修補經實機確認全部成立。
> 「VIX 能力內」的情境多在 90+(AG3 匯出往返 95.3、AG1 95.3… 抱歉 AG1 91.7、AG2 91.7),
> 扣分集中在少數真實小缺陷與「誠實限制只在程式碼/docstring、未在 CLI 輸出表面化」。

## 一、評分結果

| 情境 | Judge A | Judge B | Judge C | 平均 |
|------|--------:|--------:|--------:|-----:|
| AG1 三人團隊覆核分流 | 93 | 88 | 94 | 91.7 |
| AG2 季度品質計分卡 | 95 | 90 | 90 | 91.7 |
| AG3 匯出↔ultralytics 往返 | 96 | 93 | 97 | **95.3** |
| AG4 12 個月緩慢季節漂移 | 90 | 79 | 72 | **80.3** |
| AG5 單圖多標籤重疊缺陷 | 88 | 82 | 80 | 83.3 |
| AG6 損毀/截斷影像 | 80 | 86 | 95 | 87.0 |
| AG7 覆核週轉 SLA | 86 | 84 | 90 | 86.7 |
| AG8 對外部資料集去重 | 92 | 88 | 88 | 89.3 |
| AG9 新站點信心門檻 | 91 | 87 | 85 | 87.7 |
| AG10 兩套標註準則 A/B | 87 | 85 | 82 | 84.7 |
| **平均** | **89.8** | **86.2** | **87.3** | **87.8** ❌ |

## 二、共識(實機收斂)

1. **【P0 真缺陷】SPC 預設用「漂移序列本身」估 target/sigma → 緩慢漂移只在最後一點才警報(AG4 最低)**:`spc_monitor` 預設 `target=median(整段)`、`sigma=std(整段)`,把要偵測的漂移吸收進控制界線;實測單調上升 12 批只在 index 11 警報,失去「領先指標」意義。應改用「前段 in-control 基線」估參數,並對短序列(<8 批)加註記。
2. **【P1】損毀影像在 top-level `infer`/`embed` 噴未攔截 `OSError`/`UnidentifiedImageError`(AG6)**:`main()` 只攔 FileNotFoundError/ValueError → 仍噴 traceback;且未指明是哪個檔。
3. **【P1 一致主題】誠實限制只在程式碼/docstring,未在 CLI 輸出表面化(AG5/AG8/AG9/AG10)**:mono-label proxy(`dets[0].label`)、LSH 近似(>2000)、pixel_fallback 較弱、triage 非統計 A/B、parity 為代理且新站需自有 eval set —— 這些都對,但 `compare`/`label-noise`/`dedup`/`parity`/`cost-gate` 的 CLI 沒印出來(`drift-type`/`reviewer-audit` 已有印「注:」的好榜樣)。
4. **【P2】SOP 限制章節缺 SLA / mono-label / 季度比較需同後端 等條目;便利性缺口(review-queue --by、audit --turnaround)。**

## 三、後續方向(本輪落地,然後 Round 16 全新 AH1–AH10)

| 優先 | 補強 | 影響 |
|---|---|---|
| P0 | `spc_monitor` 用前段 in-control 基線估 target/sigma + 短序列 `short_series` 註記 | AG4 |
| P1 | `main()` 一併攔 `OSError`;embed 包裝損毀影像為具名 VIX 錯誤(指出檔案+補救) | AG6 |
| P1 | `compare`/`label-noise`/`dedup`/`parity`/`cost-gate` CLI 印「注:」誠實註記(triage/LSH/backend/需站點 eval) | AG5/8/9/10 |
| P2 | SOP 補 SLA / mono-label / 同後端比較 條目 | AG5/AG7 |

> **狀態:未達標(87.8),但已逼近。** 落地上述(含測試)→ AH1–AH10 重驗。最關鍵是 AG4 的 SPC 基線真缺陷。
