# Round 12 — 情境 AD1–AD10:**未達標(平均 70.0 < 95)**,實機測試揪出更深的真實 bug

> 本輪三位評審**實際在目標 Windows 機器上逐指令跑 CLI**(非只讀碼),因此比 R11 更嚴苛,
> 分數反而下降到 70.0 —— 暴露了「讀碼看不出、實跑才會炸」的真實缺陷。這正是要的結果。

## 一、評分結果

| 情境 | Judge A | Judge B | Judge C | 平均 |
|------|--------:|--------:|--------:|-----:|
| AD1 多廠一致性爭議 | 68 | 58 | 74 | 66.7 |
| AD2 重現上季訓練集 | 92 | 82 | 80 | 84.7 |
| AD3 橡皮圖章覆核員 | 70 | 52 | 70 | 64.0 |
| AD4 預算下選下一批 500 張 | 78 | 74 | 72 | 74.7 |
| AD5 攝影機悄悄換型 | 66 | 70 | 74 | 70.0 |
| AD6 乾淨交付外部團隊 | 95 | 84 | 92 | **90.3** |
| AD7 兩類其實是同一類 | 80 | 40 | 76 | 65.3 |
| AD8 可證明的 PII 移除 | 38 | 76 | 38 | **50.7** |
| AD9 接入 CI 擋壞批次 | 74 | 72 | 66 | 70.7 |
| AD10 量化標註者分歧 | 70 | 48 | 70 | 62.7 |
| **平均** | **73.1** | **65.6** | **71.2** | **70.0** ❌ |

## 二、共識(實機驗證收斂的真實 bug — 即「要修什麼」)

1. **【P0 嚴重】PII/有害移除沒在匯出端生效(AD8,三評審最低分)**:`dismiss`/`harmful --remove` 只加 `rejected` 標籤,但 `export` 走 `get_by_tag(GOLDEN)` **從不排除 rejected** → 被「移除」的 golden 樣本仍會匯出、`verify` 還回 ok=True。「匯出+verify 證明已移除」根本是反的。
2. **【P0 平台】cp950 主控台崩潰(AD1/AD7)**:`parity`/`merge`/`compare` 印 `⚠️` emoji,在你機器的預設 cp950 console 直接 `UnicodeEncodeError` exit 1,只有手動設 `PYTHONUTF8=1` 才正常 —— 旗艦功能在實機硬失敗。
3. **【P0】`relabel`/`--rollback` 在 `--adapter memory` 下是 no-op(AD7)**:relabel 改了記憶體內 detection 物件卻沒呼叫 `adapter.set_detections` 持久化 → 跨指令「改了 20 筆」但 coverage 不變、rollback 0 筆。R11 的持久化修補反而曝光了這個本來就斷的鏈。
4. **【P1】CI gate 無視損毀的稽核鏈(AD9)**:竄改 decision log 後 `gate` 印 `hash-chain verified: False`,卻仍 GO/exit 0。稽核鏈壞掉應直接 NO-GO。
5. **【P1】量化人的情境輸出太薄(AD3/AD10)**:`reviewer-audit` 對所有人都回 `consistency=1.00, conflicts=0`、無決策筆數、無「樣本太少無法判定」註記;`fp-rate` 只算 `dismiss` 事件、忽略 `resolve --false-alarm` → 兩條誤報路徑對不起來。
6. **【P1】`run` 摘要與 CLI 輸出漏關鍵欄位**:`run` 那行漏印 `leakage` 與 `audit_verified`;`active-learn` 的逐筆白話 `why` 在 pipeline 算了卻沒在 CLI 印出來。
7. **【P2】`restore --apply` 不重算 content_hash**(直接沿用快照存的值),分歧的重播不會被抓到;`label_noise`/`parity` 用 proxy(偵測信心、平均信心)而非嚴格量(預測類後驗、真實 CR + 顯著性)。
8. **【P2】缺「誠實限制聲明」**:AD3/AD5/AD8 明確以「系統是否誠實說出自己的極限」評分,但輸出/文件都沒有。

## 三、爭議 / 取捨
- AD6(乾淨交付)拿 90.3,證明 R11 的 `verify` 完整性 + 相對路徑修補**確實成立**且強。
- Judge B 對 AD7/AD10 給特別低分(40/48),因為實跑下功能沒作用;A/C 偏向「機制存在」給中段 —— **採 B 觀點:跨指令實跑不成立就是缺陷**,relabel 持久化列 P0。
- proxy 估計(label_noise/parity)爭議:列 P2,先在輸出/文件標明為「triage 啟發、非統計保證」,不重寫演算法本質。

## 四、後續方向(本輪要落地,然後跑 Round 13 全新 AE1–AE10 重驗)

| 優先 | 補強 | 影響 |
|---|---|---|
| P0 | `export` 排除 `rejected`(+回歸測試:dismiss golden→匯出/清單皆不見) | AD8 |
| P0 | `main()` 開頭 `sys.stdout.reconfigure(utf-8, errors=replace)`,移除 emoji 崩潰 | AD1/AD7/全部中文輸出 |
| P0 | `relabel`/`rollback` 改完呼叫 `adapter.set_detections` 持久化 | AD7 |
| P1 | 稽核鏈損毀 → `gate` NO-GO / 非零 exit | AD9 |
| P1 | `reviewer-audit` 加 `n_decisions` + 低樣本旗標;`fp-rate` 納入 resolve false_alarm | AD3/AD10 |
| P1 | `run` 摘要印 leakage+audit_verified+pixel_fallback;`active-learn` CLI 印 why | AD9/AD4 |
| P2 | `restore --apply` 重算 content_hash 並比對;輸出/文件加「誠實限制」註記 | AD2/AD3/AD5/AD8 |

> **狀態:未達標(70.0)。** 落地上述補強(含測試)→ 產生全新 AE1–AE10 → 三評審重驗,直到平均 ≥ 95。
