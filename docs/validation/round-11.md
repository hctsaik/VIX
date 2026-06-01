# Round 11 — 情境 AC1–AC10:**未達標(平均 78.1 < 95)**,三評審揪出真實缺口

> 本輪是針對「目前已完整建好(含 SOP、47 指令)的系統」做的**全新、刻意嚴格**的再驗證。
> 與前 10 輪不同:三位評審被明確要求「不得給情面分、必須對照原始碼驗證能力是否真的存在且可行動」。
> 結果分數大幅低於前一輪(R10=95.25),正說明**前輪可能偏寬鬆**,而本輪暴露了真正的工程缺口 —— 這是好事。

## 一、本輪情境(AC1–AC10,多代理設計,全新框架)

| # | 情境 | 重點探測 |
|---|------|----------|
| AC1 | 新人冷啟動(無人帶) | onboarding 自足性、`--adapter memory` 乾跑、錯誤→排解對應 |
| AC2 | 向質疑的主管辯護單一張被攔影像 | `explain`/`history`/`fp-rate` 的可解釋與信任 |
| AC3 | 已出貨的錯標批次回滾 | `audit`/`history`/`relabel --rollback`/`snapshot` 補救鏈 |
| AC4 | 兩家標註供應商二選一 | `label-noise`/`audit-labels`/`reviewer-audit`/`dedup`/`leakage` 可比較性 |
| AC5 | 模型回歸根因追到資料 | `drift-type`(covariate vs concept)/`trend`/`geometry` |
| AC6 | 週期性重建基準(anchor 過時?) | `guard --ack`/重新校準/`new-classes`/帳本一致性 |
| AC7 | 專案中途分類法變更(拆/併類) | `merge-preview`/`relabel`/`coverage`/`export` |
| AC8 | 向稽核員證明合規 | `audit` 篩選/雜湊鏈/`verify` 清單比對/`snapshot` |
| AC9 | 死線前分流 5 萬張 | 一鍵 `run`/`review-queue`/`harmful`/`cost-gate`/`gate` |
| AC10 | 上月新增的資料到底有沒有幫助 | `value`/`coverage` 前後/`dedup`/資料歸因 |

## 二、評分結果

| 情境 | Judge A(原始碼) | Judge B(工作流) | Judge C(完整性) | 平均 |
|------|--------:|--------:|--------:|-----:|
| AC1 新人冷啟動 | 72 | 62 | 62 | **65.3** |
| AC2 辯護被攔影像 | 90 | 58 | 84 | 77.3 |
| AC3 錯標批次回滾 | 88 | 78 | 86 | 84.0 |
| AC4 供應商二選一 | 70 | 70 | 70 | **70.0** |
| AC5 回歸根因 | 90 | 75 | 80 | 81.7 |
| AC6 重建基準 | 80 | 72 | 78 | 76.7 |
| AC7 分類法變更 | 82 | 80 | 81 | 81.0 |
| AC8 證明合規 | 95 | 82 | 83 | 86.7 |
| AC9 5萬張分流 | 80 | 76 | 82 | 79.3 |
| AC10 資料有沒有幫助 | 85 | 74 | 79 | 79.3 |
| **平均** | **83.2** | **72.7** | **78.5** | **78.1** ❌ |

## 三、共識(三評審收斂,即「要修什麼」)

1. **`vix embed` 獨立執行直接崩潰(真 bug)** — `cli.py` 呼叫 `adapter.compute_embeddings()` 沒帶 `model_key`,但簽章要求該參數 → `TypeError`。SOP/quickstart 第 3 步必掛。**三位評審一致點名。**
2. **記憶體 adapter 不跨指令保存狀態** — `InMemoryAdapter` 是純行程內(`self._s={}`),偵測/embedding/路由欄位在不同 `vix` 指令間遺失。導致 `explain`/`review-queue`/`coverage`/`value`/`harmful` 在 `--adapter memory` 乾跑下回空或「sample not found」。只有單行程的 `vix run` 正常。**直接拖累 AC1/AC2/AC9/AC10。**
3. **AC4 供應商比較是手工的** — 指標都在,但沒有單一「並排比較」指令,要自己拼表。
4. **`verify` 匯出完整性有破口** —(a)注入額外檔仍回 `ok=True`(沒檢查「只應有清單內的檔」);(b)清單以 basename 為鍵,不同子目錄同名檔會碰撞而漏檢。
5. **`label-noise` 名為 confident-learning 但不純** — 用 kNN 多數票當偽標籤 + YOLO 自身信心當 CL 信心,confident-joint 有循環性;應改用獨立來源(校準後的 YOLO 類別後驗)。

## 四、爭議 / 取捨

- **記憶體 adapter 該不該持久化?** Judge B/C 認為 SOP 把 `--adapter memory` 當「無 FiftyOne 學習逐步工作流」的途徑,就該跨指令可用;但這原設計只為單元測試/單行程乾跑。**結論:採 Judge 觀點 —— 為乾跑體驗加檔案持久化**(對應 SOP 承諾)。
- **drift / value 的門檻、半徑是固定啟發值**(非統計檢定):Judge C 視為「rigor 不足」,A/B 視為合理工程取捨。**結論:本輪不改演算法本質,但在文件/輸出標明為可調啟發參數**(避免過度宣稱)。
- Console 中文亂碼:確認為 Windows code page 假象,原始檔為合法 UTF-8,**不扣分、不處理**。

## 五、後續方向(本輪要落地的補強,然後跑 Round 12 全新 10 情境重驗)

| 優先 | 補強 | 影響情境 |
|---|---|---|
| P0 | 修 `vix embed` CLI 傳 `model_key`(真 bug) | AC1 |
| P0 | `InMemoryAdapter` 檔案持久化(跨指令保存偵測/embedding/路由) | AC1/AC2/AC9/AC10 |
| P1 | `verify` 完整性:偵測未預期額外檔 + 改用相對路徑為鍵 | AC8 |
| P1 | 新增 `vix compare --tag-a/--tag-b` 並排比較(noise/dup/leakage/一致性) | AC4 |
| P2 | `label-noise` 改用校準後 YOLO 後驗作獨立信號(降低循環性) | AC4 |
| P2 | `value` 同時輸出 redundant%、`explain` 列出最近 golden 鄰居 id | AC10/AC2 |

> **狀態:未達標(78.1)。** 依規則:落地上述補強(含測試)→ 產生全新 10 情境(AD1–AD10)→ 三評審重驗 → 直到平均 ≥ 95。
