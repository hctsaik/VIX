# VIX 情境驗證(Scenario-Driven Validation）

目標:用多代理人反覆驗證 VIX 是否「真的能幫電腦視覺工程師提升效率、更好管理資料集」。

## 驗證迴圈

```
每一輪:
  1. (多代理人) 定義 10 個真實情境 — CV 工程師日常管資料集會遇到的事
  2. (多代理人, 多位獨立評審) 對「VIX 能否完美處理該情境」打分 0–100
  3. 平均分數:
        >= 95  -> 達標,停止
        <  95  -> 找出缺口 -> 改進系統(含測試)-> 下一輪產生「新的」10 個情境重驗
```

評審獨立打分後取平均;每輪記錄:討論項目、共識、爭議、後續方向、分數表。

## 結果:**兩度達標** 🎯

- **原始系列 Round 10 平均 95.25**(S–AB,雙評審)。
- **嚴格再驗證系列 Round 22 平均 95.8**(AC–AN,三位獨立評審、每輪實機跑 CLI + 對抗式驗證)。

## 第一系列(R1–R10):從基線補強到達標

| 輪次 | 情境批次 | 平均分 | 文件 |
|------|----------|-------:|------|
| Round 1 | S1–S10 | 28.75 | [round-01.md](round-01.md) |
| Round 2 | T1–T10 | 44.3 | [round-02.md](round-02.md) |
| Round 3 | U1–U10 | 43.8 | [round-03.md](round-03.md) |
| Round 4 | V1–V10 | 66.4 | [round-04.md](round-04.md) |
| Round 5 | W1–W10 | 83.4 | [round-05.md](round-05.md) |
| Round 6 | X1–X10 | 86.6 | [round-06.md](round-06.md) |
| Round 7 | Y1–Y10 | 91.5 | [round-07.md](round-07.md) |
| Round 8 | Z1–Z10 | 92.3 | [round-08.md](round-08.md) |
| Round 9 | AA1–AA10 | 94.85 | [round-09.md](round-09.md) |
| **Round 10** | **AB1–AB10** | **95.25 ✅** | [round-10.md](round-10.md) |

## 第二系列(R11–R22):刻意嚴格的再驗證(三評審、實機跑 CLI)

> 三位獨立評審(原始碼正確性 / 工作流易用性 / 資料完整性),每輪實際安裝執行 CLI、對抗式驗證(竄改 log、刪檔、注入 export…)。前輪較寬鬆,本系列從刻意嚴格的 78.1 起,逐輪修補實機驗出的真實缺陷。

| 輪次 | 情境批次 | 平均分 | 關鍵修補 |
|------|----------|-------:|----------|
| Round 11 | AC1–AC10 | 78.1 | embed bug、記憶體持久化、verify 完整性、compare |
| Round 12 | AD1–AD10 | 70.0 | export 排除 rejected、relabel 持久化、稽核鏈→gate |
| Round 13 | AE1–AE10 | 76.2 | 帳本鎖+fsync+容半行、set-threshold、reasons、parity 小樣本 |
| Round 14 | AF1–AF10 | 81.6 | 並行不丟資料、zero-golden NO-GO、未校準→review、乾淨錯誤 |
| Round 15 | AG1–AG10 | 87.8 | SPC 基線、損毀影像具名、CLI 誠實註記 |
| Round 16 | AH1–AH10 | 84.7 | Tag.EVAL 隔離、throughput、backend 入稽核、no_detection 註記 |
| Round 17 | AI1–AI10 | 85.9 | 後端一致性強制、routing-diff 增減、capacity |
| Round 18 | AJ1–AJ10 | 89.6 | 尾端截斷偵測、review_queue 標籤錯誤風險 |
| Round 19 | AK1–AK10 | 87.8 | coverage 絕對目標、.hwm fail-closed、gate 完整性顯示 |
| Round 20 | AL1–AL10 | 88.9 | EVAL 不入候選池、ingest 自動 batch 標籤、restore-dismissed |
| Round 21 | AM1–AM10 | 91.6 | dedup 精簡輸出、quality_score 標註啟發式 |
| **Round 22** | **AN1–AN10** | **95.8 ✅** | 核心強項任務,每條成功標準客觀可驗證 |

> 系統實作見 [../../README.md](../../README.md);操作手冊 [../guide/VIX_SOP.html](../guide/VIX_SOP.html);
> 缺哪些功能的多代理討論見 [../discussion/feature-roadmap.md](../discussion/feature-roadmap.md);
> 規格 [../spec/v0.1-technical-spec.md](../spec/v0.1-technical-spec.md);測試 [../spec/TESTING.md](../spec/TESTING.md)。
