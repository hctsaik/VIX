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

## 結果:**達標** 🎯 — Round 10 平均 **95.25 ≥ 95**

經 10 輪、100 個情境、每輪雙評審獨立打分,系統從基線 28.75 一路補強到 95.25,滿足「平均 ≥ 95」的目標門檻。

## 輪次索引

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

> 系統實作見 [../../README.md](../../README.md)、[../../QUICKSTART.md](../../QUICKSTART.md);規格見 [../spec/v0.1-technical-spec.md](../spec/v0.1-technical-spec.md);測試見 [../spec/TESTING.md](../spec/TESTING.md)。
