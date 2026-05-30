# VIX 多代理人落地討論紀錄

本資料夾記錄「如何把 **VIX (Vision Integrity eXplainability)** 概念落地」的多代理人(multi-agent)反覆討論過程,直到達成共識為止。可能基於 [voxel51/fiftyone](https://github.com/voxel51/fiftyone) 做開發。

## 討論方法

每一輪由數個不同角色的 AI 代理人(演算法、系統架構、MLOps/治理、產品/MVP)獨立提出立場,再由主導 AI 綜整:**討論方向 → 重點 → 決策 → 共識 → 下一輪待澄清**。發散(divergent)→ 收斂(convergent)→ 共識(consensus)。

## 角色

| 代號 | 角色 | 關注焦點 |
|------|------|----------|
| A | Data-Centric ML 演算法 | embedding、OOD/距離度量、label error、cartography、conformal、drift |
| B | 系統架構師(FiftyOne 整合) | build-vs-reuse 邊界、資料模型、UI、部署拓樸 |
| C | MLOps / 資料治理 | 凍結評估集、資料版控、閉環安全、漂移治理、可重現 ablation |
| D | 產品 / 務實落地 | 分期、MVP 範圍、刪減清單、效益與風險 |

## 輪次索引

| 輪次 | 階段 | 文件 | 狀態 |
|------|------|------|------|
| Round 1 | 發散:各角色獨立立場 | [round-01.md](round-01.md) | ✅ 完成 |
| Round 2 | 收斂:衝突點對撞 | [round-02.md](round-02.md) | ✅ 完成 |
| Round 3 | 抉擇:直接用 FiftyOne vs 自建 | [round-03-use-vs-build.md](round-03-use-vs-build.md) | ✅ 完成 |
| Round 4 | 約束鎖定:air-gap / 原始碼 / 費用 | [round-04-airgap-licensing.md](round-04-airgap-licensing.md) | ✅ 完成 |
| 共識 | 四角色「有條件是」+ use-vs-build + 約束鎖定 | [consensus.md](consensus.md) | ✅ 已鎖定 |

> **結果**:經 2 輪(發散→收斂),四個角色達成**有條件共識**,9 條 binding 條件已彼此相容並納入 [consensus.md](consensus.md)。核心結論:VIX 是建立在 FiftyOne 上的 **Data-Centric AI 資料守門員**,以分期(v0.1 可見性 / v0.2 可追溯 / v1.0 可制度化)控制複雜度,以**凍結參照點**防止閉環放大模型偏差。

> 最終人機協作歷程(給同事看的說明)請見 [../collaboration-journey.html](../collaboration-journey.html)
> 拿著能開工的實作規格請見 [../spec/v0.1-technical-spec.md](../spec/v0.1-technical-spec.md)
