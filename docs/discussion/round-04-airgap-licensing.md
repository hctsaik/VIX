# Round 4 — 約束鎖定:air-gap、原始碼完整性、費用與後顧之憂

> 觸發:使用者給出三個硬約束 ——「工廠影像有保密/法規限制」「沒有新預算」「資料不能外流」,並追問「我有完整的 FiftyOne 原始碼嗎?用它有沒有費用或後顧之憂?」本輪以查證為主(正方驗證可行性 + 紅隊找缺口 + 主持人親查原始碼)。

## 一、最重要的釐清:「資料不能外流」≠「必須自建」

**FiftyOne OSS 設計上 100% 在本機/離線執行**:Python 套件 + bundled MongoDB(`~/.fiftyone/var/lib/mongo`)+ localhost App(`localhost:5151`)。預設組態下,影像/標註/metadata **沒有離開機器的路徑**。→ air-gap 可行,無須因「資料不外流」而放棄 FiftyOne。(來源:FiftyOne config 文件)

## 二、查證結果

### A. 原始碼完整性(主持人親查 GitHub)
使用者問「我有完整的程式碼嗎?」—— 歷史上 `fiftyone-brain` 曾以封裝形式發佈,故必須查證。實查 `voxel51/fiftyone-brain` repo:

| 檔案 | 大小 | 判定 |
|------|------|------|
| `fiftyone/brain/similarity.py` | 61,746 bytes | 實質實作 |
| `fiftyone/brain/visualization.py` | 37,385 bytes | 實質實作 |
| `fiftyone/brain/internal/core/uniqueness.py` | 7,243 bytes | 實質實作 |
| `fiftyone/brain/internal/core/hardness.py` | 4,142 bytes | 實質實作 |
| `fiftyone/brain/internal/core/mistakenness.py` | 19,233 bytes | 實質實作 |

**裁定:`fiftyone`(core,含 App 前端)與 `fiftyone-brain` 皆為 Apache 2.0,且 repo 含完整可建置/可稽核的實作原始碼**(非空殼)。→ 對需要原始碼稽核的法規環境,這點成立。⚠️ 僅就「當前/Apache 2.0 版本」而言;pin 舊版前應確認該版 LICENSE 與原始碼完整度。

### B. air-gap 硬化清單(正方驗證,皆有官方來源)
1. **關閉遙測**(預設只送匿名 UUID 事件、不含影像;air-gap 仍應全關):環境變數 `FIFTYONE_DO_NOT_TRACK=true` 或 `~/.fiftyone/config.json` 設 `{"do_not_track": true}`。
2. **離線安裝**:連網機 `pip download fiftyone -d /wheels/` → 離線機 `pip install --no-index --find-links=/wheels/ fiftyone`。
3. **離線模型權重**:連網機 `foz.download_zoo_model("dinov2-vitl14-torch")` → 複製權重目錄 → 離線機以 `FIFTYONE_MODEL_ZOO_DIR` 指向。
4. **MongoDB 地端**:bundled 即本機;或 `FIFTYONE_DATABASE_URI` 指向自架地端 MongoDB。**不需任何雲端/Atlas**。
5. **標註後端**:✅ FiftyOne App 原生 tagging、自架 CVAT(`FIFTYONE_CVAT_URL=localhost`)、自架 Label Studio(`LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED`,只引用本機路徑不複製影像)。❌ **禁用** Labelbox / V7 / app.cvat.ai(會上傳影像到雲)。

### C. 費用
- **OSS = $0**,無 seat/使用/授權費;Apache 2.0 對**已發佈版本不可撤回**(永遠可用、可 fork)。
- 付費只在你**主動選擇** Enterprise(RBAC/audit log/SSO/snapshot/多人 portal)時才發生 —— 而本案不需要(見 D)。

### D. 紅隊:OSS 在「保密+法規」下真正缺什麼(皆 Enterprise 付費功能)
| 缺口 | 零預算替代 | 是否補上 |
|------|-----------|----------|
| Dataset 版控/snapshot | **DVC + Git**(共識已採;Git commit 可 GPG 簽章,可追溯性更佳) | ✅ 已補 |
| 業務操作審計 | VIX **`DecisionLog`**(每筆路由/覆核決策) | ⚠️ 部分(見下) |
| RBAC / 權限分離 | **OS/網路層**:Linux 多帳號、Nginx + Basic Auth/mTLS、VPN/內網隔離 | ⚠️ 粗粒度可行 |
| SSO / 集中式 IdP | 無原生;Nginx 認證不等於 IdP | ❌ 缺 |
| 不可竄改稽核日誌 | DecisionLog 需加 append-only / WORM 儲存才有法律效力 | ❌ 需自建 |

**真正的紅線**(若法規明確強制):①強制集中式 SSO/IdP;②法庭級不可竄改 audit log;③per-user 資料集存取稽核。此三者若強制 → 要嘛自建那一塊(extend DecisionLog 成 append-only audit store),要嘛買 Enterprise。多數「工廠內部工具」情境不會踩到。

## 三、本輪共識

1. **三約束不推翻先前共識,反而強化「混合路徑 + adapter 解耦」**:合規邏輯(audit/權限)留在 VIX 自己的 service layer(adapter 邊界之內),FiftyOne 只當地端的視覺化/查詢引擎。日後抽換或升級 Enterprise,合規層不必重寫。(紅隊與正方在此交集)
2. **§9 的工作假設「無 Enterprise 預算」「單人/地端」由『假設』升級為『已確認約束』**;新增「air-gap / 資料不外流」為一級約束。
3. **v0.1 增列 air-gap 硬化清單**(本文件 §二.B)為交付前置。
4. **`DecisionLog` 規格升級**:加上「覆核者身分 + DB server 時間戳 + append-only」三屬性,以便在零預算下逼近 audit log 的合規效力。

## 四、待澄清(僅剩法規細節,使用者可答)

- 你的法規是否**強制**:集中式 SSO?法庭級不可竄改 audit log?per-user 存取稽核?
  - 若**否**(多數內部工具):OSS + DVC + append-only DecisionLog + OS 層存取控制 = 零預算合規可行。
  - 若**是**:把「自建 append-only audit store」排進 v0.2,或重新評估預算。

## 五、使用者裁決(2026-05-30,本輪結案)

> **SSO 可先跳過,並非關鍵。** → v0.1 **不需**強制 SSO / 法庭級不可竄改稽核 / per-user 存取稽核。「自建 append-only audit store」**不列入** v0.2 硬需求;`DecisionLog` 仍採 append-only,並**選用**(非必須)逐行 hash-chain 提供近乎零成本的防竄改證據。存取控制交給 OS/網路層。→ 決策層全部收斂,進入 v0.1 技術規格(見 [../spec/v0.1-technical-spec.md](../spec/v0.1-technical-spec.md))。
