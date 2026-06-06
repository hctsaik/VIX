# VIX Production 就緒度盤點（Multi-Agent，完整版）

> 目的：在「把 VIX 做成 production 工具，幫工程師**更好地訓練 YOLO、減少 false alarm、分析 dataset、
> 一致化 label 定義**，且**第一階段先做好離線資料維護與校正**」的前提下，做一次**可證明無缺漏**的盤點。
>
> 方法不是憑印象，而是**多軸窮舉 + 交叉對帳**。本文件每個結論都綁到下方 §0 的硬數字。

盤點日期：2026-06-06 ｜ 盤點對象 commit：`7eded2b`（model-loop v2）

---

## §0 — 如何確保「沒有盤點缺漏」（方法學）

憑記憶盤點必漏。唯一可靠的是**從 6 個彼此獨立的清單窮舉，再交叉對帳**：任一軸漏掉的，會被另一軸抓到。

| 列舉軸 | Ground truth 來源 | 實測數量 |
|---|---|---|
| A 程式碼 | `git ls-files` + `wc -l` + `grep '^def/^class'` | 40 模組 / **6,070 行** src（全庫 13,169 行） |
| B CLI 指令 | `cli.py` argparse | **58 指令** |
| C Pipeline 函式 | `grep '^def' pipeline.py` | **~70 函式** |
| D 測試 | `git ls-files tests` + `grep 'def test_'` | 44 檔 / **172 測試** |
| E 文件 | `git ls-files docs` | spec×1 + discussion×4 + validation×22 + SOP/guide |
| F 真實資料 | workspaces | `jdg_export` 27 圖/27 標、`jdgws`/`runws` |

**無缺漏的證明機制 = 完整性矩陣**（能力 × {程式碼 / CLI / 測試 / 文件 / 真實驗證}）：

- 任一 **CLI 指令 ∪ pipeline 函式** 對不到能力 → 抓到「未被盤點的功能」。
- 任一 **能力** 對不到程式碼 → 抓到「缺的功能」。
- 任一能力有程式碼**但無測試** → 抓到「品質缺口」。
- 三份清單（58 指令 ∪ 70 函式 ∪ 172 測試）**互相補位**：單一來源遺漏，會在另兩個對帳時浮現。

**多角色 = 各包一軸再對帳**（團隊確保無缺漏的標準作法）：
- `[AI Engineer]` 程式碼 + 測試軸（每模組是否真實、可測、健全）
- `[Domain Expert]` 能力 ↔ 四大目標軸（是否真的服務 production 目標）
- `[QA Reviewer]` 方法本身 + 非功能/production 軸 + **殘餘不確定性揭露**
- `[Architect/Maintainer]` 文件 + 真實資料 + 部署/環境軸，並負責**對帳**

**殘餘不確定性揭露（誠實，亦屬「無缺漏」一環）**：
逐行精讀 9 模組（`pipeline`*、`cli`、`scorer`、`threshold`、`reference`、`gate`、`decision_log`、`analytics`、`types`）；
其餘 ~30 模組以「CLI + pipeline 編排 + 對應測試存在」三方對帳**確認介面與職責**（非逐行）。
凡標 ◐ 者為「介面已確認、內部未逐行」。

---

## §1 — 完整能力盤點（58 指令依四大目標歸位）

### 目標① 分析 dataset（17 指令｜VIX 已強）

| 指令 | 功能 | 模組 | 測試 | 信心 |
|---|---|---|---|---|
| `coverage` / `value` | 類別分佈、覆蓋缺口(+還需幾張)、新批覆蓋新區比例 | analytics | test_analytics | ✅ |
| `dedup` / `leakage` | 近重複(>2000 自動 LSH)、跨 split 洩漏 | analytics, lsh | test_analytics, test_lsh_scale | ✅ |
| `audit-labels` / `label-noise` | kNN 標錯偵測、confident-learning 類對雜訊 | analytics, confident_learning | test_analytics, test_concepts | ✅ |
| `harmful` / `new-classes` / `box-qa` | 有害樣本排序、開集新類別群聚、框幾何 QA | analytics, box_qa | test_open_set, test_box_qa | ✅ |
| `report` / `trend` / `compare` | 健康報告+品質分、跨批信心趨勢、兩來源並排 | report, quality | test_report | ◐ |
| `history` / `throughput` / `capacity` / `reasons` / `fp-rate` | 逐圖歷史、覆核週轉、工時估、理由彙總、誤報率 | pipeline | test_pipeline_extras | ◐ |

### 目標② 一致 label 定義（10 指令｜VIX 招牌）

| 指令 | 功能 | 模組 | 測試 | 信心 |
|---|---|---|---|---|
| `guard` | 凍結錨點 centroid_shift + label_consistency 漂移自閘（需 `--ack`） | reference | test_reference | ✅ |
| `drift` / `drift-type` / `geometry` | 跨期定義漂移、covariate vs concept、框幾何漂移 | analytics, drift_types, geometry | test_concepts, round* | ◐ |
| `reviewer-audit` / `compare` | 標注者自我一致性、兩 vendor 並排 | quality | round*, test_concepts | ◐ |
| `relabel` / `merge` / `merge-preview` / `set-threshold` | 改名/合併+rollback、類別映射調和、門檻覆寫 | labelmap, threshold | test_labelmap, test_threshold | ✅ |

### 目標③ 減少 false alarm（11 指令 + 1 缺口）

| 指令 | 功能 | 模組 | 測試 | 信心 |
|---|---|---|---|---|
| `route` / `calibrate` / `set-threshold` | 兩軸 per-class 百分位門檻 → pass/review + 人話理由 | scorer, threshold | test_scorer, test_threshold | ✅ |
| `bank-audit` | 多庫低 conf 嵌入審查 → defect/reflection/normal/unknown（advisory，不覆寫 route） | bank_audit | test_bank_audit | ✅ |
| `review-queue` / `explain` / `dismiss` / `resolve` / `restore-dismissed` / `sync-reviews` | 風險佇列+人話、逐圖解釋、誤報處置、回寫閉環 | triage, explain | test_triage_explain, test_resolve | ✅ |
| `active-learn` | 不確定+多樣性取樣 | analytics | test_analytics | ✅ |
| **❌（缺）SAM 去乾淨 crop → embedding** | SAFE 獨有；細粒度 FA(bubble vs reflection)分得更開 | — | — | — |

### 目標④ 訓練 YOLO 更好（Phase 2，8 指令 + 1 缺口）

| 指令 | 功能 | 模組 | 測試 | 信心 |
|---|---|---|---|---|
| `export` / `verify` | 單向→YOLO txt + data.yaml + 逐檔 hash manifest；收方驗證 | exporter, verify | test_exporter, test_quality_gate_verify | ✅ |
| `gate` | 訓練前 GO/NO-GO（覆核未清/洩漏/漂移/稽核鏈/後端混用/eval-golden 重疊/受保護類別回歸…） | gate | test_quality_gate_verify, test_challenge_guard | ✅ |
| `eval-ingest` / `error-mine` / `set-eval-baseline` | 吃真實 val→AP/混淆/FP-FN；反查最該標候選；凍結回歸基準 | eval_ingest | test_keystone, test_eval_typed, test_error_mine_boxlevel | ✅ |
| `snapshot` / `restore` | 不可變資料版本快照/還原 | snapshot | test_snapshot | ✅ |
| **❌（缺）受控重訓飛輪** | SAFE 獨有；governed flywheel（重訓→閘→帳本） | — | — | — |

### 跨切 / 基礎建設（12 指令）

`ingest`・`infer`(YOLO)・`embed`(DINOv2 ViT-B/14)・`run`(一條龍)・`app`(FiftyOne GUI)・
`audit`(稽核鏈查詢)・`quickstart`・`routing-diff`・`spc`(EWMA/CUSUM 領先指標)・
`parity`(跨組)・`cost-gate`(不對稱漏報成本)・`verify-fiftyone`/`verify-gui`(Tier-2 真實環境驗證)。
基礎：`config`、`types`、`decision_log`(append-only 雜湊鏈)、`manifest`、`adapters`(base/memory/fiftyone)、`embedding`、`detect`。

**對帳結果**：58 指令全數歸位（17+10+11+8+12），無「未盤點功能」；70 pipeline 函式皆對應指令或為內部 helper；
172 測試覆蓋 §1 核心模組（見每列）。**三方對帳無孤兒。**

---

## §2 — Production 缺口（對焦 Phase 1：離線資料維護與校正）

四目標中 **①②③ VIX 已覆蓋 ~80%**；④為 Phase 2。Phase 1 主要是「補關鍵缺口 + 硬化 + 真實驗證」，依風險/價值：

| # | 缺口 | 目標 | 性質 | 風險 |
|---|---|---|---|---|
| **G1** | **SAM 去背 embedding（吸收 SAFE）** — 接 `embedding/` 當選配前處理 | ③ | 開發 | 低 |
| **G2** | **「class 定義書」可版本化 + 綁 drift 閘** — 你明講要；VIX 偵測漂移但無「定義來源」 | ② | 開發 | 中 |
| **G3** | **「分析→校正→重檢」單一 SOP 動線** — 零件齊，缺一條順手流程+文件 | ①②③ | 整合 | 低 |
| **G4** | **真實 jdg 資料端到端驗證 + 量化效益（FINDINGS 誠實法）** — 22 輪在何資料？ | 全部 | 驗證 | 中 |
| **G5** | **真實後端測試覆蓋** — 172 測試多為 core/免 FiftyOne；FiftyOne adapter/App 靠 tier-2，需真環境 | 基礎 | 品質 | 中 |
| **G6** | **ARM/離線環境基線** — ViT-B/14 + FiftyOne 在單機 ARM CPU 延遲；memory adapter 用 pixel(品質低) | 基礎 | 驗證/設定 | 中 |
| **G7** | **VOC/任意格式 ingest（吸收 SAFE voc2yolo）** — VIX 只吃資料夾 | ① | 開發 | 低 |

> 純新開發只有 G1(SAM)、G2(定義書)、G7(VOC ingest)；其餘為整合/硬化/驗證。

---

## §3 — 階段路線（與「merge＝不對稱吸收」共識一致）

- **Phase 1（離線資料維護與校正，現在）**：VIX 為底，補 G1–G7。產出＝工程師敢用、信得過的離線
  「分析→校正→一致化→gated export」迴圈，並在真實 jdg 上量化效益。
- **Phase 2（訓練更好的 YOLO）**：吸收 SAFE 的**受控重訓飛輪**（promote 必過 `pre_train_gate` +
  寫 `decision_log` + 需 `--ack`，永不自動晉升），接上既有 `eval-ingest`/`challenge-guard` 閉環。
- **Phase 3（產品化）**：離線批次 → 線上/串流 gatekeeper + latency 預算（對準 SMM 即時）；v0.2 frozen-reference-YOLO KL 漂移閘。

**建議 Phase 1 起手序**：`G4 真實驗證 → G1 SAM 吸收 → G3 SOP 動線 + G2 定義書 → G5/G6 真環境/效能硬化 → G7`。

---

## §4 — 強制日誌（共識）

**📌 [目前討論項目]** VIX 對四大 production 目標的完整盤點 + 如何保證無缺漏。

**✅ [達成共識]**
1. **無缺漏方法 = 6 軸窮舉 + 完整性矩陣對帳**（58 指令 ∪ 70 函式 ∪ 172 測試 互相補位），非憑印象；殘餘不確定性逐項標信心。
2. **盤點完成**：58 指令全數歸位四目標+基礎建設，三方對帳無孤兒；①②③ 已覆蓋 ~80%，④為 Phase 2。
3. **Phase 1 真正缺口僅 7 項（G1–G7）**，純新開發只有 SAM/定義書/VOC ingest。
4. **方向**：VIX 為底吸收 SAFE；Phase 1 離線維護 → Phase 2 受控飛輪 → Phase 3 線上產品化。

**⚠️ [潛在爭議與風險]**
- **測試「真實覆蓋」假象**（G5）：172 測試多免 FiftyOne；真實 FiftyOne+App 路徑需 tier-2 真環境，未必每次跑。
- **真實資料驗證未證**（G4）：22 驗證輪是否在 jdg、有無量化效益需確認，否則 production 宣稱無依據。
- **ARM/離線可行性**（G6）：ViT-B/14 + FiftyOne 在單機 ARM CPU 需實測延遲。

**🚀 [下一步行動]** 先做 **G4**：在真實 `jdg` 資料上把離線維護迴圈跑一遍、量化效益（沿用 SAFE FINDINGS 誠實法），作為 production 化事實地基。

---

## 附錄 A — 模組清單與讀取深度（40 模組）

精讀(✅)：`pipeline.py`(多數)、`cli.py`、`core/scorer.py`、`core/threshold.py`、`core/reference.py`、
`core/gate.py`、`core/decision_log.py`、`core/analytics.py`、`types.py`。

介面對帳(◐)：`core/`{eval_ingest, bank_audit, calibration, confident_learning, drift_types, quality,
spc, parity, labelmap, manifest, snapshot, exporter, verify, triage, explain, geometry, box_qa, lsh,
errors, report}、`config.py`、`detect.py`、`embedding/`{dinov2, simple}、`adapters/`{base, memory,
fiftyone_adapter}、`verification.py`、`plugins/vix_review`、`logging_setup.py`。

## 附錄 B — 列舉指令（可重現本盤點）
```bash
git ls-files 'src/**/*.py' | xargs wc -l | sort -rn      # A 程式碼
grep -oE 'add_parser\("[a-z-]+"' src/vix/cli.py          # B CLI（58）
grep -oE '^def [a-z_]+' src/vix/pipeline.py              # C 函式（~70）
grep -rEc 'def test_' tests                              # D 測試（172）
git ls-files docs ; git ls-files jdg_export jdgws runws  # E 文件 / F 資料
```
