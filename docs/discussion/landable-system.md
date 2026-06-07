# 什麼系統能「真的落地」並幫 CV 工程師更有效率地管理資料集

> /goal:用 multi-agent 思考到底什麼系統能真的落地、幫 CV 工程師增加效率、更好管理資料集。每輪記錄討論項目/共識/爭議/後續方向;有共識後開發;再用 multi-agent 定義 10 情境打分,未達平均 95 就 +10 反覆驗證。
> 標的系統 = 現有的 VIX(c:/code/claude/VIX):離線、零訓練的 CV 資料策展 + 模型弱點診斷工具。

---

## Round 1 — 五個 lens 的發散(grounded 讀碼)

五個獨立 lens:① 實戰 CV 工程師日常 ② data-centric AI 版圖 ③ 工具為何被採用 vs 死掉 ④ 懷疑論/反鍍金讀碼 ⑤ 端到端整合接縫。

### 討論項目 + 各 lens 的核心發現
- **① 工程師日常**:VIX 站在工程師迴圈「旁邊」而非「裡面」。最有價值的 verb(`eval-ingest`/`weakness-report`/`error-mine`/`hardneg`)全在 `yolo val` 右側,但「跑完 val → VIX 讀得到」的橋**只存在於 `docs/examples/dogfood_eval_yolo.py`,不是 CLI verb**。要先手寫 ~40 行 glue 才碰得到最好的功能。
- **② data-centric 版圖**:VIX 默默重造了它自己相依(FiftyOne Brain)+ cleanlab 已有的 commodity 層(near-dup/label-error/hardness;`analytics.py`、`confident_learning.py`)。但**埋在裡面的 `core/consistency.py`(GT×embedding 失敗歸因:taxonomy vs model vs label_noise)是全版圖沒人做的差異化 wedge**,卻被當成報告裡的腳註。
- **③ 採用學**:**沒有 Monday-morning entry point**。要先有 golden/anchor/YOLO .pt/Py3.11+FiftyOne 才跑得動 9 步;唯一 5 分鐘能跑的 `--synthetic` 自己標明「NOT real inference」。time-to-first-real-value = 數小時到數天,而終點是治理產物不是「wow」時刻。
- **④ 懷疑論**:這其實是**兩到三個產品塞進一個 CLI**(curation gate 8 verbs + 模型弱點診斷 10 verbs + MLOps 教科書層 spc/parity/cost-gate/...)。真正 spine 只有 ~12 verb + 4 core module(DecisionLog/encoder_fingerprint/snapshot/gate)。`_GoldenPathParser` 那個藏 70-verb 牆的 hack 本身就是 smell。
- **⑤ 整合接縫**:**VIX 沒有 GT label importer** —— `ingest` 只記影像,從不讀 sibling `labels/*.txt`/`data.yaml`/COCO。一個帶著既有標註資料集來的工程師**根本無法載入自己的標註來稽核**。證據:walkthrough 自己在 demo 腳本裡重寫了兩次 VOC parser。

### 浮現的共識(5 lens 高度一致)
1. **最大的 gap 是「on-ramp / 入口」,不是缺功能。** 4/5 lens 獨立指向同一點:高價值功能被埋在 9 步世界觀 + 要手組的 JSONL 後面。
2. **缺兩個關鍵接縫**:(a) GT label importer(`ingest --labels yolo/coco`);(b) `yolo val → VIX` eval 橋(promote `dogfood_eval_yolo.py` 成 verb)。這兩個讓「我有資料/模型」能一鍵接到 VIX 最好的功能。
3. **74-verb 表面過度建造、傷採用。** 全 5 lens 都說收斂成 ~6–15 verb 兩層 CLI;治理/MLOps-textbook 葉子 verb 隱藏或刪除。
4. **差異化 wedge = 失敗「歸因」引擎(consistency)+ 誠實防竄改 decision log。** 沒有 incumbent 做「為什麼這類會錯 + 該怎麼辦 + 上鏈可稽核」。但現在被埋住。
5. **weakness-report HTML 才是真正的「wow」產物**,卻被埋在 SOP B9 一堆前置之後。

### 浮現的「真的落地」假說(待 Round 2 收斂)
> 能落地的不是更多功能,而是**大幅收窄 + 一個真入口**:一個指令 `vix diagnose <影像資料夾> [--labels yolo] [--weights model.pt]` → 載入工程師**既有的標註**(補缺的 GT 接縫)→ 可選跑他的模型得到**真 eval**(補 yolo-val 橋)→ 直接產出排序好的**弱點/歸因報告**(那個 wedge)→ **不需要**預先 golden/anchor/FiftyOne 世界觀。然後把 74-verb 收成小的對外 core,治理 spine 保留但降級。

### 爭議(Round 2 要解)
- **A. diagnose 要不要 FiftyOne?** ③ 說用 memory adapter 免 FiftyOne;① 警告 memory 的 pixel-fallback embedding 對稽核近乎無用。→ 需釐清:diagnose 應用「真 YOLO + 真 DINOv2 on memory adapter」,免 FiftyOne 但不退化成 pixel。
- **B. 刪 vs 藏?** ④ 主張直接刪 spc/parity/cost-gate/capacity/throughput/calibrate-confidence/drift-type/compare(wired into nothing);其餘 lens 偏向「藏進 advanced」。
- **C. wedge(歸因)先還是 on-ramp(diagnose)先?** 兩者互補:on-ramp 正是「交付」wedge 的載體。
- **D. 重造 FiftyOne 怎麼辦?** ② 主張 FiftyOneAdapter 委派給 Brain/cleanlab;但這和「離線/零相依」護城河有張力。保留 pure core 當離線 fallback 是否兩全?
- **E. 治理 spine(DecisionLog/encoder-fp/snapshot)是 IC 工程師要的嗎?** ①③ 說那是 team/合規賣點不是 first-win;④ 力保它是「信任 GO 判定」的命脈。→ 共識方向:保留但**不要拿它當開場**。

### 後續方向(進入 Round 2)
Round 2 做 cross-examination + 收斂:壓力測試上面的假說、解 A–E 爭議、產出一份**有範圍、反鍍金**的可建造 spec(要蓋哪些 verb、藏/刪哪些、diagnose 的確切契約、wedge 怎麼浮上來),為 Round 3 定稿與開發鋪路。

---

## Round 2 — cross-examination(Builder vs Adversary)

兩個對抗角色:Builder 把 Round 1 假說變成具體 spec 並裁決爭議 A–E;Adversary 唯一任務是「殺掉」這計畫。我(主持)收斂。

### Builder 的 spec 重點
- 新 `vix import-labels`(GT importer):讀 sibling `labels/*.txt`+`data.yaml` / COCO `instances.json` / VOC xml,轉成 `Detection(BBox)` 並 `set_detections`。新 pure module `core/label_import.py`,~40 行,複用 `exporter.py:45` 與 dogfood 已寫兩次的 VOC/COCO 數學。
- 新 `vix eval-run --weights`:把 `dogfood_eval_yolo.py` 升級成 verb,跑模型→組 `{vix_hash,gt,pred}`→`eval_ingest`(順手讓 `eval_ingest` 收 list 或 path)。
- 新 `vix diagnose <folder> [--labels] [--weights]`:**純 orchestrator**(ingest→import-labels→compute_embeddings→eval-run→weakness_report),不需 golden/anchor/calibrate/route/gate。
- 裁決:**A** diagnose 預設真 DINOv2(torch.hub on InMemoryAdapter,免 FiftyOne),`--embed pixel` 當逃生口;只需把 `make_adapter`(cli.py:74-83)那條線接上 `DinoV2Embedder.embed`。**B** 葉子 verb 一律**藏**(`--advanced`)不刪,v1 零刪除零測試churn。**C** on-ramp 先(wedge 已寫好、只是搆不到)。**D** 不引入 cleanlab/Brain,pure core 當離線預設(護城河)。**E** 治理 spine 保留接線、只從敘事降級。

### Adversary 的三發致命質疑(grounded)
1. **🔴 歸因 wedge 結構上無法在 on-ramp 點火。** `consistency_findings` 只吃 `_emb_by_class(adapter,{Tag.GOLDEN})`(pipeline.py:1429;consistency.py:6-13 明說要「human-confirmed GOLDEN per class + 混淆矩陣」)。on-ramp 標榜「免 golden」,且它要升級的 dogfood 路徑只貼 `Tag.EVAL` + `pixel_embedding`(8×8 灰階)→ 歸因表**空的或在像素亮度上算 separability(垃圾)**。賣點在開場模式下不會亮。
2. **🔴 「離線/免 FiftyOne」要嘛退化成垃圾、要嘛第一次跑就要連網。** 唯一免 FiftyOne 的 embedder 是 pixel-fallback(simple.py 自己說 not for production);真 DINOv2(dinov2_torch.py:44 `torch.hub.load`)新機第一次要下載權重(test 自己 `pytest.skip` air-gapped)。真正阻礙是 **Py3.11 + torch + 權重 + 信任**,不是 verb 數。
3. **🔴 `vix_hash` 是檔案位元組 SHA-256(manifest.py:18-24),但 eval/dogfood 用 `p.stem`(檔名)對齊 → 兩個 key space 不相容,且 `except: pass`(pipeline.py:1038)把錯誤吞掉。** 結果:mAP 照印(只靠 JSONL),但 per-image FP/FN attach 全部默默 no-op → 失敗面是空的卻給假信心。**Builder 的 risk #2 與 Adversary kill-shot #3 獨立撞同一個 bug = 確實存在。**

### Adversary 也承認「殺不掉」的部分
- `core/eval_ingest.py` 是真資產:pure stdlib、COCO-style 類別配對、all-point AP、**去重的 typed FP/FN(classification/localization/missed/background)**。給 predictions+GT JSONL 就能跑,免 embedding。
- 收窄後的賣點「一個指令:跑我的模型在我的標籤上 → typed per-class FP/FN 報告」**真的會落地**——只要靠 eval 路徑 + 統一 hash。
- CLI collapse 必須 **hide 不 delete**(308 test functions across 63 files,run_pipeline spine + plugin 都穿過這些 verb,刪 = 高爆炸半徑)。
- consistency.py 的數學是保守誠實的(support tier、Wilson/bootstrap CI、taxonomy_watch 而非自信合併)——問題在**輸入**不在邏輯。

### Round 2 收斂決定(主持裁決)
能落地的系統 = **誠實分層的 `vix diagnose` on-ramp**:
- **Tier A(headline,穩健、近乎離線、零 DINOv2):** `vix diagnose <folder> --labels yolo|coco|voc --weights model.pt` → 匯入既有 GT(新 `import-labels`,複用 bbox 數學)→ 跑模型(新 `eval-run`,**content-hash 統一、join 失敗大聲報錯**)→ **typed per-class FP/FN + AP + 混淆**報告(現成 `eval_ingest`+`weakness_report`)。免 golden 世界觀、免 DINOv2、免 FiftyOne。**這才是我們承諾與展示的。** 連模型都不想跑的人可直接餵 predictions JSONL(純 stdlib 路徑,真離線)。
- **Tier B(advanced,明確標示,需 DINOv2):** embedding 標籤稽核 + 歸因 wedge,經 `--audit`/`--embed real` 開啟,帶誠實 banner。
- **裁決修正 vs Builder:** 採 A/B/C/D/E,但 **A 改為「分層」**(headline = eval 路徑,不依賴 DINOv2);**歸因不放進零設定承諾**(化解 kill-shot 1、2)。
- **必做去風險(kill-shot 3):** `import-labels`+`eval-run` 共用 `compute_hash`(content SHA-256);diagnose 的 join 移除 silent swallow,referenced image 不在 manifest 就**大聲 warn/error**。
- **誠實護欄:** 0 boxes 匯入、name-set 不一致、pixel-fallback 退化、DINOv2 首次下載 —— 全部硬警告。

### 殘留待 Round 3 鎖定的爭議
- **F.(誠實性,新)** 把工程師「既有(可能錯的)標籤」當 GOLDEN 餵歸因 → 循環論證(用受審標籤當審查基準)。consistency 對 separability/taxonomy 尚可(只問「這兩類在空間可分嗎」),但對 label_noise 較危險。需定:匯入標籤 = **provisional reference 非 human-confirmed golden**,且歸因輸出要對應 hedge。
- **G.** Tier A headline 到底是 `diagnose` 一個 verb 帶 `--weights`,還是 eval-only 的更小承諾?確認產品定位一句話。
- Round 3:鎖定去風險後 spec + Definition of Done,並特別審 F 的誠實邊界,然後開發。

---

## Round 3 — 共識鎖定(Honesty / DoD / Positioning 三審)

三個並行審查:① Honesty Auditor(解 F)② Definition-of-Done + 測試計畫 ③ Positioning + will-it-land(解 G)。三者皆 grounded 讀碼。

### ① 誠實裁決(F)—— 這是專案的身分,最關鍵
- **F1(Tier A FP/FN):允許,但必須相對化。** eval 數學對「與你標籤的一致度」是誠實的;一旦 prose 暗示標籤=真實、模型有錯就變謊。措施:匯入且未覆核的標籤一律標示「**你提供的標籤(匯入,未經 VIX 覆核)**」;不可用裸 "GT"/"false positive";`background` FP 必帶雙因 hedge「= 你的標籤此處沒框 → 可能模型幻覺,**也可能漏標的 GT**」;頂部誠實 banner(與既有 PROXY 戳記並列)。**不否決。**
- **F2(Tier B 歸因):不可把匯入標籤標 GOLDEN。** `Tag.GOLDEN`(types.py:29)餵了 12+ 處(calibrate/route/frozen/label-audit/coverage/snapshot/training-pool-hash/export/**gate NO-GO**),自動標 golden 會毒化全部。新增 **`Tag.PROVISIONAL`**(只供 embedding 幾何運算,絕不觸發不可逆/設門檻動作)。consistency 各裁決在 provisional 參照下的存活:
  - `separable/inseparable_embedding`、`separability(sep_err+CI)`:**存活**(可證偽的幾何陳述,不主張標籤對錯;維持 encoder-bound 措辭)。
  - `clean`:存活(純諮詢)。
  - `model` → 降為 `model_watch`(永不阻擋);`taxonomy` → `taxonomy_watch`(去掉 merge/stop 祈使,排除出 gate)。
  - **`label_noise`:在 provisional 下整個抑制**(純循環論證:用受審標籤定罪受審標籤)→ 改發 `label_audit_needed`「要判定是否雜訊,先人工覆核這些匯入標籤」。
- **Gate firewall:** consistency NO-GO(pipeline.py:795-801)只在 confirmed-golden 參照才武裝;provisional → advisory,**永不阻擋重訓**。
- **否決(VETO):** 自動把匯入標籤標 GOLDEN;`label_noise` 對 provisional 參照開火;gate 用 provisional 參照製造 NO-GO。

### ② Definition of Done + 測試(baseline 確認 308 tests)
- DoD 19 項皆可觀察可測;關鍵:**content-hash join 真的把 per-image FP/FN 掛上正確影像**(KS3),stem-keyed JSONL 對 content-hash manifest **大聲報錯**;Tier A 不 import DINOv2/FiftyOne;two-tier help 用 `argparse.SUPPRESS` 隱藏不刪除,**所有 verb 仍可 dispatch(零刪除)**;export→import-labels round-trip 等價;全 308 + 新測試綠。
- 6 個新測試檔(全 InMemoryAdapter + 合成 fixture,免網路/FiftyOne/torch):test_label_import / test_import_labels_cli / test_eval_ingest_listmode / **test_eval_run_hashjoin(KS3)** / test_diagnose_orchestrator / test_two_tier_help。
- **既有測試風險避法:** `eval_ingest(results)` 用 `isinstance(results,(str,Path))` 多型分派、path 分支位元不變;strict-join 用 `strict_join=True` 只套 diagnose 路徑(裸 `eval-ingest` 保持 best-effort 向後相容);two-tier help 只改 `help=SUPPRESS` 不動 `add_parser` 名與 dispatch;重相依一律 handler 內 lazy import。
- 建造順序(小→大):label_import → eval_ingest 多型 → import-labels → strict-join+eval-run(KS3)→ diagnose → two-tier help → 全套回歸。

### ③ Positioning(G)+ will-it-land
- **一句話定位:** 「**`yolo val` 給你 mAP;VIX 給你該修什麼的清單** —— 指向你的資料夾與 .pt,離線一行得到 per-class FP/FN + AP + 混淆 + 該先看的影像,免重訓、免服務。」
- **落地判定:** Tier A 乾淨清除阻礙(1 入口)(3 verb 過載)(4 與 FiftyOne 競爭卻又依賴它)——因 headline **完全不需 FiftyOne/DINOv2**;有條件清除(2 安裝)——**前提:diagnose 預設不走 `--adapter auto`、不碰 FiftyOne/DINOv2 安裝圖**(memory adapter 當純偵測 sink,如 dogfood_eval_yolo.py:55-76)。殘留:GT 正確性(本質,靠 hedge)、語言(CLI 全 zh-Hant → 既有慣例,對齊 owner 團隊)。
- **減法裁決:** 不是 docs-only。VOC/YOLO/COCO parser **被困在 example 腳本裡**(dogfood_eval_yolo.py:34-44、dogfood_pathole.py:33-45),不在 src、不可 import、非 verb —— 那正是 round 1 指的「埋起來的價值」。**要建,但只建一個 `diagnose` verb + 一個 label importer module**,90% 復用既有 `eval_ingest`/`weakness_report`/renderer。
- **killer demo:** 真 pothole set(`C:\code\claude\patHole_Dataset`,665 VOC)→ `vix diagnose ... --labels voc --weights best.pt` → 把單一 mAP 變成「漏 47、glare 誤報 61、框鬆、最自信 12 個錯 + 該看的影像」,截圖進 README。
- **必改/否決:** ⓋETO `diagnose` 走 `--adapter auto`;headline 的 "FP" 必帶 GT-correctness hedge;語言對齊既有 zh-Hant 慣例(owner 團隊);不加新子系統。

### 🔒 鎖定的建造 spec(進入開發)
1. `core/label_import.py`(純):`yolo_txt_to_dets`/`coco_to_dets`/`voc_to_dets` → `dict[image_path -> [Detection]]`,複用 `BBox` + exporter/dogfood 數學;非法輸入大聲 `ValueError`。
2. `types.py`:新增 `Tag.PROVISIONAL`。
3. `pipeline.import_labels` + `vix import-labels`(content-hash join、未知影像大聲報錯、`--as reference|eval`,reference→PROVISIONAL)。
4. `pipeline.eval_ingest` 收 list 或 path(多型、附加);diagnose 路徑 `strict_join=True` 大聲報錯(移除該路徑 silent `except: pass`)。
5. `pipeline.eval_run`(跑 YOLO、**content-hash keyed**,lazy import ultralytics)。
6. `pipeline.diagnose` + `vix diagnose`(orchestrator;Tier A 預設 memory-only 免 FiftyOne/DINOv2;Tier B `--audit`/`--embed real` opt-in)。
7. 誠實框架:weakness_report 加 `reference=provisional|golden` 旗標 → banner + 重貼 FP/FN 標頭 + background 雙因 hedge。
8. 誠實 firewall:consistency 在 provisional 抑制 `label_noise`(改 `label_audit_needed`)、`taxonomy/model`→`*_watch`;gate NO-GO 需 confirmed-golden。
9. two-tier `--help`(SUPPRESS 隱藏、零刪除)。
10. 6 新測試檔 + 全 308 回歸綠。

**Definition of Done(一句話):** 陌生 CV 工程師在自己現有 Python+YOLO 環境裡、不裝 FiftyOne、`vix diagnose <資料夾> --labels <fmt> --weights <pt>` 一行,就拿到對「他自己的標籤」誠實相對化的 per-class FP/FN+AP+混淆報告與該先看的影像;且 multi-agent 十情境平均 ≥95。

---

## 開發完成 + Round 4 驗收(10 情境 × 3 獨立評分)

### 已建造(全套 **332** 綠 = 308 baseline + 24 新;含 live Tier-2 GUI/FiftyOne + 真 DINOv2)
- `core/label_import.py`(純 yolo/voc/coco parser)+ `Tag.PROVISIONAL`(匯入標籤=診斷參照,**永不** golden)。
- `pipeline.import_labels`(**content-hash join**、未匯入影像大聲報錯)、`eval_run`(yolo-val 橋,content-hash keyed,lazy ultralytics)、`diagnose`(orchestrator;Tier A 免 FiftyOne/DINOv2,Tier B `--audit` opt-in)。
- `eval_ingest` 收 list|path + `strict_join`(stem-keyed 對 content-hash manifest **大聲報錯**,KS3)。
- 誠實:firewall(provisional 參照 → 抑制 `label_noise`→`label_audit_needed`、`taxonomy/model`→`*_watch`、永不 gate-block);報告未覆核參照 banner + FP/FN 重貼標 + background 雙因 hedge。
- CLI:`diagnose`/`import-labels`/`eval-run` 三 verb + two-tier `--help`(16 core 顯示、~60 隱藏、**76 全可 dispatch、零刪除**)。

### 10 情境(多代理定義,grounded):Tier-A happy / COCO / VOC / 免-FiftyOne / 標籤錯時的誠實 / content-hash join(KS3)/ 歸因 firewall / time-to-first-value / 0-box 邊界 / 誠實 over-claim 壓測。

### 三個獨立評分(grounded 讀碼 + 跑測試)
| 評分者 | 平均 | 最弱項 |
|---|---|---|
| A | **99.0** | #1 `out["eval"].per_class_ap` 字母序非弱→強 |
| B | **96.5** | #10 provenance positional evs[-2];#3 VOC 零 size 靜默丟框 |
| C | **95.5** | #9 非標準 layout 0-box 訊息不夠精準 |

**總平均 = 97.0,三者皆 ≥95 → 通過 95 分閘門(第一輪 10 情境即達標,無需再生 10 個)。**

### 採納評分回饋的低風險修補(只升不降)
1. `diagnose` 的 `out["eval"].per_class_ap` 改為**弱→強排序**(對齊報告表 + CLI/JSON 首屏)。
2. **VOC + COCO**:帶標註但 `<size>`/width/height 無效 → **大聲 ValueError 命名檔案**(消除靜默丟框,與「無靜默不一致」哲學一致);box-free 無 size 仍 OK。
3. `diagnose` 0-box 失敗訊息改為**診斷式**(「找到 N 張影像但 0 個標籤配對成功」+ 各格式找過哪些路徑 + 用 `--label-dir`),化解 scorer C 指出的首次採用風險。
→ 修補後新測試 24 綠、全套 **332** 綠。

**結論:能「真的落地」的不是更多功能,而是一個誠實分層的 on-ramp —— 把已存在的弱點/歸因引擎,用工程師既有的標籤與模型、一行指令、零 FiftyOne 世界觀地交付,並對「參照是未覆核標籤」這件事全程誠實。**

---

## Round 5 — 閉環(loop)增量(on-ramp 已 97 分後的下一步)

三個 lens:① close-the-action-loop ② did-my-fix-help(資料歸因)③ 懷疑論/反鍍金。

### 各 lens
- **① 閉環**:diagnose 是單向街——產出報告/worklist 卻**無離線 action verb**;且 `export` 只吃 `Tag.GOLDEN`(pipeline.py:348),diagnose 全 `PROVISIONAL` → **diagnose 後 `vix export` 直接報「尚無 golden」**(已讀碼證實)。提議 `vix fix`(worklist CSV 套用)+ provisional-aware export。
- **② did-my-fix-help**:數字(per_class_ap/eval_set_hash)+ 誠實比較器(`regression_check`、`_report_provenance`)**都已存在**,但沒有「成對 per-class delta、錨定 baseline」的面。提議 `vix diagnose-compare`(read-only、eval_set_hash 閘、per-class Δ、support gate、無因果語)。
- **③ 懷疑論(最強)**:迴圈**用既有 verb 已可閉**(resolve/export/ap-trend/report prev→cur),真缺口是**可發現性/文件**不是能力。警告:「did-my-fix-help」子系統**重造 ap-trend + provenance** 且引誘 overclaim(你是外部重訓,非 VIX 造成);「act-on-worklist」逼近**自動改標(禁止)**。建議:文件 + 一行 next-step nudge,幾乎不寫碼。

### 爭議裁決(主持)
- 懷疑論在「別建新子系統」上對,但漏了 ① 的真 bug(**diagnose 後 export 死路**)。② 的新 verb 確會重造 ap-trend → **不建 `vix diagnose-compare`、不建 `vix fix`、不建 provisional-export 子系統**(反鍍金)。
- **沒人講清楚的關鍵誠實洞見:`eval_set_hash` 含 GT**(eval_ingest.py:35-45)。所以**修了 eval set 的標籤 → hash 變 → 自動「不可比較」**。誠實的「修了有沒有幫助」迴圈**必須用固定(frozen)held-out eval set**:修*訓練*標籤 → 外部重訓 → 在**凍結的 eval set** 上 re-diagnose → 才可比。這正是 ap-trend/provenance 既有「eval 變了就拒比」的所以然。

### Round 5 共識(誠實、最小、反鍍金)
1. **報告內 per-class before/after**:擴充既有 `_report_provenance` 帶 `prev_per_class_ap`,在 weakness report 加「自上次可比較執行的 Δ」欄(md+html),**僅在 eval_set_hash 相符時**顯示,框定為「在此固定 eval set 上量到;你於外部重訓;非 VIX 造成」。**不開新 verb**(避開懷疑論的重造陷阱),就地回答 ②「哪一類動了」。
2. **關鍵誠實洞見上桌**:comparability banner + quickstart 寫明「改了 eval set 的標籤會破壞可比性 → 用凍結 held-out eval set」。
3. **diagnose next-step nudge**(CLI):印出誠實的回程(修標→外部重訓→在固定 eval set 再 diagnose→ap-trend),含可比性 caveat。
4. **export 死路 → 診斷式錯誤**:有 PROVISIONAL 但無 GOLDEN 時,訊息說明「diagnose 匯入的是參照標籤;你本就擁有這些標籤檔;要匯出請先 resolve→golden」。(指引,非新能力。)
5. **文件**:_QUICKSTART + 本檔寫出閉環敘事 + 凍結 eval 規則。
- **不建**:`vix fix` CSV 自動套用、`vix diagnose-compare` 新 verb、provisional-export 子系統。

### 後續:建造上述 1-5 → 多代理出 10 個更難的情境(聚焦閉環 + 誠實可比性)→ 3 評分者 ≥95。

### 建造 + Round 5 驗收(已完成,全套 338 綠 = 308 + 30 新)
- 建:`_report_provenance` 帶 `prev_per_class_ap`(僅 comparable 時);weakness_report per-class **Δ(同 eval set)** 欄(md+html,僅 eval_set_hash 相符時顯示)+ 凍結-eval 誠實框定;`diagnose` next-step nudge(state-aware);export 死路 → 診斷式錯誤;`_QUICKSTART` 閉環區塊。
- 10 個更難情境(多代理定義):happy loop / **改 eval 標籤→拒比(關鍵誠實閘)** / 一類退步一類進步都顯示 / 首份無 Δ / 低支撐 swing / keep-vs-move-on / 可發現性 / export 死路 / 跨格式 loop / 對抗式三向 over-claim 壓測。
- 3 獨立評分(grounded 讀碼 + 跑測試 + repro):**A 97.2、B 94.9、C 96.9 → 平均 96.3 ≥95,通過**。關鍵誠實閘(改 eval set 後系統拒絕顯示 Δ/假 +mAP)三人皆驗證成立(報告 `_report_provenance` 與 gate `regression_check` 兩道獨立防線)。
- 採納三人收斂回饋的兩個低風險修補(只升不降):(1) **低支撐 Δ 不畫 ↑/↓ 箭頭**、改標「n少不穩」(`_delta_cell`,沿用 gate 的 min_support=20)—— 防止 n_gt=4 的 +0.02 看起來像 n_gt=200 的真進步;(2) **nudge 改 state-aware** —— 首份/不可比時不再承諾「會看到 Δ」,而是「凍結此 eval set 當基準」。修補後 30 新測試綠、全套 **338** 綠。

**Round 5 結論:閉環的價值不是新子系統,而是讓既有的 before/after 在報告內就地呈現、且對「什麼才可比」(凍結 eval set;改了 eval 標籤就不可比)與「什麼才算真的動了」(低支撐不畫箭頭)全程誠實。懷疑論的「別建子系統」是對的;真正缺的是把誠實的可比性擺到工程師眼前。**
