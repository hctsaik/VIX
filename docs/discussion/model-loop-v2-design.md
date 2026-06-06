# Model-loop v2 — 設計共識(design of record)

> **狀態:共識草案 → 經 2 位 review 代理壓力測試 → §6 收斂所有 must-fix → ✅ 已實作 + 測試。**
> §2 為原始規格;**§6 的修訂為最終約束(衝突以 §6 為準)**。
> 交付:Tier 0(`_require_known` 守門 + `resolve --label` 可逆)、T1a(`core/eval_ingest.py` 錯誤分型 + IoU sweep + `errors.py` 共用 `iou()`)、
> T1b(`pipeline.error_mine` 誤差框 IoU 反查 + 退回)、T1c(`core/box_qa.py` + `vix box-qa`)、
> T2(`core/gate.py` `regression_check` + `pipeline.set_eval_baseline` + `eval_set_hash` + gate 接線 + `vix set-eval-baseline`)。
> 測試 +20:`test_resolve.py`(B1/B2)、`test_eval_typed.py`、`test_challenge_guard.py`、`test_box_qa.py`、`test_error_mine_boxlevel.py`。**全套 172 pytest 綠。** 操作見 SOP §B8。
> 來源:四個視角代理(YOLO 訓練 / Data-Centric 策展 / MLOps 治理 / SMM 現場操作)獨立
> grounded 分析後**強烈收斂**到同一條主線。本文把共識收斂成可直接實作的具體規格。

## 0. 一句話問題陳述

VIX 已用 `eval-ingest` 把模型表現接回來(roadmap #1,已出貨),但**接回來的解析度太低**:
`evaluate()` 只在單一 IoU 算 per-class AP,FP/FN 只留**整數 count**——
[`_match_image`](../../src/vix/core/eval_ingest.py) 內部其實算得出每個錯誤框是「定位誤差 / 類別混淆 / 背景誤報 / 漏檢」,
卻在回傳前丟掉。連帶兩個後果:
1. `error_mine` 只能用「整張圖所有偵測 embedding 的平均」反查——一張晶圓上一個小氣泡混在五個大反光裡時,平均被反光主導、訊號近乎雜訊。
2. `gate` **完全不吃 mAP**:任何策展動作都可能偷偷讓 mAP 掉,gate 仍說 GO。

外加代理在程式碼中**證實的兩個正確性 bug**(非功能缺口):
- `resolve`/`dismiss`/`restore-dismissed` 對不存在的 id **靜默 no-op,卻仍印成功 + 寫進不可竄改帳本** → 幻影確認、樣本卡在佇列。
- `resolve --label` 改標籤**無 undo 記錄**(只有 `relabel` 有)→ 覆核迴圈單筆改標不可逆。

`core/errors.py`(T5,分類/定位/漏檢分型)是**死碼**(零呼叫者),且自帶第二套 `bbox_iou` 與 `eval_ingest.iou` 重複。

## 1. 四視角共識(收斂點)

| 視角 | 各自 #1 | 收斂到的共同根因 |
|---|---|---|
| YOLO 訓練 | 逐框錯誤分型 + 用錯誤框 crop 反查(非整圖平均) | **eval 解析度** |
| Data-Centric | box-QA(逐框幾何健檢) | 錯誤分型的幾何基礎(部分在 errors.py)|
| MLOps 治理 | challenge-guard(mAP-drop 接進 gate 硬擋)| 需要**穩定的逐類 AP** 才守得住 |
| SMM 現場 | 靜默 no-op 是 ledger 完整性 bug;`batch-status` | 信任 / 可逆 |

→ **修「eval 丟掉錯誤框細節」這一件事,同時解鎖三個 #1。** 這就是本設計的主軸。

> **修正一個代理的措辭**:box-QA 的「退化框 / 截斷框 / 長寬比離群」幾何**不在** `errors.py`(那是 pred-vs-GT 的分類/定位分型)。
> errors.py 的貢獻是 eval 的**錯誤分型 taxonomy**;box-QA 的靜態框驗證是**新的**(但極便宜,純 `BBox` 算術)。本設計分開處理,不混為一談。

---

## 2. 規格

### Tier 0 — 兩個正確性 bug(先做,各約 5–10 行,純修復)

**B1 — 存在性守門(fail-closed)。** 新增純 helper:
```python
def _require_known(adapter, ids) -> None:
    known = {h for h, *_ in adapter.samples()}
    missing = [i for i in ids if i not in known]
    if missing:
        raise ValueError(f"找不到 vix_hash {missing};App 顯示的是 sample id,"
                         f"請用 vix_hash(見 vix history / explain),或用檔名(待 Tier-3 支援)")
```
在 `resolve_review`、`dismiss`、`restore_dismissed` **動作與寫稽核之前**呼叫。沒有樣本就乾淨報錯,**絕不**印成功、**絕不**寫帳本。

**B2 — `resolve --label` 可逆。** `resolve_review(decision="confirm", label=...)` 覆寫 `d.label` 時,
把每個被改的偵測以 `relabel_dataset` **相同格式**附加到 `relabel_changes.jsonl`:
```python
{"id": f"{vix_hash}:{i}", "old": old_label, "new": label}
```
如此 `relabel_rollback` 既有路徑即可反轉**所有**改標(不只 `relabel`)。同時 review 稽核記錄維持不變。

> 硬約束 T0-1:Tier 0 不得改變既有成功路徑的行為(既有測試全綠);只新增「找不到就報錯」與「改標就記 undo」。

### Tier 1 — eval 解析度鍵石(主軸)

**T1a — 錯誤分型 + IoU sweep(改 `core/eval_ingest.py`,純函式)。**
擴充 `_match_image` 回傳**帶型別的 FP/FN 框**(它已有 `used[]` 與 preds,資訊都在,只是沒回傳):
- FP 型別:`classification`(此 FP pred 在 ≥thr 蓋住一個未匹配的**異類** GT)| `background`(其餘)。
- FN 型別:`classification`(該未匹配 GT 被異類 pred 在 ≥thr 蓋住)| `localization`(被**同類** pred 在 IoU∈[`loc_band`,thr) 蓋住)| `missed`(無有意義重疊)。
  這正是 `errors.py` 的 taxonomy;將其**併入**這個更完整的 matcher(class-specific + confusion + AP 都在這)。
- `evaluate()` **加性**回傳(不破壞既有 key):
  - `map_by_iou: {0.5: .., 0.75: ..}` 與 `loc_gap = mAP@0.5 - mAP@0.75`(定位尾巴可見化)。
  - `fp_detail`/`fn_detail`: `{vix_hash: [{label, bbox, type, best_iou}]}`。
  - 既有 `mAP`(=@iou_thr)、`per_class_ap`、`confusion`、`per_image`(counts)、`fn_hashes`/`fp_hashes` **原樣保留**。
- `errors.py`:`diagnose_image`/`diagnose_errors` 改成**呼叫共用 matcher 的薄包裝**(刪掉重複的 `bbox_iou`),T5 不再是死碼且零重複。

**T1b — 用錯誤框反查(改 `pipeline.error_mine`)。**
不再對「整張圖所有偵測」取平均。對每張誤差圖:
- 取**該圖 FP 框自身的偵測 embedding**(FP 就是一個 pred,embedding 已算好——精準)。
- FN 框:若有**同圖 pred 與之重疊**(定位誤差 FN,正是 mAP 尾巴的高價值處),用該 pred 的 embedding;若為純 `missed`(零重疊,無 pred 可代表),**退回**用整圖平均並記一筆 caveat。
- 反查仍是「候選對最近誤差區的 cosine」。
> 硬約束 T1-1:無真實 embedding 後端(`--adapter memory`/pixel_fallback)時行為退化但不崩。
> 硬約束 T1-2:`eval_results.json` 結構**加性**;舊 `error_mine`/讀者不得因新 key 失效。

**T1c — `vix box-qa`(新純模組 `core/box_qa.py`)。**
對 golden 偵測逐框靜態健檢,回傳排序問題清單(只**標記**不自動刪,沿用 `harmful` 無 `--remove` 的姿態):
- `degenerate`:`w*h < area_eps` 或 `w<eps`/`h<eps`。
- `truncated`:框越界 [0,1](`cx-w/2<=0` 或 `cx+w/2>=1` …)→ 建議 `ignore`。
- `aspect_outlier`/`area_outlier`:超出**該類** golden 包絡 [p01,p99];需 `min_support`(沿用既有 `low_support` 想法)否則不對小樣本類別發噪。
新增 `pipeline.box_qa` + CLI `vix box-qa` + `health_report` 一列計數。

### Tier 2 — challenge-guard(把準度接進 gate)

**T2a — 純回歸檢查(`core/gate.py`,新函式)。**
```python
def regression_check(current_ap: dict, baseline_ap: dict, current_map: float, baseline_map: float,
                     map_drop_thr: float = 0.02, protected_drops: dict[str,float] | None = None,
                     eval_support: dict[str,int] | None = None, min_support: int = 20)
    -> tuple[list[str], list[str]]:  # (blocking_reasons, advisory_warnings)
```
- 整體 `baseline_map - current_map > map_drop_thr` → blocking。
- 受保護類別(`protected_drops` 指定逐類門檻,沿用 `set_threshold` 的逐類 policy 風格)AP 掉超過該門檻 → blocking。
- **小樣本誠實**:`eval_support[c] < min_support` 的類別**不阻擋**,只進 advisory warning(單一 IoU、小 N 的 AP delta 會抖,沿用 `parity`/`spc` 既有「樣本不足」慣用語)。

**T2b — baseline 凍結 + 接線。**
- `vix set-eval-baseline`:把當前 `eval_results.json` 連同 snapshot `content_hash` + **eval set 的內容雜湊**寫成 `eval_baseline.json`(eval set 偷改也偵測得到)。審計記一筆。
- `pre_train_gate` 加 keyword 參數 `map_drop`/`protected_class_drops`,把 `regression_check` 的 blocking 併入既有 `reasons`、advisory 併入 `checks`。`pre_train_gate_stage` 在 `eval_results.json` 與 `eval_baseline.json` 都在時自動帶入。
> 硬約束 T2-1:無 baseline 或無 eval 時,gate 行為與今日**完全一致**(此檢查 opt-in,不得讓既有 GO 變 NO-GO)。
> 硬約束 T2-2:baseline 一律綁 `content_hash`+`eval_set_hash`;eval set 變了就不能宣稱比較有效(避免「換了 eval set 的假 mAP 漲」)。

---

## 3. 測試計畫(對「重要更新寫 testing」)

- **Tier 0**:`resolve`/`dismiss`/`restore-dismissed` 對未知 id → raise 且 **decision log 無新記錄**(讀 log 斷言);`resolve --confirm --label` 後 `relabel_rollback` 能還原;既有成功路徑回歸不變。
- **T1a**:一張人造圖含 1 定位誤差 + 1 類別混淆 + 1 背景 FP + 1 純漏檢 → `fp_detail`/`fn_detail` 型別精確;`map_by_iou` 在「框系統性偏鬆」案例呈現 `loc_gap>0`;`errors.diagnose_image` 與 matcher 一致。
- **T1b**:1 FN 混 N 反光的圖 → 反查用 FN 對應 pred 的 embedding,排序與「整圖平均」**不同且更靠近**目標;memory adapter 退化不崩。
- **T1c**:退化框/截斷框/長寬比離群各一 → 命中;小樣本類別不誤報。
- **T2**:baseline 後 mAP 掉超門檻 → NO-GO 且 reason 明確;受保護類別掉 → NO-GO;小樣本類別掉 → 只 advisory 不擋;無 baseline → 行為同今日(opt-in 回歸)。

## 4. 誠實邊界(寫進 SOP)

- recall / mAP 只在**凍結、有標籤**的 eval set 上經 eval-ingest 才算數;engineer-confirmed-rate 是 precision-like,不可當 recall。
- 錯誤分型與 box-QA 是**幾何/IoU 真值**,不靠 embedding,對「DINOv2 是泛視覺、非缺陷判別」這個已知限制免疫;但 error-mine 的反查仍吃 embedding,故 FN 反查的可靠性受此限制(已在 T1b 記 caveat)。
- challenge-guard 守的是「**別讓策展偷偷掉準**」,不是「保證漲準」;小 eval set 下用 advisory 而非硬擋,避免 GO/NO-GO 亂跳。

## 5. 不做 / 降級(共識)

- #9 influence:沒真重訓就是 `harmful` 換皮,延後。
- #8 release registry:在有可信 per-delta mAP 前是空轉;`set-eval-baseline` 先提供最小版的「版本綁 mAP+eval雜湊」,完整 registry 待真有發布週期再說。
- 資料模型 GT/框 provenance(解鎖 fix-labels #6):是更大的 schema 改動,本輪**不含**,但 T1a 的 `fp_detail`/`fn_detail` 為其鋪路。

---

## 6. Round-1 review 收斂(最終約束,覆寫 §2 衝突處)

兩位 review 代理(correctness + buildability)獨立指出同一批問題。以下為**綁定**決議:

**R1 [CRITICAL]｜T1b 的「用錯誤框自身 embedding」前提不成立 → 改為明確 IoU 比對 + 退回。**
eval JSON 的 `pred` 框是**外部**的:不帶 embedding,也不保證等於 adapter 既存 `Detection`(可能是不同推論回合/門檻/順序)。決議:
- `error_mine` 對每個 `fp_detail`/`fn_detail` 框,在**同一 `vix_hash`** 內用 `eval_ingest.iou` 比對既存 detections,取 IoU ≥ `emb_match_iou`(預設 0.5)且 `embedding is not None` 者中 IoU 最高的那個 detection 的 embedding;
- 比不到(純 `missed` FN、或外部框無對應)→ **退回該圖偵測 embedding 平均**並記 caveat;
- 即使退回,也已是改進:反查**只**在 `fp_hashes∪fn_hashes` 的誤差圖上做(今日是更糊的全集)。
- **T1b 最後做、風險最高**;`--adapter memory`/pixel_fallback 下退化不崩(硬約束 T1-1 擴充涵蓋「無可比對 embedding」)。

**R2 [HIGH]｜錯誤 taxonomy 不得重複計數;FP 改三分類。**
同類近失框(同類 pred 在 IoU∈[`loc_band`,thr))今日落入 `fp_boxes`,若 FP 只分 classification/background 會被同時記成「background FP」+「localization FN」=同一錯誤算兩次。決議:FP 三分類 `classification`(蓋未匹配**異類** GT)/`localization`(蓋**同類** GT 於 [`loc_band`,thr))/`background`(其餘);且一個定位誤差**只報一次**(報為 FN-`localization`,其配對的 localization-FP 連結不另計為 background)。測試:系統性鬆框案例只產生 1 個 localization 錯誤。

**R3 [MED]｜已匹配 GT 上的異類 FP = `background`(刻意,duplicate 桶),寫進文件而非「the rest」默默吸收。** 不宣稱 taxonomy 是語意真值(見 R9)。

**R4 [HIGH]｜`errors.py` 不是「行為保留的薄包裝」→ 只共用 `iou()`。**
`diagnose_image` 是 **GT 中心、class-agnostic best-overlap、`min_overlap=0.1`、回傳 legacy 字串** `ok/classification_error/localization_error/missed`,且有 [test_errors.py](../../tests/test_errors.py) 鎖死。matcher 是 pred 中心、class-specific、conf 貪婪。**不合併語意**。決議:`errors.bbox_iou` 改為委派 `eval_ingest.iou(a.as_tuple(), b.as_tuple())`(刪重複數學),`diagnose_*` 公開 API 與字串**原樣保留**(test_errors 全綠)。「wiring」= 錯誤 taxonomy 進入 live eval 路徑(matcher 的 typed detail)+ 移除重複 IoU;不是把 diagnose 接到 CLI。

**R5 [MED]｜IoU sweep 必須在每個 IoU **重跑** `_match_image`**(TP@0.5 可能是 FP@0.75);`_ap` 可重用。順手刪 `_ap` 內未使用的 `best_prec` 迴圈(eval_ingest.py:89–92)。`loc_gap` 註明為**艦隊級**訊號、非逐類因果。

**R6 [HIGH]｜`eval_set_hash` 必須真的綁 GT 且在 gate 時比對。**
今日 `eval_results.json` 不含此 hash、`regression_check` 簽章也無 hash → 換 eval set 的作弊偵測不到。決議:
- `eval-ingest` 時把 `eval_set_hash = sha256(json.dumps(sorted [(vix_hash, sorted gt {label,bbox})], sort_keys=True))` 寫進 `eval_results.json`;`set-eval-baseline` 一併存進 `eval_baseline.json`。
- `pre_train_gate_stage` 比對 current vs baseline 的 `eval_set_hash`;**不一致 → 不套用 regression 為 blocking,改發 advisory「eval set 已變,比較無效」**(誠實 > 假擋)。
- 與 snapshot `content_hash` **分開**(後者是 golden/thr,非 eval set),不混用。

**R7 [HIGH]｜小樣本 vs 受保護類別:fail-closed。**
受保護類別(`protected_drops` 指定)**一律**評估阻擋,**不受** `min_support` 豁免;若受保護類別 eval 支撐 < `min_support` → 該情況本身就是 blocking reason(「受保護類別 C 的 eval 覆蓋不足,無法認證」)。`min_support→advisory` 只適用**非**受保護類別。baseline 有、current 缺的類別(類別消失)視為 eval-set 變動(連動 R6),不靜默放行。

**R8 [MED]｜Tier 0 精確化。**
- B1 `_require_known` 在**所有副作用之前**(含 `apply_tags`/`set_detections`,非只 log);`dismiss`/`restore_dismissed` 先驗整批 ids 再動作(不半套)。
- B2:`resolve --label` 會把該圖**所有** detection 設為同一 label → undo 必須 `enumerate(dets)` **逐 detection** 記一筆 `{"id":"h:i","old":<該框原 label>,"new":label}`、且在覆寫**之前**擷取 old;沿用 `relabel_rollback` 的 `:i` 位置索引(註明:若之後重排 detections,位置映射會失準,與既有 `relabel` 同限制)。
- `restore_dismissed` 需 optional `remove_tags`;先檢查可用性,否則 raise `ValueError`(避免 raw `NotImplementedError` 繞過 cli.py 的 `(ValueError,OSError)` 乾淨退出)。

**R9 [LOW]｜§4 誠實措辭軟化。** 「**量測**(IoU/面積/長寬比)精確;**分類**用固定 IoU band,是確定性 policy、非語意真值」;並揭露 R1 的退回(error-mine 在外部框無對應時退回整圖平均)。

**R10 [build]｜雜項落地。**
- `eval_results.json` 體積:`fp_detail`/`fn_detail` 只存 `{label,bbox,type}`(`best_iou` 不落地、需要時即算);keyed by hash;文件註明預期大小。`per_image` 維持被 strip;新 key 通過既有 filter(已驗證 `error_mine` 只讀 `fn_hashes`/`fp_hashes`,新增讀 `fp_detail`/`fn_detail`)。
- Config 新增 `eval_results_path`、`eval_baseline_path` property(沿用既有 `@property` 慣例)。
- `box_qa.py` 為**新** core 模組,**可**用 numpy(p01/p99);**不得**把 numpy 帶進 stdlib-only 的 `eval_ingest.py`。`box-qa` **唯讀**(回傳/印出 ranked issues,不寫 tag/ledger),與 `harmful` 無 `--remove` 一致。
- `eval_support` 來源 = `evaluate()` 既有回傳的 `n_gt`(已在檔案中)。
- `set-eval-baseline` 審計用 `extra` dict 存 hashes/mAP,非塞進 `decision` 字串。
- `pre_train_gate` 新 keyword 參數預設 no-op(`None`);`health_report` 與 `pre_train_gate_stage` 兩處呼叫點都不得因此改變既有判定(T2-1 opt-in)。

**最終建置順序(reviewer 建議,採納):**
**Tier 0 → T1a(matcher+typed detail+sweep+errors 共用 iou)→ T2(regression_check+set-eval-baseline+gate 接線)→ T1c(box-qa)→ T1b(embedding 反查,最後、最險)。**
每階段獨立綠燈再進下一階段。
