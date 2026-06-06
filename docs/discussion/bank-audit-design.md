# Bank-Audit 設計討論(Low-conf YOLO Proposal Mining + DINO Embedding Audit）

> 多代理討論「要做哪些事」直到共識。四個視角:偵測級聯 / embedding 度量 / VIX 架構 / MLOps-SMM 營運。
> 目標:把 proposal 落地成 VIX 內的離線 POC,先有共識再動工。

## Round 1 — 四方主張(摘要)

**偵測級聯**:`run_yolo` 已用 conf=0.001(低分框本來就在),真缺口是 NMS-off 重疊爆炸、crop 太緊(透明小 defect 訊號近零)。要:① 低分 proposal 變一級(distinct tag,絕不污染 golden 路由/KPI)② 類別無關 loose NMS(IoU~0.7)+ 進 audit 前做 proposal 去重(一個 proposal = 一個 audit 單位)③ crop **context padding + min-size 上採樣**(SMM recall 最高槓桿)④ conf 下限用 **sweep**(recovered-defect vs review-load 曲線)選,別猜;建議 0.05。

**embedding 度量**:cosine 鎖定(與 VIX 一致)。核心風險是**銀行不平衡**(Defect 小而緊 vs Reflection 大而雜 → raw 最近距離不可比)。要:① 每銀行**距離校準**(用各銀行自身 intra-bank LOO 尺度,重用 `intra_class_knn_distances`)② **各銀行各自 Top-K**(非全域池化,否則大銀行靠密度灌票)③ 距離加權投票 ④ **abstain 兩道閘**:全銀行皆遠→unknown/novel(novelty radius ~0.3);票差 < margin τ→unknown ⑤ **銀行衛生不可砍**(對銀行本身做 dedup + label-noise,一張錯標就毒化全部)。Normal 應為**真銀行**(否則 novel 與 clean 混淆)。

**VIX 架構**:~80% 已有。新建很小:① 純核心 `core/bank_audit.py`(多銀行 Top-K 投票,只吃向量,FiftyOne-free,可測)② `pipeline.bank_audit`(選低信心偵測、用 tag 建銀行、attach 欄位、`defect_like`→`hard_positive` tag、記稽核)③ CLI `bank-audit`(仿 `error-mine`)。銀行=**可設定 tag**(`--defect-tag` 預設 golden、`--reflection-tag` 預設 rejected、`--normal-tag` 預設 pass);**POC 先不加 Tag.NORMAL**。低分 proposal**已存在**,不另建低信心推論路徑——以信心區間挑選即可。`bank_verdict` 為**諮詢欄位**,不覆寫 route;hard-positive 是 staging tag,**人工 resolve→golden,不自動晉升**。

**MLOps/SMM**:① **review-load 預算**是成敗關鍵(超低 conf 會 10–100×)→ 每 lot top-N cutoff + **abstain 為主**(只有 defect-like + 薄 unknown 進人工佇列,其餘記錄不顯示)+ 用 `spc` EWMA 當控制迴路。② **沒有 GT 不能宣稱 recall**:用 `--eval` 凍結有標籤 challenge set,經 `eval-ingest` 算 **per-class recall/AP delta**(唯一誠實的 recall);engineer-confirmed-rate 是 precision-like,**分開報**。③ Phase 1–4 分階段、各設 gate。④ KPI 驗收門檻:confirmed-defect-rate ≥ 25%、eval 上 recall delta ≥ +3pts、review load 在分析工時預算內才 GO。⑤ 銀行會隨 recipe/tool 漂移→打指紋、用 `guard`/`parity` 把關。

## 共識(四方一致,無異議)

1. **新建面很小**:純核心 `core/bank_audit.py` + `pipeline.bank_audit` + CLI `bank-audit`;其餘全部重用(embedding、kNN、tags、review-queue、explain、App Embeddings 面板、eval-ingest/error-mine、dedup/label-noise)。
2. **低分 proposal 已存在**(run_yolo conf=0.001),以**信心區間挑選**操作,不另建推論路徑。
3. **cosine、raw DINOv2**,POC 不做 metric-learning fine-tune。
4. **Top-K + 投票 + abstain** → defect-like / reflection-like / normal-like / unknown;unknown=novel/blind-spot 池;**hard-positive = defect-like ∪ unknown,人工 resolve→golden,不自動晉升**。
5. **銀行衛生強制**(對銀行做 dedup + label-noise)再 audit。
6. **proposal 去重**(loose NMS + embedding/IoU)在 audit 前;一 proposal=一單位。
7. **review-load 預算**:每 lot top-N + abstain 為主;重用 review-queue + spc 控制迴路。
8. **recall 只在凍結 `--eval` set 上經 eval-ingest 量**;confirmed-rate 另計、不可冒充 recall。
9. **離線 POC**:不動 production 路由、不上即時路徑、不處理 zero-proposal safety net、不自動晉升;**核心保持 FiftyOne-free**。
10. **crop 品質**:context padding + min-size 上採樣(透明小 defect)。
11. **Phase 1–4 分階段 + KPI 驗收門檻**才談 productionize。

## 待解爭議(Round 2 要收斂)

- **D1 投票演算法(核心)**:架構派的「全域池化 Top-K + 以銀行來源多數票」(簡單,但大銀行靠密度灌票) vs embedding 派的「各銀行各自 Top-K + 每銀行距離校準 + 距離加權」(嚴謹,較多碼)。ops 想「一個旋鈕」。
- **D2 Normal 是否真銀行**:embedding 派要真 Normal 銀行;架構派擔心 `pass`=「尚未覆核」非「已驗證 normal」會污染投票。→ 真銀行但可設定、且建議用**已驗證**的 normal(非 raw pass);未提供時退化為 2 銀行 + novelty radius。
- **D3 crop/proposal 處理 + 諮詢性**:偵測派要 `proposal_tier` 欄位 + NMS 參數 + crop padding/min-size(動 detect.py/dinov2.py);架構派要最小變更、選信心區間即可。→ 需定:哪些動 detect/dinov2、低分 proposal 是否打 distinct tag、verdict 維持諮詢不覆寫 route。
- **D4 conf 下限 + NMS IoU**:多數同意 0.05、類別無關 loose NMS、用 sweep 選 → 近共識,確認即可。
- **D5 具體交付清單**:把「現在要建的」與「你要跑的 POC 分析」分清;sweep 工具、銀行衛生 gating、eval tie-in 是否一起建。
- **D6 銀行漂移**:打指紋 + guard/parity gating 是 phase 1 還是 phase 2。

> 下一步:Round 2 針對 D1–D5 收斂(各方看過彼此主張後投最終票 + 是否仍有 blocker)。

## Round 2 — 對 strawman 逐點投票:**四方全部 ACCEPT(達成共識)**

D1–D5 全數通過。僅四個「實作時必須遵守」的硬約束(非設計爭議):

1. **(偵測)crop 變換對稱**:建銀行與 audit 查詢必須用**同一套 crop 變換**,否則 query 與 bank 向量來自不同前處理,cosine 投票全偏。→ POC 直接用**既有已存的 embedding**(銀行與 proposal 都來自同一次 `compute_embeddings`,皆為 bare crop → 自動對稱);crop padding/min-size 列為**選用增強**,若啟用則須對銀行與 proposal **一起重算 embedding**。
2. **(embedding)scale 下限**:`scale_b = max(median(intra-bank LOO dist @K), 1e-3)`,避免近重複銀行 scale→0 產生 NaN/inf。
3. **(架構)純投票器收「預先算好的 scales」**:`pipeline.bank_audit` 在 build-time 算一次 `build_bank_scales`,把 scales 傳進 `bank_vote`;voter 不自校準(否則每次查詢都重算、與「build-time 自動校準」矛盾)。
4. **(營運)銀行指紋進 BUILD-NOW**:每次 audit 記錄 bank fingerprint(hash + 各銀行筆數 + embedding 模型版本)+ staleness 時戳;drift **gating** 才延後到 phase 2(否則 phase 2 無 baseline)。另:review 預算要**硬上限**(`per_lot_top_n` + `unknown_slice_frac` 兩個 actuator,EWMA 監控 review-rate);recall 驗收須**逐類別**(pooled +3pts 且**無單一目標類別退步**);`--eval` set 須版本化/雜湊化並記各類 support。

## 最終共識規格(Design of Record,可直接實作)

### 純核心 `src/vix/core/bank_audit.py`(FiftyOne-free,numpy-only,可單元測試)
```python
@dataclass class BankVerdict: verdict; winning_bank; margin; min_raw_dist; per_bank; topk_evidence
build_bank_scales(banks: dict[str,np.ndarray], k) -> dict[str,float]   # max(median(LOO@K), 1e-3),重用 scorer.intra_class_knn_distances
bank_vote(query, banks, scales, cfg, bank_label_map) -> BankVerdict     # 各銀行各自 Top-K → s_b=exp(-d_b/scale_b);verdict=argmax,經兩道閘:
                                                                       #   ① novelty radius(min_b d_b > r_novel,raw cosine,~0.3)→ unknown
                                                                       #   ② margin τ(s_b*−s_b2 < τ)→ unknown;否則 = bank_label_map[winning]
audit_batch(queries, banks, scales, cfg, bank_label_map) -> list[BankVerdict]
loose_nms(dets, iou_thr=0.7) -> list[Detection]                        # 類別無關 loose NMS(純函式)
```
- 銀行標籤映射:`{defect_tag:"defect_like", reflection_tag:"reflection_like", normal_tag:"normal_like"}`;abstain→`unknown`。
- cosine、raw DINOv2;距離加權(soft)分數;單一 ops 旋鈕 = `tau`。

### `pipeline.bank_audit(adapter, cfg, normal_tag=None, conf_band=(0.05,None), tau=0.10, pad=0.0, min_size=0, dedup_distance=0.05, per_lot_top_n=50, unknown_slice_frac=0.2, top=None)`
1. 用 tag 建銀行:Defect=`_emb_by_class({golden})`、Reflection=`{rejected}`、Normal=`{normal_tag}`(可選;None→2 銀行+novelty radius,並**warn 若用 raw pass**)。
2. **銀行衛生**(在 build_bank_scales 之前):對每個銀行做 `dedup` + `label-noise`,污染即警告。
3. `scales = build_bank_scales(banks, cfg.knn_k)`。
4. 選 conf 區間偵測 → 打 `Tag.PROPOSAL`;`loose_nms` + `near_duplicate_groups` 去重(一 proposal=一單位)。
5. 逐 proposal `bank_vote(...)` → `attach_fields(h, {bank_verdict, bank_evidence})`(**唯讀諮詢,絕不覆寫 routing_decision**);`verdict∈{defect_like,unknown}` → 打 `Tag.HARD_POSITIVE`(staging,**人工 resolve→golden,不自動晉升**)。
6. review 預算:只浮出 `per_lot_top_n` 的 defect-like + 薄 `unknown_slice_frac`;其餘記錄不顯示。
7. `DecisionLog.append("bank_audit", extra={bank_fingerprint, counts, mAP_link...})`。

### CLI `vix bank-audit`(仿 error-mine):`--normal-tag --tau --conf-lo --conf-hi --pad --min-size --dedup-distance --top`
### 動到的檔案:`core/bank_audit.py`(新)、`pipeline.py`(+stage)、`cli.py`(+子指令)、`embedding/dinov2.py`(crop_detection 加 `pad`/`min_size` 參數,預設關)、`types.py`(`Tag.PROPOSAL`/`Tag.HARD_POSITIVE`)、`tests/test_bank_audit.py`(新)。**不動 adapters/scorer/analytics/detect。**

### 砍掉(POC 不做):production 路由變更、即時路徑、zero-proposal safety net、自動晉升 golden、Tag.NORMAL 常數、bank-drift **gating**、metric-learning fine-tune、LSH(POC 銀行 <2000 用 brute force)。

### POC 你要跑的(用這個功能,非新碼):Phase1 低 conf mine → Phase2 audit → Phase3 App Embeddings 面板 + @vix/review 看 Top-K 證據覆判 → Phase4 `eval-ingest` 算逐類 recall delta + KPI;conf 下限用 sweep 選。
### 驗收 GO 門檻:engineer-confirmed-defect-rate ≥ 25%(yield,另計、不冒充 recall)、凍結 `--eval` 上 pooled recall delta ≥ +3pts 且無單一目標類別退步、review load 在分析工時預算內。

> **狀態:共識達成 → ✅ 已實作 + 測試。**
> 交付:`src/vix/core/bank_audit.py`(純投票器:`build_bank_scales`/`bank_vote`/`audit_batch`/`loose_nms`)、
> `pipeline.bank_audit`、CLI `vix bank-audit`、`dinov2.crop_detection` 加 `pad`/`min_size`、
> `Tag.PROPOSAL`/`Tag.HARD_POSITIVE`、`tests/test_bank_audit.py`(6 測試,含 core 投票/abstain/scale-floor/NMS + pipeline 端對端 + 非代表框 staging 回歸)。
> 四個硬約束全部落地;經兩位審查代理對照規格 + 對抗式正確性審查,修掉 1 真 bug(多偵測影像 hard-positive 只看代表框→改成任一 proposal 為 defect/unknown 即 staging)+ 3 防禦性 edge(NaN 銀行消毒、零向量 query 守門、novelty 註解澄清)。SOP §B7 有操作說明。152 pytest 全綠。
