# batch-gate:「這一批能不能進訓練?哪些要清?」(多代理結論 + 已實作 v1)

> 工程師每週真正的問題。經多輪多代理討論(實戰 / MLOps / data-centric / 懷疑者 → 中立主席收斂)。

## 結論:**值得做(對工程師 HIGH、對模型 MEDIUM-保護性)→ 已實作 v1。**
懷疑者「90% 是包裝、10% 一個新檢查」事實上對(VIX ~85-90% 已有批次原語);但**戰略上**:價值單位不是 LOC,是**疲憊時刻的一個決策**。今天工程師得**記住並手動拼六個指令** + 心算一個根本不存在的檢查(批次↔凍結 eval 洩漏)。**靠記性的安全檢查,會在最忙時失效。** 把它合成「一個每週都會跑的指令」+ 那個唯一的因果性檢查 = 對工程師 HIGH、對模型保護性中等。

**誠實邊界(全體共識)**:這是**資料衛生 + 洩漏安全**判定,**不是「進訓練會漲 mAP」的保證**(VIX 不重訓,無法證明一批有沒有幫助)。HURT 訊號(洩漏、退化框)近乎**因果確定**;HELP 訊號(覆蓋、新穎度)是**弱代理**。**不做** batch value score / dashboard(把確定傷害和弱代理平均是不誠實、且會被過度信任)。命名 `batch-gate`(衛生),不叫 `batch-readiness`(暗示價值)。

## 已實作 v1(`vix batch-gate <id>`)
| 類別 | 檢查 | 來源 |
|---|---|---|
| **BLOCK①(唯一新)** | 批次 → 凍結 eval/golden **近重複洩漏**(汙染會悄悄灌高 gate 信任的 mAP) | reuse `cross_split_leakage`,改 key 成 batch-vs-frozen |
| **BLOCK②** | 批次內**退化框**(零面積=壞訓練目標) | reuse `box_qa.audit_boxes`(batch-scoped) |
| **clean-list(諮詢、不擋)** | 未覆核數、疑似標錯、批次內近重複、自信卻錯 | reuse `suspected_label_errors`/`near_duplicate_groups`/`hardneg --batch` |
| **判定** | `BLOCK`(因果傷害)>`PARTIAL`(無凍結 eval/golden → 洩漏不可檢,**絕不靜默 PASS**)>`CLEAN`(無 block 但有可清項)>`PASS` | `core/gate.batch_gate_verdict`(純) |

**誠實契約**:每個判定都帶「protective 非 additive」「洩漏=唯一可量化主張」;**無 eval → PARTIAL 不是 PASS**(保護模型的主檢查不可用);pixel_fallback 後端 → 標註「近重複偵測較粗」。審計記 `batch_gate` 事件(`batch_id` + 各檢查計數 + backend)。

檔案:`core/gate.batch_gate_verdict`、`pipeline.batch_gate`、CLI;測試 `tests/test_batch_gate.py`(8:純判定 + 洩漏/退化 BLOCK + 無凍結 PARTIAL + 乾淨 PASS + 審計 + 未知批次報錯)。全套 222 綠。

## 放大價值的下一步(amplify)
1. **`batch_admit` hash 鏈帳本事件**(治理基石,刻意延後到 v1 被採用後):admit 動作時寫一筆綁 {verdict, 各檢查, **admit 前/後 snapshot content_hash**, eval_set_hash, backend} 到 `batch:w23` → 使准入**可辯護 + 可逆**(un-admit = restore admit 前 snapshot)+ **可查詢**(「哪些批次在訓練集、為何」)。~90% 靠既有 `DecisionLog.batch_id`/`snapshot.content_hash`/`restore_apply`。**先讓 verdict 被信任、被每週用,再給它帳本**(沒被採用的 verdict 配帳本是 audit theater)。
2. App 內一眼呈現 batch-gate verdict;3. 跨週 batch 品質趨勢(在 verdict 被信任後)。
