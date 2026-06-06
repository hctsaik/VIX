# SAFE ↔ VIX:merge 範圍決策(多代理共識,decision of record)

> 問題:VIX 與姊妹專案 **SAFE**(YOLO→SAM→DINO 安全網 + 會重訓 YOLO 的 data flywheel)是否該 merge?留哪個?未來方向?
> 經三輪多代理辯論(架構師 / 產品策略 / 務實維護者 → 收斂 → 中立主席鎖案),再經 owner 兩次範圍收斂而定案。

## 關鍵約束(owner 拍板)
1. **YOLO 訓練不進 VIX** —— 訓練一次太久,不適合放進工具。
2. owner 要的是**離線診斷**,不是閉環重訓:
   - (A) 知道 **YOLO 的弱點**;
   - (B) 知道**哪些未標註資料最能幫到 YOLO 訓練**。
3. owner 通常**有一個小的有標註(GT)val set**。

## 結論
- **產品 = VIX。不 merge SAFE 的訓練飛輪。** SAFE 的 `retrain_loop`/`distill` 畢業機制/`autolabel`(為訓練產標籤)/三面向升級閘/GUI **全部不取**。
- 移除「重訓」需求後,SAFE 與 VIX 的重疊(bank-audit ≈ match_audit 早已手動搬過)使得「值得從 SAFE 搬進來的」**只剩兩件**,且幾乎全是 VIX 內部的呈現/整合,**唯一真正新演算法 = hardneg(~25 行)**。
- **不要再搬 `hardpos`** —— VIX 的 `bank_audit`(低信心帶 → 多銀行投票 → `HARD_POSITIVE`)已經是那個 recall 引擎。
- **誠實邊界**:不重訓 = **沒有證明,只有代理**。「該標哪些」永遠是嫌疑/優先排序,不是實測 mAP 增益。唯有「這批資料**確實**讓 mAP 漲」需要真的重訓一次量測(VIX 不做,交給外部;`challenge-guard` 只在 owner 重訓後把 eval 丟回來把關)。

## 已實作(本次)
| 項目 | 內容 | 檔案 |
|---|---|---|
| **hardneg** | 「YOLO 最自信卻錯」挖掘。GT 模式:依 conf 排序**已證實的 eval-FP**;GT-free 模式:`conf≥conf_thr` 但 `knn_dist>dist_thr`(嵌入翻盤)的自信誤報,全部由**已存** route 欄位算,無推論/訓練 | `core/hardneg.py`、`pipeline.hardneg`、CLI `vix hardneg`、`eval_ingest` 在 `fp_detail` 加 `conf` |
| **weakness-report** | 兩模式「YOLO 哪裡弱 / 去標這些」報告:GT 區塊(per-class AP+混淆+FP/FN 分型+loc_gap+自信卻錯)+ GT-free 區塊(嵌入翻盤)+ **逐弱類「去標這些」佇列** + 全程 PROXY 標記 | `core/weakness_report.py`、`pipeline.weakness_report`、CLI `vix weakness-report` |
| **class-aware error-mine** | `error_mine(for_class=…)`:把誤差區域限定到單一弱類的框 → 弱類 C 產出「最接近 C 失敗處的候選」,而非 class-blind 全域池 | `pipeline.error_mine` |

owner 有 val set → **GT 區塊為中心**(含 `conf`-in-`fp_detail` 撈「自信卻錯」eval-FP);GT-free 區塊與佇列為輔,未標註新資料也能跑。測試 +6(`test_hardneg.py`、`test_weakness_report.py`),全套綠。操作見 SOP §B9。

## 未採用 / 已知限制
- SAFE 的訓練飛輪、SAM 級、Streamlit GUI、重複的 analytics 腳本:不取(VIX 已有對應或不需要)。
- 「無 GT 下偵測未標註資料中**被漏抓**的物件」(開放世界):真正的新建模、需推論 → **記為已知限制,不做**;部分由 `bank_audit` 的 `HARD_POSITIVE` recall 訊號補。
- 完整三輪辯論脈絡:見本目錄其他討論檔與對話紀錄。
