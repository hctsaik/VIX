# VIX FiftyOne App — 完整 GUI 測試計畫 + 驗收結果(多代理,迭代至平均 ≥95)

> 用 Playwright 做非常完整的 GUI 測試;多代理列測試計畫、定義情境、打分數;未達平均 95 不停,反覆加情境到滿足。

## 結果:**平均 95.93 / 100(三位評審 96.1 / 96.4 / 95.3),≥95 達標。**
- **40 個情境全綠**;全套 **289 passed**。
- **環境**:本機 .venv311 同時跑得起 live FiftyOne App + Mongo + Chromium,所以是**真正的端到端 GUI 測試**(不是只測靜態 HTML)。
- **過程中找到並修掉 4 個真實 robustness bug**(見下)。

## 如何確保測試「完整」(completeness 框架)
多代理定義的覆蓋矩陣:**SURFACES × STATES × EFFECTS × INTEGRITY × FRAMING**。
- **SURFACES**:panel vix_report(regen/worklist 按鈕)、panel vix_queue(inspect/confirm/dismiss 列動作、refresh)、operator confirm_golden/dismiss_false_alarm/explain_sample、plugin 註冊。
- **STATES**:正常、無 golden、未校準、無選取、單選、多選、已解決、未知/失效 hash、無 vix_hash 的樣本。
- **EFFECTS**:tag 變更、grid reload、佇列列移除、panel markdown、set_view 導覽、relabel、恰好一筆 log。
- **INTEGRITY**:verify_chain、is_truncated、**精確事件數**、reviewer 出處、**GUI==CLI 同一寫入路徑**。
- **FRAMING**:PROXY 措辭、絕不自動改標/merge。
- **「完美處理」= 四者同時成立**:正確效果 + 優雅失敗(命名原因、零幻影寫入、不崩) + 稽核完整 + 誠實框架。
- **評分**:三位多代理評審(嚴格 QA / 稽核完整性 / 覆蓋完整性)各自 0–100 打分;95+ 需精確事件數 + verify_chain + is_truncated + 出處 + 對抗孿生 + 框架 + 決定性(輪詢非定時睡) + 隔離。

## 測試架構(誠實的雙層)
- **真 Playwright 瀏覽器層**(`tests/test_gui_browser.py` + `vix verify-gui`):啟動 live App,Playwright 用 `domcontentloaded`(FO websocket 永不 idle),斷言 grid 在 DOM 渲染、兩個 panel 在瀏覽器掛載、瀏覽器內鍵盤執行 confirm_golden → golden + **恰好一筆鏈式 log**。
- **Ledger 錨定 handler/整合層**(`tests/test_gui_e2e.py`,35 情境):用真 FiftyOneAdapter + pipeline + DecisionLog 對**活的 Mongo dataset** 驅動真正的 panel/operator handler,斷言錨在 dataset tags + **精確事件數/欄位形狀** + verify_chain + is_truncated + 對抗孿生。多代理共識:React 自訂 panel 列動作用選擇器點擊會 flaky,改以 ledger 錨定的 handler 測試做 CI 級確定性驗證,瀏覽器層證明掛載 + operator-browser 路徑。

## 找到並修掉的真實 bug(GUI robustness)
1. **E1** `VixQueuePanel._resolve` 無 try/except → 失效/未知 hash 觸發 `_require_known` raise → panel 崩。修:caught + 友善 err,零寫入,鏈完整。
2. **E2** `_selected_hashes` 對非 VIX 樣本 `["vix_hash"]` → KeyError。修:safe get_field、跳過。
3. **E3** `VixReportPanel.on_worklist` 繞過錯誤捕捉 → 無 golden 時崩。修:try/except → 友善訊息。
4. **S7** `_sample_id_for_hash` 用 `.first()`,空查詢會 **raise**(非 None)→ inspect 失效 hash 崩。修:try/except → None(導覽 no-op)。

## 兩輪迭代
- **Round 1**:20 情境 → 平均 **88.0**(QA 92.2 / 稽核 82.65 / 覆蓋 89.15)。評審指出:許多情境只差 verify_chain/is_truncated/no-write 斷言(已成立但沒斷)、以及真實覆蓋缺口。
- **Round 2**:強化既有 20(全加 verify_chain/is_truncated/no-write/值級 GUI==CLI 對等)+ 新增 20(relabel、多選、dismiss operator、mixed batch、refresh、**竄改偵測 canary**、operator schema、saved-views、5 個瀏覽器 DOM)→ **40 情境,平均 95.93**。

## 40 情境清單(全綠)
**Happy/render**:GUI-01 grid DOM〔瀏覽器〕· GUI-02 報告+PROXY · GUI-03 regen 1 事件 · GUI-04/DI-5 worklist tags==views · GUI-05 佇列列==review_queue · GUI-06 inspect→set_view 選中該樣本 · GUI-07 confirm→golden+恰好1事件+出處+鏈+未截斷+移除 · GUI-08 dismiss→rejected(孿生) · GUI-09/B4 瀏覽器內鍵盤 confirm→golden+1 鏈式事件〔瀏覽器〕· GUI-10 explain 下鑽+零 review 寫入。
**Error/edge**:S1 無 golden 優雅 · S3 未校準優雅 · S4 空佇列≠錯誤 · S5 confirm 無選取→友善+零寫入 · S6 explain 無選取→友善 · S7 inspect 失效 hash no-op · E1 confirm 未知 hash 捕捉+零寫入+鏈 · E2 無 vix_hash 跳過 · E3 worklist 無 golden 優雅。
**Integrity**:DI-1/3 GUI confirm == CLI **值級**欄位對等(非第二寫入路徑) · DI-2 混合序列鏈恆有效+未截斷 · DI-4 重覆 confirm 各自入帳/golden 一次。
**Round 2**:relabel(可逆 relabel_changes + decision==label)· relabel 同標籤不寫多餘 · 多選 confirm/dismiss · dismiss **operator**(happy + 無選取)· mixed batch 跳過非 VIX · refresh 反映外部解決 · 兩次連續 confirm 各自事件 · **竄改 canary**:改中間記錄→verify_chain False、截斷→is_truncated True · operator config + resolve_input/output schema · _row_hash id-fallback · **saved-views**:vixq tag→具名 saved view 出現在 list_saved_views 並解析到樣本。
**Browser**:B1 grid DOM · B5 兩 panel 並排掛載 · B2 vix_queue 掛載 · B3 vix_report 掛載 · B4 瀏覽器內 confirm + ledger。

## 已知殘留(誠實、不阻擋 ≥95)
- 自訂 panel 的 **TableView 列動作經 React DOM 點擊**未端到端(已記:選擇器 flaky);列動作**邏輯**在 handler 層全覆蓋、**id-fallback** 有測、panel **掛載**在瀏覽器有證、operator-browser **瀏覽器**路徑由 B4 端到端證。
- 瀏覽器層 framing 以 DOM 文字存在斷言(非版面幾何)。
- `is_truncated` 用 `.hwm` 錨,程式碼誠實聲明不防檔案系統級對手(canary 只測所宣稱的防禦)。

## 重現
- Handler/整合:`FIFTYONE_DO_NOT_TRACK=true .venv311/Scripts/python -m pytest tests/test_gui_e2e.py`
- 瀏覽器:`... pytest tests/test_gui_browser.py`(需 Chromium)
- 端到端煙霧:`vix verify-gui`
