# @vix/review — VIX 覆核工作台(FiftyOne 外掛)

把通用的 FiftyOne App 升級成 **VIX 專屬覆核工作台**:在 App 裡直接

- **確認 → 併入 golden**(可選填更正類別 = 重標)
- **標記誤報並排除**(→ rejected,退出覆核佇列)
- **為何被攔(下鑽解釋)** — 顯示該張的各軸數值/門檻/敏感度

每個動作都呼叫 `vix.pipeline.resolve_review` / `explain_one`,寫進同一份 append-only、hash-chain 稽核軌跡 —— GUI 只是核心之上的表現層。

## 需求
- `pip install 'vix[fiftyone]'`(Python 3.10–3.11)
- 一個已用 VIX 建好的 FiftyOne dataset(`vix ingest/infer/embed/route`)

## 啟用
最簡單:直接用 VIX CLI —— 它會**自動**設好 `FIFTYONE_PLUGINS_DIR`(指向本外掛)與
`VIX_WORKSPACE`(絕對路徑,讓 App 觸發的 operator 與 CLI 共用同一稽核/紀錄位置)。
```powershell
vix --workspace .\vixws app        # 開 App,plugin 已自動載入;按 ` 叫出 VIX 動作
```
若改用原生 `fiftyone` 指令啟動,才需手動設(等效):
```powershell
$env:FIFTYONE_PLUGINS_DIR = (Resolve-Path src\vix\plugins)
$env:VIX_WORKSPACE = (Resolve-Path .\vixws)
fiftyone plugins list              # 應看到 @vix/review
```

> 已用 Playwright 自動驗證(`vix verify-gui`):選取樣本 → 執行 `confirm_golden` → 樣本變 golden +
> append-only hash-chain 留痕。見 [../../../docs/spec/VERIFY_FIFTYONE.md](../../../docs/spec/VERIFY_FIFTYONE.md)。

## 用法
1. `vix app` 開啟後,用左側 `review_queue` 視圖 + Embeddings 面板 lasso 圈選要處理的影像。
2. 按 `` ` `` 叫出 operator browser,選:
   - **VIX: 確認 → 併入 golden**(選填更正類別)
   - **VIX: 標記誤報並排除**
   - **VIX: 為何被攔(下鑽解釋)**
3. 動作完成後資料集自動重整;`vix audit` 可查每筆覆核留痕。

> 純 CLI 等效指令(免 GUI、可測試):`vix resolve <hash> --confirm [--label X]` 或 `vix resolve <hash> --false-alarm`。
