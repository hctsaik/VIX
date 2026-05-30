# 執行手冊(RUNNING)

VIX 分兩層,先搞懂這個就不會卡:

| 層 | 內容 | 需要 | 在這台機器 |
|----|------|------|-----------|
| **Tier 1 — 核心** | 全部資料治理邏輯(路由/標錯/去重/覆蓋/漂移/版本/稽核/報告/閘…)、CLI、89 個測試 | 只要 `numpy + pytest + pillow + pyyaml` | ✅ 現在就能跑 |
| **Tier 2 — 視覺化/重模型** | FiftyOne App(GUI 覆核)、DINOv2 真實 embedding、LanceDB、YOLO 推論 | `pip install 'vix[fiftyone,yolo]'`,**Python 3.10–3.11**(FiftyOne 尚不支援 3.14) | ⛔ 需另建環境 |

> 設計上 core 完全不依賴 FiftyOne,FiftyOne 藏在 `adapters/fiftyone_adapter.py` 後面,所以 Tier 1 可獨立驗證、離線可跑。

---

## 1. 最快:跑測試(證明邏輯可用)
```powershell
cd c:\code\claude\VIX
python -m pytest -q          # 預期 89 passed,<2 秒,不需 FiftyOne/GPU/網路
```

## 2. 看它端到端跑一遍(離線 demo)
```powershell
$env:PYTHONIOENCODING="utf-8"     # Windows 讓中文正常顯示(或先 chcp 65001)
python examples\demo.py
```
會用合成影像跑 calibrate→route→dedup→coverage→gate→report→export 並印出結果。

## 3. 用 CLI
免安裝:
```powershell
$env:PYTHONPATH="src"; $env:PYTHONIOENCODING="utf-8"
python -m vix.cli quickstart      # 概念詞彙 + 最短工作流
python -m vix.cli --help          # 全部指令
```
裝成 `vix` 指令:
```powershell
pip install -e .                  # 之後可直接打 vix quickstart
```

---

## 4. 跑在真實資料上(Tier 2)
在 **Python 3.10/3.11** 虛擬環境:
```powershell
py -3.11 -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -e ".[fiftyone,yolo,dev]"
```
一條龍(最常用):
```powershell
vix run --input .\incoming --batch w22 --weights yolo.pt --export .\train_ready
```
分步(需要細控時):
```powershell
vix ingest .\golden  --batch init --golden      # 黃金集
vix ingest .\anchor  --batch init --anchor      # 凍結錨點(漂移基準)
vix ingest .\incoming --batch w22               # 新批次
vix infer  --weights yolo.pt                     # YOLO 偵測
vix embed                                        # DINOv2 + LanceDB kNN
vix calibrate                                    # per-class 門檻
vix route                                        # pass/review + 理由
vix app                                          # ← GUI 覆核(見下)
vix gate                                         # 能不能訓練? GO/NO-GO
vix report .\out                                 # 品質分數 + 報告
vix export .\train_ready                         # YOLOv8 + 逐檔 hash
```

---

## 5. GUI 在哪?

**現在就有 GUI:`vix app`**(Tier 2)。它直接用 **FiftyOne App**,提供:
- 互動式 embedding 散佈圖(UMAP)— 圈選(lasso)一群點 → 批次打 tag
- 影像縮圖牆 + bbox/分數 overlay
- 側欄依任意欄位/tag 過濾;預設已建好 `review_queue` / `passed` 視圖
- 在瀏覽器操作(`localhost:5151`),覆核結果寫回資料集

啟動:`vix app`(需先 `pip install 'vix[fiftyone]'`)。

**VIX 專屬覆核工作台(已實作,外掛):`@vix/review`**。把 App 升級成有按鈕的覆核台:
- **確認 → 併入 golden**(可選填更正類別 = 重標)
- **標記誤報並排除**(→ rejected)
- **為何被攔(下鑽解釋)**(各軸數值/門檻/敏感度)

啟用:直接 `vix app` 即可 —— CLI 會**自動**設好 `FIFTYONE_PLUGINS_DIR`(指向內建 `@vix/review`)與 `VIX_WORKSPACE`(絕對,App operator 與 CLI 共用稽核位置)。在 operator browser(按 `` ` ``)叫出三個 VIX 動作,每個寫進同一份 hash-chain 稽核軌跡。詳見 [src/vix/plugins/vix_review/README.md](src/vix/plugins/vix_review/README.md)。

**一鍵驗收(已在真機 FiftyOne 1.16 跑過)**:
- `vix verify-fiftyone` —— headless 驗 FiftyOneAdapter 全鏈 + sync_reviews 回寫閉環(9/9 PASS)。
- `vix verify-gui` —— Playwright 驅動 App + **實際點擊執行** `confirm_golden`(rev1→golden + 留痕);`--no-execute` 只截圖。詳見 [docs/spec/VERIFY_FIFTYONE.md](docs/spec/VERIFY_FIFTYONE.md)。

**免 GUI 等效(可測試、可嵌 CI)**:`vix resolve <hash> --confirm [--label X]` 或 `vix resolve <hash> --false-alarm` —— 同樣關閉 review→golden 閉環並留痕。

**報告**目前是 Markdown + JSON(`vix report`),可直接看或轉 HTML/PDF。

> GUI 策略:**core 與 GUI 完全解耦**(adapter 模式)。覆核工作台是 FiftyOne App 上的外掛;若日後要改成獨立網頁儀表板(走 v0.2 的 FastAPI 層),核心邏輯一行都不用動。

---

## 6. 常見坑

- **`--adapter memory` 不跨指令保存**:它是離線單進程用的(資料只在記憶體)。適合 `vix run --adapter memory`、`examples/demo.py`、測試;**不要**分開下 `vix ingest --adapter memory` 再 `vix route --adapter memory`(每次都是空的)。要跨指令持久化、要用 `vix app`,請用 Tier 2 預設的 FiftyOne adapter(`--adapter auto`,狀態存本機 MongoDB)。
- **Windows 中文亂碼**:`$env:PYTHONIOENCODING="utf-8"` 或 `chcp 65001`。
- **`pip install fiftyone` 失敗**:多半是 Python 版本太新,改用 3.10/3.11。
- **air-gap / 沒裝 FiftyOne**:`--adapter auto` 會自動降級成像素 embedder 並在報告/稽核標記 `embedding_backend: pixel_fallback`,流程不中斷。

> 更多:[QUICKSTART.md](QUICKSTART.md)(概念+最短路徑)、[docs/spec/v0.1-technical-spec.md](docs/spec/v0.1-technical-spec.md)(規格)、[docs/spec/TESTING.md](docs/spec/TESTING.md)(測試分層)、[docs/validation/](docs/validation/)(10 輪情境驗證)。
