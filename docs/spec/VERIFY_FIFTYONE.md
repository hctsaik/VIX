# Tier-2 驗收(FiftyOne App + 覆核工作台)

> ## ✅ 已實際執行並通過(2026-05-31)
> 開發機其實有 **Python 3.11.9**,建了 `.venv311` 真的跑起來驗:
> - 裝 **FiftyOne 1.16.0** + Playwright + chromium。
> - **`vix verify-fiftyone`**(headless,真實 FiftyOne 後端)→ **9/9 全 PASS**:adapter 往返、calibrate、route、get_by_tag=16、health_report(品質分數 ~58)、**sync_reviews:rev1→golden、rev2→rejected**、hash-chain 完整。
> - **`vix verify-gui`**(Playwright 驅動 App)→ **PASS**:`plugins discovered: ['@vix/review']`,且 Playwright **實際點擊執行 `confirm_golden`** → `rev1` 由 `['review']` → `['review','golden']` + append-only hash-chain 留痕。截圖見 `<workspace>/gui_shots/`(app.png / operators.png / after_execute.png)。
> - 修到兩個真問題:① `fiftyone.yml` 非 ASCII 在 Windows cp950 下載不進 plugin → 改純 ASCII;② operator 用相對 workspace → CLI 現在自動設 `VIX_WORKSPACE`(絕對)與 `FIFTYONE_PLUGINS_DIR`,App 與 CLI 共用 workspace、`vix app` 自動載入 plugin。
>
> 唯一仍需人手的只剩「主觀目視確認標註對錯」(業務判斷,非系統能力)。

---

## 環境(Python 3.10 或 3.11)
```bash
py -3.11 -m venv .venv311 && . .venv311/Scripts/activate     # Linux: source .venv311/bin/activate
pip install -e ".[fiftyone,yolo,dev]"
pip install playwright && playwright install chromium        # 只有 verify-gui 需要
python -m pytest -q                                          # 先確認 102 tests 仍綠
```

## 一鍵驗收(兩個子指令)
```bash
vix --workspace ./vixws verify-fiftyone      # headless:FiftyOneAdapter 全鏈 + sync_reviews 回寫閉環
vix --workspace ./vixws verify-gui           # Playwright:截圖 App + 實際執行 confirm_golden operator
vix --workspace ./vixws verify-gui --no-execute   # 只截圖、不點 Execute
```
- 兩者**自動**建合成 dataset(不需 GPU/YOLO 權重/下載 DINOv2,用像素 fallback embedding)、跑完自動清除。
- 通過 = `=== 全部 PASS ===` / `=== GUI 驗證 PASS ===`,exit code 0(NO-GO/FAIL 為非零,可嵌 CI)。
- CLI 會自動設 `VIX_WORKSPACE`(絕對)與 `FIFTYONE_PLUGINS_DIR`(指向內建 `@vix/review`)。

## 人工 GUI 操作(可選,需要人眼判斷時)
```bash
vix --workspace ./vixws app          # 開 App;plugin 已自動載入
```
在 App 按 `` ` `` 叫出 operator browser,選 **VIX: 確認→併入 golden / 標記誤報並排除 / 為何被攔(下鑽解釋)**,選取樣本後執行;之後 `vix sync-reviews` 或 operator 即時回寫並留痕(`vix audit --event review` 可查)。

## (選用)接真實 YOLO + DINOv2
```bash
vix ingest ./imgs --batch t1 --golden
vix infer --weights yolo.pt          # 真實 YOLO
vix embed                            # 真實 DINOv2 ViT-B/14 + LanceDB
vix run --export ./train_ready       # 一條龍
```

> 驗證邏輯在 `src/vix/verification.py`(fiftyone/playwright 皆 lazy import,核心與 Py3.14 pytest 不受影響)。
