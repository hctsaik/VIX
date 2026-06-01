# VIX 圖解教學

逐步教你在 FiftyOne App 用 **Embeddings(特徵地圖)** 看影像分群、並反查「某一群是什麼影像」。

## 怎麼看

用瀏覽器打開 **[EMBEDDINGS_HOWTO.html](EMBEDDINGS_HOWTO.html)** —— 0~7 步,每步都有一張標了紅框/紅圈/步驟編號的截圖,照著點即可。

直接開檔(Windows):

```powershell
start docs\guide\EMBEDDINGS_HOWTO.html
```

## 內容

| 步驟 | 動作 | 圖 |
|---|---|---|
| 0 | 看懂特徵地圖(分群長怎樣) | `../spec/img/animals_umap.png` |
| 1 | 切換到 `vix_animals` 資料集 | `img/step1.png` |
| 2 | 打開 Embeddings 面板 | `img/step2.png` |
| 3 | 選 `feat_umap` + Color by `ground_truth` | `img/step3.png` |
| 4 | 套索框選一群 | `img/step4.png` |
| 5 | Only show selected(只顯示選取) | `img/step5.png` |
| 6 | 看這群是什麼影像 | `img/step6.png` |
| 7 | 清除還原 | `img/step7.png` |

> 截圖由 `examples/guide_boxes.py`(擷取+抓座標)與 `examples/guide_render.py`(畫紅框)自動產生,
> 可重跑以更新。App 需在 `http://localhost:5151` 服務 `vix_animals`(見 `examples/serve_existing.py`)。
