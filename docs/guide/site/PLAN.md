# VIX 新手文件網站 — 計畫(multi-agent Phase 1 共識)

目標:讓第一次使用的 CV 工程師「看著文件就懂」。多步驟流程用 Playwright 實際操作 + 截圖。最後成一份結構清晰的 HTML 網站。

## 站台結構(多頁、共用側欄、深色 zh-Hant、file:// 可開、相對路徑、沿用既有截圖)
- `index.html` — 首頁 / 這是什麼 + 5 分鐘 Quickstart(`vix diagnose` 一行)★
- `install.html` — 安裝(兩軌:core-only Tier A;Tier-2 .venv311 給 App/DINOv2)
- `diagnose.html` — 診斷你的模型(on-ramp)★ + 報告截圖
- `report.html` — 讀懂弱點報告 + 誠實 hedge(H1–H4)
- `formats.html` — 輸入格式 yolo/voc/coco(`--names`/`--data-yaml`/`--label-dir`/0-box 診斷)
- `loop.html` — 修了有沒有幫助?(凍結 eval set 的 Δ;Round 5)
- `audit.html` — 稽核標籤(Tier B `--audit` + near-dup-labels;沿用 DINO 圖)
- `app.html` — 在 FiftyOne App 裡覆核(沿用 GUI_WALKTHROUGH 10 張圖)+ 看 Embeddings 分群(沿用)
- `honesty.html` — 誠實邊界與限制(H1–H7 彙整)
- `reference.html` — 參考(連到 VIX_SOP.html、`vix --help`、設計文件)

## 截圖(Playwright)
- 新:`vix diagnose` 弱點報告 HTML(離線,無需 FiftyOne)——最高價值,本站新增。多類別、含未覆核 banner、typed FP/FN、混淆、confident-wrong、跨兩次可比執行的 per-class Δ。
- 沿用(勿重拍,Tier-2):App = docs/guide/walkthrough/*.png(GUI_WALKTHROUGH);Embeddings = docs/guide/img/step*.png;DINO 稽核 = docs/guide/dino_labelaudit/*.png。

## 誠實要點(文件必須教,與程式碼一致)
H1 匯入標籤=未覆核參照(非 golden) H2 background FP 可能是你漏標的框 H3 可比性需凍結 eval set(eval_set_hash 含 GT) H4 排序是 PROXY 非實測 mAP 增益 H5 Tier B 對未覆核標籤只 label_audit_needed H6 低支撐 Δ 不畫箭頭 H7 memory pixel-fallback 不適合稽核。

## 不做
不重抄 VIX_SOP(連過去);不轉錄 76 verb 表(指向 `vix --help`);不重複 PNG(就地相對連結);頁尾註明「CLI 以 vix --help 為準」。
