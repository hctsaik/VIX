# VIX 快速上手(新人 5 分鐘)

VIX 幫你把「物件偵測資料集」管好:篩 edge case、找標錯、去重、覆蓋分析、漂移監控、版本、稽核、放行。全程**地端、離線、$0**。

## 核心概念(先懂這 5 個)

| 概念 | 意思 |
|------|------|
| **golden** | 已確認、可進訓練的資料 —— 事實基準 |
| **anchor** | 從 golden 凍結的一小份,**永不訓練**,用來偵測「類別定義有沒有被改掉」 |
| **review** | 被雙軸(信心低 / 外觀離群)攔下,待人工覆核 |
| **pass** | 自動通過,品質足夠 |
| **rejected** | 經 `dismiss` 標記的誤報或有害樣本(自動排除於覆核佇列) |
| **quality score / gate** | 0–100 品質分數 + GO/NO-GO 放行結論 |

## 最短工作流

> 想完全離線、不裝 FiftyOne 試跑?每個指令加 `--adapter memory`(用像素 fallback embedder)。

```bash
# 1) 匯入
vix ingest ./golden   --batch init --golden     # 黃金集
vix ingest ./anchor   --batch init --anchor     # 凍結錨點(漂移基準)
vix ingest ./incoming --batch 2026w22           # 待檢新批次

# 2) 推論 + 特徵
vix infer --weights yolo.pt                       # YOLO -> 偵測框
vix embed                                         # DINOv2 ViT-B/14 + LanceDB kNN 索引

# 3) 篩選 + 覆核
vix calibrate                                     # per-class 百分位門檻
vix route                                         # 標 pass/review + 人類可讀理由
vix review-queue --top 40                         # 風險最高的先覆核
vix dismiss <id> ...                              # 標記誤報(之後不再出現)

# 4) 放行 + 交付
vix gate                                          # 能不能訓練? GO / NO-GO + 理由
vix report ./out                                  # 品質分數 + 導覽報告(自動對比上一份)
vix export ./train_ready                          # 匯出 YOLOv8 + 逐檔 hash(可驗證)
```

## 一鍵一條龍

```bash
vix run --input ./incoming --batch 2026w22 --weights yolo.pt --export ./train_ready
# 依序 ingest→infer→embed→calibrate→route→dedup→audit-labels→guard→gate→report→export
# 任一步失敗即停,每步寫入不可竄改的稽核軌跡;NO-GO 時 exit code = 2(可嵌 CI)
```

## 常用診斷指令

| 想做的事 | 指令 |
|----------|------|
| 找標錯 | `vix audit-labels` |
| 找近似重複 | `vix dedup` |
| 類別分布 / 還需幾張 | `vix coverage` |
| 選下一批要標的 | `vix active-learn --budget 500` |
| 定義漂移 / 跨時期 | `vix drift --from <tagA> --to <tagB>` / `vix geometry --from .. --to ..` |
| 為何這張被攔 | `vix explain <hash>` |
| 稽核某時段操作 | `vix audit --since .. --event ..` |
| 這張圖的歷史 | `vix history <hash>` |
| 版本快照 / 還原 | `vix snapshot --version v1` / `vix restore <snap> --apply` |
| 完整指令清單 | `vix --help` 或 `vix quickstart` |

詳見 [README](README.md)、規格 [docs/spec/v0.1-technical-spec.md](docs/spec/v0.1-technical-spec.md)、設計歷程 [docs/validation/](docs/validation/)。
