# Round 14 — 情境 AF1–AF10:**未達標(平均 81.6 < 95)**,但軌跡持續上升

> 軌跡:R11 78.1 → R12 70.0 → R13 76.2 → **R14 81.6**。R13 的韌性修補經實機確認成立
> (AF3 當機半行+BOM 優雅降級=94、AF6 稽核鏈可獨立離線重算到相同結論=90.7、AF9 force-delete 偵測=92)。

## 一、評分結果

| 情境 | Judge A | Judge B | Judge C | 平均 |
|------|--------:|--------:|--------:|-----:|
| AF1 零 golden 冷啟動 | 62 | 62 | 72 | **65.3** |
| AF2 並行競爭鏈是否存活 | 90 | 58 | 55 | **67.7** |
| AF3 當機半行+BOM 降級 | 95 | 92 | 95 | 94.0 |
| AF4 匯入外部分類法 | 80 | 88 | 80 | 82.7 |
| AF5 「該停止標注了嗎」 | 72 | 84 | 82 | 79.3 |
| AF6 稽核員獨立重驗鏈 | 90 | 90 | 92 | 90.7 |
| AF7 標註者一致性 | 85 | 86 | 80 | 83.7 |
| AF8 新產品線領域漂移 | 70 | 89 | 84 | 81.0 |
| AF9 磁碟被強刪復原 | 92 | 91 | 93 | 92.0 |
| AF10 證明從未進訓練+model card | 88 | 80 | 70 | 79.3 |
| **平均** | **82.4** | **82.0** | **80.3** | **81.6** ❌ |

## 二、共識(實機收斂的真實缺口)

1. **【P0 嚴重,R13 引入的回歸】並行鎖在 Windows 競態會丟資料(AF2)**:`_LockFile.__exit__` 的 `unlink` 與另一執行緒的 `os.open(O_EXCL)` 競態,拋出**未被攔截的 `PermissionError`(Errno 13)**(retry 迴圈只攔 `FileExistsError`)→ append 直接拋例外、**靜默丟失決策**(實測 200 筆丟 1–96 筆)。鏈本身一致,但「所有決策都在」沒做到。R13 測試只測單執行緒,沒抓到。
2. **【P0】零/空資料集假 GO(AF1)**:`pre_train_gate` 無 zero-golden 守門 → 全空資料集回 **GO/exit 0「all checks passed」**;且空統計的類別得 `dist_thr=inf` → 極新樣本**靜默 PASS**。對守門員最危險的錯答。
3. **【P1】階段亂序丟原始 traceback(usability)**:`vix route` 在 calibrate 前、`vix verify` 缺 manifest → 噴 Python 堆疊而非「先跑 calibrate」。手動逐步跑的工程師會撞牆。
4. **【P1】route 覆核率過高警告只進 log 不印 stdout(AF8)**:`counts["warning"]` 有算,但 CLI 只印 pass/review。
5. **【P1】active-learn 的 why 是硬編碼、會自相矛盾(AF5)**:novelty≈0 仍印「與既有資料差異大」。
6. **【P2】snapshot content_hash 機器相關但被當內容身分(AF10)**:摻入絕對路徑+時間戳,同內容跨機器不同 hash;SOP 已誠實揭露,但程式碼本身未註明。

## 三、後續方向(本輪落地,然後 Round 15 全新 AG1–AG10)

| 優先 | 補強 | 影響 |
|---|---|---|
| P0 | `_LockFile` 攔 `PermissionError` 一併 retry(修並行丟資料)+ 多執行緒回歸測試 | AF2 |
| P0 | `pre_train_gate` 加 zero-golden 守門→NO-GO;`calibrate` 空統計類別標未校準→route 進 review | AF1 |
| P1 | `main()` 包裝:預期的 FileNotFoundError/ValueError 印一行訊息 + 非零 exit(堆疊留給 DEBUG) | usability |
| P1 | route handler 印 `counts["warning"]`;active-learn why 依實際 novelty/uncertainty 動態生成 | AF8/AF5 |
| P2 | snapshot content_hash 加程式碼註記(機器相關,跨機比對用 golden 組成+verify) | AF10 |

> **狀態:未達標(81.6)。** 落地上述(含測試)→ AG1–AG10 重驗。最嚴重者為 AF2 並行丟資料(R13 自己引入),優先修。
