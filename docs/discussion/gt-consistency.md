# GT × 嵌入:一致性歸因(多代理結論 + 已實作 v1)

> 問題:除了 DINO embedding,若還有 labeling GT,能幫這系統什麼?對資料一致性如何設計?是否有價值?如何放大?
> 經三輪多代理辯論(實戰工程師 / data-centric 研究者 / 產品懷疑者 / 架構師 → 收斂)。

## 結論:**確實有價值(HIGH,邊界清楚)→ 走「放大」路線**
連最硬的懷疑者都從 MEDIUM 升到 HIGH(他原本的反對逐一被拆:不是獨立 pillar 而是整合進既有 weakness-report、不是 commodity 因為 separability 沒人做、不受「不重訓無法證明」限制因為 separability 是可證偽的幾何陳述)。

**核心洞見**:VIX 既有的標籤訊號都是**自我參照**(只能說「和鄰居不一樣」),所以在「兩類重疊處」結構性失明 —— 正是 SAFE pothole/crack 0.27 的死因。**GT 打破循環**,把「偵測」升級成「**歸因**」:這個失敗是 **taxonomy / model / label** 問題?三者在 AP 數字上長一樣,但正確處置完全相反(停止多標 / 沿邊界補資料 / 重新裁決),今天工程師只能用直覺選、錯約 ⅓ —— 這就是 painkiller。

額外:類別定義爛掉會污染**每個** embedding 驅動訊號(drift/coverage/error-mine/active-learn/bank-audit),所以一致性歸因是**驗證地基**的最高槓桿,而非 sidecar。

## 已實作 v1(整合進 weakness-report,非獨立 pillar;諮詢式 + CI + 支撐閘)
| 能力 | 做法 | 檔案 |
|---|---|---|
| **可分性 separability** | 逐類對 LOO-kNN 誤差(cosine, k=max(3,⌊√n_min⌋) 取奇);高=「**在目前 embedding 空間**不可分」(綁定編碼器,非「taxonomy 壞了」)+ Wilson CI | `core/consistency.py` |
| **重疊 × 混淆 2×2** | O[i→j](golden GT 點的 k-NN 落在 j 的比例,bootstrap CI)對 model 混淆 C[i→j](eval,Wilson CI);Δ=O−C 的保守 CI → **taxonomy / model / label_noise / clean** | `core/consistency.py` |
| **支撐階梯** | n_min<10 或 pair<25 → `insufficient_support`(不給判決);10–20 → `provisional`(禁建議 merge);≥20 → `supported` | `core/consistency.py` |
| **整合 + 介面** | findings 進 `weakness_report`(.md + **.html**);`vix consistency` 直接看;審計記判定 | `pipeline.consistency` / `weakness_report`、CLI |

**誠實契約**(對應研究者的小樣本疑慮 + 懷疑者的 encoder 疑慮):永遠附 support + CI;`0∈CI(Δ)` 不給方向性歸因;**merge 建議需 supported + taxonomy + CI 半寬足夠窄**;separability 措辭一律綁「目前 embedding 空間」;**絕不自動改標 / 自動 merge**。資料來源 = 人工確認的 **golden** 嵌入(=GT 標籤的真實例)+ eval 混淆矩陣;不需多標註者(κ 路線需 `Detection.source` schema 變更 + 多標註者資料,**延後**)。

## 測試
`tests/test_consistency.py`(估計子 + taxonomy/model/label_noise/insufficient 判定 + pipeline + HTML 寫出)、
`tests/test_weakness_report_gui.py`(**Playwright** 開 headless Chromium 載入真實 `weakness_report.html`,斷言一致性表格 + `taxonomy` 判定 + 樣式化判定格 + PROXY 標記在瀏覽器中渲染)。全套綠。

## 放大價值的下一步(優先序)
1. **佇列命中率回饋飛輪**:把「上週佇列 N% 命中、趨勢」回饋進下一輪排序 → 會自我變準的策展(實戰派的「不可或缺」鍵)。
2. **領域自適應 embedding**(最大槓桿):用 golden GT 在凍結 DINO 上學輕量投影頭 / metric learning(秒~分鐘、CPU、$0、**非訓練 YOLO**)→ 提高可分性天花板、**同時銳化整個 embedding stack**、直接打 SAFE 0.27。一致性層**指出何時編碼器是瓶頸**,這步**修它**。
3. **去風險檢查**:出強勢「停止標註」措辭前,對被判不可分的對做一次離線「真的訓個小偵測器看分不分得開」sanity check —— 唯一能讓結論翻盤的條件,檢查一次買保險。
