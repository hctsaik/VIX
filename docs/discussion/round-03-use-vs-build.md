# Round 3 — 抉擇:直接用 FiftyOne vs 自建類似系統

> 觸發:使用者質問共識的根基假設「基於 FiftyOne 開發」——「我需要直接用 FiftyOne,還是能自己開發類似功能?優缺點是什麼?」這是 buy/reuse vs build 的根本架構抉擇,以三方辯論(正方直接用 / 反方自建 / 裁判決策框架)處理。

## 一、本輪討論方向

不是「用不用 FiftyOne」的二元題,而是釐清:
1. 直接用,**免費得到什麼**、**真實代價**是什麼。
2. 自建,**真實成本**多少、**何時才合理**。
3. **授權與鎖定風險**這兩個最常被忽略卻最致命的因素。
4. 有沒有一條**兼得**的路。

## 二、各方重點

### 正方(直接用 FiftyOne)
- 免費得到:Dataset 模型、App(Embeddings panel/Lasso/tagging/SavedView)、Brain 全套、Model Zoo DINOv2、evaluate_detections、CVAT/Labelbox 標註整合。
- 自建這些 = 向量 DB + embedding pipeline + 互動式 UI + 相似度索引 + 標註整合 + 版控,**保守 3–6 個月**,且維護/相容/社群/文件全要自扛;用 FiftyOne 等於把這些**外包給 Voxel51**。
- 客製化用 **plugin/operator/Panel** 疊上去,**不需 fork**,與核心升級解耦。
- 誠實代價:App 不易嵌入既有 portal(獨立 SPA);MongoDB 強依賴。但 VIX 是獨立守門站,這兩點可控。

### 反方(自建 / 警惕鎖定)
- **Enterprise 付費牆**:OSS 定位是「單人本地工具」。多人協作 portal、**Dataset versioning/snapshot**、RBAC、SSO、審計日誌、air-gap 部署 → 全在 Enterprise(per-seat 訂閱、無公開報價)。授權雖不限商用,但**功能壁壘等同商業鎖定**。
- **鎖定風險**:metadata/embedding/tag 全活在 FiftyOne 的 MongoDB ODM schema(「FiftyOne 的 MongoDB 方言」);跨版本要 `fiftyone migrate`、不向下相容;想遷出要完整 ETL;App 嵌入性差。
- **自建成本誠實拆分**:embedding 存取 + Faiss kNN + tag 狀態 + 簡單 grid UI =「**一週級**」;閉環寫回訓練本來就要自己寫;真正不該重造的是**精緻的互動式 embedding 投影 + lasso**、多模態瀏覽器、豐富 eval 儀表板——但 VIX 初期用不到。「為目前需要的 10% 引入整套 90% 的依賴」。
- 何時自建才對:資料不得進第三方資料模型(資安/法規/air-gap)、需深度嵌入既有 MES/QC portal、需求極簡、團隊已有資料平台。

### 裁判(決策框架 + 混合路徑)
- 提出**決策矩陣**(6 因子)。
- 核心主張:**把 FiftyOne 當函式庫/SDK 用,但 VIX 的領域邏輯一律不寫進 FiftyOne schema,用一層薄 adapter 解耦**。
- 點出正反方的真正共識交集:「**爭的不是用不用 FiftyOne,而是業務邏輯要不要寫進 FiftyOne schema——兩邊答案都是『不要』**。」

## 三、關鍵事實查證(主持人裁決)

⚠️ **三方對 `fiftyone-brain` 授權出現矛盾**:正方/反方說 Apache 2.0,裁判說「閉源 freeware,禁反編譯/轉賣」。主持人**親自查證**(PyPI metadata + GitHub LICENSE):

| 套件 | 授權(查證於 2026-05-30) | 來源 |
|------|--------------------------|------|
| `fiftyone`(core) | **Apache License 2.0** | GitHub LICENSE / PyPI classifier |
| `fiftyone-brain` | **Apache License 2.0**(LICENSE 檔首 20 行為標準 Apache 2.0;PyPI `License :: OSI Approved :: Apache Software License`) | GitHub raw LICENSE / PyPI JSON metadata |

**裁定:裁判代理人錯誤。** `fiftyone-brain` 目前是 Apache 2.0。裁判的說法是該套件**早期版本**的舊狀況(曾為 source-available/限制性授權),Voxel51 後來重新授權為 Apache 2.0。

> **衍生注意**(寫入共識):若 pin 到**舊版** `fiftyone-brain`,可能落在舊的限制性授權;商用時應確認所用版本的 LICENSE,並以 Apache 2.0 版本為準。

## 四、共識(三方收斂)

> **直接用 FiftyOne,但當「函式庫/SDK」用,並從第一天就用一層薄 adapter 把 VIX 領域邏輯與 FiftyOne 解耦(混合路徑)。**

理由整合:
1. **授權無阻**:core + brain 皆 Apache 2.0,商用自由(只需避免 pin 到舊版限制性 brain)。
2. **時間價值**:自建等效 3–6 個月;真正難的(互動式 embedding 投影 + lasso、多模態瀏覽)不值得重造。
3. **鎖定可控**:把 `OutlierScorer / ThresholdPolicy / FrozenReference / DecisionLog / DatasetExporter` 寫成**獨立、可單獨存活**的 VIX module,只透過 `FiftyOneAdapter` 接觸 FiftyOne;日後抽換只改一個檔。
4. **與既有 VIX 共識一致**:先前共識已把版控交給 **DVC(非 FiftyOne snapshot)**、v0.1 單人本地、業務邏輯獨立——本輪只是把「解耦」明文化為架構紀律。
5. **Enterprise 付費牆不早咬**:OSS 足夠撐到 v1.0;多標註者治理階段才評估 Enterprise vs 自建 portal vs adapter-swap。

## 五、決策矩陣(何時該偏向自建)

| 決策因子 | → 直接用 FiftyOne | → 偏向自建 |
|----------|-------------------|-----------|
| 商用授權容忍度 | core+brain Apache 2.0,可商用 | 合約要求 100% 開源可稽核、禁任何第三方 binary |
| 資料可否進第三方資料模型/雲 | 資料留地端、App 本地跑 → 無虞 | 法規/資安禁止外流且 Enterprise on-prem 成本過高、需 air-gap |
| 是否需深度嵌入既有系統 | 獨立審核站,以 manifest/API 銜接 | 必須內嵌進既有 MES/QC/標註 portal |
| 團隊規模與時程 | 小團隊、v0.1 要 4–8 週交付 | 大組織、有專屬平台工程師、可承受 6+ 月 |
| 預期資料量級 | < 數百萬樣本、單機/小叢集 | PB 級、多專案共池、需分散式索引 |
| 客製化深度 | 客製集中在門檻/漂移/參照等業務邏輯 | 需完全自定義 UI/標註流程/多模態比較 |

## 六、混合路徑的 adapter 邊界(寫入共識)

| 直接委託 FiftyOne | 必須是 VIX 自己的(可獨立存活) |
|-------------------|-------------------------------|
| `compute_embeddings/similarity/uniqueness` | `OutlierScorer`(YOLO 信心 + DINO 距離的分數定義) |
| `fo.Dataset` 作工作空間 | `ThresholdPolicy`(門檻策略 + 版本紀錄) |
| `fo.launch_app()` 覆核視覺化 | `FrozenReference`(凍結參照 manifest,存 JSON/parquet,不靠 fo tag) |
| 標註整合(CVAT/Label Studio) | `DecisionLog`(每筆路由決策紀錄,可獨立查詢) |
| | `DatasetExporter`(輸出格式與 fo 無關) |

`FiftyOneAdapter` 為唯一膠水層;抽換 FiftyOne 只改這一檔,核心 module 零修改。

## 七、對應 VIX 分期

- **v0.1**:直接吃 FiftyOne SDK + App;VIX 核心邏輯寫成獨立 module,僅透過 adapter 接觸;FrozenReference 存 JSON 不靠 fo tag。
- **v0.2**:導入 `DatasetExporter` 完善閉環;若需多人協作再評估 FiftyOne Enterprise(自托管)vs 自建;adapter 介面保持穩定。
- **v1.0**:才評估 (a) 自建 plugin 深化 UI;(b) 若授權/效能/規模出現瓶頸,以 adapter-swap 換自建 embedding 服務;(c) 是否脫離 FiftyOne 後端。

## 八、下一輪/待澄清

本輪共識穩固,無新衝突。唯一仍取決於使用者實況的因子(沿用先前工作假設,可覆寫):
- 工廠影像是否有**保密/不得外流**限制?→ 若極嚴格且需 air-gap,把「自建 or Enterprise on-prem」評估提前到 v0.2。
- VIX 覆核 UI 是否**必須內嵌**既有 MES/QC portal?→ 若是,App 嵌入性差會把天平推向自建 UI(但仍可用 FiftyOne 做後端運算)。
- 預期**資料量級**?→ 決定何時碰到 FiftyOne 後端天花板。
