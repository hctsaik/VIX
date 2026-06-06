# 接下來值得做什麼:Panel 功能 + 易用性(多代理三輪 → 首批已實作)

> 問題:特別是「增加 Panel 功能」或「讓產品更好用」,還有什麼值得做?經三輪多代理討論。

## 結論:產品是 **CLI 深、App 薄**;最高槓桿一半其實在 CLI。
~70 個 CLI 動詞藏著全部深度,整個 App 只有 1 個報告 Panel + 3 個 operator。R1 四視角打 **32/32/42/88**——**核心架構是對的(瘦 App/厚 CLI 是測試邊界)**,缺口是「導覽黏合 + 上手坡道」。

## 三輪軌跡
- **R1(獨立四視角:落地工程師 / FiftyOne App 專家 / 產品易用性 / 架構懷疑者)** — 工程師與 App 專家**各自獨立**選同一個 #1:**App 唯一超能力「點一列→視圖跳到該圖」完全沒用上**。
- **R2(對抗三角色:建構者 / 紅隊 / 仲裁者)** — 用程式碼修正 R1 的過時假設,並對 admit 按鈕、圖表達成「DROP」共識。
- **R3(主席讀碼裁決)** — 親驗 cli.py 行為。

## 讀碼裁決(R3)
- bare `vix` 確實撞 `required=True` 吐 verb 牆;`_QUICKSTART` 字串已存在只是沒接上;前置失敗只給通用 `錯誤:{e}` 洩漏內部檔名。
- **admit 按鈕(P4):三位一致 DROP**——admit 寫訓練池帳本,一鍵侵蝕 CLI 的刻意性(PARTIAL 是危險中間態、`force` 必須是打字旗標)。改唯讀檢視 + 複製提示。
- **AP 趨勢線(P6):殺**——VIX 不重訓,上升線=偽因果(紅隊:「人只看線、不看註腳」);快照長條 DEFER(per-class AP 表已存在)。
- **硬接「下一步」字串(U3):REJECT**——pipeline 是分支 DAG;改進 `vix status` 的條件式分支。
- **P5(App 狀態列)併進 U4**(別在不可測面做第二份)。

## 架構護欄(全體接受)
- **G1** 任何判定/閘/分數/門檻**不得在 plugin 內計算**——必須呼叫 `pipeline.*`。
- **G2** 所有 DecisionLog 寫入走 CLI 同一條(已測)路徑(保護 hash 鏈)。
- **G3** App 的工作 = 選取 / 導覽 / 視覺分流;互動結果是一組 `vix_hash` 交回核心,不是新產物格式。
- **G4** 「讀核心寫的檔來顯示」→ Panel;「新計算」→ 先當 CLI verb 出貨+測,Panel 再 render。

## ✅ 首批已實作(commit 待填;全套 **249** 綠)
| 標籤 | 是什麼 | 為何 |
|---|---|---|
| **U1** | bare `vix` / 未知動詞 → 印 9 步黃金路徑(`_GoldenPathParser.error` + `required=False`),不再吐 70-verb 牆 | 每個新人第一鍵就撞牆;字串已存在,純路由,離線可測。最寬 blast radius。 |
| **U2** | `route`/`calibrate`/`export` 前置守衛:`route` 缺門檻 → 「請先執行 vix calibrate」(非 `Errno 2`);`calibrate` 無 golden、`export` 無 golden 同理 | 新人最常見的順序錯誤變自我修正。純核心、可測。 |
| **U4** | `vix status` —— 讀 tag + 工作區產物,印「樣本統計 + 你在這 → 下一步跑 X」(**條件式分支**,非硬接) | 每週「我上次到哪」的缺失動詞;吸收 P5+U3 誠實版。`pipeline.status` 純核心、可測。 |
| **P1(旗艦)** | `VixQueuePanel`(`@vix/review/vix_queue`):review queue 變**可點 `TableView`**,列動作 `inspect`→`ctx.ops.set_view` 跳到該圖、`confirm`/`dismiss`→既有 `resolve_review` 就地處理 | 開啟 App 唯一沒用上的超能力。**Panel 內零邏輯**(row 來自 `review_queue`、動作走 `resolve_review`,皆已測核心)。 |

**P1 的紅隊條件(已落實)**:Panel 零邏輯;唯一 live-only 黏合 `_sample_id_for_hash` 保持 3 行(漂移時最易找);列仍顯示 `vix_hash` 作降級路徑(`set_view` 壞了可回 CLI);verify-gui 斷言兩個面板都在 live runtime 註冊。離線已驗:面板註冊 + TableView render schema `to_json` 乾淨序列化(`row_actions` 正確)。**live row-action 綁定(`ctx.params["row"]`、`ctx.panel.data` 路徑)只能在 `vix verify-gui` 的真 App 確認**,可能需小幅調整。

## DEFER(真的、稍後)
- **P4-status**:唯讀 batch-gate 判定 + `batch_ledger` 檢視(可點 offender + 複製 `vix batch-admit w23` 提示,**無**按鈕)。
- **P3**:由 worklist 真正建立 `vixq:*` saved views(`worklist_views` 已是純函式);需 verify-gui 加 `list_saved_views()` 斷言。
- **P2**:`resolve_placement` 把 confirm/dismiss 釘到 grid 動作列(隨後續搭車)。

## ❌ REJECT
admit 按鈕 / AP 趨勢線 / 硬接下一步 / auto-relabel-merge / in-panel UMAP(原生 Embeddings 面板已有)/ 重算式 dashboard / 門檻滑桿 / TUI 精靈 / plugin 自存註記。理由:侵蝕誠實或把可測 CLI 邏輯搬到不可測面。

## 為何先做這四個
U1+U2+U4 是最便宜、可測、受惠最廣(同時幫新人與每週使用者)且 claim 不出證明不了的東西;P1 是當之無愧的旗艦 Panel、**零新核心邏輯**——一個 PR 同時兌現「Panel 訴求」+ 三個高槓桿可測修補。
