# 「把專案改回叫 fiftyone」可行性稽核(多代理,3 lens)

> 問題:(a) 用 multi-agent 評估把專案從 VIX/vix 改名為 `fiftyone`;(b) 有多少 hard-code 需要改;(c) 如何確保全部改完。
> **決議:report only — 暫不執行任何改名。** 本文件即評估結論。

## TL;DR(一句話)
**不要照字面把 Python 套件改名為 `fiftyone`。** literal 套件名會和本專案所包裝的真 voxel51 函式庫(18 檔 `import fiftyone`)自我衝突,CLI `fiftyone` 也撞到真函式庫自己的指令;而 `vix_hash`(×579)等是**已落地、進雜湊鏈**的識別碼,改名會默默弄壞既有資料。可行的最小破壞讀法是改成**非衝突名 `fiftyone_vix`(CLI `fovix`)並凍結所有已落地字串**——但這不是字面上的「改成 fiftyone」。

## 1. 有多少 hard-code 需要改(分類,repo-wide ~2,600 處)

| # | 類別 | 數量 | 判定 |
|---|---|---|---|
| A | 套件 + import:`src/vix/`、`from vix`/`import vix`(300,~70 檔)、pyproject `name="vix"` + console script、`prog="vix"`、logger/help 文字 | ~300 | ✅ 機械式安全 |
| B | **🔴 `fiftyone` 衝突**:18 檔 `import fiftyone as fo` / `.brain` / `.zoo` / `.operators`(**真 voxel51 函式庫**) | 18 檔 | ⛔ 硬阻斷 |
| C | **`vix_hash`**:manifest/snapshot/decision-log 的 JSON key + FiftyOne sample 欄位,且折進 `_hash_record` 雜湊鏈 | **579** | ⚠️ 資料風險(最嚴重) |
| D | **`@vix/review`** plugin URI(`fiftyone.yml name:` + registry) | 22 | ⚠️ registry 失效 |
| E | **`vixq:`** 已落地 sample tag 前綴(撐住 saved views) | 59 | ⚠️ 資料風險 |
| F | `VIX_WORKSPACE` env + `vix_workspace` 目錄;dataset 名(`vix_verify`…)、brain key(`vix_sim/umap`)、`info["vix_encoder_fp"]`、panel `vix_report/queue` | ~50 | ⚠️ 資料/registry |
| G | 文件(~600,如 `VIX_SOP.html` ×149)、測試(~430,62 檔)、範例(~60) | ~1,090 | 混合:prose 安全;CLI/欄位文字需小心 |
| H | **無法 find/replace**:repo 目錄改名 + GitHub repo URL、檔名改(`test_vix_panel.py`、`VIX_SOP.html`)、**24 張螢幕截圖把「VIX:」/`vixq:` 烤進像素**、tracked fixtures `jdgws/`+`runws/` | — | 手動 / 重拍 |

## 2. 為什麼 literal `fiftyone` 不可行(三 lens 獨立同結論)

- **`import fiftyone` 自我衝突**:套件叫 `fiftyone` 時,專案內(以及 `fiftyone-brain`、`fiftyone.zoo`、plugin loader)的每個 `import fiftyone` 都會解析到**你的**套件而非 voxel51 的——它沒有 `fo.Dataset / Detection / launch_app / zoo / brain / operators`。adapter、App plugin、整條 Tier-2 全壞。**無法** alias 第三方函式庫自己的內部 self-import。
- **`fiftyone` CLI 指令也已被佔用**:真函式庫的 console script 就叫 `fiftyone`;發佈 `fiftyone = ...` 會 shadow 掉 `fiftyone plugins list`,而那正是本專案 plugin-enable 流程依賴的。
- **`vix_hash`(×579)是落地 + 雜湊鏈**:改名會讓既有每一份 decision log 的 `verify_chain()`/`gate` 失敗,並 orphan 所有既有 dataset/manifest/snapshot——是**靜默資料破壞**,不是大聲報錯。

### 持久化識別碼的裁決(若哪天真要動)
**程式碼可自由改名**(套件、模組、符號、區域變數——包括叫 `vix_hash` 的 Python 變數可改成 `content_hash` 而不動到磁碟)。但**凍結每個已寫到磁碟 / 進 FiftyOne dataset schema / 進 plugin registry / 進使用者環境的字串**。預設 = 保留(向後相容);只有有具體需求才 migrate,且必須附資料遷移步驟;env var / tag 的中間方案是 alias(read-old / write-new)。

## 3. 如何確保改完(8 項「done」gate;真要執行時用)
1. ripgrep:無殘留舊套件/script/module 參照(`from vix`、`import vix`、`src/vix`、pyproject 舊名)→ 全部回 **零**。
2. allow-list:殘存的 `fiftyone` 命中只剩**真函式庫 import**(`import fiftyone as fo`、`.brain/.zoo/.operators`、`fiftyone.yml`)。
3. 持久化識別碼:選保留 → 仍存在;選 migrate → 有遷移腳本 + 測試。
4. `pip install -e .` + smoke:新套件可 import **且**真 `fiftyone` 仍可解析;新 console script 可跑;舊套件已不存在。
5. 全套 **303** pytest == 改名前基線,零回歸。
6. 既有 `jdgws/`+`runws/` 落地檔仍可載入(或遷移 round-trip 通過)。
7. `@vix/review` plugin + operators/panels 重新註冊、列得出來。
8. live GUI / Tier-2 E2E 綠(`.venv311`)。

## 建議名稱(若日後執行)
import 套件 `fiftyone_vix`、發行名 `fiftyone-vix`、console script `fovix`(或 `fiftyone-vix`)。產品/文件可描述為「FiftyOne-native / 基於 FiftyOne 的策展工具」,但**不要**把套件或 CLI 取字面 `fiftyone`。
