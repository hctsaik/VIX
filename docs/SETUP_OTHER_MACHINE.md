# 在別台電腦重現 VIX(看影像分群 demo)

程式碼在 GitHub,clone 即可;**但資料集(vix_animals 的 embedding/UMAP)存在本機 FiftyOne 資料庫、不在 git 裡**,所以新機器要在當地**重新產生一次**(下載 CIFAR-10 → 算特徵 → 算 UMAP,幾分鐘、完全可重現,不需搬資料)。

---

## 前提

| 項目 | 需求 |
|---|---|
| **Python** | **3.11**(關鍵:FiftyOne 目前**不支援 3.13/3.14**;用 3.11) |
| git | 任意版本 |
| 網路 | 第一次需連網下載 CIFAR-10(約 170MB)與 pip 套件 |
| OS | Windows / macOS / Linux 皆可(CPU 即可,不需 GPU) |

> 確認 Python 3.11 有裝:Windows 打 `py -3.11 --version`;mac/Linux 打 `python3.11 --version`。

---

## A. 一鍵設定(Windows)

```powershell
git clone https://github.com/hctsaik/VIX.git
cd VIX
.\scripts\setup_tier2.ps1
```

跑完後產生示範資料集並啟動 App:

```powershell
.\.venv311\Scripts\python.exe examples\serve_animals.py
```

看到 `READY: 'vix_animals' ...` 後,瀏覽器開 <http://localhost:5151>,
照著 **[docs/guide/EMBEDDINGS_HOWTO.html](guide/EMBEDDINGS_HOWTO.html)** 從步驟 1 操作即可。

---

## B. 手動設定(macOS / Linux,或想逐步了解)

```bash
git clone https://github.com/hctsaik/VIX.git
cd VIX

# 1) 建立 Python 3.11 虛擬環境
python3.11 -m venv .venv311
source .venv311/bin/activate            # Windows: .\.venv311\Scripts\Activate.ps1

# 2) 裝核心 + Tier-2 套件
pip install --upgrade pip
pip install -e .                         # 核心(numpy/pyyaml/pillow)
pip install -r requirements-tier2.txt    # fiftyone/torch/playwright...

# 3) 下載 Playwright 的 Chromium(只有「自動截圖自檢」才需要,手動操作可略過)
python -m playwright install chromium

# 4) 產生示範資料集 + 啟動 App(第一次會下載 CIFAR-10)
python examples/serve_animals.py
```

開 <http://localhost:5151>,跟著圖解教學操作。

---

## C. 各腳本用途

| 指令 | 作用 |
|---|---|
| `python examples/serve_animals.py` | **首次用這個**:下載 CIFAR-10 → ResNet50 算 embedding → 算 UMAP → 啟 App。資料會存進本機 FiftyOne DB(persistent)。 |
| `python examples/serve_existing.py` | 資料已建好後,**只載入不重算**、快速重啟 App。 |
| `python examples/gui_check_animals.py` | 自我檢查:算 silhouette 分數 + 自畫 UMAP 散佈圖 + Playwright 截圖。 |
| `vix --help` / `vix quickstart` | 看 VIX CLI 全部指令 / 新手流程(這部分**不需要 FiftyOne**,任何機器都能跑)。 |

---

## D. 只想用「純核心」(任何機器、不裝 FiftyOne)

VIX 的核心邏輯零 FiftyOne 相依,任何 Python ≥3.10 都能跑:

```bash
pip install -e .
vix --help
vix quickstart
python -m pytest -q        # 需要 pip install pytest;102 項測試
```

---

## E. 替代法:直接搬既有資料集(不重算)

若不想重新下載/計算,可在來源機器匯出、目標機器匯入:

```bash
# 來源機器
python -c "import fiftyone as fo; fo.load_dataset('vix_animals').export(export_dir='vix_animals_export', dataset_type=fo.types.FiftyOneDataset)"
# 把 vix_animals_export/ 複製到目標機器後
python -c "import fiftyone as fo; fo.Dataset.from_dir(dataset_dir='vix_animals_export', dataset_type=fo.types.FiftyOneDataset, name='vix_animals', persistent=True)"
```

> 注意:UMAP 的 brain results 會一起匯出/匯入。一般情況**建議用 A/B 重新產生**,最乾淨。

---

## 疑難排解

- **裝不起來 / FiftyOne 報錯**:多半是 Python 版本不對。確認用的是 **3.11**,不是 3.13/3.14。
- **App 開了卻是黑白方塊**:你停在舊的 `vix_verify` 測試資料集,把網址改成 `http://localhost:5151/datasets/vix_animals`(見圖解教學步驟 1)。
- **port 5151 被佔用**:關掉其他 FiftyOne 視窗,或改 `serve_animals.py` 裡的 `PORT`。
