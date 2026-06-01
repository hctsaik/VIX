"""把擷取到的截圖加上紅框 / 箭頭 / 步驟編號,輸出逐步教學圖,並組成一頁 HTML。
中文說明放在 HTML(瀏覽器原生支援中文),圖上只畫紅框與阿拉伯數字徽章。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

GUIDE = ROOT / "docs" / "guide"
IMG = GUIDE / "img"
RED = (230, 40, 40)
WHITE = (255, 255, 255)


def _font(size: int) -> ImageFont.FreeTypeFont:
    for p in (r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\arial.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def box(d: ImageDraw.ImageDraw, xy, w: int = 5) -> None:
    x0, y0, x1, y1 = xy
    for i in range(w):
        d.rectangle([x0 - i, y0 - i, x1 + i, y1 + i], outline=RED)


def ellipse(d: ImageDraw.ImageDraw, xy, w: int = 5) -> None:
    x0, y0, x1, y1 = xy
    for i in range(w):
        d.ellipse([x0 - i, y0 - i, x1 + i, y1 + i], outline=RED)


def arrow(d: ImageDraw.ImageDraw, p0, p1, w: int = 6) -> None:
    import math

    d.line([p0, p1], fill=RED, width=w)
    ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    for a in (ang + 2.6, ang - 2.6):
        d.line([p1, (p1[0] - 22 * math.cos(a), p1[1] - 22 * math.sin(a))], fill=RED, width=w)


def badge(d: ImageDraw.ImageDraw, cx: int, cy: int, n: str, r: int = 26) -> None:
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=RED, outline=WHITE, width=3)
    f = _font(int(r * 1.3))
    tb = d.textbbox((0, 0), n, font=f)
    d.text((cx - (tb[2] - tb[0]) / 2, cy - (tb[3] - tb[1]) / 2 - tb[1]), n, font=f, fill=WHITE)


def load(name: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    im = Image.open(name).convert("RGB")
    return im, ImageDraw.Draw(im)


def main() -> None:
    app = IMG / "raw_app.png"
    umap = ROOT / "docs" / "spec" / "img" / "animals_umap.png"
    selected = ROOT / "docs" / "spec" / "img" / "animals_selected_cluster.png"

    # STEP 1 切換資料集
    im, d = load(app)
    box(d, (166, 12, 264, 58))
    badge(d, 130, 35, "1")
    arrow(d, (148, 35), (170, 35))
    im.save(IMG / "step1.png")

    # STEP 2 開 Embeddings 面板(框右半邊面板)
    im, d = load(app)
    box(d, (938, 80, 1592, 520))
    badge(d, 905, 100, "2")
    arrow(d, (928, 100), (945, 100))
    im.save(IMG / "step2.png")

    # STEP 3 選 brain key + color by(右上兩個下拉)
    im, d = load(app)
    box(d, (944, 86, 1028, 120))      # feat_umap
    box(d, (1030, 86, 1150, 120))     # ground_truth
    badge(d, 986, 150, "3")
    badge(d, 1090, 150, "3")
    arrow(d, (986, 138), (986, 122))
    arrow(d, (1090, 138), (1090, 122))
    im.save(IMG / "step3.png")

    # STEP 4 套索圈一群(用清楚的彩色 UMAP 示意)
    im, d = load(umap)
    ellipse(d, (470, 10, 660, 230))   # 框右上 ship 群
    badge(d, 690, 60, "4")
    arrow(d, (676, 75), (655, 110))
    im.save(IMG / "step4.png")

    # STEP 5 Only show selected(格狀上方工具列會出現選取數)
    im, d = load(app)
    box(d, (298, 80, 950, 130))
    badge(d, 270, 104, "5")
    arrow(d, (290, 104), (300, 104))
    im.save(IMG / "step5.png")

    # STEP 6 看這群是什麼(只剩選取的格狀)
    im, d = load(selected)
    W = im.width
    badge(d, 40, 40, "6")
    im.save(IMG / "step6.png")

    # STEP 7 還原(view bar 的 + add stage / ✕)
    im, d = load(app)
    box(d, (270, 8, 372, 60))
    badge(d, 240, 34, "7")
    arrow(d, (258, 34), (272, 34))
    im.save(IMG / "step7.png")

    print("[OK] step1..step7.png 已輸出到", IMG)


if __name__ == "__main__":
    main()
