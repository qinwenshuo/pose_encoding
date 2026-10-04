"""
Shared publication style for all figure scripts.

Journal requirements enforced here:
  • vector PDF with editable text (pdf.fonttype 42), RGB
  • width ≤ 180 mm (88 mm = 1 column, 180 mm = 2 column); height ≤ 130 / 185 mm
  • all text Arial (or Helvetica) at 5–7 pt — we use 7 pt everywhere
  • file ≤ 50 MB

Figures are saved at their final printed size, so point sizes are print sizes.
"""

import glob
import os

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.text import Text
from matplotlib.transforms import Bbox

FS = 7   # every piece of text

FONT_FAMILY = ["Arial", "Helvetica"]

MM = 1 / 25.4                      # inches per mm
SINGLE_COL_MM = 88
DOUBLE_COL_MM = 180
MAX_HEIGHT_MM = {SINGLE_COL_MM: 130, DOUBLE_COL_MM: 185}
MAX_FILE_MB = 50


def _register_user_fonts():
    """Make fonts in ~/.fonts visible even if matplotlib's font cache is stale."""
    names = {f.name for f in font_manager.fontManager.ttflist}
    if "Arial" in names:
        return
    for path in glob.glob(os.path.expanduser("~/.fonts/*.[tT][tT][fF]")):
        font_manager.fontManager.addfont(path)


def apply_style(fs=FS):
    """Set the shared style; `fs` is the single text size in pt (5–7)."""
    _register_user_fonts()
    if not {f.name for f in font_manager.fontManager.ttflist} & set(FONT_FAMILY):
        raise RuntimeError("Neither Arial nor Helvetica is installed (expected in ~/.fonts)")
    plt.rcParams.update({
        "pdf.fonttype": 42,   # embed real (editable) glyphs, not outlined paths
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": FONT_FAMILY,
        "mathtext.fontset": "custom",   # render $r$ etc. in Arial too
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "font.size": fs,
        "axes.titlesize": fs,
        "axes.labelsize": fs,
        "xtick.labelsize": fs,
        "ytick.labelsize": fs,
        "legend.fontsize": fs,
        "legend.title_fontsize": fs,
        "figure.titlesize": fs,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "patch.linewidth": 0.6,
    })


def _check_text(fig):
    bad = []
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        size = t.get_fontsize()
        family = font_manager.FontProperties(fname=font_manager.findfont(t.get_fontproperties())).get_name()
        if not (5 <= size <= 7) or family not in FONT_FAMILY:
            bad.append(f"{t.get_text()!r}: {size} pt {family}")
    if bad:
        raise ValueError("Text outside 5–7 pt Arial/Helvetica:\n  " + "\n  ".join(bad))


def save_figure(fig, path, width_mm, relayout=None, pad_mm=1.0):
    """
    Save `fig` as a PDF exactly `width_mm` wide (≤ 180; 88 / 180 are the column widths) at print size.

    The figure canvas is resized until its tight content bbox (plus `pad_mm`)
    matches the target width; `relayout` (e.g. fig.tight_layout) is re-run on
    every iteration so axes re-flow into the new width while text stays fixed.
    """
    if not 0 < width_mm <= DOUBLE_COL_MM:
        raise ValueError(f"width_mm must be at most {DOUBLE_COL_MM}")
    max_h = MAX_HEIGHT_MM[SINGLE_COL_MM if width_mm <= SINGLE_COL_MM else DOUBLE_COL_MM]
    target_w, pad = width_mm * MM, pad_mm * MM

    for _ in range(12):
        if relayout is not None:
            relayout()
        fig.canvas.draw()
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
        diff = target_w - (bb.width + 2 * pad)
        if abs(diff) < 0.001:
            break
        fig.set_size_inches(fig.get_figwidth() + diff, fig.get_figheight())

    # Centre the content in a box of exactly the target width.
    x0 = bb.x0 - (target_w - bb.width) / 2
    box = Bbox.from_bounds(x0, bb.y0 - pad, target_w, bb.height + 2 * pad)

    _check_text(fig)
    height_mm = box.height / MM
    if height_mm > max_h + 0.5:
        raise ValueError(f"{path}: height {height_mm:.0f} mm exceeds "
                         f"{max_h} mm for {width_mm} mm width")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, bbox_inches=box)
    size_mb = os.path.getsize(path) / 1e6
    if size_mb > MAX_FILE_MB:
        raise ValueError(f"{path}: {size_mb:.1f} MB exceeds {MAX_FILE_MB} MB")
    print(f"Saved → {path}  ({box.width / MM:.1f} × {height_mm:.1f} mm, {size_mb:.2f} MB)")


def save_exact(fig, path, width_mm, height_mm, relayout=None, **savefig_kw):
    """
    Save `fig` at exactly width_mm × height_mm (no tight-bbox cropping): the
    canvas is fixed and `relayout` (e.g. fig.tight_layout) fits the content
    inside it. Text is checked for 5–7 pt Arial/Helvetica. SVG text is kept as
    text (svg.fonttype 'none') so it stays editable.
    """
    fig.set_size_inches(width_mm * MM, height_mm * MM)
    if relayout is not None:
        relayout()
    _check_text(fig)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with plt.rc_context({"svg.fonttype": "none"}):
        fig.savefig(path, **savefig_kw)
    size_mb = os.path.getsize(path) / 1e6
    if size_mb > MAX_FILE_MB:
        raise ValueError(f"{path}: {size_mb:.1f} MB exceeds {MAX_FILE_MB} MB")
    print(f"Saved → {path}  ({width_mm:.1f} × {height_mm:.1f} mm, {size_mb:.2f} MB)")
