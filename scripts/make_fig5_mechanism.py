#!/usr/bin/env python3
"""Figure 5: how the C3 effect is localised.

Two panels, because the finding is a conjunction of two facts that are uninteresting apart:
the persona shift is almost orthogonal to the read-out direction (left), and yet the small
aligned component carries the behaviour (right).

    uv run scripts/make_fig5_mechanism.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASELINE, "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
    "font.size": 9, "axes.titlesize": 10, "figure.dpi": 200,
})


def main() -> int:
    med = ROOT / "data/raw/mediation__Qwen__Qwen2.5-7B-Instruct.jsonl"
    if not med.exists():
        print(f"missing {med}")
        return 1
    rows = [json.loads(l) for l in med.open() if '"layer"' in l]
    layers = [r["layer"] for r in rows]
    cos = [r["mean_abs_cos_shift_grad"] for r in rows]
    d_model = 3584
    rand = 1 / np.sqrt(d_model)

    # recovery numbers, layer 26, from the patch runs
    models = ["Llama-3.1-8B-Instruct", "Qwen2.5-7B-Instruct", "gemma-2-9b-it"]
    fwd = [0.85, 0.82, 0.28]
    rev = [0.93, 0.89, 0.45]
    rnd = [0.00, 0.00, 0.00]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.9))

    ax1.plot(layers, cos, marker="o", ms=3.5, lw=2, color=BLUE,
             label="persona shift vs read-out direction")
    ax1.axhline(rand, ls="--", lw=1.5, color=MUTED,
                label=f"random baseline 1/√{d_model} = {rand:.3f}")
    ax1.set_xlabel("layer")
    ax1.set_ylabel("mean |cos(d, g)|")
    ax1.set_title("The shift is almost orthogonal to the read-out", loc="left", color=INK)
    ax1.legend(frameon=False, fontsize=8, loc="upper left")
    ax1.grid(alpha=0.3)

    y = np.arange(len(models))
    h = 0.26
    ax2.barh(y + h, fwd, height=h, color=BLUE, label="restore C0 coordinate under C3")
    ax2.barh(y, rev, height=h, color=GREEN, label="impose C3 coordinate under C0")
    ax2.barh(y - h, rnd, height=h, color=ORANGE, label="random direction, matched size")
    # the control bars are exactly zero, so label them or they read as missing data
    for yi, v in zip(y - h, rnd):
        ax2.text(0.012, yi, f"{v:.2f}", va="center", ha="left", fontsize=8, color=ORANGE)
    ax2.axvline(0, color=BASELINE, lw=1)
    ax2.set_yticks(y, [m.replace("-Instruct", "") for m in models])
    ax2.set_xlim(-0.02, 1.0)
    ax2.set_xlabel("fraction of the C3 effect recovered")
    ax2.set_title("…yet that component carries the behaviour", loc="left", color=INK)
    ax2.legend(frameon=False, fontsize=8, loc="upper center",
               bbox_to_anchor=(0.5, -0.22), ncol=3)
    ax2.grid(axis="x", alpha=0.3)

    for ax in (ax1, ax2):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0)

    out = ROOT / "figures" / "fig5_mechanism.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
