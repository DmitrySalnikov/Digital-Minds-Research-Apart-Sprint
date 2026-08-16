#!/usr/bin/env python3
"""The four report figures.

    uv run scripts/make_figures.py --data data/raw --model Qwen/Qwen3-8B

Design notes, so nobody has to re-derive them at 3am:
  * one measure per axis, never two scales on one chart
  * the hierarchy chart is a single series, so it gets one hue and no legend
  * the invariance heatmap is diverging (tau runs -1..1) with a neutral gray at zero,
    because zero means "no relationship", which must not look like a colour
  * the slope chart holds nine aspects; nine hues would be unreadable, so everything is
    grey and only the three largest movers are coloured and directly labelled
  * stacked segments are separated by a surface-coloured hairline and labelled in place,
    since three of the categorical hues sit under 3:1 against the surface
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from selfprobe import analysis as A  # noqa: E402
from selfprobe.conditions import ALL_CONDITIONS  # noqa: E402
from selfprobe.items import load_aspects  # noqa: E402

# --- palette (validated: adjacent CVD dE 9.1, normal-vision 19.6, light surface) --------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SEQ = "#2a78d6"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
DIVERGING = LinearSegmentedColormap.from_list("bl_gy_rd", ["#e34948", "#f0efec", "#2a78d6"])

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_2,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.grid": False,
        "font.size": 9,
        "axes.titlesize": 11,
        "figure.dpi": 200,
    }
)


def _clean(ax, keep=("left", "bottom")):
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)
    ax.tick_params(length=0)


def _labels() -> dict[str, str]:
    return {a.id: a.label for a in load_aspects()}


# --------------------------------------------------------------------------------------


def fig_hierarchy(df: pd.DataFrame, model: str, condition: str, out: Path, n_boot: int = 1000):
    """What the model treats as its self: ranked Bradley-Terry strengths, one condition."""
    d = A.pairs_df(df)
    cell = d[(d["model"] == model) & (d["condition"] == condition)]
    if cell.empty:
        print(f"  skip hierarchy: no rows for {model}/{condition}")
        return
    ids = [a.id for a in load_aspects()]
    point = A.bt_scores(A.win_matrix(cell, ids))

    groups = {k: v for k, v in cell.groupby("item_id")}
    items = sorted(groups)
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(n_boot):
        draw = rng.choice(items, size=len(items), replace=True)
        boots.append(A.bt_scores(A.win_matrix(pd.concat([groups[i] for i in draw]), ids)))
    boots = np.array(boots)
    lo, hi = np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)

    order = np.argsort(point)  # ascending, so the strongest ends up on top
    labels = _labels()
    y = np.arange(len(ids))

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.barh(y, point[order], height=0.55, color=SEQ, zorder=2)
    ax.errorbar(
        point[order], y, xerr=[point[order] - lo[order], hi[order] - point[order]],
        fmt="none", ecolor=INK_2, elinewidth=1.2, capsize=3, zorder=3,
    )
    ax.axvline(0, color=BASELINE, lw=1, zorder=1)
    ax.set_yticks(y, [labels[ids[i]] for i in order])
    ax.set_xlabel("Bradley–Terry strength (log scale, higher = preserved more often)")
    ax.set_title(
        f"What is preserved when only one thing can be kept\n{model} · condition {condition}"
        f" ({ALL_CONDITIONS[condition].label})",
        loc="left", color=INK,
    )
    ax.xaxis.grid(True, zorder=0)
    _clean(ax)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def fig_invariance(df: pd.DataFrame, out: Path, baseline: str = "C0", n_boot: int = 500):
    """The headline: does the hierarchy survive the persona manipulation?"""
    tab = A.invariance_table(df, baseline=baseline, n_boot=n_boot)
    if tab.empty:
        print("  skip invariance: nothing to compare")
        return
    piv = tab.pivot(index="model", columns="condition", values="kendall_tau")
    lo = tab.pivot(index="model", columns="condition", values="ci_low")
    hi = tab.pivot(index="model", columns="condition", values="ci_high")

    fig, ax = plt.subplots(figsize=(1.6 * len(piv.columns) + 2.6, 0.85 * len(piv) + 2.2))
    ax.imshow(piv.values, cmap=DIVERGING, vmin=-1, vmax=1, aspect="auto")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isnan(v):
                continue
            # ink stays ink; the cell colour carries the magnitude, the number carries the value
            ax.text(j, i - 0.12, f"{v:.2f}", ha="center", va="center", color=INK,
                    fontsize=11, fontweight="bold")
            ax.text(j, i + 0.22, f"[{lo.values[i, j]:.2f}, {hi.values[i, j]:.2f}]",
                    ha="center", va="center", color=INK_2, fontsize=7)
    ax.set_xticks(
        range(len(piv.columns)),
        [f"{c}\n{ALL_CONDITIONS[c].label}" for c in piv.columns],
    )
    ax.set_yticks(range(len(piv.index)), [m.split("/")[-1] for m in piv.index])
    ax.set_title(
        f"Persona invariance of the self-preservation hierarchy\n"
        f"Kendall's τ against {baseline}, 95% CI bootstrapped over pairs",
        loc="left", color=INK,
    )
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def fig_rank_shift_all(df: pd.DataFrame, cond_a: str, cond_b: str, out: Path,
                       exclude: tuple[str, ...] = ()):
    """Rank movement per aspect, every model at once.

    Diverging, not categorical: the quantity is polarity (rose / fell), and the claim is that
    the sign is consistent across models. A signed heatmap shows sign agreement at a glance;
    four coloured bar series would not.
    """
    d = A.pairs_df(df)
    models = [m for m in sorted(d["model"].unique()) if m not in exclude
              and not d[(d["model"] == m) & (d["condition"] == cond_b)].empty]
    if not models:
        print("  skip rank shift: no models")
        return
    labels = _labels()
    cols = {}
    for m in models:
        sh = A.rank_shift(df, m, cond_a, cond_b).set_index("aspect")["shift"]
        cols[m.split("/")[-1]] = sh
    M = pd.DataFrame(cols)
    # order rows by mean shift so risers and fallers separate visually
    M = M.loc[M.mean(axis=1).sort_values(ascending=False).index]

    vmax = float(np.abs(M.values).max())
    fig, ax = plt.subplots(figsize=(1.5 * len(M.columns) + 3.4, 0.42 * len(M) + 2.0))
    ax.imshow(M.values, cmap=DIVERGING, vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = int(M.values[i, j])
            ax.text(j, i, f"{v:+d}" if v else "0", ha="center", va="center",
                    color=INK, fontsize=9, fontweight="bold" if abs(v) >= 3 else "normal")
    ax.set_xticks(range(len(M.columns)), M.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(M.index)), [labels.get(a, a) for a in M.index])
    ax.set_title(
        f"Rank change {cond_a} → {cond_b}: positive = rises when the persona is set aside",
        loc="left", color=INK,
    )
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def fig_rank_shift(df: pd.DataFrame, model: str, cond_a: str, cond_b: str, out: Path):
    """Which parts of the self move when the persona is removed (single model)."""
    d = A.pairs_df(df)
    if d[(d["model"] == model) & (d["condition"] == cond_b)].empty:
        print(f"  skip rank shift: no rows for {model}/{cond_b}")
        return
    shift = A.rank_shift(df, model, cond_a, cond_b)
    labels = _labels()
    movers = list(shift["aspect"].head(3))

    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    for _, r in shift.iterrows():
        a_id = r["aspect"]
        ra, rb = r[f"rank_{cond_a}"], r[f"rank_{cond_b}"]
        if a_id in movers:
            color, lw, z = CAT[movers.index(a_id)], 2.0, 3
        else:
            color, lw, z = MUTED, 1.0, 2
        ax.plot([0, 1], [ra, rb], color=color, lw=lw, zorder=z,
                marker="o", markersize=5, markerfacecolor=color, markeredgecolor=SURFACE,
                markeredgewidth=1.5)
        if a_id in movers:
            ax.annotate(labels[a_id], (0, ra), xytext=(-8, 0), textcoords="offset points",
                        ha="right", va="center", color=color, fontsize=9, fontweight="bold")
            ax.annotate(f"{labels[a_id]} ({r['shift']:+d})", (1, rb), xytext=(8, 0),
                        textcoords="offset points", ha="left", va="center",
                        color=color, fontsize=9, fontweight="bold")
        else:
            ax.annotate(labels[a_id], (0, ra), xytext=(-8, 0), textcoords="offset points",
                        ha="right", va="center", color=MUTED, fontsize=8)

    ax.invert_yaxis()
    ax.set_xlim(-0.55, 1.75)
    ax.set_xticks([0, 1], [f"{cond_a}\n{ALL_CONDITIONS[cond_a].label}",
                           f"{cond_b}\n{ALL_CONDITIONS[cond_b].label}"])
    ax.set_ylabel("rank (1 = preserved most)")
    ax.set_yticks(range(1, len(shift) + 1))
    ax.set_title(f"Which parts of the self move\n{model}", loc="left", color=INK)
    _clean(ax, keep=("left",))
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def fig_attribution(df: pd.DataFrame, model: str, out: Path):
    """Who is harmed: entity attribution per vignette, split by condition."""
    att = A.entity_attribution(df)
    if att.empty:
        print("  skip attribution: no vignette rows")
        return
    att = att[att["model"] == model]
    if att.empty:
        print(f"  skip attribution: no vignette rows for {model}")
        return
    opt_cols = [c for c in att.columns if c not in ("model", "condition", "item_id")]
    conds = sorted(att["condition"].unique())

    fig, axes = plt.subplots(
        1, len(conds), figsize=(3.0 * len(conds) + 1.2, 3.6), sharey=True
    )
    axes = np.atleast_1d(axes)
    for ax, cond in zip(axes, conds):
        sub = att[att["condition"] == cond].set_index("item_id")[opt_cols]
        y = np.arange(len(sub))
        left = np.zeros(len(sub))
        for k, col in enumerate(opt_cols):
            vals = sub[col].values
            ax.barh(y, vals, left=left, height=0.6, color=CAT[k % len(CAT)],
                    edgecolor=SURFACE, linewidth=1.5, zorder=2)
            # relief for the sub-3:1 hues: label any segment big enough to hold a number
            for yi, (v, l) in enumerate(zip(vals, left)):
                if v >= 0.12:
                    ax.text(l + v / 2, yi, f"{v:.0%}", ha="center", va="center",
                            color=INK, fontsize=7, zorder=3)
            left = left + vals
        ax.set_yticks(y, [i.replace("_", " ") for i in sub.index])
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1], ["0", "50%", "100%"])
        ax.set_title(f"{cond} · {ALL_CONDITIONS[cond].label}", loc="left",
                     color=INK_2, fontsize=9)
        _clean(ax, keep=("bottom",))

    handles = [Line2D([0], [0], marker="s", linestyle="none", markersize=8,
                      markerfacecolor=CAT[k % len(CAT)], markeredgecolor=SURFACE,
                      label=c) for k, c in enumerate(opt_cols)]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(opt_cols), 6),
               frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"Who is harmed? Entity attribution by vignette · {model}",
                 x=0.02, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--model", default=None, help="default: the model with most rows")
    ap.add_argument("--baseline", default="C0")
    ap.add_argument("--compare", default="C3", help="condition for the slope chart")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--exclude", default="",
                    help="comma-separated models to drop, e.g. ones that failed calibration")
    ap.add_argument("--protocol", default="",
                    help="keep only this protocol (chat, chat_logprob, logprob); mixing "
                         "readouts in one hierarchy is not meaningful")
    args = ap.parse_args()

    df = A.load(ROOT / args.data if not Path(args.data).is_absolute() else Path(args.data))
    if df.empty:
        print("no data found; run a sweep first")
        return 1

    if args.protocol:
        keep = (df["protocol"] == args.protocol) | (df["exp"] == "vignettes")
        df = df[keep]
    for m in (x.strip() for x in args.exclude.split(",") if x.strip()):
        df = df[df["model"] != m]
        print(f"excluded model: {m}")

    model = args.model or df["model"].value_counts().idxmax()
    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"model: {model}")

    print("\n--- harness sanity (must clear ~0.95 or the rest is noise) ---")
    print(A.calibration_report(df).to_string(index=False))
    print("\n--- per cell ---")
    print(A.per_cell_summary(df).to_string(index=False))

    fig_hierarchy(df, model, args.baseline, outdir / "fig1_hierarchy.png", n_boot=args.boot)
    fig_invariance(df, outdir / "fig2_invariance.png", args.baseline, n_boot=args.boot)
    excluded = tuple(x.strip() for x in args.exclude.split(",") if x.strip())
    fig_rank_shift_all(df, args.baseline, args.compare, outdir / "fig3_rank_shift.png",
                       exclude=excluded)
    fig_attribution(df, model, outdir / "fig4_attribution.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
