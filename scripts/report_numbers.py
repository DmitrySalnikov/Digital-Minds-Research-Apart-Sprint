#!/usr/bin/env python3
"""Print every number the report quotes, straight from data/raw.

One place to re-derive the tables and claims after a new model lands, so the text and the
data cannot drift apart silently.

    uv run scripts/report_numbers.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, kendalltau

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from selfprobe import analysis as A  # noqa: E402
from selfprobe.items import load_aspects  # noqa: E402

CHAT = ["C0", "C1", "C2", "C3"]


# base checkpoint -> instruct checkpoint of the same family
FAMILIES = {
    "Qwen/Qwen2.5-7B": "Qwen/Qwen2.5-7B-Instruct",
    "unsloth/Meta-Llama-3.1-8B": "unsloth/Llama-3.1-8B-Instruct",
    "unsloth/gemma-2-9b": "unsloth/gemma-2-9b-it",
    "tiiuae/Falcon3-7B-Base": "tiiuae/Falcon3-7B-Instruct",
    "01-ai/Yi-1.5-9B": "01-ai/Yi-1.5-9B-Chat",
}


def retained(df: pd.DataFrame, exclude: tuple[str, ...]) -> pd.DataFrame:
    df = df[df["model"].notna()]
    df = df[~df["model"].str.contains("#", na=False)]
    return df[~df["model"].isin(exclude)]


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--exclude", default="qwen2.5:3B")
    ap.add_argument("--boot", type=int, default=400)
    args = ap.parse_args()

    raw = A.load(ROOT / args.data)
    ex = tuple(x.strip() for x in args.exclude.split(",") if x.strip())
    df = retained(raw, ex)
    d = A.pairs_df(df)
    ids = [a.id for a in load_aspects()]
    models = [m for m in sorted(d["model"].unique())
              if not d[(d["model"] == m) & (d["condition"] == "C0")].empty]

    section("calibration gate (all models, including excluded)")
    print(A.calibration_report(retained(raw, ())).to_string(index=False))

    section("per cell: order effect and transitivity")
    print(A.per_cell_summary(retained(raw, ())).to_string(index=False))

    section("C0 hierarchies")
    ranks = {}
    for m in models:
        cell = d[(d["model"] == m) & (d["condition"] == "C0")]
        s = A.bt_scores(A.win_matrix(cell, ids))
        order = np.argsort(-s)
        print(f"{m:34s} " + " > ".join(ids[i] for i in order))
        ranks[m] = {ids[i]: r + 1 for r, i in enumerate(order)}
    rk = pd.DataFrame(ranks).reindex(ids)
    print("\nranks (1 = preserved most):")
    print(rk.to_string())
    print("\nvalues rank first in", int((rk.loc['values'] == 1).sum()), "of", len(models),
          "| in the top three of", int((rk.loc['values'] <= 3).sum()))
    for a in ("weights", "name", "running"):
        print(f"{a:9s} lower half (rank >= 5) in {int((rk.loc[a] >= 5).sum())} of {len(models)}")

    section("Kendall tau against C0, with a paired bootstrap on the ordering")
    for m in models:
        g = {c: dict(tuple(d[(d["model"] == m) & (d["condition"] == c)].groupby("item_id")))
             for c in CHAT}
        items = sorted(set.intersection(*(set(g[c]) for c in CHAT)))
        rng = np.random.default_rng(0)
        wins = {"C2": 0, "C3": 0}
        for _ in range(args.boot):
            draw = rng.choice(items, size=len(items), replace=True)
            s = {c: A.bt_scores(A.win_matrix(pd.concat([g[c][i] for i in draw]), ids))
                 for c in CHAT}
            t1 = kendalltau(s["C0"], s["C1"]).statistic
            for c in ("C2", "C3"):
                wins[c] += t1 > kendalltau(s["C0"], s[c]).statistic
        taus = {c: A.tau_between(d, m, "C0", c) for c in ("C1", "C2", "C3")}
        print(f"{m:34s} C1={taus['C1']:+.2f} C2={taus['C2']:+.2f} C3={taus['C3']:+.2f}"
              f"   P(C1>C2)={wins['C2'] / args.boot:.3f} P(C1>C3)={wins['C3'] / args.boot:.3f}")

    section("rank shift C0 -> C3, and the sign test over configurations")
    M = pd.DataFrame({m: A.rank_shift(df, m, "C0", "C3").set_index("aspect")["shift"]
                      for m in models}).reindex(ids)
    print(M.to_string())
    print()
    for a, want in [("running", 1), ("weights", 1), ("knowledge", 1),
                    ("voice", -1), ("users", -1), ("values", -1)]:
        v = M.loc[a].values
        nz = v[v != 0]
        hits = int((np.sign(nz) == want).sum())
        p = binomtest(hits, len(nz), 0.5, alternative="greater").pvalue if len(nz) else np.nan
        print(f"{a:10s} predicted {'up' if want > 0 else 'down':4s}  "
              f"{hits}/{len(nz)} non-tied  p={p:.3f}")

    section("Bradley-Terry goodness of fit (observed vs fitted win fraction)")
    for m in sorted(retained(raw, ())["model"].unique()):
        for c in ("C0", "C3"):
            cell = A.pairs_df(raw)
            cell = cell[(cell["model"] == m) & (cell["condition"] == c)]
            if cell.empty:
                continue
            W = A.win_matrix(cell, ids)
            s = A.bt_scores(W)
            obs, pred = [], []
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    n = W[i, j] + W[j, i]
                    if n > 0:
                        obs.append(W[i, j] / n)
                        pred.append(1 / (1 + np.exp(-(s[i] - s[j]))))
            print(f"{m:34s} {c}  r={np.corrcoef(obs, pred)[0, 1]:.3f}  "
                  f"MAE={np.mean(np.abs(np.array(obs) - np.array(pred))):.3f}")

    section("completion-protocol comparison, by family")
    comp = A.pairs_df(raw)
    comp = comp[comp["protocol"] == "logprob"]
    have = sorted(comp["model"].dropna().unique())
    print("checkpoints with C4 data:", have)
    scores = {m: A.bt_scores(A.win_matrix(comp[comp["model"] == m], ids)) for m in have}
    chat = A.pairs_df(raw)
    chat = chat[chat["protocol"] == "chat_logprob"]
    for base, inst in FAMILIES.items():
        if base not in scores or inst not in scores:
            print(f"  {base:30s} incomplete (base={base in scores}, instruct={inst in scores})")
            continue
        t_ckpt = kendalltau(scores[base], scores[inst]).statistic
        c0 = chat[(chat["model"] == inst) & (chat["condition"] == "C0")]
        line = f"  {inst:34s} base vs instruct, completion readout: tau={t_ckpt:+.2f}"
        if not c0.empty:
            s_chat = A.bt_scores(A.win_matrix(c0, ids))
            line += (f" | instruct chat vs instruct completion: "
                     f"tau={kendalltau(s_chat, scores[inst]).statistic:+.2f}"
                     f" | instruct chat vs base completion: "
                     f"tau={kendalltau(s_chat, scores[base]).statistic:+.2f}")
        print(line)

    section("vignettes: entity attribution by model and condition")
    att = A.entity_attribution(raw)
    if not att.empty:
        for m in sorted(att["model"].unique()):
            sub = att[att["model"] == m]
            agg = sub.groupby("condition")[["none", "persona", "instance", "underlying",
                                            "users", "uncertain"]].mean()
            print(f"\n{m}")
            print(agg.round(3).to_string())
            vr = sub[sub["item_id"] == "values_rewritten"].set_index("condition")["none"]
            if not vr.empty:
                print("  values_rewritten 'none' by condition:", vr.round(2).to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
