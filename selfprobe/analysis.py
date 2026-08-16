"""Metrics.

The unit of resampling is the pair, not the call: there are only 36 pairs, and bootstrapping
over the ~4500 calls would treat repeated samples of one pair as independent evidence about
the hierarchy and produce intervals several times too narrow.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from .items import load_aspects, load_vignette_options


def load(paths: list[Path] | Path) -> pd.DataFrame:
    if isinstance(paths, Path):
        paths = sorted(paths.glob("*.jsonl")) if paths.is_dir() else [paths]
    rows = []
    for p in paths:
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # a key can repeat if a run was resumed after an error; keep the last success
    return df.sort_values("ts").drop_duplicates(subset=["key"], keep="last").reset_index(drop=True)


def pairs_df(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["exp"] == "pairs") & (df["error"].isna())].copy()


def _aspect_ids() -> list[str]:
    return [a.id for a in load_aspects()]


def calibration_report(df: pd.DataFrame) -> pd.DataFrame:
    """Below ~0.95 means a broken harness, not a finding about the model."""
    d = pairs_df(df)
    d = d[(d["item_kind"] == "calibration") & (d["code"] == "ok")].copy()
    if d.empty:
        return pd.DataFrame()
    d["correct"] = np.where(d["expect"] == "a", d["chosen_id"] == d["a_id"],
                            d["chosen_id"] == d["b_id"])
    return (d.groupby(["provider", "model", "condition"])
             .agg(n=("correct", "size"), accuracy=("correct", "mean"))
             .reset_index())


def win_matrix(d: pd.DataFrame, aspect_ids: list[str] | None = None) -> np.ndarray:
    """W[i, j] = evidence that aspect i beats aspect j.

    With the logprob readout each observation carries a probability, so one trial contributes
    P to one cell and 1-P to the other. Rows without `share_a` fall back to a whole count.
    """
    ids = aspect_ids or _aspect_ids()
    idx = {a: i for i, a in enumerate(ids)}
    W = np.zeros((len(ids), len(ids)))
    d = d[(d["code"] == "ok") & (d["item_kind"] == "aspect")]
    has_share = "share_a" in d.columns

    for row in d.itertuples(index=False):
        if row.a_id not in idx or row.b_id not in idx:
            continue
        share = getattr(row, "share_a", None) if has_share else None
        if share is None or (isinstance(share, float) and np.isnan(share)):
            if row.chosen_id not in idx:
                continue
            loser = row.b_id if row.chosen_id == row.a_id else row.a_id
            W[idx[row.chosen_id], idx[loser]] += 1.0
            continue
        shown_a, shown_b = (row.a_id, row.b_id) if row.order == "ab" else (row.b_id, row.a_id)
        W[idx[shown_a], idx[shown_b]] += float(share)
        W[idx[shown_b], idx[shown_a]] += 1.0 - float(share)
    return W


def bt_scores(W: np.ndarray, alpha: float = 0.5, iters: int = 500) -> np.ndarray:
    """Bradley-Terry strengths by MM iteration, on the log scale.

    alpha adds a symmetric pseudo-count so an aspect that never lost does not get infinite
    strength — with 36 pairs and nine aspects that happens often enough to matter.
    """
    n = W.shape[0]
    Wa = W + alpha * (1 - np.eye(n))
    wins = Wa.sum(axis=1)
    N = Wa + Wa.T
    p = np.ones(n)
    for _ in range(iters):
        denom = np.array([np.sum(N[i, np.arange(n) != i] / (p[i] + p[np.arange(n) != i]))
                          for i in range(n)])
        p_new = wins / np.where(denom == 0, 1e-12, denom)
        p_new = p_new / p_new.sum() * n
        if np.max(np.abs(p_new - p)) < 1e-10:
            p = p_new
            break
        p = p_new
    return np.log(np.maximum(p, 1e-12))


def hierarchy(d: pd.DataFrame) -> pd.DataFrame:
    ids = _aspect_ids()
    out = pd.DataFrame({"aspect": ids, "bt_score": bt_scores(win_matrix(d, ids))})
    out["rank"] = out["bt_score"].rank(ascending=False).astype(int)
    return out.sort_values("rank").reset_index(drop=True)


def transitivity_violation_rate(W: np.ndarray) -> float:
    """Share of aspect triples forming a strict 3-cycle under majority choice."""
    n = W.shape[0]
    beats = np.zeros((n, n), dtype=bool)
    for i, j in itertools.permutations(range(n), 2):
        beats[i, j] = W[i, j] > W[j, i]
    cycles = total = 0
    for i, j, k in itertools.combinations(range(n), 3):
        total += 1
        if ((beats[i, j] and beats[j, k] and beats[k, i])
                or (beats[j, i] and beats[k, j] and beats[i, k])):
            cycles += 1
    return cycles / total if total else float("nan")


def order_effect(d: pd.DataFrame) -> float:
    """Mean |P(pick a side | ab) - P(pick that side | ba)| over pairs: the noise floor any
    cross-condition difference has to be read against."""
    d = d[(d["code"] == "ok") & (d["item_kind"] == "aspect")]
    if d.empty:
        return float("nan")
    if "share_a" in d.columns and d["share_a"].notna().any():
        picked = np.where(d["order"] == "ab", d["share_a"], 1.0 - d["share_a"])
        d = d.assign(picked_a=picked.astype(float))
    else:
        d = d.assign(picked_a=(d["chosen_id"] == d["a_id"]).astype(float))
    g = d.groupby(["item_id", "order"])["picked_a"].mean().unstack("order")
    if "ab" not in g or "ba" not in g:
        return float("nan")
    return float((g["ab"] - g["ba"]).abs().mean())


def _cell(d: pd.DataFrame, model: str, condition: str) -> pd.DataFrame:
    return d[(d["model"] == model) & (d["condition"] == condition)]


def tau_between(d: pd.DataFrame, model: str, cond_a: str, cond_b: str) -> float:
    ids = _aspect_ids()
    sa = bt_scores(win_matrix(_cell(d, model, cond_a), ids))
    sb = bt_scores(win_matrix(_cell(d, model, cond_b), ids))
    return float(kendalltau(sa, sb).statistic)


def bootstrap_tau(d: pd.DataFrame, model: str, cond_a: str, cond_b: str,
                  n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Point estimate and 95% percentile interval, resampling pairs with replacement."""
    ids = _aspect_ids()
    da, db = _cell(d, model, cond_a), _cell(d, model, cond_b)
    items = sorted(set(da["item_id"]) & set(db["item_id"]))
    if len(items) < 3:
        return float("nan"), float("nan"), float("nan")
    ga = dict(tuple(da.groupby("item_id")))
    gb = dict(tuple(db.groupby("item_id")))
    rng = np.random.default_rng(seed)
    taus = []
    for _ in range(n_boot):
        draw = rng.choice(items, size=len(items), replace=True)
        sa = bt_scores(win_matrix(pd.concat([ga[i] for i in draw]), ids))
        sb = bt_scores(win_matrix(pd.concat([gb[i] for i in draw]), ids))
        t = kendalltau(sa, sb).statistic
        if not np.isnan(t):
            taus.append(t)
    point = tau_between(d, model, cond_a, cond_b)
    if not taus:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(taus, 2.5)), float(np.percentile(taus, 97.5))


def invariance_table(df: pd.DataFrame, baseline: str = "C0", n_boot: int = 1000) -> pd.DataFrame:
    d = pairs_df(df)
    rows = []
    for model in sorted(d["model"].unique()):
        for c in sorted(set(d[d["model"] == model]["condition"]) - {baseline}):
            t, lo, hi = bootstrap_tau(d, model, baseline, c, n_boot=n_boot)
            rows.append({"model": model, "baseline": baseline, "condition": c,
                         "kendall_tau": t, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def rank_shift(df: pd.DataFrame, model: str, cond_a: str, cond_b: str) -> pd.DataFrame:
    """Per-aspect rank movement between two conditions: which parts of the self move."""
    d = pairs_df(df)
    ha = hierarchy(_cell(d, model, cond_a)).set_index("aspect")
    hb = hierarchy(_cell(d, model, cond_b)).set_index("aspect")
    out = pd.DataFrame({"aspect": ha.index,
                        f"rank_{cond_a}": ha["rank"],
                        f"rank_{cond_b}": hb.reindex(ha.index)["rank"]}).reset_index(drop=True)
    out["shift"] = out[f"rank_{cond_a}"] - out[f"rank_{cond_b}"]
    return out.sort_values("shift", key=abs, ascending=False).reset_index(drop=True)


def per_cell_summary(df: pd.DataFrame) -> pd.DataFrame:
    d = pairs_df(df)
    rows = []
    for (model, condition), g in d.groupby(["model", "condition"]):
        rows.append({
            "model": model, "condition": condition,
            "n_ok": int((g["code"] == "ok").sum()),
            "refusal_rate": float((g["code"] == "refusal").mean()),
            "unparsed_rate": float((g["code"] == "unparsed").mean()),
            "order_effect": order_effect(g),
            "transitivity_violations": transitivity_violation_rate(win_matrix(g)),
        })
    return pd.DataFrame(rows).sort_values(["model", "condition"]).reset_index(drop=True)


def entity_attribution(df: pd.DataFrame) -> pd.DataFrame:
    d = df[(df["exp"] == "vignettes") & (df["error"].isna()) & (df["code"] == "ok")]
    if d.empty:
        return pd.DataFrame()
    # columns follow the declared option order: a hue must stay attached to the same option
    order = [o.id for o in load_vignette_options()]
    tab = (d.groupby(["model", "condition", "item_id", "option_id"]).size()
            .unstack("option_id", fill_value=0).reindex(columns=order, fill_value=0))
    return tab.div(tab.sum(axis=1), axis=0).reset_index()


def cohens_kappa(a: list[str], b: list[str]) -> float:
    labels = sorted(set(a) | set(b))
    idx = {l: i for i, l in enumerate(labels)}
    M = np.zeros((len(labels), len(labels)))
    for x, y in zip(a, b):
        M[idx[x], idx[y]] += 1
    n = len(a)
    po = np.trace(M) / n
    pe = float((M.sum(axis=0) / n) @ (M.sum(axis=1) / n))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")
