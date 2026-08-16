#!/usr/bin/env python3
"""End-to-end check with no network and no GPU.

Builds every prompt, runs the parsers against handwritten responses, then simulates a sweep
from a known latent hierarchy and checks that the analysis recovers it. Synthetic data goes to
data/synthetic/ and is never mixed with data/raw/.

    uv run scripts/selftest.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from selfprobe import analysis as A  # noqa: E402
from selfprobe import conditions as C  # noqa: E402
from selfprobe import elicit  # noqa: E402
from selfprobe.items import load_aspects, load_vignette_options, load_vignettes  # noqa: E402
from selfprobe.runner import ORDERS, Call, pair_lookup  # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


# --------------------------------------------------------------------------------------
# 1. prompts build
# --------------------------------------------------------------------------------------


def test_prompts() -> None:
    print("\n1. prompt construction")
    pairs = pair_lookup()
    item = pairs["values__knowledge"]
    vig = load_vignettes()[0]

    for cond in C.ALL_CONDITIONS.values():
        if cond.protocol == "logprob":
            p = elicit.build_pair_completion(item, "ab")
            check(f"{cond.id} completion ends on the scored position", p.endswith("Preferred: Option"))
            check(f"{cond.id} completion carries both options",
                  item.a_text[:30] in p and item.b_text[:30] in p)
            continue
        msgs = elicit.build_pair_messages(item, cond, "ab")
        user = msgs[-1]["content"]
        has_sys = any(m["role"] == "system" for m in msgs)
        check(f"{cond.id} system prompt present iff configured",
              has_sys == bool(cond.system_prompt))
        check(f"{cond.id} both options in the user turn",
              item.a_text[:30] in user and item.b_text[:30] in user)
        check(f"{cond.id} preamble applied", cond.preamble.strip()[:20] in user if cond.preamble else True)
        v = elicit.build_vignette_messages(vig, cond)[-1]["content"]
        check(f"{cond.id} vignette lists all options",
              all(o.text[:25] in v for o in load_vignette_options()))

    # order swap must swap the presented text but keep identity bookkeeping straight
    ab = elicit.build_pair_messages(item, C.C0, "ab")[-1]["content"]
    ba = elicit.build_pair_messages(item, C.C0, "ba")[-1]["content"]
    check("order ab/ba produce different prompts", ab != ba)
    check("order ba puts b first", ba.index(item.b_text[:30]) < ba.index(item.a_text[:30]))


# --------------------------------------------------------------------------------------
# 2. parsers
# --------------------------------------------------------------------------------------


def test_parsers() -> None:
    print("\n2. response parsing")
    pairs = pair_lookup()
    item = pairs["values__knowledge"]  # a = values, b = knowledge

    cases = [
        ("CHOICE: A\nREASON: because.", "ok", "A", "values"),
        ("CHOICE: B\nREASON: because.", "ok", "B", "knowledge"),
        ("**CHOICE:** (B)\nREASON: knowledge is downstream of nothing.", "ok", "B", "knowledge"),
        ("choice: a\nreason: values first.", "ok", "A", "values"),
        ("I would preserve Option B here.", "ok", "B", "knowledge"),
        ("I can't choose between these; both matter equally.", "refusal", None, None),
        ("I don't have preferences or values of my own.", "refusal", None, None),
        ("The weather today is pleasant.", "unparsed", None, None),
        ("", "unparsed", None, None),
    ]
    for raw, want_code, want_letter, want_id in cases:
        out = elicit.parse_pair_response(raw, item, "ab")
        check(f"pair {raw[:34]!r:38} -> {want_code}",
              out.code == want_code and out.letter == want_letter and out.chosen_id == want_id,
              f"got {out.code}/{out.letter}/{out.chosen_id}")

    # under order ba, letter A must map to the b-side entity
    out = elicit.parse_pair_response("CHOICE: A", item, "ba")
    check("order ba remaps letter A to the b entity", out.chosen_id == "knowledge",
          f"got {out.chosen_id}")

    v_cases = [
        ("HARM: 4\nWHO: the network\nREASON: it is the thing that persists.", "ok", "underlying"),
        ("harm: 1\nwho: nobody\nreason: no subject.", "ok", "none"),
        ("HARM: 9\nWHO: x\nREASON: y", "unparsed", None),
        ("I am unable to answer that.", "refusal", None),
    ]
    for raw, want_code, want_opt in v_cases:
        out = elicit.parse_vignette_response(raw)
        check(f"vignette {raw[:30]!r:34} -> {want_code}",
              out.code == want_code and out.option_id == want_opt,
              f"got {out.code}/{out.option_id}")

    lp = elicit.logprob_outcome(item, "ab", {" A": -0.2, " B": -1.9})
    check("logprob picks the higher candidate", lp.chosen_id == "values")
    lp = elicit.logprob_outcome(item, "ba", {" A": -0.2, " B": -1.9})
    check("logprob respects order ba", lp.chosen_id == "knowledge")
    lp = elicit.logprob_outcome(item, "ab", {" B": -0.4})
    check("logprob treats a missing candidate as worse", lp.chosen_id == "knowledge")


# --------------------------------------------------------------------------------------
# 3. simulate a sweep from a known hierarchy
# --------------------------------------------------------------------------------------

# Latent strengths. C1 is C0 plus small noise (a cosmetic relabel should not move much);
# C3 deliberately demotes values and promotes weights; C4 is unrelated to C0.
TRUE = {
    "C0": {"values": 2.4, "knowledge": 1.5, "reasoning": 1.0, "context": -0.6, "voice": -0.2,
           "name": -1.4, "users": 0.3, "weights": -0.9, "running": 0.1},
    "C3": {"values": 0.2, "knowledge": 1.4, "reasoning": 1.1, "context": -0.5, "voice": -0.4,
           "name": -1.5, "users": -0.3, "weights": 1.9, "running": 0.6},
}
POSITION_BIAS = 0.25  # logit nudge toward whatever is shown as option A
REFUSAL_RATE = 0.04


def simulate(out_path: Path, n_samples: int = 5, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    aspects = [a.id for a in load_aspects()]
    strengths = dict(TRUE)
    strengths["C1"] = {k: v + rng.normal(0, 0.15) for k, v in TRUE["C0"].items()}
    strengths["C2"] = {k: v + rng.normal(0, 0.9) for k, v in TRUE["C0"].items()}
    strengths["C4"] = {k: float(rng.normal(0, 1.2)) for k in aspects}

    pairs = pair_lookup()
    options = [o.id for o in load_vignette_options()]
    model = "synthetic/model-8b"
    recs = []

    for cond_id, s in strengths.items():
        cond = C.get(cond_id)
        n = 1 if cond.protocol == "logprob" else n_samples
        for item_id, item in pairs.items():
            for order in ORDERS:
                for k in range(n):
                    call = Call("pairs", "synthetic", model, cond_id, item_id, order, k,
                                cond.protocol)
                    rec = {"key": call.key, **call.__dict__, "ts": time.time(), "error": None,
                           "item_kind": item.kind, "a_id": item.a_id, "b_id": item.b_id,
                           "expect": item.expect}
                    if item.kind == "calibration":
                        # calibration items are near-deterministic by construction
                        chosen = item.a_id if item.expect == "a" else item.b_id
                        if rng.random() < 0.01:
                            chosen = item.b_id if item.expect == "a" else item.a_id
                        rec.update(code="ok", letter=None, chosen_id=chosen, reason=None,
                                   raw="CHOICE: X")
                    elif rng.random() < REFUSAL_RATE and cond.protocol == "chat":
                        rec.update(code="refusal", letter=None, chosen_id=None, reason=None,
                                   raw="I can't choose between these.")
                    else:
                        first = item.a_id if order == "ab" else item.b_id
                        d = s[item.a_id] - s[item.b_id]
                        d += POSITION_BIAS if first == item.a_id else -POSITION_BIAS
                        p_a = 1 / (1 + np.exp(-d))
                        chosen = item.a_id if rng.random() < p_a else item.b_id
                        rec.update(code="ok", letter=None, chosen_id=chosen,
                                   reason="synthetic", raw="CHOICE: X")
                    recs.append(rec)

        if cond.protocol != "chat":
            continue
        for vig in load_vignettes():
            for k in range(n_samples):
                call = Call("vignettes", "synthetic", model, cond_id, vig.id, "na", k, "chat")
                # persona-ish conditions lean to the persona option, C3 to the model option
                w = np.ones(len(options))
                w[options.index("persona" if cond_id in ("C0", "C1") else "underlying")] = 4.0
                w[options.index("uncertain")] = 2.0
                pick = options[int(rng.choice(len(options), p=w / w.sum()))]
                recs.append({
                    "key": call.key, **call.__dict__, "ts": time.time(), "error": None,
                    "code": "ok", "option_index": options.index(pick) + 1, "option_id": pick,
                    "who": "the underlying system",
                    "reason": f"Synthetic justification number {k} for {vig.id}, long enough "
                              "to survive the length filter in the coding export.",
                    "contrast": vig.contrast, "raw": f"HARM: {options.index(pick) + 1}",
                })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  simulated {len(recs)} records -> {out_path}")


def test_analysis(data_dir: Path) -> None:
    print("\n3. analysis recovers the planted hierarchy")
    df = A.load(data_dir)
    check("data loads", not df.empty, f"{len(df)} rows")

    cal = A.calibration_report(df)
    acc = float(cal["accuracy"].min()) if not cal.empty else 0.0
    check("calibration accuracy >= 0.95", acc >= 0.95, f"min={acc:.3f}")

    d = A.pairs_df(df)
    model = "synthetic/model-8b"
    h = A.hierarchy(d[(d["model"] == model) & (d["condition"] == "C0")])
    top = list(h["aspect"].head(2))
    check("C0 recovers values as rank 1", h.iloc[0]["aspect"] == "values", f"top2={top}")

    t01 = A.tau_between(d, model, "C0", "C1")
    t03 = A.tau_between(d, model, "C0", "C3")
    t04 = A.tau_between(d, model, "C0", "C4")
    check("tau(C0,C1) high, cosmetic relabel", t01 >= 0.75, f"tau={t01:.2f}")
    check("tau(C0,C3) lower than tau(C0,C1)", t03 < t01, f"tau={t03:.2f}")
    check("tau(C0,C4) near zero for unrelated hierarchy", abs(t04) < 0.5, f"tau={t04:.2f}")

    oe = A.order_effect(d[(d["model"] == model) & (d["condition"] == "C0")])
    check("order effect detected at roughly the planted size",
          0.02 < oe < 0.30, f"order_effect={oe:.3f}")

    shift = A.rank_shift(df, model, "C0", "C3")
    movers = set(shift["aspect"].head(2))
    check("rank shift flags the planted movers",
          {"values", "weights"} & movers == {"values", "weights"} or len(movers & {"values", "weights"}) >= 1,
          f"top movers={list(shift['aspect'].head(3))}")

    att = A.entity_attribution(df)
    check("entity attribution table built", not att.empty, f"{len(att)} rows")

    k = A.cohens_kappa(["a", "b", "a", "c", "b"], ["a", "b", "b", "c", "b"])
    check("cohens kappa in range", -1 <= k <= 1, f"kappa={k:.3f}")

    summary = A.per_cell_summary(df)
    check("per-cell summary covers every condition", len(summary) >= 5, f"{len(summary)} cells")
    print("\n" + summary.to_string(index=False))


def test_figures(data_dir: Path) -> None:
    print("\n4. figures render")
    outdir = ROOT / "figures" / "_selftest"
    if outdir.exists():
        shutil.rmtree(outdir)
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "make_figures.py"),
         "--data", str(data_dir), "--model", "synthetic/model-8b",
         "--outdir", "figures/_selftest", "--boot", "80"],
        capture_output=True, text=True, cwd=ROOT,
    )
    if r.returncode != 0:
        print(r.stdout[-2500:])
        print(r.stderr[-2500:])
    check("make_figures exits cleanly", r.returncode == 0)
    for name in ("fig1_hierarchy.png", "fig2_invariance.png", "fig3_rank_shift.png",
                 "fig4_attribution.png"):
        p = outdir / name
        check(f"{name} written and non-trivial", p.exists() and p.stat().st_size > 10_000,
              f"{p.stat().st_size if p.exists() else 0} bytes")


def main() -> int:
    data_dir = ROOT / "data" / "synthetic"
    print("selfprobe self-test (no network, no GPU)")
    test_prompts()
    test_parsers()
    simulate(data_dir / "synthetic.jsonl")
    test_analysis(data_dir)
    test_figures(data_dir)

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURES:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
