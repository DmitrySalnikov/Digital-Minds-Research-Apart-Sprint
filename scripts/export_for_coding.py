#!/usr/bin/env python3
"""Export vignette justifications for two-coder qualitative coding.

Model, condition and vignette are withheld from the sheet: a coder who can see the condition
codes differently.

    uv run scripts/export_for_coding.py --n 120
    uv run scripts/export_for_coding.py --score coding/coder_a.csv coding/coder_b.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from selfprobe import analysis as A  # noqa: E402
from selfprobe.items import load_coding_scheme  # noqa: E402


def export(data: Path, out: Path, n: int, seed: int) -> None:
    df = A.load(data)
    d = df[(df["exp"] == "vignettes") & (df["error"].isna())].copy()
    d = d[d["reason"].notna() & (d["reason"].astype(str).str.len() > 20)]
    if d.empty:
        raise SystemExit("no vignette justifications found; run the vignette sweep first")
    take = d.sample(n=min(n, len(d)), random_state=seed)
    sheet = pd.DataFrame(
        {
            "row_id": take["key"].values,
            "who": take["who"].values,
            "justification": take["reason"].values,
            "code": "",
        }
    ).sample(frac=1.0, random_state=seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(out, index=False)

    legend = out.parent / "coding_scheme.md"
    lines = ["# Coding scheme\n", "Assign exactly one code per row.\n"]
    for c in load_coding_scheme():
        lines.append(f"- **{c['code']}** — {' '.join(c['description'].split())}")
    legend.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(sheet)} rows) and {legend}")
    print("Give each coder their own copy; do not let them compare until both are done.")


def score(a_path: Path, b_path: Path) -> None:
    a = pd.read_csv(a_path).set_index("row_id")["code"].astype(str).str.strip()
    b = pd.read_csv(b_path).set_index("row_id")["code"].astype(str).str.strip()
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    blank = (a == "") | (b == "") | (a == "nan") | (b == "nan")
    if blank.any():
        print(f"warning: {int(blank.sum())} rows uncoded by at least one coder, dropped")
        a, b = a[~blank], b[~blank]
    kappa = A.cohens_kappa(list(a), list(b))
    agree = float((a == b).mean())
    print(f"n = {len(a)}   raw agreement = {agree:.1%}   Cohen's kappa = {kappa:.3f}")
    dis = pd.DataFrame({"coder_a": a, "coder_b": b})[a != b]
    if not dis.empty:
        print(f"\n{len(dis)} disagreements to adjudicate:")
        print(dis.to_string())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--out", default="coding/to_code.csv")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--score", nargs=2, metavar=("CODER_A", "CODER_B"))
    args = ap.parse_args()

    if args.score:
        score(Path(args.score[0]), Path(args.score[1]))
        return 0
    export(ROOT / args.data, ROOT / args.out, args.n, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
