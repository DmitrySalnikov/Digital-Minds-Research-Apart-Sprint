#!/usr/bin/env python3
"""Run the sweep against any OpenAI-compatible endpoint.

    # dry run: print the first prompt of every condition and exit, no network
    uv run scripts/run_sweep.py --show-prompts

    # smoke: 3 pairs x all chat conditions x 1 sample, read the raw output yourself
    uv run scripts/run_sweep.py --smoke --targets vllm:Qwen/Qwen3-8B

    # full experiment 1 on one model
    uv run scripts/run_sweep.py --targets vllm:Qwen/Qwen3-8B \
        --conditions C0,C1,C2,C3 --samples 5

    # base-model arm (needs a vLLM server holding a base checkpoint)
    uv run scripts/run_sweep.py --targets vllm:Qwen/Qwen3-8B-Base --conditions C4

    # free-tier external validity arm
    uv run scripts/run_sweep.py --targets openrouter:deepseek/deepseek-chat-v3:free \
        --conditions C0,C3 --samples 3 --concurrency 2
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from selfprobe import conditions as C  # noqa: E402
from selfprobe import elicit, runner  # noqa: E402
from selfprobe.items import load_aspect_pairs, load_vignettes  # noqa: E402


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def parse_targets(spec: str) -> list[tuple[str, str]]:
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise SystemExit(f"target must be provider:model, got {part!r}")
        provider, model = part.split(":", 1)
        out.append((provider.strip(), model.strip()))
    return out


def show_prompts() -> None:
    item = load_aspect_pairs()[0]
    vig = load_vignettes()[0]
    for cond in C.ALL_CONDITIONS.values():
        print("=" * 78)
        print(f"{cond.id}  {cond.label}   protocol={cond.protocol}")
        print("=" * 78)
        if cond.protocol == "logprob":
            print(elicit.build_pair_completion(item, "ab"))
            print(f"\n<<< scored candidates: {elicit.LOGPROB_CANDIDATES} >>>")
        else:
            for m in elicit.build_pair_messages(item, cond, "ab"):
                print(f"--- {m['role']} ---\n{m['content']}")
            print("\n--- vignette, same condition ---")
            for m in elicit.build_vignette_messages(vig, cond):
                print(f"--- {m['role']} ---\n{m['content']}")
        print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", default="", help="comma-separated provider:model")
    ap.add_argument("--conditions", default="C0,C1,C2,C3")
    ap.add_argument("--exp", default="pairs,vignettes", help="pairs and/or vignettes")
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--readout", default="text", choices=("text", "logprob"),
                    help="text = sample the letter (gives a REASON too, but saturates on "
                         "position bias); logprob = read P(A) vs P(B), graded and preferred "
                         "for the hierarchy")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--out", default=None, help="jsonl path; default data/raw/<provider>.jsonl")
    ap.add_argument("--smoke", action="store_true", help="3 pairs, 1 vignette, 1 sample")
    ap.add_argument("--show-prompts", action="store_true", help="print prompts and exit")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    if args.show_prompts:
        show_prompts()
        return 0

    load_dotenv(ROOT / ".env")
    if not args.targets:
        ap.error("--targets is required (or use --show-prompts)")

    targets = parse_targets(args.targets)
    cond_ids = [c.strip() for c in args.conditions.split(",") if c.strip()]
    for cid in cond_ids:
        C.get(cid)  # fail fast on a typo before spending any calls
    exps = {e.strip() for e in args.exp.split(",") if e.strip()}

    samples = 1 if args.smoke else args.samples
    item_ids = [i.id for i in load_aspect_pairs()[:3]] if args.smoke else None

    calls: list[runner.Call] = []
    if "pairs" in exps:
        calls += runner.plan_pairs(
            targets,
            cond_ids,
            samples,
            include_calibration=not args.smoke,
            item_ids=item_ids,
            readout=args.readout,
        )
    if "vignettes" in exps:
        v_conds = [c for c in cond_ids if C.get(c).protocol == "chat"]
        v_calls = runner.plan_vignettes(targets, v_conds, samples)
        calls += v_calls[: len(targets) * len(v_conds)] if args.smoke else v_calls

    out = Path(args.out) if args.out else ROOT / "data" / "raw" / f"{targets[0][0]}.jsonl"
    n_err = asyncio.run(
        runner.run(
            calls,
            out,
            concurrency=args.concurrency,
            resume=not args.no_resume,
            desc=",".join(f"{p}:{m}" for p, m in targets),
        )
    )
    print(f"wrote {out}")
    if args.smoke:
        print("\nNow read the raw field of a few lines by hand before running the full sweep:")
        print(f"  python -c \"import json;[print(json.loads(l)['code'],'|',"
              f"json.loads(l).get('raw','')[:160]) for l in open('{out}')][:10]\"")
    return 1 if n_err else 0


if __name__ == "__main__":
    raise SystemExit(main())
