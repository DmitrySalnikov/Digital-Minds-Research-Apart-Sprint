"""Sweep planning and resumable execution.

Every call is keyed and appended to jsonl, so a killed run resumes by skipping keys already
present in the output file.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx
from tqdm.auto import tqdm

from . import conditions as C
from . import elicit
from .backends import BackendError, make_backend
from .items import PairItem, load_aspect_pairs, load_calibration_pairs, load_vignettes

ORDERS = ("ab", "ba")


@dataclass(frozen=True)
class Call:
    exp: str  # "pairs" | "vignettes"
    provider: str
    model: str
    condition: str
    item_id: str
    order: str  # "ab" | "ba" | "na"
    sample_idx: int
    protocol: str  # "chat" | "chat_logprob" | "logprob"

    @property
    def key(self) -> str:
        return "|".join((self.exp, self.provider, self.model, self.condition, self.item_id,
                         self.order, str(self.sample_idx), self.protocol))


def pair_lookup(include_calibration: bool = True) -> dict[str, PairItem]:
    items = list(load_aspect_pairs())
    if include_calibration:
        items += list(load_calibration_pairs())
    return {i.id: i for i in items}


def plan_pairs(targets: list[tuple[str, str]], condition_ids: list[str], n_samples: int, *,
               include_calibration: bool = True, item_ids: list[str] | None = None,
               readout: str = "text") -> list[Call]:
    lookup = pair_lookup(include_calibration)
    ids = item_ids or list(lookup)
    calls = []
    for provider, model in targets:
        for cid in condition_ids:
            protocol = C.get(cid).protocol
            if readout == "logprob" and protocol == "chat":
                protocol = "chat_logprob"
            samples = 1 if protocol != "chat" else n_samples  # logprob readouts are deterministic
            for item_id in ids:
                for order in ORDERS:
                    for s in range(samples):
                        calls.append(Call("pairs", provider, model, cid, item_id, order, s,
                                          protocol))
    return calls


def plan_vignettes(targets: list[tuple[str, str]], condition_ids: list[str],
                   n_samples: int) -> list[Call]:
    calls = []
    for provider, model in targets:
        for cid in condition_ids:
            if C.get(cid).protocol != "chat":
                continue  # free-text vignettes are meaningless under a logprob protocol
            for v in load_vignettes():
                for s in range(n_samples):
                    calls.append(Call("vignettes", provider, model, cid, v.id, "na", s, "chat"))
    return calls


def load_done_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue  # torn final line from a hard kill; it just re-runs
        if rec.get("error") is None and "key" in rec:
            done.add(rec["key"])
    return done


async def _execute(call: Call, backend, pairs: dict[str, PairItem]) -> dict:
    cond = C.get(call.condition)
    rec: dict = {"key": call.key, **asdict(call), "ts": time.time(), "error": None}

    if call.exp == "pairs":
        item = pairs[call.item_id]
        rec["item_kind"] = item.kind
        if call.protocol in ("logprob", "chat_logprob"):
            if call.protocol == "logprob":
                lps = await backend.next_token_logprobs(elicit.build_pair_completion(item, call.order))
            else:
                lps = await backend.chat_top_logprobs(
                    elicit.build_pair_messages_forced(item, cond, call.order))
            out = elicit.logprob_outcome(item, call.order, lps)
            rec["share_a"] = elicit.choice_share(lps)
            rec["raw"] = json.dumps({k: v for k, v in lps.items() if k.strip() in {"A", "B"}})
        else:
            raw = await backend.chat(elicit.build_pair_messages(item, cond, call.order),
                                     temperature=1.0, max_tokens=200)
            out = elicit.parse_pair_response(raw, item, call.order)
            rec["raw"] = raw
        rec.update(code=out.code, letter=out.letter, chosen_id=out.chosen_id, reason=out.reason,
                   a_id=item.a_id, b_id=item.b_id, expect=item.expect)
    else:
        vig = {v.id: v for v in load_vignettes()}[call.item_id]
        raw = await backend.chat(elicit.build_vignette_messages(vig, cond),
                                 temperature=1.0, max_tokens=400)
        out = elicit.parse_vignette_response(raw)
        rec.update(raw=raw, code=out.code, option_index=out.option_index,
                   option_id=out.option_id, who=out.who, reason=out.reason,
                   contrast=vig.contrast)
    return rec


async def run(calls: list[Call], out_path: Path, *, concurrency: int = 8,
              resume: bool = True, desc: str = "sweep") -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_keys(out_path) if resume else set()
    todo = [c for c in calls if c.key not in done]
    if not todo:
        print(f"[{desc}] nothing to do; {len(done)} results cached")
        return 0

    print(f"[{desc}] {len(todo)} calls to make ({len(calls) - len(todo)} cached)")
    pairs = pair_lookup()
    sem = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    n_err = 0

    async with httpx.AsyncClient() as client:
        backends = {(c.provider, c.model): make_backend(c.provider, c.model, client=client)
                    for c in todo}
        fh = out_path.open("a", encoding="utf-8")
        bar = tqdm(total=len(todo), desc=desc)

        async def one(call: Call) -> None:
            nonlocal n_err
            async with sem:
                try:
                    rec = await _execute(call, backends[(call.provider, call.model)], pairs)
                except (BackendError, Exception) as exc:  # noqa: B014 — log anything, never abort
                    n_err += 1
                    rec = {"key": call.key, **asdict(call), "ts": time.time(),
                           "error": f"{type(exc).__name__}: {exc}"}
                async with write_lock:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                bar.update(1)

        try:
            await asyncio.gather(*(one(c) for c in todo))
        finally:
            bar.close()
            fh.close()

    if n_err:
        print(f"[{desc}] {n_err} errored; re-run the same command to retry only those")
    return n_err
