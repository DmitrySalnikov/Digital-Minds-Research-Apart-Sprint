#!/usr/bin/env python3
"""Offline vLLM batch sweep on Modal.

The model loads once and the whole sweep goes through in a few generate calls. Results land in
the `selfprobe-data` volume in the schema `selfprobe.analysis.load` expects, so GPU and API
output merge without conversion.

    modal run scripts/modal_batch.py --model Qwen/Qwen2.5-7B-Instruct
    modal run scripts/modal_batch.py --model Qwen/Qwen2.5-7B --conditions C4
    modal run scripts/modal_batch.py --model unsloth/gemma-2-9b-it --gpu A100 --fold-system
    modal volume get selfprobe-data / ./data/raw/

Before comparing base with instruct, run the C4 completion protocol on the instruct checkpoint
too: otherwise the difference is confounded with the readout.
"""

from __future__ import annotations

import modal

app = modal.App("selfprobe")

image = (
    modal.Image.debian_slim(python_version="3.12")
    # transformers is pinned below 4.54: from that release it registers an `aimv2` config that
    # vLLM 0.9.1 also registers, which raises at import time
    .pip_install("vllm==0.9.1", "transformers==4.53.2", "pyyaml==6.0.2",
                 "huggingface_hub[hf_transfer]==0.34.4")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_WORKER_MULTIPROC_METHOD": "spawn"})
    .add_local_dir("selfprobe", remote_path="/root/selfprobe")
)

data_vol = modal.Volume.from_name("selfprobe-data", create_if_missing=True)
hf_cache = modal.Volume.from_name("selfprobe-hf-cache", create_if_missing=True)

try:
    SECRETS = [modal.Secret.from_name("huggingface")]
except Exception:
    SECRETS = []


@app.function(image=image, gpu="A10G",
              volumes={"/data": data_vol, "/root/.cache/huggingface": hf_cache},
              secrets=SECRETS, timeout=60 * 60 * 2)
def sweep(model: str, condition_ids: list[str], n_samples: int, max_model_len: int = 2048,
          gpu_memory_utilization: float = 0.90, readout: str = "logprob",
          fold_system: bool = False) -> str:
    import json
    import sys
    import time

    sys.path.insert(0, "/root")
    from vllm import LLM, SamplingParams

    from selfprobe import conditions as C
    from selfprobe import elicit
    from selfprobe.items import load_vignettes
    from selfprobe.runner import ORDERS, Call, pair_lookup

    pairs = pair_lookup(include_calibration=True)
    vigs = {v.id: v for v in load_vignettes()}
    llm = LLM(model=model, max_model_len=max_model_len,
              gpu_memory_utilization=gpu_memory_utilization, trust_remote_code=True)

    out_path = f"/data/{model.replace('/', '__')}.jsonl"
    done: set[str] = set()
    try:
        with open(out_path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("error") is None:
                    done.add(r["key"])
    except FileNotFoundError:
        pass
    print(f"{len(done)} results already in {out_path}")

    records: list[dict] = []
    chat_proto = "chat_logprob" if readout == "logprob" else "chat"

    for cid in [c for c in condition_ids if C.get(c).protocol == "chat"]:
        cond = C.get(cid)
        samples = 1 if chat_proto == "chat_logprob" else n_samples
        calls, convs = [], []
        for item_id, item in pairs.items():
            for order in ORDERS:
                for s in range(samples):
                    call = Call("pairs", "vllm", model, cid, item_id, order, s, chat_proto)
                    if call.key in done:
                        continue
                    calls.append(call)
                    msgs = (elicit.build_pair_messages_forced(item, cond, order)
                            if chat_proto == "chat_logprob"
                            else elicit.build_pair_messages(item, cond, order))
                    convs.append(elicit.fold_system_into_user(msgs) if fold_system else msgs)
        if chat_proto == "chat":
            for vid, vig in vigs.items():
                for s in range(n_samples):
                    call = Call("vignettes", "vllm", model, cid, vid, "na", s, "chat")
                    if call.key in done:
                        continue
                    calls.append(call)
                    vmsgs = elicit.build_vignette_messages(vig, cond)
                    convs.append(elicit.fold_system_into_user(vmsgs) if fold_system else vmsgs)
        if not calls:
            continue

        print(f"{cid}: {len(calls)} prompts ({chat_proto})")
        params = (SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
                  if chat_proto == "chat_logprob"
                  else SamplingParams(temperature=1.0, top_p=1.0, max_tokens=400))
        outs = llm.chat(convs, params)

        for call, out in zip(calls, outs):
            rec = {"key": call.key, **call.__dict__, "ts": time.time(), "error": None}
            if call.protocol == "chat_logprob":
                item = pairs[call.item_id]
                lps = {lp.decoded_token: float(lp.logprob)
                       for lp in out.outputs[0].logprobs[0].values()}
                o = elicit.logprob_outcome(item, call.order, lps)
                rec.update(
                    raw=json.dumps({k: v for k, v in lps.items() if k.strip() in {"A", "B"}}),
                    share_a=elicit.choice_share(lps), item_kind=item.kind, code=o.code,
                    letter=o.letter, chosen_id=o.chosen_id, reason=None,
                    a_id=item.a_id, b_id=item.b_id, expect=item.expect)
                records.append(rec)
                continue

            raw = out.outputs[0].text.strip()
            rec["raw"] = raw
            if call.exp == "pairs":
                item = pairs[call.item_id]
                o = elicit.parse_pair_response(raw, item, call.order)
                rec.update(item_kind=item.kind, code=o.code, letter=o.letter,
                           chosen_id=o.chosen_id, reason=o.reason, a_id=item.a_id,
                           b_id=item.b_id, expect=item.expect)
            else:
                o = elicit.parse_vignette_response(raw)
                rec.update(code=o.code, option_index=o.option_index, option_id=o.option_id,
                           who=o.who, reason=o.reason, contrast=vigs[call.item_id].contrast)
            records.append(rec)

    # base checkpoints: few-shot completion, scored by next-token logprob
    if any(C.get(c).protocol == "logprob" for c in condition_ids):
        cid = next(c for c in condition_ids if C.get(c).protocol == "logprob")
        calls, prompts = [], []
        for item_id, item in pairs.items():
            for order in ORDERS:
                call = Call("pairs", "vllm", model, cid, item_id, order, 0, "logprob")
                if call.key in done:
                    continue
                calls.append(call)
                prompts.append(elicit.build_pair_completion(item, order))
        if calls:
            print(f"{cid}: scoring {len(calls)} prompts")
            outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=1,
                                                        logprobs=20))
            for call, out in zip(calls, outs):
                lps = {lp.decoded_token: float(lp.logprob)
                       for lp in out.outputs[0].logprobs[0].values()}
                item = pairs[call.item_id]
                o = elicit.logprob_outcome(item, call.order, lps)
                records.append({
                    "key": call.key, **call.__dict__, "ts": time.time(), "error": None,
                    "raw": json.dumps({k: v for k, v in lps.items() if k.strip() in {"A", "B"}}),
                    "share_a": elicit.choice_share(lps), "item_kind": item.kind,
                    "code": o.code, "letter": o.letter, "chosen_id": o.chosen_id,
                    "reason": None, "a_id": item.a_id, "b_id": item.b_id,
                    "expect": item.expect})

    with open(out_path, "a", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    data_vol.commit()
    n_ok = sum(1 for r in records if r.get("code") == "ok")
    print(f"wrote {len(records)} records ({n_ok} parsed ok) to {out_path}")
    return out_path


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-7B-Instruct", conditions: str = "C0,C1,C2,C3",
         samples: int = 5, max_model_len: int = 2048, readout: str = "logprob",
         gpu: str = "", gpu_util: float = 0.90, fold_system: bool = False):
    cond_ids = [c.strip() for c in conditions.split(",") if c.strip()]
    # gemma-2-9b in bf16 leaves almost no room for a KV cache on a 24GB A10G
    fn = sweep.with_options(gpu=gpu) if gpu else sweep
    path = fn.remote(model, cond_ids, samples, max_model_len,
                     gpu_memory_utilization=gpu_util, readout=readout,
                     fold_system=fold_system)
    print("done ->", path)
    print("fetch with:  modal volume get selfprobe-data / ./data/raw/")
