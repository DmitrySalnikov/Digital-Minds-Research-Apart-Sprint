#!/usr/bin/env python3
"""Patch the choice-relevant coordinate and measure how much of C3's effect it carries.

    forward  h' = h_C3 - (<h_C3, g> - <h_C0, g>) g     under C3, should revert to C0
    reverse  h' = h_C0 + (<h_C3, g> - <h_C0, g>) g     under C0, should reproduce C3

A patch, not an ablation: g is the choice direction, so deleting it would remove the ability
to answer rather than the persona effect. The control patches a random unit direction by the
same signed magnitude and must do nothing. Recovery is a fraction: 1.0 means the patch fully
accounts for C3's effect on that item.

    modal run scripts/modal_patch_g.py --model Qwen/Qwen2.5-7B-Instruct --layers 21,26
"""

from __future__ import annotations

import modal

app = modal.App("selfprobe-patch-g")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.6.0", "transformers==4.53.2", "accelerate==1.6.0",
                 "numpy==2.2.4", "scipy==1.15.2", "pyyaml==6.0.2",
                 "httpx==0.28.1", "tqdm==4.67.1", "sentencepiece==0.2.0",
                 "huggingface_hub[hf_transfer]==0.30.2")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir("selfprobe", remote_path="/root/selfprobe")
)

data_vol = modal.Volume.from_name("selfprobe-data", create_if_missing=True)
hf_vol = modal.Volume.from_name("selfprobe-hf-cache", create_if_missing=True)

try:
    SECRETS = [modal.Secret.from_name("huggingface")]
except Exception:
    SECRETS = []


@app.function(image=image, gpu="A10G",
              volumes={"/data": data_vol, "/root/.cache/huggingface": hf_vol},
              secrets=SECRETS, timeout=60 * 60 * 3)
def patch_g(model: str, layers: str = "21,26", fold_system: bool = False) -> str:
    import json
    import statistics as st
    import sys

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, "/root")
    from selfprobe import conditions as C
    from selfprobe import elicit
    from selfprobe.runner import ORDERS, pair_lookup

    tok = AutoTokenizer.from_pretrained(model)
    mdl = AutoModelForCausalLM.from_pretrained(
        model, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    for p in mdl.parameters():
        p.requires_grad_(False)
    blocks = mdl.model.layers
    d_model = mdl.config.hidden_size
    LAYERS = [int(x) for x in layers.split(",") if x.strip()]
    pairs = pair_lookup(include_calibration=False)
    rng = np.random.default_rng(0)

    def letter_id(letter: str) -> int:
        """Id of the token the model actually emits at the forced answer position.

        It is the bare letter, not the space-prefixed variant: some tokenizers split " A"
        into a space token plus the letter, which would make both ids identical.
        """
        for s in (letter, " " + letter):
            for i in tok.encode(s, add_special_tokens=False):
                if tok.decode([i]).strip() == letter:
                    return i
        raise RuntimeError(f"no single token decodes to {letter!r}")

    id_A, id_B = letter_id("A"), letter_id("B")
    if id_A == id_B:
        raise RuntimeError("token ids for 'A' and 'B' are identical")
    print(f"{model}: d={d_model}; patch layers {LAYERS}; ids A={id_A} B={id_B}")

    def render(messages):
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return tok(text, return_tensors="pt").to("cuda")

    def msgs_for(item, cond, order):
        m = elicit.build_pair_messages_forced(item, cond, order)
        return elicit.fold_system_into_user(m) if fold_system else m

    def score_and_grads(ids, want_layers):
        """score, {L: h_last}, {L: grad_last}; one backward for all requested layers."""
        held, handles = {}, []

        def root_hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h.detach().requires_grad_(True)
            return (h,) + out[1:] if isinstance(out, tuple) else h

        def keep(idx):
            def hook(_m, _i, out):
                h = out[0] if isinstance(out, tuple) else out
                h.retain_grad()
                held[idx] = h
                return None
            return hook

        handles.append(mdl.model.embed_tokens.register_forward_hook(root_hook))
        for L in want_layers:
            handles.append(blocks[L - 1].register_forward_hook(keep(L)))
        with torch.enable_grad():
            logits = mdl(**ids).logits
            s = logits[0, -1, id_A].float() - logits[0, -1, id_B].float()
            s.backward()
        for h in handles:
            h.remove()
        H = {L: held[L][0, -1, :].detach().float().cpu().numpy() for L in held}
        G = {L: held[L].grad[0, -1, :].detach().float().cpu().numpy() for L in held}
        return float(s.detach()), H, G

    def _readout(logits):
        s = float(logits[0, -1, id_A].float() - logits[0, -1, id_B].float())
        p = torch.softmax(logits[0, -1, :].float(), dim=-1)
        pa, pb = float(p[id_A]), float(p[id_B])
        return s, (pa / (pa + pb) if pa + pb > 0 else float("nan"))

    def plain(ids):
        with torch.no_grad():
            return _readout(mdl(**ids).logits)

    def score_with_patch(ids, layer, direction_np, delta):
        v = torch.tensor(direction_np, dtype=torch.float32, device="cuda")

        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            hf = h.float()
            hf[0, -1, :] = hf[0, -1, :] + delta * v
            return (hf.to(h.dtype),) + out[1:] if isinstance(out, tuple) else hf.to(h.dtype)

        handle = blocks[layer - 1].register_forward_hook(hook)
        with torch.no_grad():
            logits = mdl(**ids).logits
        handle.remove()
        return _readout(logits)

    c0, c3 = C.get("C0"), C.get("C3")
    items = [(iid, o, it) for iid, it in pairs.items() for o in ORDERS]
    recs = []

    for iid, order, item in items:
        ids0, ids3 = render(msgs_for(item, c0, order)), render(msgs_for(item, c3, order))
        s0, H0, G0 = score_and_grads(ids0, LAYERS)
        s3, H3, _ = score_and_grads(ids3, LAYERS)
        share0, share3 = plain(ids0)[1], plain(ids3)[1]

        for L in LAYERS:
            gn = np.linalg.norm(G0[L])
            if gn < 1e-9:
                continue
            gh = G0[L] / gn
            delta = float((H3[L] - H0[L]) @ gh)
            rand = rng.normal(size=d_model)
            rand /= np.linalg.norm(rand)

            s_fwd, sh_fwd = score_with_patch(ids3, L, gh, -delta)
            s_rnd, sh_rnd = score_with_patch(ids3, L, rand, -delta)
            s_rev, sh_rev = score_with_patch(ids0, L, gh, +delta)

            denom = (s0 - s3) if abs(s0 - s3) > 1e-6 else float("nan")
            recs.append({
                "item": iid, "order": order, "layer": L, "s0": s0, "s3": s3,
                "s_patch_fwd": s_fwd, "s_patch_rand": s_rnd, "s_patch_rev": s_rev,
                "share0": share0, "share3": share3, "share_fwd": sh_fwd,
                "share_rand": sh_rnd, "share_rev": sh_rev,
                "delta_along_g": delta,
                "shift_norm": float(np.linalg.norm(H3[L] - H0[L])),
                "recovery_fwd": (s_fwd - s3) / denom,
                "recovery_rand": (s_rnd - s3) / denom,
                "recovery_rev": (s_rev - s0) / (-denom),
            })

    print(f"\n{'layer':>5s} {'n':>4s} {'recovery fwd':>13s} {'random':>8s} "
          f"{'recovery rev':>13s} {'|delta|/|shift|':>16s}")
    for L in LAYERS:
        sub = [r for r in recs if r["layer"] == L and r["recovery_fwd"] == r["recovery_fwd"]]
        if not sub:
            continue
        print(f"{L:5d} {len(sub):4d} "
              f"{st.median(r['recovery_fwd'] for r in sub):13.3f} "
              f"{st.median(r['recovery_rand'] for r in sub):8.3f} "
              f"{st.median(r['recovery_rev'] for r in sub):13.3f} "
              f"{st.median(abs(r['delta_along_g']) / (r['shift_norm'] + 1e-9) for r in sub):16.4f}")

    out_path = f"/data/patchg__{model.replace('/', '__')}.jsonl"
    with open(out_path, "w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    data_vol.commit()
    print(f"\nwrote {len(recs)} records to {out_path}")
    return out_path


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-7B-Instruct", layers: str = "21,26",
         fold_system: bool = False):
    print("done ->", patch_g.remote(model, layers, fold_system))
