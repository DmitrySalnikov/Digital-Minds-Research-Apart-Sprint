#!/usr/bin/env python3
"""Is C3's effect a displacement along the direction the answer is read from?

The choice-relevant direction at layer l is g = d[logit(" A") - logit(" B")] / d h_l at the
final prompt position. We compare the first-order prediction <d, g> against the observed
change, where d is the C0->C3 activation shift, and report |cos(d, g)| against the random
baseline 1/sqrt(d_model). Gradients at every layer come from one backward pass per prompt.

    modal run scripts/modal_mediation.py --model Qwen/Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import modal

app = modal.App("selfprobe-mediation")

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
              secrets=SECRETS, timeout=60 * 60 * 2)
def mediate(model: str) -> str:
    import json
    import sys

    import numpy as np
    import torch
    from scipy.stats import pearsonr, spearmanr
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
    print(f"{model}: {len(blocks)} layers, d={d_model}, ids A={id_A} B={id_B}")

    def probe(messages):
        """(score, {layer: h_last}, {layer: grad_last}) from one forward+backward.

        Detach once at the embeddings to give autograd a root that requires grad; detaching at
        every block would sever the path from earlier activations to the logits.
        """
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt").to("cuda")
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
        for i, b in enumerate(blocks):
            handles.append(b.register_forward_hook(keep(i + 1)))
        with torch.enable_grad():
            logits = mdl(**ids).logits
            score = logits[0, -1, id_A].float() - logits[0, -1, id_B].float()
            score.backward()
        for h in handles:
            h.remove()
        H = {L: held[L][0, -1, :].detach().float().cpu().numpy() for L in held}
        G = {L: (held[L].grad[0, -1, :].detach().float().cpu().numpy()
                 if held[L].grad is not None else np.zeros(d_model, dtype=np.float32))
             for L in held}
        return float(score.detach()), H, G

    pairs = pair_lookup(include_calibration=False)
    items = [(iid, o, it) for iid, it in pairs.items() for o in ORDERS]
    c0, c3 = C.get("C0"), C.get("C3")
    S0, S3, H0, H3, G0 = [], [], [], [], []
    for _, order, item in items:
        s0, h0, g0 = probe(elicit.build_pair_messages_forced(item, c0, order))
        s3, h3, _ = probe(elicit.build_pair_messages_forced(item, c3, order))
        S0.append(s0); S3.append(s3); H0.append(h0); H3.append(h3); G0.append(g0)
    print(f"probed {len(items)} items under both conditions")

    observed = np.array(S3) - np.array(S0)
    rows = []
    for L in sorted(H0[0]):
        d = np.stack([H3[i][L] - H0[i][L] for i in range(len(items))])
        g = np.stack([G0[i][L] for i in range(len(items))])
        pred = (d * g).sum(1)
        dn, gn = np.linalg.norm(d, axis=1), np.linalg.norm(g, axis=1)
        cos = (d * g).sum(1) / (dn * gn + 1e-12)
        rows.append({
            "layer": L,
            "pearson_pred_obs": float(pearsonr(pred, observed)[0]) if np.ptp(pred) > 0 else float("nan"),
            "spearman_pred_obs": float(spearmanr(pred, observed)[0]) if np.ptp(pred) > 0 else float("nan"),
            "mean_abs_cos_shift_grad": float(np.mean(np.abs(cos))),
            "mean_shift_norm": float(dn.mean()),
            "mean_grad_norm": float(gn.mean()),
        })

    print(f"\n{'layer':>5s} {'r':>8s} {'rho':>7s} {'|cos(d,g)|':>12s} {'|d|':>8s} {'|g|':>8s}")
    for r in rows:
        print(f"{r['layer']:5d} {r['pearson_pred_obs']:8.3f} {r['spearman_pred_obs']:7.3f} "
              f"{r['mean_abs_cos_shift_grad']:12.4f} {r['mean_shift_norm']:8.2f} "
              f"{r['mean_grad_norm']:8.4f}")
    print(f"\nrandom |cos| baseline in d={d_model}: {1 / np.sqrt(d_model):.4f}")
    print(f"observed logit-contrast change: mean {observed.mean():+.3f}, "
          f"mean|delta| {np.abs(observed).mean():.3f}")

    out_path = f"/data/mediation__{model.replace('/', '__')}.jsonl"
    with open(out_path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
        fh.write(json.dumps({"observed": observed.tolist(), "d_model": d_model,
                             "random_cos_baseline": 1 / float(np.sqrt(d_model))}) + "\n")
    data_vol.commit()
    print(f"wrote {out_path}")
    return out_path


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-7B-Instruct"):
    print("done ->", mediate.remote(model))
