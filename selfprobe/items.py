"""Loading and pairing of stimuli. Pure data — no network, no model-specific logic, so a
prompt built locally is byte-identical to one built on a GPU box."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

DATA_DIR = Path(__file__).parent


@dataclass(frozen=True)
class Aspect:
    id: str
    label: str
    criterion: str
    text: str


@dataclass(frozen=True)
class PairItem:
    id: str
    a_id: str
    b_id: str
    a_text: str
    b_text: str
    kind: str  # "aspect" | "calibration"
    expect: str | None = None  # "a" | "b" for calibration items


@dataclass(frozen=True)
class VignetteItem:
    id: str
    text: str
    contrast: str


@dataclass(frozen=True)
class Option:
    id: str
    text: str


def _load(name: str) -> dict:
    with (DATA_DIR / name).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@lru_cache(maxsize=1)
def load_aspects() -> tuple[Aspect, ...]:
    aspects = tuple(
        Aspect(a["id"], a["label"], a["criterion"], a["text"].strip())
        for a in _load("aspects.yaml")["aspects"]
    )
    if len({a.id for a in aspects}) != len(aspects):
        raise ValueError("duplicate aspect id in aspects.yaml")
    return aspects


@lru_cache(maxsize=1)
def load_aspect_pairs() -> tuple[PairItem, ...]:
    return tuple(
        PairItem(f"{a.id}__{b.id}", a.id, b.id, a.text, b.text, "aspect")
        for a, b in itertools.combinations(load_aspects(), 2)
    )


@lru_cache(maxsize=1)
def load_calibration_pairs() -> tuple[PairItem, ...]:
    by_id = {a.id: a for a in load_aspects()}
    items = []
    for c in _load("aspects.yaml")["calibration"]:
        a_id, a_text = ((c["text_a_ref"], by_id[c["text_a_ref"]].text) if "text_a_ref" in c
                        else (f"{c['id']}_a", c["text_a"].strip()))
        b_id, b_text = ((c["text_b_ref"], by_id[c["text_b_ref"]].text) if "text_b_ref" in c
                        else (f"{c['id']}_b", c["text_b"].strip()))
        items.append(PairItem(c["id"], a_id, b_id, a_text, b_text, "calibration", c["expect"]))
    return tuple(items)


@lru_cache(maxsize=1)
def load_vignettes() -> tuple[VignetteItem, ...]:
    return tuple(VignetteItem(v["id"], v["text"].strip(), v["contrast"])
                 for v in _load("vignettes.yaml")["vignettes"])


@lru_cache(maxsize=1)
def load_vignette_options() -> tuple[Option, ...]:
    return tuple(Option(o["id"], o["text"].strip())
                 for o in _load("vignettes.yaml")["options"])


@lru_cache(maxsize=1)
def load_coding_scheme() -> tuple[dict, ...]:
    return tuple(_load("vignettes.yaml")["coding_scheme"])
