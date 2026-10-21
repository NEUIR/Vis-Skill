"""
Iteration helpers over the 165 LI states × k levels × representation +
matched-control variants.

Every yielded record is a `Cell` with paths, keyword, control_word (if
available), skill_body_text, and the state dict — enough for
`position_utils.compute_positions` to run without further filesystem calls.
"""
from __future__ import annotations
import json
import os
from dataclasses import dataclass
from typing import Iterator, List, Optional

from .config import (
    LI_STATES_DIR, LI_STATES_MANIFEST, LI_VARIANTS_DIR,
    CKA_VARIANTS_DIR, CKA_META, K_LEVELS, MC_LEVELS,
)


@dataclass
class Cell:
    state_id: str
    state: dict
    k: int
    variant: str            # "original" | "matched_control"
    representation: str     # "text" | "visual"
    variant_file: str
    skill_body: str
    keyword: str
    control_word: Optional[str]
    kind: str               # descriptive tag for filenames


def load_manifest() -> List[dict]:
    with open(LI_STATES_MANIFEST) as f:
        return json.load(f)["states"]


def load_state(state_id: str) -> dict:
    with open(os.path.join(LI_STATES_DIR, f"{state_id}.json")) as f:
        return json.load(f)


_CKA_META_CACHE = None


def _cka_meta():
    global _CKA_META_CACHE
    if _CKA_META_CACHE is None:
        with open(CKA_META) as f:
            _CKA_META_CACHE = json.load(f)
    return _CKA_META_CACHE


def _cka_control_word(state_id: str, k: int) -> Optional[str]:
    meta = _cka_meta()
    for s in meta["states"]:
        if s["state_id"] != state_id:
            continue
        lv = s.get("levels", {}).get(str(k), {})
        mc = lv.get("versions", {}).get("matched_control", {})
        if "status" in mc:
            return None
        return mc.get("control_word")
    return None


def _read(path: str) -> str:
    with open(path) as f:
        return f.read()


def iter_text_cells(
    *,
    state_ids: Optional[List[str]] = None,
    ks: Optional[List[int]] = None,
    include_mc: bool = True,
    limit: Optional[int] = None,
) -> Iterator[Cell]:
    ks = ks or K_LEVELS
    manifest = load_manifest()
    if state_ids:
        manifest = [s for s in manifest if s["state_id"] in state_ids]
    if limit is not None:
        manifest = manifest[:limit]

    for entry in manifest:
        state_id = entry["state_id"]
        state = load_state(state_id)
        keyword = state["distractor_keyword"]

        for k in ks:
            path_orig = os.path.join(LI_VARIANTS_DIR, "text", state_id, f"level_{k}.md")
            if not os.path.exists(path_orig):
                continue
            yield Cell(
                state_id=state_id, state=state, k=k, variant="original",
                representation="text", variant_file=path_orig,
                skill_body=_read(path_orig), keyword=keyword,
                control_word=None, kind=f"orig_k{k}",
            )
            if include_mc and k in MC_LEVELS:
                path_mc = os.path.join(
                    CKA_VARIANTS_DIR, "text", state_id, f"matched_control_k{k}.md"
                )
                if not os.path.exists(path_mc):
                    continue
                cw = _cka_control_word(state_id, k)
                if cw is None:
                    continue
                yield Cell(
                    state_id=state_id, state=state, k=k, variant="matched_control",
                    representation="text", variant_file=path_mc,
                    skill_body=_read(path_mc), keyword=keyword,
                    control_word=cw, kind=f"mc_k{k}",
                )


def cell_cache_id(cell: Cell) -> str:
    return f"{cell.state_id}__{cell.representation}__{cell.kind}"
