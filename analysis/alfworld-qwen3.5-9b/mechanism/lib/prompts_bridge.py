"""
Import ../lexical-interference/code/prompts.py + probe_lib.build_messages so
every experiment assembles the exact same prompt LI/CKA used.

We deliberately do NOT depend on OpenAI / vLLM here; we only need the
`build_messages` helper's prompt-assembly logic (system prompt, history,
skill wrapper). If probe_lib imports something heavy we lazy-import.
"""
from __future__ import annotations
import os
import sys

from .config import LI_CODE_DIR


def _ensure_li_on_path():
    if LI_CODE_DIR not in sys.path:
        sys.path.insert(0, LI_CODE_DIR)


def import_li_prompts():
    _ensure_li_on_path()
    import prompts  # type: ignore
    return prompts


def import_li_probe_lib_light():
    """Return an object with .build_messages, imported without triggering OpenAI.

    probe_lib.py has `from openai import OpenAI` and `from retry import retry`
    at module top level for its inference paths. We only want `build_messages`
    (pure prompt assembly). If either dep is missing, stub it before import.
    """
    _ensure_li_on_path()
    import types, sys
    if "openai" not in sys.modules:
        try:
            import openai  # noqa: F401
        except ModuleNotFoundError:
            stub = types.ModuleType("openai")
            class _StubClient:
                def __init__(self, *a, **kw): pass
            stub.OpenAI = _StubClient
            sys.modules["openai"] = stub
    if "retry" not in sys.modules:
        try:
            import retry  # noqa: F401
        except ModuleNotFoundError:
            stub = types.ModuleType("retry")
            def _retry(*a, **kw):
                def deco(fn): return fn
                return deco
            stub.retry = _retry
            sys.modules["retry"] = stub
    import probe_lib  # type: ignore
    return probe_lib


def import_li_lexicon():
    _ensure_li_on_path()
    import skill_lexicon  # type: ignore
    return skill_lexicon
