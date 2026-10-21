#!/usr/bin/env python3
"""
Tokenizer sanity check for symbol-encoding on Gemma-4-31B-it.

The whole symbol-encoding design rides on the fact that changing the casing of a keyword
in a text document produces a DIFFERENT token sequence.  Under Qwen's BPE that
is true by construction (`cabinet` -> 1 token, `CABINET` -> 3, `CaBiNeT` -> ~4).
Under Gemma's SentencePiece it is EMPIRICALLY true (SentencePiece is not
case-normalising by default), but that is exactly the sort of thing a config
flag can change silently.  If it turns out to be false here, then the five
casing arms would all tokenise identically and the whole experiment is a null
intervention -- worse, one that would produce numbers that LOOK meaningful.

So this script refuses to proceed unless every arm produces a DIFFERENT token
count from `native` for at least one of the LI keywords.  It also prints the
per-keyword token counts so the human can eyeball them.

No model weights loaded; tokenizer only; runs in a few seconds on CPU.
"""
import os
import sys
from collections import defaultdict

GEMMA = os.environ.get("GEMMA_MODEL", os.path.join(os.environ.get("MODELS_DIR", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "models"))), "Gemma-4-31B-it"))

# The 7 keywords used in LI (from lexical-interference/code/skill_lexicon.py).
KEYWORDS = ["cabinet", "drawer", "fridge", "microwave", "safe",
            "desklamp", "sinkbasin"]

# What "arm" means to symbol-encoding, applied to the keyword itself.
def native(w):    return w
def title(w):     return w.title()
def upper(w):     return w.upper()
def altcase(w):   return "".join(c.upper() if i % 2 else c.lower() for i, c in enumerate(w))

ARMS = [("native", native), ("title", title), ("upper", upper), ("altcase", altcase)]


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(GEMMA, trust_remote_code=True)

    print(f"Tokenizer: {GEMMA}  (class: {type(tok).__name__})")
    print(f"is_fast={tok.is_fast}")
    print()

    # Include the surrounding context ` open {kw} 4` — this is how the keyword
    # appears in the document, and SentencePiece treats leading whitespace as
    # part of the piece so token counts can look different from tokenising the
    # bare word.
    hits = defaultdict(list)
    print(f"{'keyword':>12}  " + "  ".join(f"{a:>18}" for a, _ in ARMS))
    print(f"{'':>12}  " + "  ".join(f"{'ids (n)':>18}" for _ in ARMS))
    for kw in KEYWORDS:
        row = [f"{kw:>12}"]
        native_ids = None
        for arm, fn in ARMS:
            ids = tok(f" open {fn(kw)} 4", add_special_tokens=False)["input_ids"]
            row.append(f"{str(ids):>18}")
            if arm == "native":
                native_ids = ids
            elif ids != native_ids:
                hits[arm].append(kw)
        print("  ".join(row))
        print("  ".join([" " * 12] + [f"{'n=%d' % len(tok(' open '+fn(kw)+' 4', add_special_tokens=False)['input_ids']):>18}"
                                      for _, fn in ARMS]))

    print()
    print("Arms that produced a DIFFERENT token sequence from native, per keyword:")
    for arm, _ in ARMS[1:]:
        n = len(hits[arm])
        marker = "OK" if n >= 4 else "WEAK" if n >= 1 else "FAIL"
        print(f"  {arm:>10}  {n}/{len(KEYWORDS)} keywords differ  [{marker}]")

    if any(len(hits[a]) == 0 for a, _ in ARMS[1:]):
        print()
        print("*** GATE FAIL: at least one arm never changes the token sequence. ***")
        print("*** symbol-encoding on this tokenizer is a null intervention -- do NOT run.  ***")
        sys.exit(2)
    if all(len(hits[a]) >= 4 for a, _ in ARMS[1:]):
        print()
        print("GATE PASS: every arm differs on the majority of keywords. symbol-encoding is meaningful.")
        sys.exit(0)
    print()
    print("GATE WEAK: some arms only differ on a minority of keywords. Inspect the")
    print("per-keyword table above before running the full experiment.")
    sys.exit(1)


if __name__ == "__main__":
    main()
