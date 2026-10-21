"""
Step 2c — render each constructed skill document into RASTERIZED (plain-text
screenshot) cards: the third representation of the lexical-interference design.

    text        — the document as characters (lexical-interference, existing)
    structural  — the document parsed into styled skill cards: title, section
                  headings, bullets/numbering, code boxes  (= render_cards.py,
                  the main experiment's Visual Skill, existing)
    rasterized  — the SAME markdown source drawn as plain monospace text lines:
                  one font, one size, one colour, no headings/bullets/boxes
                  ("plain-text screenshot", this script)

The contrast that makes the experiment interpretable:

  text vs rasterized       — the words stop being language tokens and become
                             pixels, while the visible character stream stays
                             byte-identical to the text condition (the raw
                             markdown is drawn verbatim, '#'/'-'/'```' included).
                             Any difference is the MODALITY change alone.
  rasterized vs structural — same modality (images), same source document, but
                             structural adds the typographic organisation
                             (heading sizes/colours, bullet glyphs, code boxes,
                             description styling). Any difference is the
                             STRUCTURE rendering alone.

Fidelity is therefore exact by construction, and `--verify` (default on)
enforces it: the drawn non-whitespace character count must EQUAL the source's
(render_cards.py tolerates 2% because markdown syntax legitimately disappears
into styling; here nothing may disappear — a mismatch is a bug, not a rounding
 artefact, and is reported as a hard failure).

Card geometry mirrors render_cards.py: 1000px wide, one card per skill part
(a state's 5 skills -> 5 images, inside the servers' image limit), DejaVu fonts.
The parts files are read exactly as render_cards.py reads them, so this script
works on any variants dir whose text/{state_id}/ subfolders hold *.parts.json —
level_{k}.parts.json here, {version}_k{k}.parts.json in the
causal-keyword-ablation experiment.

Outputs:
  variants/{out_subdir}/{state_id}/{stem}/card{i}_{skill_name}.png
  variants/{out_subdir}/render_verification.json      (with --verify)
"""
import json
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


# ══════════════════════════════════════════════════════════════════════
# Fonts (local ./fonts first, then the system DejaVu install)
# ══════════════════════════════════════════════════════════════════════

FONTS_DIR = Path(__file__).resolve().parent / "fonts"


def _load_font(size, mono=False, bold=False):
    if mono:
        names = ["DejaVuSansMono.ttf"]
    elif bold:
        names = ["DejaVuSans-Bold.ttf"]
    else:
        names = ["DejaVuSans.ttf"]
    roots = [FONTS_DIR, Path("/usr/share/fonts/truetype/dejavu"),
             Path("/usr/share/fonts/dejavu")]
    for root in roots:
        for n in names:
            p = root / n
            if p.exists():
                return ImageFont.truetype(str(p), size=size)
    raise FileNotFoundError(f"No DejaVu font for size={size} mono={mono} bold={bold}")


CANVAS_W = 1000
MARGIN_X = 40
TOP_PAD = 32
BOTTOM_PAD = 36
LINE_GAP = 8          # uniform for every line: no block/section rhythm
MONO_SIZE = 24        # ~63 chars/line at 920px content width

BG = (255, 255, 255)
TEXT_COLOR = (15, 18, 22)   # the ONLY colour: headings draw like body text

FONT_MONO = _load_font(MONO_SIZE, mono=True)


# ══════════════════════════════════════════════════════════════════════
# Source lines: the raw document, verbatim
# ══════════════════════════════════════════════════════════════════════

def card_source_lines(skill_name, description, markdown):
    """The exact lines a plain-text screenshot of this part shows.

    Mirrors the assembled full document's per-skill section ("=== Skill: X ==="
    / "Description: ..." / raw markdown), so the union of the cards carries the
    same character stream as the text condition's document — the only content
    the text condition has that the cards do not is the collection header
    "# Relevant Skills for This Task", which the structural condition also
    drops (its cards start at the skill title). The wrapper line "[Skill: name]"
    is added as TEXT by build_messages in both image conditions, identically.
    """
    lines = [f"=== Skill: {skill_name} ==="]
    if description:
        lines.append(f"Description: {description}")
    lines.append("")
    lines.extend(markdown.rstrip("\n").split("\n"))
    return lines


def _nonspace(s):
    return sum(1 for ch in s if not ch.isspace())


# ══════════════════════════════════════════════════════════════════════
# Wrapping (word-wrap with indent preservation; hard cut for overlong words)
# ══════════════════════════════════════════════════════════════════════

def wrap_line(draw, line, font, max_w):
    """Word-wrap one source line to max_w pixels, preserving its leading
    indentation on every wrapped row. Whitespace RUNS are collapsed by the
    wrap (a screenshot of text re-flowed at a fixed window width); the
    non-whitespace character stream is untouched, which is what the
    verification counts."""
    if not line.strip():
        return [""]
    indent = line[:len(line) - len(line.lstrip())]
    words = line.split()
    out, cur = [], []

    def width(s):
        return draw.textlength(s, font=font)

    def emit_hard_cut(word):
        """A single word wider than max_w: cut it at character boundaries."""
        rest = word
        while rest:
            lo, hi = 1, len(rest)
            while lo < hi:                       # longest fitting prefix
                mid = (lo + hi + 1) // 2
                if width(indent + rest[:mid]) <= max_w:
                    lo = mid
                else:
                    hi = mid - 1
            cut = max(1, lo)
            piece, rest = rest[:cut], rest[cut:]
            out.append(indent + piece)

    for w in words:
        trial = indent + " ".join(cur + [w]) if cur else indent + w
        if width(trial) <= max_w:
            cur.append(w)
            continue
        if cur:
            out.append(indent + " ".join(cur))
            cur = []
        if width(indent + w) <= max_w:
            cur = [w]
        else:
            emit_hard_cut(w)
    if cur:
        out.append(indent + " ".join(cur))
    return out or [""]


# ══════════════════════════════════════════════════════════════════════
# Card renderer
# ══════════════════════════════════════════════════════════════════════

def _layout(draw, lines, dry=True):
    content_w = CANVAS_W - 2 * MARGIN_X
    lh = FONT_MONO.getmetrics()
    line_h = lh[0] + lh[1]
    y = TOP_PAD
    for ln in lines:
        for row in wrap_line(draw, ln, FONT_MONO, content_w):
            if not dry:
                draw.text((MARGIN_X, y), row, font=FONT_MONO, fill=TEXT_COLOR)
            y += line_h + LINE_GAP
    return y + BOTTOM_PAD - LINE_GAP


def render_rasterized_card(skill_name, description, markdown, out_path):
    """-> (size, drawn_nonspace_chars, source_nonspace_chars)."""
    lines = card_source_lines(skill_name, description, markdown)
    probe = Image.new("RGB", (CANVAS_W, 64), BG)
    height = int(_layout(ImageDraw.Draw(probe), lines, dry=True))
    img = Image.new("RGB", (CANVAS_W, max(height, 120)), BG)
    _layout(ImageDraw.Draw(img), lines, dry=False)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    drawn = sum(_nonspace(r)
                for ln in lines
                for r in wrap_line(ImageDraw.Draw(probe), ln, FONT_MONO,
                                   CANVAS_W - 2 * MARGIN_X))
    source = sum(_nonspace(ln) for ln in lines)
    return img.size, drawn, source


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main(args):
    variants_dir = Path(args.variants_dir)
    text_dir = variants_dir / "text"
    out_root = variants_dir / args.out_subdir
    if not text_dir.exists():
        raise SystemExit(f"no text variants at {text_dir}; run construct_variants.py "
                         f"(or link the lexical-interference variants) first")

    report = []
    mismatches = []
    n_cards = 0
    tall = []
    for state_dir in sorted(p for p in text_dir.iterdir() if p.is_dir()):
        for parts_file in sorted(state_dir.glob(args.parts_glob)):
            stem = parts_file.name[:-len(".parts.json")]
            parts = json.loads(parts_file.read_text(encoding="utf-8"))
            out_dir = out_root / state_dir.name / stem
            for i, part in enumerate(parts):
                out_path = out_dir / f"card{i}_{part['skill_name']}.png"
                if out_path.exists() and not args.overwrite:
                    n_cards += 1
                    continue
                size, drawn, source = render_rasterized_card(
                    part["skill_name"], part.get("description", ""),
                    part["markdown"], out_path)
                n_cards += 1
                if size[1] > args.warn_height:
                    tall.append((str(out_path), size[1]))
                if args.verify:
                    if drawn != source:
                        mismatches.append({
                            "path": str(out_path.relative_to(out_root)),
                            "drawn_chars": drawn, "source_chars": source})
                    report.append({
                        "path": str(out_path.relative_to(out_root)),
                        "height": size[1],
                        "drawn_chars": drawn,
                        "source_chars": source,
                    })
        print(f"  rendered {state_dir.name}")

    print(f"\nRendered/kept {n_cards} rasterized cards -> {out_root}")
    if tall:
        print(f"[WARN] {len(tall)} cards taller than {args.warn_height}px "
              f"(max {max(h for _, h in tall)}px) — check the servers' image token budget")
    if args.verify and report:
        out = out_root / "render_verification.json"
        out.write_text(json.dumps(
            {"cards": len(report), "mismatches": len(mismatches),
             "mismatch_detail": mismatches[:50], "report": report},
            indent=2, ensure_ascii=False), encoding="utf-8")
        if mismatches:
            print(f"[FAIL] {len(mismatches)}/{len(report)} cards do NOT preserve the "
                  f"source characters exactly (details in {out}) — rasterization must "
                  f"be lossless; fix before probing.")
            raise SystemExit(1)
        print(f"Verification: all {len(report)} cards preserve the source character "
              f"stream exactly; details in {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--variants_dir", required=True,
                   help="directory whose text/{state_id}/ holds *.parts.json "
                        "(lexical-interference variants, or the "
                        "causal-keyword-ablation version variants)")
    p.add_argument("--out_subdir", default="rasterized",
                   help="sub-directory of variants_dir to write the cards into")
    p.add_argument("--parts_glob", default="*.parts.json",
                   help="which parts files to render (level_*.parts.json, "
                        "*_k*.parts.json, ...)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--verify", action="store_true", default=True)
    p.add_argument("--no_verify", dest="verify", action="store_false")
    p.add_argument("--warn_height", type=int, default=6000)
    main(p.parse_args())
