"""
Step 2b — render each constructed skill document into skill-card PNGs.

One card per skill (so a state's 5 skills become 5 images, matching what the
`vis-skill` main experiment feeds the model and staying inside the servers'
`--limit-mm-per-prompt '{"image": 6}'`), 1000px wide, DejaVu fonts.

This is a v5-*like* card, deliberately RE-WRITTEN rather than reused. The main
experiment's `render_v5.py` classifies each section into one of paragraph /
bullets / numbered and keeps only the winning kind, so a section that has both a
list and prose silently loses the prose — which is exactly where this
experiment's injected sentences live (the same flaw voided part of
an earlier pilot analysis). The layout therefore differs from the main cards in style
(no section boxes; content width 920 instead of TEXT_MAX_W=760; FONT_DESC 28 vs
30, FONT_CODE 27 vs 30, LINE_GAP 10 vs 12) while matching them in geometry
(1000px wide, ~2.3-3.9k px tall) and in being fed as one PNG per skill.

The renderer is deliberately **faithful**: every block of the markdown is drawn
(headings, paragraphs, bullets, numbered lists, code fences), because the text
and visual conditions must carry the *same* information for the comparison to
mean anything. `--verify` re-derives the visible character count from the source
and reports any card whose drawn characters differ, so silent content loss cannot
pass unnoticed.
"""
import os
import re
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
INNER_PAD = 20
LINE_GAP = 10
BLOCK_GAP = 16
SECTION_GAP = 22

BG = (255, 255, 255)
TEXT_COLOR = (15, 18, 22)
SUBTEXT = (55, 60, 68)
BORDER = (200, 206, 214)
LIGHT_BOX = (250, 251, 253)
CODE_BOX = (244, 246, 249)
TITLE_COLOR = (20, 75, 130)
RULE_COLOR = (50, 120, 180)
ACCENT = (25, 100, 160)

FONT_TITLE = _load_font(46, bold=True)
FONT_DESC = _load_font(28)
FONT_H2 = _load_font(34, bold=True)
FONT_H3 = _load_font(31, bold=True)
FONT_BODY = _load_font(30)
FONT_CODE = _load_font(27, mono=True)


# ══════════════════════════════════════════════════════════════════════
# Markdown -> blocks
# ══════════════════════════════════════════════════════════════════════

def _strip_inline(text):
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    text = re.sub(r"\[(.+?)\]\((.+?)\)", r"\1 (\2)", text)
    return text.strip()


def parse_blocks(md):
    """-> ordered list of blocks: {"kind": h2|h3|para|bullets|numbered|code,
    "text"/"items"}. Every non-empty source line ends up in exactly one block."""
    blocks = []
    lines = md.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        s = line.strip()

        if not s:
            i += 1
            continue

        if s.startswith("```"):
            buf, i = [], i + 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            blocks.append({"kind": "code", "text": "\n".join(buf).rstrip()})
            continue

        if s.startswith("#"):
            level = len(s) - len(s.lstrip("#"))
            blocks.append({"kind": "h2" if level <= 2 else "h3",
                           "text": _strip_inline(s.lstrip("#").strip())})
            i += 1
            continue

        if s.startswith("=== Skill:"):
            blocks.append({"kind": "h2", "text": _strip_inline(s.strip("= ").strip())})
            i += 1
            continue

        m_num = re.match(r"^(\d+)[.)]\s+(.*)$", s)
        m_bul = re.match(r"^[-*+]\s+(.*)$", s)
        if m_num or m_bul:
            kind = "numbered" if m_num else "bullets"
            items = []
            while i < len(lines):
                s2 = lines[i].strip()
                if not s2:
                    break
                mn = re.match(r"^(\d+)[.)]\s+(.*)$", s2)
                mb = re.match(r"^[-*+]\s+(.*)$", s2)
                if mn and kind == "numbered":
                    items.append(_strip_inline(mn.group(2)))
                elif mb and kind == "bullets":
                    items.append(_strip_inline(mb.group(1)))
                elif (mn or mb):
                    break
                elif items:
                    items[-1] += " " + _strip_inline(s2)   # continuation line
                else:
                    break
                i += 1
            blocks.append({"kind": kind, "items": items})
            continue

        buf = []
        while i < len(lines) and lines[i].strip() and not re.match(
                r"^(#|```|=== Skill:|\d+[.)]\s|[-*+]\s)", lines[i].strip()):
            buf.append(lines[i].strip())
            i += 1
        blocks.append({"kind": "para", "text": _strip_inline(" ".join(buf))})

    return [b for b in blocks if b.get("text") or b.get("items")]


def visible_chars(blocks):
    """Non-whitespace characters the renderer is going to draw."""
    n = 0
    for b in blocks:
        if b["kind"] in ("h2", "h3", "para", "code"):
            n += len(re.sub(r"\s", "", b["text"]))
        else:
            n += sum(len(re.sub(r"\s", "", it)) for it in b["items"])
    return n


def _alnum(s):
    """Lower-cased alphanumeric content of a string (markdown syntax, list
    markers, punctuation and whitespace are presentation, not content)."""
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def drawn_content_chars(blocks):
    """Alphanumeric characters actually drawn, INCLUDING the numbering of
    ordered-list items (the renderer draws its own '1.', '2.', ... prefixes;
    bullets contribute no alphanumeric prefix). This is the quantity that must
    match the source's alphanumeric content for the card to be faithful."""
    out = []
    for b in blocks:
        if b["kind"] in ("h2", "h3", "para", "code"):
            out.append(b["text"])
        elif b["kind"] == "numbered":
            out.extend(f"{i} {item}" for i, item in enumerate(b["items"], 1))
        else:
            out.extend(b["items"])
    return len(_alnum(" ".join(out)))


# ══════════════════════════════════════════════════════════════════════
# Drawing primitives
# ══════════════════════════════════════════════════════════════════════

def _size(draw, text, font):
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def _line_h(draw, font):
    return _size(draw, "Ag", font)[1]


def _wrap(draw, text, font, max_w):
    words = str(text).split()
    if not words:
        return [""]
    lines, cur = [], []
    for w in words:
        trial = " ".join(cur + [w])
        if _size(draw, trial, font)[0] <= max_w or not cur:
            cur.append(w)
        else:
            lines.append(" ".join(cur))
            cur = [w]
    if cur:
        lines.append(" ".join(cur))
    return lines


def _draw_para(draw, x, y, text, font, fill, max_w, dry):
    lh = _line_h(draw, font)
    for ln in _wrap(draw, text, font, max_w):
        if not dry:
            draw.text((x, y), ln, font=font, fill=fill)
        y += lh + LINE_GAP
    return y


def _draw_list(draw, x, y, items, font, fill, max_w, dry, numbered):
    lh = _line_h(draw, font)
    prefix_w = _size(draw, "99. " if numbered else "• ", font)[0]
    for i, item in enumerate(items, 1):
        prefix = f"{i}. " if numbered else "• "
        wrapped = _wrap(draw, item, font, max_w - prefix_w)
        if not dry:
            draw.text((x, y), prefix, font=font, fill=fill)
            draw.text((x + prefix_w, y), wrapped[0], font=font, fill=fill)
        y += lh + LINE_GAP
        for ln in wrapped[1:]:
            if not dry:
                draw.text((x + prefix_w, y), ln, font=font, fill=fill)
            y += lh + LINE_GAP
        y += 4
    return y


def _draw_code(draw, x, y, text, max_w, dry):
    lh = _line_h(draw, FONT_CODE)
    raw_lines = []
    for ln in text.split("\n"):
        raw_lines.extend(_wrap(draw, ln, FONT_CODE, max_w - 2 * 14) if ln.strip() else [""])
    box_h = len(raw_lines) * (lh + 6) + 2 * 12
    if not dry:
        draw.rounded_rectangle([x, y, x + max_w, y + box_h], radius=8,
                               fill=CODE_BOX, outline=BORDER, width=1)
        cy = y + 12
        for ln in raw_lines:
            draw.text((x + 14, cy), ln, font=FONT_CODE, fill=TEXT_COLOR)
            cy += lh + 6
    return y + box_h


# ══════════════════════════════════════════════════════════════════════
# Card renderer
# ══════════════════════════════════════════════════════════════════════

def _layout(draw, title, description, blocks, dry=True):
    content_w = CANVAS_W - 2 * MARGIN_X
    x = MARGIN_X
    y = TOP_PAD

    y = _draw_para(draw, x, y, title, FONT_TITLE, TITLE_COLOR, content_w, dry)
    if not dry:
        draw.line((x, y + 2, CANVAS_W - MARGIN_X, y + 2), fill=RULE_COLOR, width=2)
    y += 12

    if description:
        y = _draw_para(draw, x, y, description, FONT_DESC, SUBTEXT, content_w, dry)
        y += BLOCK_GAP

    for b in blocks:
        if b["kind"] == "h2":
            y += SECTION_GAP
            y = _draw_para(draw, x, y, b["text"], FONT_H2, ACCENT, content_w, dry)
            y += 2
        elif b["kind"] == "h3":
            y += BLOCK_GAP
            y = _draw_para(draw, x, y, b["text"], FONT_H3, ACCENT, content_w, dry)
        elif b["kind"] == "para":
            y = _draw_para(draw, x, y, b["text"], FONT_BODY, TEXT_COLOR, content_w, dry)
            y += BLOCK_GAP
        elif b["kind"] in ("bullets", "numbered"):
            y = _draw_list(draw, x, y, b["items"], FONT_BODY, TEXT_COLOR, content_w,
                           dry, numbered=(b["kind"] == "numbered"))
            y += BLOCK_GAP
        elif b["kind"] == "code":
            y = _draw_code(draw, x, y, b["text"], content_w, dry)
            y += BLOCK_GAP
    return y + BOTTOM_PAD


def render_card(title, description, markdown, out_path):
    blocks = parse_blocks(markdown)
    probe = Image.new("RGB", (CANVAS_W, 64), BG)
    height = int(_layout(ImageDraw.Draw(probe), title, description, blocks, dry=True))
    img = Image.new("RGB", (CANVAS_W, max(height, 120)), BG)
    _layout(ImageDraw.Draw(img), title, description, blocks, dry=False)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return img.size, visible_chars(blocks)


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main(args):
    variants_dir = Path(args.variants_dir)
    text_dir = variants_dir / "text"
    vis_dir = variants_dir / "visual"
    if not text_dir.exists():
        raise SystemExit(f"no text variants at {text_dir}; run construct_variants.py first")

    report = []
    n_cards = 0
    tall = []
    for state_dir in sorted(p for p in text_dir.iterdir() if p.is_dir()):
        for parts_file in sorted(state_dir.glob("level_*.parts.json")):
            level = parts_file.name.replace(".parts.json", "")
            parts = json.loads(parts_file.read_text(encoding="utf-8"))
            out_dir = vis_dir / state_dir.name / level
            for i, part in enumerate(parts):
                out_path = out_dir / f"card{i}_{part['skill_name']}.png"
                if out_path.exists() and not args.overwrite:
                    n_cards += 1
                    continue
                size, drawn = render_card(
                    part["skill_name"], part.get("description", ""),
                    part["markdown"], out_path)
                n_cards += 1
                if size[1] > args.warn_height:
                    tall.append((str(out_path), size[1]))
                if args.verify:
                    # Faithfulness = equal ALPHANUMERIC content on both sides.
                    # The previous raw non-whitespace comparison counted markdown
                    # syntax ('#', fences, list markers) as source content while
                    # the drawn side excludes the drawn prefixes — a systematic
                    # ~14% bias that flagged every card. Compare content only.
                    vblocks = parse_blocks(part["markdown"])
                    src_n = len(_alnum(_strip_inline(
                        part["skill_name"] + part.get("description", "") + part["markdown"])))
                    drawn_n = (len(_alnum(part["skill_name"] + part.get("description", "")))
                               + drawn_content_chars(vblocks))
                    report.append({
                        "path": str(out_path.relative_to(vis_dir)),
                        "height": size[1],
                        "drawn_chars": drawn_n,
                        "source_chars": src_n,
                    })
        print(f"  rendered {state_dir.name}")

    print(f"\nRendered/kept {n_cards} cards -> {vis_dir}")
    if tall:
        print(f"[WARN] {len(tall)} cards taller than {args.warn_height}px "
              f"(max {max(h for _, h in tall)}px) — check the servers' image token budget")
    if args.verify and report:
        # markdown syntax removal makes an exact match impossible; flag >2% loss
        bad = [r for r in report
               if r["source_chars"] and
               abs(r["drawn_chars"] - r["source_chars"]) / r["source_chars"] > 0.02]
        out = vis_dir / "render_verification.json"
        out.write_text(json.dumps(
            {"cards": len(report), "flagged": len(bad), "report": report},
            indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Verification: {len(bad)}/{len(report)} cards deviate >2% in visible "
              f"characters; details in {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--variants_dir", required=True,
                   help="directory produced by construct_variants.py (contains text/)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--verify", action="store_true", default=True)
    p.add_argument("--no_verify", dest="verify", action="store_false")
    p.add_argument("--warn_height", type=int, default=6000)
    main(p.parse_args())
