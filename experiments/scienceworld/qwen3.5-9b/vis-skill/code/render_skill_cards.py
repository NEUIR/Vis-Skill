"""
ScienceWorld Visual-Skill renderer v6: structure-preserving, reasoning-aligned.

Goals
-----
1) Keep SKILL.md semantic content and reading order unchanged.
2) Preserve Markdown structure instead of classifying/re-writing sections.
3) Make ordered procedures visually explicit without adding new instructions.
4) Use a compact, natural-document layout to reduce blank visual tokens.
5) Align the final image dimensions to Qwen3.5's effective 32 px spatial grid
   (vision patch_size=16, spatial_merge_size=2).

Input : ../skills_scienceworld/<skill>/SKILL.md
Output: ../images/skill_cards_v6/<frontmatter-name>.png
"""

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import yaml
from PIL import Image, ImageDraw, ImageFont


# -----------------------------------------------------------------------------
# Fonts
# -----------------------------------------------------------------------------

FONTS_DIR = Path(__file__).resolve().parent / "fonts"


def _load_font(size: int, mono: bool = False, bold: bool = False):
    if mono:
        candidates = [
            FONTS_DIR / "DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
        ]
    elif bold:
        candidates = [
            FONTS_DIR / "NotoSans-Bold.ttf",
            FONTS_DIR / "DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
            "/usr/share/fonts/opentype/noto/NotoSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        ]
    else:
        candidates = [
            FONTS_DIR / "NotoSans-Regular.ttf",
            FONTS_DIR / "DejaVuSans.ttf",
            "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
            "/usr/share/fonts/opentype/noto/NotoSans-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        ]
    for p in candidates:
        if os.path.exists(p):
            return ImageFont.truetype(str(p), size=size)
    raise FileNotFoundError(f"No usable font found. Checked: {[str(x) for x in candidates]}")


# -----------------------------------------------------------------------------
# Geometry: use multiples of 32 for Qwen3.5 visual grid alignment
# -----------------------------------------------------------------------------

GRID = 32
CANVAS_W = 1024
MARGIN_X = 48
TOP_PAD = 40
BOTTOM_PAD = 48
CONTENT_W = CANVAS_W - 2 * MARGIN_X  # 928 px

# Compact enough to keep related steps near each other, but comfortably readable.
FONT_TITLE = _load_font(42, bold=True)
FONT_DESC = _load_font(28)
FONT_H1 = _load_font(36, bold=True)
FONT_H2 = _load_font(32, bold=True)
FONT_H3 = _load_font(29, bold=True)
FONT_BODY = _load_font(28)
FONT_BODY_BOLD = _load_font(28, bold=True)
FONT_CODE = _load_font(26, mono=True)
FONT_META = _load_font(22)

TEXT = (20, 23, 28)
SUBTEXT = (65, 70, 78)
HEADING = (22, 62, 102)
RULE = (212, 217, 224)
CODE_BG = (246, 247, 249)
CODE_BORDER = (216, 220, 226)
INLINE_CODE_BG = (241, 243, 246)
NUMBER_FILL = (238, 244, 250)
NUMBER_BORDER = (128, 156, 184)
BULLET = (55, 77, 102)
BG = (255, 255, 255)

PARA_GAP = 12
BLOCK_GAP = 18
HEADING_GAP = 12
LINE_GAP = 8
LIST_ITEM_GAP = 12
INDENT = 34


# -----------------------------------------------------------------------------
# Data model
# -----------------------------------------------------------------------------

@dataclass
class Block:
    kind: str
    text: str = ""
    level: int = 0
    items: Optional[list] = None
    lines: Optional[list] = None


# -----------------------------------------------------------------------------
# Markdown parsing: preserve order and content; do not semantically rewrite.
# -----------------------------------------------------------------------------

FENCE_RE = re.compile(r"^\s*```")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
ORDERED_RE = re.compile(r"^(\s*)(\d+)([.)])\s+(.*)$")
BULLET_RE = re.compile(r"^(\s*)[-*+]\s+(.*)$")
HR_RE = re.compile(r"^\s*(?:---+|___+|\*\*\*+)\s*$")
LABEL_RE = re.compile(r"^\s*(?:\*\*|__)[^\n:]{1,80}:.*(?:\*\*|__)?\s*$")


def _strip_inline_markdown(text: str) -> str:
    # Only removes visual markup characters; leaves all semantic words/symbols.
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    return text


def _parse_frontmatter(content: str):
    name = ""
    description = ""
    metadata = {}
    body = content

    if content.startswith("---"):
        # Require the closing fence on its own line.
        m = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", content, flags=re.S)
        if m:
            raw_yaml, body = m.group(1), m.group(2)
            try:
                data = yaml.safe_load(raw_yaml)
                if isinstance(data, dict):
                    metadata = data
                    name = str(data.get("name", ""))
                    description = str(data.get("description", ""))
            except Exception:
                # If YAML cannot be parsed, keep it in the body so no content is lost.
                body = content
                metadata = {}
                name = ""
                description = ""

    return name, description, metadata, body


def parse_markdown_blocks(body: str) -> List[Block]:
    lines = body.splitlines()
    blocks: List[Block] = []
    i = 0

    while i < len(lines):
        raw = lines[i]
        s = raw.strip()

        if not s:
            blocks.append(Block(kind="blank"))
            i += 1
            continue

        # Fenced code: preserve exact line text and indentation.
        if FENCE_RE.match(raw):
            fence = raw.strip()
            code_lines = []
            i += 1
            while i < len(lines) and not FENCE_RE.match(lines[i]):
                code_lines.append(lines[i].rstrip("\n"))
                i += 1
            if i < len(lines):
                i += 1
            blocks.append(Block(kind="code", lines=code_lines, text=fence[3:].strip()))
            continue

        m = HEADING_RE.match(raw)
        if m:
            blocks.append(Block(kind="heading", level=len(m.group(1)), text=m.group(2).strip()))
            i += 1
            continue

        if HR_RE.match(raw):
            blocks.append(Block(kind="hr"))
            i += 1
            continue

        # Markdown table: retain as preformatted lines to preserve columns/symbols.
        if s.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].rstrip())
                i += 1
            blocks.append(Block(kind="table", lines=table_lines))
            continue

        mo = ORDERED_RE.match(raw)
        if mo:
            items = []
            while i < len(lines):
                mm = ORDERED_RE.match(lines[i])
                if not mm:
                    break
                indent = len(mm.group(1).replace("\t", "    "))
                items.append({
                    "indent": indent,
                    "num": mm.group(2),
                    "punct": mm.group(3),
                    "text": mm.group(4).strip(),
                })
                i += 1
            blocks.append(Block(kind="ordered", items=items))
            continue

        mb = BULLET_RE.match(raw)
        if mb:
            items = []
            while i < len(lines):
                mm = BULLET_RE.match(lines[i])
                if not mm:
                    break
                indent = len(mm.group(1).replace("\t", "    "))
                items.append({
                    "indent": indent,
                    "text": mm.group(2).strip(),
                })
                i += 1
            blocks.append(Block(kind="bullets", items=items))
            continue

        # Blockquote: preserve text, visually indent it.
        if s.startswith(">"):
            q_lines = []
            while i < len(lines) and lines[i].lstrip().startswith(">"):
                q_lines.append(lines[i].lstrip()[1:].lstrip())
                i += 1
            blocks.append(Block(kind="quote", text=" ".join(q_lines)))
            continue

        # Labeled lines such as **Scenario:** / **Observation:** / **Action:** /
        # **Result:** are kept on their own row. This preserves the procedural trace
        # geometry without changing any wording.
        if LABEL_RE.match(raw):
            blocks.append(Block(kind="paragraph", text=raw.strip()))
            i += 1
            continue

        # Normal Markdown paragraph. A single newline inside a paragraph is a space
        # semantically, so group consecutive plain lines until a structural boundary.
        para = [raw.strip()]
        i += 1
        while i < len(lines):
            nxt = lines[i]
            ns = nxt.strip()
            if not ns:
                break
            if (
                FENCE_RE.match(nxt)
                or HEADING_RE.match(nxt)
                or HR_RE.match(nxt)
                or ORDERED_RE.match(nxt)
                or BULLET_RE.match(nxt)
                or ns.startswith("|")
                or ns.startswith(">")
                or LABEL_RE.match(nxt)
            ):
                break
            para.append(ns)
            i += 1
        blocks.append(Block(kind="paragraph", text=" ".join(para)))

    # Collapse repeated blank blocks; keeps separation but avoids token-wasting whitespace.
    compact = []
    for b in blocks:
        if b.kind == "blank" and compact and compact[-1].kind == "blank":
            continue
        compact.append(b)
    return compact


def parse_skill_md(skill_dir: Path):
    md_path = Path(skill_dir) / "SKILL.md"
    if not md_path.exists():
        return None

    content = md_path.read_text(encoding="utf-8")
    name, description, metadata, body = _parse_frontmatter(content)
    if not name:
        name = Path(skill_dir).name

    return {
        "name": name,
        "description": description,
        "metadata": metadata,
        "body": body,
        "blocks": parse_markdown_blocks(body),
        "source": content,
    }


# -----------------------------------------------------------------------------
# Inline layout: supports **bold** and `inline code` without changing wording.
# -----------------------------------------------------------------------------

INLINE_RE = re.compile(r"(\*\*.*?\*\*|__.*?__|`[^`]*`)")


def _text_size(draw, text, font):
    if text == "":
        return 0, draw.textbbox((0, 0), "Ag", font=font)[3]
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def _inline_runs(text: str, base_font=FONT_BODY):
    parts = INLINE_RE.split(text)
    runs = []
    for part in parts:
        if not part:
            continue
        if (part.startswith("**") and part.endswith("**")) or (part.startswith("__") and part.endswith("__")):
            runs.append((part[2:-2], FONT_BODY_BOLD, "bold"))
        elif part.startswith("`") and part.endswith("`"):
            runs.append((part[1:-1], FONT_CODE, "code"))
        else:
            runs.append((part, base_font, "normal"))
    return runs


def _tokenize_runs(runs):
    tokens = []
    for text, font, style in runs:
        # Keep whitespace tokens so inline formatting doesn't concatenate words.
        for tok in re.findall(r"\s+|\S+", text):
            tokens.append((tok, font, style))
    return tokens


def _split_long_token(draw, token, font, max_w):
    if _text_size(draw, token, font)[0] <= max_w:
        return [token]
    out, cur = [], ""
    for ch in token:
        test = cur + ch
        if cur and _text_size(draw, test, font)[0] > max_w:
            out.append(cur)
            cur = ch
        else:
            cur = test
    if cur:
        out.append(cur)
    return out


def layout_inline(draw, text: str, max_w: int, base_font=FONT_BODY):
    tokens = _tokenize_runs(_inline_runs(text, base_font=base_font))
    lines = []
    cur = []
    cur_w = 0

    for tok, font, style in tokens:
        # Drop leading spaces on a wrapped line.
        if tok.isspace() and not cur:
            continue

        pieces = _split_long_token(draw, tok, font, max_w)
        for piece in pieces:
            pw = _text_size(draw, piece, font)[0]
            if cur and cur_w + pw > max_w:
                # Trim trailing whitespace at line end.
                while cur and cur[-1][0].isspace():
                    cur.pop()
                lines.append(cur)
                cur = []
                cur_w = 0
                if piece.isspace():
                    continue
            cur.append((piece, font, style))
            cur_w += pw

    while cur and cur[-1][0].isspace():
        cur.pop()
    if cur or not lines:
        lines.append(cur)
    return lines


def draw_inline(draw, x, y, text, max_w, base_font=FONT_BODY, fill=TEXT, measure_only=False, line_gap=LINE_GAP):
    lines = layout_inline(draw, text, max_w, base_font=base_font)
    for line in lines:
        fonts = [font for _, font, _ in line] or [base_font]
        lh = max(_text_size(draw, "Ag", f)[1] for f in fonts)
        xx = x
        for token, font, style in line:
            tw, th = _text_size(draw, token, font)
            if not measure_only:
                if style == "code" and token.strip():
                    draw.rounded_rectangle(
                        [xx - 2, y - 1, xx + tw + 2, y + th + 3],
                        radius=3,
                        fill=INLINE_CODE_BG,
                    )
                draw.text((xx, y), token, font=font, fill=fill)
            xx += tw
        y += lh + line_gap
    return y


# -----------------------------------------------------------------------------
# Preformatted text layout (code / tables)
# -----------------------------------------------------------------------------


def wrap_preformatted(draw, line: str, font, max_w: int):
    # Preserve all characters. Only wrap when a line physically exceeds the box.
    if _text_size(draw, line, font)[0] <= max_w:
        return [line]
    out = []
    cur = ""
    for ch in line:
        test = cur + ch
        if cur and _text_size(draw, test, font)[0] > max_w:
            out.append(cur)
            cur = ch
        else:
            cur = test
    out.append(cur)
    return out


def measure_pre_block(draw, lines, max_w, font=FONT_CODE):
    lh = _text_size(draw, "Ag", font)[1] + 6
    n = 0
    for line in lines:
        n += max(1, len(wrap_preformatted(draw, line, font, max_w)))
    return n * lh


# -----------------------------------------------------------------------------
# Block renderer
# -----------------------------------------------------------------------------


def _heading_font(level: int):
    if level <= 1:
        return FONT_H1
    if level == 2:
        return FONT_H2
    return FONT_H3


def render_block(draw, block: Block, x: int, y: int, w: int, measure_only=False):
    if block.kind == "blank":
        return y + 6

    if block.kind == "hr":
        if not measure_only:
            draw.line((x, y + 5, x + w, y + 5), fill=RULE, width=2)
        return y + 16

    if block.kind == "heading":
        font = _heading_font(block.level)
        # Headings are visually styled; wording stays exactly the same.
        plain = _strip_inline_markdown(block.text)
        y += 4
        y = draw_inline(draw, x, y, plain, w, base_font=font, fill=HEADING, measure_only=measure_only, line_gap=6)
        if block.level == 2:
            if not measure_only:
                draw.line((x, y + 1, x + w, y + 1), fill=RULE, width=1)
            y += 8
        return y + HEADING_GAP

    if block.kind == "paragraph":
        y = draw_inline(draw, x, y, block.text, w, base_font=FONT_BODY, fill=TEXT, measure_only=measure_only)
        return y + PARA_GAP

    if block.kind == "quote":
        qx = x + 24
        qw = w - 24
        y0 = y
        y = draw_inline(draw, qx + 18, y + 8, block.text, qw - 24, base_font=FONT_BODY, fill=SUBTEXT, measure_only=measure_only)
        if not measure_only:
            draw.rectangle([qx, y0 + 4, qx + 5, y - 4], fill=RULE)
        return y + BLOCK_GAP

    if block.kind in ("code", "table"):
        lines = block.lines or []
        pad_x, pad_y = 18, 16
        inner_w = w - 2 * pad_x
        h = measure_pre_block(draw, lines or [""], inner_w, FONT_CODE) + 2 * pad_y
        if not measure_only:
            draw.rounded_rectangle([x, y, x + w, y + h], radius=8, fill=CODE_BG, outline=CODE_BORDER, width=1)
            ty = y + pad_y
            lh = _text_size(draw, "Ag", FONT_CODE)[1] + 6
            for line in lines or [""]:
                wrapped = wrap_preformatted(draw, line, FONT_CODE, inner_w)
                for part in wrapped:
                    draw.text((x + pad_x, ty), part, font=FONT_CODE, fill=(35, 39, 45))
                    ty += lh
        return y + h + BLOCK_GAP

    if block.kind == "ordered":
        items = block.items or []
        base_num_w = 48
        for item in items:
            level = min(3, item["indent"] // 2)
            ix = x + level * INDENT
            iw = w - level * INDENT
            num = item["num"]
            # Circle contains the original list number; no new wording is added.
            circle = 36
            text_x = ix + base_num_w
            text_w = iw - base_num_w
            item_y = y
            content_end = draw_inline(
                draw, text_x, y + 3, item["text"], text_w,
                base_font=FONT_BODY, fill=TEXT, measure_only=measure_only,
            )
            item_h = max(circle + 4, content_end - item_y)
            if not measure_only:
                cy = item_y + 20
                draw.ellipse([ix, cy - 18, ix + 36, cy + 18], fill=NUMBER_FILL, outline=NUMBER_BORDER, width=2)
                nf = FONT_META
                nw, nh = _text_size(draw, num, nf)
                draw.text((ix + 18 - nw / 2, cy - nh / 2 - 1), num, font=nf, fill=HEADING)
            y = item_y + item_h + LIST_ITEM_GAP
        return y + 4

    if block.kind == "bullets":
        items = block.items or []
        for item in items:
            level = min(3, item["indent"] // 2)
            ix = x + level * INDENT
            iw = w - level * INDENT
            bullet_x = ix + 8
            text_x = ix + 32
            text_w = iw - 32
            item_y = y
            content_end = draw_inline(
                draw, text_x, y, item["text"], text_w,
                base_font=FONT_BODY, fill=TEXT, measure_only=measure_only,
            )
            if not measure_only:
                draw.ellipse([bullet_x, item_y + 11, bullet_x + 8, item_y + 19], fill=BULLET)
            y = content_end + 5
        return y + 6

    return y


# -----------------------------------------------------------------------------
# Equivalence audit
# -----------------------------------------------------------------------------


def _normalized_source_fragments(body: str):
    """Extract semantic text fragments from source in reading order.

    Markdown control syntax is ignored, but every textual fragment must survive.
    This is intentionally conservative: if the parser silently drops prose, headings,
    list items, code, or table lines, the audit fails.
    """
    frags = []
    in_code = False
    for raw in body.splitlines():
        s = raw.strip()
        if FENCE_RE.match(raw):
            in_code = not in_code
            continue
        if not s:
            continue
        if in_code:
            frags.append(re.sub(r"\s+", " ", raw.strip()))
            continue
        m = HEADING_RE.match(raw)
        if m:
            frags.append(re.sub(r"\s+", " ", _strip_inline_markdown(m.group(2).strip())))
            continue
        mo = ORDERED_RE.match(raw)
        if mo:
            frags.append(re.sub(r"\s+", " ", _strip_inline_markdown(mo.group(4).strip())))
            continue
        mb = BULLET_RE.match(raw)
        if mb:
            frags.append(re.sub(r"\s+", " ", _strip_inline_markdown(mb.group(2).strip())))
            continue
        if HR_RE.match(raw):
            continue
        if s.startswith("|"):
            # Table lines are rendered as preformatted text (raw), so the
            # audit keeps them raw as well.
            frags.append(re.sub(r"\s+", " ", s))
            continue
        if s.startswith(">"):
            s = s[1:].strip()
        frags.append(re.sub(r"\s+", " ", _strip_inline_markdown(s)))
    return [f for f in frags if f]


def _normalized_block_fragments(blocks: List[Block]):
    frags = []
    for b in blocks:
        if b.kind == "heading":
            frags.append(re.sub(r"\s+", " ", _strip_inline_markdown(b.text.strip())))
        elif b.kind in ("paragraph", "quote"):
            # Paragraph parser may join consecutive source lines. Split checking is
            # done through a containment-based sequence below.
            frags.append(re.sub(r"\s+", " ", _strip_inline_markdown(b.text.strip())))
        elif b.kind in ("code", "table"):
            for line in b.lines or []:
                if line.strip():
                    frags.append(re.sub(r"\s+", " ", line.strip()))
        elif b.kind in ("ordered", "bullets"):
            for it in b.items or []:
                frags.append(re.sub(r"\s+", " ", _strip_inline_markdown(it["text"].strip())))
    return [f for f in frags if f]


def verify_no_content_loss(body: str, blocks: List[Block]):
    src = _normalized_source_fragments(body)
    dst = _normalized_block_fragments(blocks)

    # Compare flattened semantic streams. This tolerates Markdown paragraph line
    # joining but catches omissions and reordering.
    src_stream = " ".join(src)
    dst_stream = " ".join(dst)
    if src_stream != dst_stream:
        # Find a useful prefix for debugging.
        n = min(len(src_stream), len(dst_stream))
        pos = next((i for i in range(n) if src_stream[i] != dst_stream[i]), n)
        context_a = src_stream[max(0, pos - 80):pos + 120]
        context_b = dst_stream[max(0, pos - 80):pos + 120]
        raise ValueError(
            "Content-equivalence audit failed. The visual parser changed/dropped text.\n"
            f"SOURCE around mismatch: {context_a!r}\n"
            f"PARSED around mismatch: {context_b!r}"
        )


# -----------------------------------------------------------------------------
# Main render
# -----------------------------------------------------------------------------


def _round_up(v: int, factor: int = GRID):
    return int(math.ceil(v / factor) * factor)


def render_skill(skill: dict, out_path: Path):
    verify_no_content_loss(skill["body"], skill["blocks"])

    tmp = Image.new("RGB", (CANVAS_W, 128), BG)
    md = ImageDraw.Draw(tmp)

    # First pass: exact height measurement.
    y = TOP_PAD
    y = draw_inline(md, MARGIN_X, y, skill["name"], CONTENT_W, base_font=FONT_TITLE, fill=HEADING, measure_only=True, line_gap=6)
    if skill["description"]:
        y += 4
        y = draw_inline(md, MARGIN_X, y, skill["description"], CONTENT_W, base_font=FONT_DESC, fill=SUBTEXT, measure_only=True, line_gap=7)
    y += 16
    y += 2 + 18  # header rule + gap

    for block in skill["blocks"]:
        y = render_block(md, block, MARGIN_X, y, CONTENT_W, measure_only=True)

    natural_h = y + BOTTOM_PAD
    canvas_h = _round_up(natural_h, GRID)

    img = Image.new("RGB", (CANVAS_W, canvas_h), BG)
    draw = ImageDraw.Draw(img)

    # Second pass: draw.
    y = TOP_PAD
    y = draw_inline(draw, MARGIN_X, y, skill["name"], CONTENT_W, base_font=FONT_TITLE, fill=HEADING, line_gap=6)
    if skill["description"]:
        y += 4
        y = draw_inline(draw, MARGIN_X, y, skill["description"], CONTENT_W, base_font=FONT_DESC, fill=SUBTEXT, line_gap=7)
    y += 16
    draw.line((MARGIN_X, y, CANVAS_W - MARGIN_X, y), fill=RULE, width=2)
    y += 20

    for block in skill["blocks"]:
        y = render_block(draw, block, MARGIN_X, y, CONTENT_W, measure_only=False)

    img.save(out_path, optimize=True)
    return img


def approx_qwen35_visual_tokens(w: int, h: int):
    # Qwen3.5 config: patch_size=16, spatial_merge_size=2 => ~32x32 px/token.
    return math.ceil(w / GRID) * math.ceil(h / GRID)


def main():
    project = Path(__file__).resolve().parent.parent
    skills_dir = project / "skills_scienceworld"
    out_dir = project / "images" / "skill_cards_v6"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'Skill':<52} {'Size':>12} {'~VisTok':>9}")
    print("-" * 78)

    total_vis = 0
    count = 0
    for sd in sorted(skills_dir.iterdir()):
        if not sd.is_dir():
            continue
        skill = parse_skill_md(sd)
        if skill is None:
            continue
        out = out_dir / f"{skill['name']}.png"
        img = render_skill(skill, out)
        w, h = img.size
        vt = approx_qwen35_visual_tokens(w, h)
        total_vis += vt
        count += 1
        print(f"{skill['name']:<52} {w:>4}x{h:<6} {vt:>9}")

    print("-" * 78)
    print(f"Rendered {count} skills. Avg ~visual tokens: {total_vis // count if count else 0}")


if __name__ == "__main__":
    main()
