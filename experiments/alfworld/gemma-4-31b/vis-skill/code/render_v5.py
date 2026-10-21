"""
Skill card renderer v5.

Fixes two bugs found in v4's renderer:
1. Font loading silently fell back to PIL's tiny fixed-size default font
   because its candidate paths (/usr/share/fonts/truetype/dejavu/...,
   matplotlib's bundled copy) may not exist on the host. Every font role
   (title/section/body/etc.) rendered at the same ~10pt size regardless of
   the size requested in code, so the intended size hierarchy never showed
   up and text looked tiny relative to the fixed-pixel paddings/margins.
   Fixed by bundling real DejaVu ttf files inside this project (code/fonts)
   so rendering no longer depends on system or conda-env font paths that
   may not survive an environment rebuild.
2. _estimate_height() computed box heights with its own separate arithmetic
   that didn't exactly match what _draw_* actually draws (extra hardcoded
   "+12"/"+10" padding, and a stray "lh_ex + 10" line in the example-section
   estimate with no counterpart in the real drawing code), leaving 22-52px
   of dead space at the bottom of every section box. Fixed by making the
   estimate call the exact same _draw_paragraph/_draw_numbered/_draw_bullets
   routines used for real rendering (in a measure_only mode that skips the
   actual draw.text() calls), so estimated and actual heights can't drift.

Canvas width and overall layout are unchanged from v4.
"""
import os
import re
import yaml
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ── Font loading ──

FONTS_DIR = Path(__file__).resolve().parent / "fonts"

def _load_font(size, mono=False, bold=False):
    if mono:
        candidates = [
            FONTS_DIR / "DejaVuSansMono.ttf",
            "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        ]
    elif bold:
        candidates = [
            FONTS_DIR / "DejaVuSans-Bold.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
    else:
        candidates = [
            FONTS_DIR / "DejaVuSans.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    for p in candidates:
        if os.path.exists(p):
            return ImageFont.truetype(str(p), size=size)
    raise FileNotFoundError(
        f"No usable DejaVu font found for size={size}, mono={mono}, bold={bold}. "
        f"Checked: {[str(c) for c in candidates]}"
    )

CANVAS_W = 1000
TEXT_MAX_W = 760

MARGIN_X = 40
TOP_PAD = 32
BOTTOM_PAD = 36
SECTION_GAP = 24
INNER_PAD = 22
LINE_GAP = 12

BG = (255, 255, 255)
TEXT_COLOR = (15, 18, 22)
SUBTEXT = (50, 55, 62)
BORDER = (200, 206, 214)
LIGHT_BOX = (250, 251, 253)
EXAMPLE_BG = (253, 250, 240)
TRACE_BG = (244, 245, 248)
TITLE_COLOR = (20, 75, 130)

# Same size hierarchy as v4's original design intent (title 46 > section 33
# > body/code 30 > example-label 25 > example 23 > trace 22). Now that real
# DejaVu fonts actually load, this hierarchy is finally realized visually.
FONT_TITLE = _load_font(46, bold=True)
FONT_DESC = _load_font(30)
FONT_SECTION = _load_font(33, bold=True)
FONT_BODY = _load_font(30)
FONT_CODE = _load_font(30, mono=True)
FONT_EXAMPLE_LABEL = _load_font(25, bold=True)
FONT_EXAMPLE = _load_font(23)
FONT_TRACE = _load_font(22, mono=True)

# ── Drawing utilities ──

def _text_size(draw, text, font):
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def _wrap(draw, text, font, max_width):
    words = text.split()
    lines, cur = [], []
    for w in words:
        test = " ".join(cur + [w])
        if _text_size(draw, test, font)[0] <= max_width:
            cur.append(w)
        else:
            if cur:
                lines.append(" ".join(cur))
            cur = [w]
    if cur:
        lines.append(" ".join(cur))
    return lines or [""]


def _draw_paragraph(draw, x, y, text, font, fill, max_w, gap=LINE_GAP, measure_only=False):
    lh = _text_size(draw, "Ag", font)[1]
    for line in _wrap(draw, text, font, max_w):
        if not measure_only:
            draw.text((x, y), line, font=font, fill=fill)
        y += lh + gap
    return y


def _draw_numbered(draw, x, y, items, font, fill, max_w, gap=LINE_GAP, measure_only=False):
    num_w = _text_size(draw, "99. ", font)[0]
    lh = _text_size(draw, "Ag", font)[1]
    for i, item in enumerate(items, 1):
        prefix = f"{i}. "
        wrapped = _wrap(draw, item, font, max_w - num_w)
        if not measure_only:
            draw.text((x, y), prefix, font=font, fill=fill)
            draw.text((x + num_w, y), wrapped[0], font=font, fill=fill)
        y += lh + gap
        for line in wrapped[1:]:
            if not measure_only:
                draw.text((x + num_w, y), line, font=font, fill=fill)
            y += lh + gap
        y += 5
    return y


def _draw_bullets(draw, x, y, items, font, fill, max_w, gap=LINE_GAP, measure_only=False):
    bw = _text_size(draw, "• ", font)[0]
    lh = _text_size(draw, "Ag", font)[1]
    for item in items:
        wrapped = _wrap(draw, item, font, max_w - bw)
        if not measure_only:
            draw.text((x, y), "• ", font=font, fill=fill)
            draw.text((x + bw, y), wrapped[0], font=font, fill=fill)
        y += lh + gap
        for line in wrapped[1:]:
            if not measure_only:
                draw.text((x + bw, y), line, font=font, fill=fill)
            y += lh + gap
        y += 5
    return y


# ── SKILL.md Parser (unchanged from v4) ──

def _strip_md(text):
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'`(.+?)`', r'\1', text)
    return text.strip()


def _section_accent(name):
    lo = name.lower()
    if any(k in lo for k in ["instruction", "when to use", "trigger"]):
        return (25, 100, 160)
    if any(k in lo for k in ["workflow", "execution", "procedure", "core"]):
        return (30, 110, 180)
    if "action" in lo and ("format" in lo or "decision" in lo):
        return (30, 150, 110)
    if any(k in lo for k in ["error", "recovery", "handling"]):
        return (200, 55, 45)
    if any(k in lo for k in ["example", "scenario"]):
        return (195, 150, 20)
    if any(k in lo for k in ["post-condition", "postcondition", "post condition", "output"]):
        return (35, 125, 190)
    if any(k in lo for k in ["assumption", "rule", "key"]):
        return (120, 65, 150)
    if any(k in lo for k in ["tip", "note"]):
        return (190, 130, 25)
    return (25, 100, 160)


def _parse_example(lines):
    scenario = ""
    trace = []
    result = ""
    in_code = False
    code_lines = []
    remaining = []

    for line in lines:
        s = line.strip()
        if s.startswith("```"):
            if in_code:
                trace = code_lines[:]
                code_lines = []
            in_code = not in_code
            continue
        if in_code:
            code_lines.append(_strip_md(s) if s else "")
            continue

        if s.startswith("**Scenario:") or s.startswith("**Task:"):
            scenario = _strip_md(s.split(":", 1)[1]) if ":" in s else _strip_md(s)
        elif s.startswith("**Result:"):
            result = _strip_md(s.split(":", 1)[1]) if ":" in s else _strip_md(s)
        elif s:
            remaining.append(_strip_md(s))

    if not scenario and remaining:
        scenario = remaining[0]
        remaining = remaining[1:]
    if not result and remaining:
        result = remaining[-1]

    if not trace:
        trace = ["(No trace available)"]

    return {
        "name": "Example",
        "type": "example",
        "scenario": scenario or "See trace below.",
        "trace": [t for t in trace if t.strip()],
        "result": result or "Task completed.",
        "accent": (195, 150, 20),
    }


def _classify_and_parse(name, lines):
    numbered = []
    bullets = []
    paragraphs = []
    subsection_items = []

    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if not s:
            i += 1
            continue

        if s.startswith("### "):
            header_text = _strip_md(s[4:])
            sub_content = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("## ") and not lines[i].strip().startswith("### "):
                sub_content.append(lines[i])
                i += 1
            flat = " ".join(_strip_md(l) for l in sub_content if l.strip() and not l.strip().startswith("```"))
            subsection_items.append(f"{header_text}: {flat}" if flat else header_text)
            continue

        if s.startswith("```"):
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i].rstrip())
                i += 1
            i += 1
            paragraphs.append(" ".join(_strip_md(l) for l in code_lines if l.strip()))
            continue

        m = re.match(r'^(\d+)\.\s+(.*)', s)
        if m:
            numbered.append(_strip_md(m.group(2)))
            i += 1
            continue

        m2 = re.match(r'^[-*]\s+(.*)', s)
        if m2:
            bullets.append(_strip_md(m2.group(1)))
            i += 1
            continue

        paragraphs.append(_strip_md(s))
        i += 1

    if subsection_items:
        return {
            "name": name,
            "type": "numbered",
            "content": subsection_items,
            "accent": _section_accent(name),
        }

    if numbered and len(numbered) >= len(bullets):
        return {
            "name": name,
            "type": "numbered",
            "content": numbered,
            "accent": _section_accent(name),
        }

    if bullets:
        is_action = "action" in name.lower() and "format" in name.lower()
        sec = {
            "name": name,
            "type": "bullets",
            "content": bullets,
            "accent": _section_accent(name),
        }
        if is_action:
            sec["monospace_all"] = True
        return sec

    full_text = " ".join(paragraphs) if paragraphs else ""
    return {
        "name": name,
        "type": "paragraph",
        "content": full_text,
        "accent": _section_accent(name),
    }


def parse_skill_md(skill_dir):
    skill_dir = Path(skill_dir)
    md_path = skill_dir / "SKILL.md"
    if not md_path.exists():
        return None

    content = md_path.read_text(encoding="utf-8")
    name = skill_dir.name
    description = ""
    body = content

    if content.strip().startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            try:
                h = yaml.safe_load(parts[1])
                if isinstance(h, dict):
                    name = h.get("name", name)
                    description = h.get("description", "")
            except Exception:
                pass
            body = parts[2]

    raw_lines = body.split("\n")
    sections_raw = []
    current_name = None
    current_lines = []

    for line in raw_lines:
        s = line.strip()
        if s.startswith("## "):
            if current_name is not None:
                sections_raw.append((current_name, current_lines))
            current_name = s[3:].strip()
            current_lines = []
        elif s.startswith("# ") and not s.startswith("## "):
            if current_name is not None:
                sections_raw.append((current_name, current_lines))
            heading = s[2:].strip()
            if heading.lower() not in ("instructions",):
                current_name = None
                current_lines = []
            else:
                current_name = heading
                current_lines = []
        else:
            current_lines.append(line)

    if current_name is not None:
        sections_raw.append((current_name, current_lines))

    sections = []
    for sec_name, sec_lines in sections_raw:
        if not sec_lines or all(not l.strip() for l in sec_lines):
            continue
        lo = sec_name.lower()
        if "example" in lo or "scenario" in lo:
            sections.append(_parse_example(sec_lines))
            sections[-1]["name"] = sec_name
        else:
            sections.append(_classify_and_parse(sec_name, sec_lines))

    title = name.replace("alfworld-", "").replace("-", " ").title()
    return {"title": title, "description": description, "sections": sections}


# ── Section Renderer ──
#
# _estimate_height() reuses the exact same _draw_* calls (in measure_only
# mode) that _draw_section() uses to render real content, so the box height
# allocated for the background rectangle can never drift from what's
# actually drawn inside it — this is what eliminates v4's dead-space bug.

def _estimate_height(draw, section, content_w):
    text_w = min(content_w - 36, TEXT_MAX_W)
    cy = INNER_PAD
    if section["type"] == "paragraph":
        cy = _draw_paragraph(draw, 0, cy, section["content"], FONT_BODY, TEXT_COLOR, text_w, measure_only=True)
    elif section["type"] == "numbered":
        cy = _draw_numbered(draw, 0, cy, section["content"], FONT_BODY, TEXT_COLOR, text_w, measure_only=True)
    elif section["type"] == "bullets":
        font = FONT_CODE if section.get("monospace_all") else FONT_BODY
        cy = _draw_bullets(draw, 0, cy, section["content"], font, TEXT_COLOR, text_w, measure_only=True)
    elif section["type"] == "example":
        cy = _draw_paragraph(draw, 0, cy, section["scenario"], FONT_EXAMPLE_LABEL, TEXT_COLOR, text_w, gap=7, measure_only=True)
        cy += 8
        trace_lines = []
        for t in section["trace"]:
            trace_lines.extend(_wrap(draw, t, FONT_TRACE, text_w - 20))
        tlh = _text_size(draw, "Ag", FONT_TRACE)[1]
        trace_h = INNER_PAD * 2 + len(trace_lines) * (tlh + 4)
        cy += trace_h + 10
        cy = _draw_paragraph(draw, 0, cy, section["result"], FONT_EXAMPLE_LABEL, TEXT_COLOR, text_w, gap=7, measure_only=True)
    return cy + INNER_PAD


def _draw_section(draw, x, y, w, section):
    accent = section["accent"]
    box_h = _estimate_height(draw, section, w)
    text_w = min(w - 36, TEXT_MAX_W)

    draw.text((x, y), section["name"], font=FONT_SECTION, fill=accent)
    title_h = _text_size(draw, section["name"], FONT_SECTION)[1]
    y += title_h + 8

    bx, by = x, y
    bg = EXAMPLE_BG if section["type"] == "example" else LIGHT_BOX
    draw.rounded_rectangle([bx, by, bx + w, by + box_h], radius=10, fill=bg, outline=BORDER, width=1)
    draw.rectangle([bx, by, bx + 7, by + box_h], fill=accent)

    cx = bx + 18
    cy = by + INNER_PAD

    if section["type"] == "paragraph":
        cy = _draw_paragraph(draw, cx, cy, section["content"], FONT_BODY, TEXT_COLOR, text_w)
    elif section["type"] == "numbered":
        cy = _draw_numbered(draw, cx, cy, section["content"], FONT_BODY, TEXT_COLOR, text_w)
    elif section["type"] == "bullets":
        font = FONT_CODE if section.get("monospace_all") else FONT_BODY
        cy = _draw_bullets(draw, cx, cy, section["content"], font, TEXT_COLOR, text_w)
    elif section["type"] == "example":
        cy = _draw_paragraph(draw, cx, cy, section["scenario"], FONT_EXAMPLE_LABEL, TEXT_COLOR, text_w, gap=7)
        cy += 8

        trace_lines = []
        for t in section["trace"]:
            trace_lines.extend(_wrap(draw, t, FONT_TRACE, text_w - 20))
        tlh = _text_size(draw, "Ag", FONT_TRACE)[1]
        trace_h = INNER_PAD * 2 + len(trace_lines) * (tlh + 4)
        draw.rounded_rectangle([cx, cy, cx + text_w, cy + trace_h], radius=8,
                               fill=TRACE_BG, outline=(190, 196, 206), width=1)
        ty = cy + INNER_PAD
        for t in section["trace"]:
            for line in _wrap(draw, t, FONT_TRACE, text_w - 20):
                draw.text((cx + 10, ty), line, font=FONT_TRACE, fill=(40, 45, 52))
                ty += tlh + 4
        cy += trace_h + 10

        cy = _draw_paragraph(draw, cx, cy, section["result"], FONT_EXAMPLE_LABEL, TEXT_COLOR, text_w, gap=7)

    return by + box_h + SECTION_GAP + title_h + 8


# ── Main Renderer ──

def render_skill(skill_dict, out_path):
    content_w = CANVAS_W - 2 * MARGIN_X

    tmp = Image.new("RGB", (CANVAS_W, 100), BG)
    td = ImageDraw.Draw(tmp)

    y_est = TOP_PAD
    y_est += _text_size(td, skill_dict["title"], FONT_TITLE)[1] + 24
    desc_lines = _wrap(td, skill_dict["description"], FONT_DESC, min(content_w, TEXT_MAX_W))
    y_est += len(desc_lines) * (_text_size(td, "Ag", FONT_DESC)[1] + 8) + 20

    for sec in skill_dict["sections"]:
        box_h = _estimate_height(td, sec, content_w)
        title_h = _text_size(td, sec["name"], FONT_SECTION)[1]
        # matches _draw_section's actual consumption exactly: title drawn
        # once (title_h+8), box drawn (box_h), gap, then the same
        # (title_h+8) offset again in its return value — mirroring both
        # occurrences here is what keeps canvas_h from ever being
        # under-allocated (v4's version only counted this once, so its
        # final crop() extended past the real image bounds and PIL padded
        # that overhang with solid black rows).
        y_est += 2 * (title_h + 8) + box_h + SECTION_GAP

    canvas_h = y_est + BOTTOM_PAD

    img = Image.new("RGB", (CANVAS_W, canvas_h), BG)
    draw = ImageDraw.Draw(img)
    y = TOP_PAD

    draw.text((MARGIN_X, y), skill_dict["title"], font=FONT_TITLE, fill=TITLE_COLOR)
    y += _text_size(draw, skill_dict["title"], FONT_TITLE)[1] + 12
    draw.line((MARGIN_X, y, CANVAS_W - MARGIN_X, y), fill=(50, 120, 180), width=2)
    y += 12

    y = _draw_paragraph(draw, MARGIN_X, y, skill_dict["description"], FONT_DESC, SUBTEXT, min(content_w, TEXT_MAX_W), gap=8)
    y += 20

    for sec in skill_dict["sections"]:
        y = _draw_section(draw, MARGIN_X, y, content_w, sec)

    img = img.crop((0, 0, CANVAS_W, y + BOTTOM_PAD))
    img.save(out_path)
    return img


# ── Entry Point ──

def main():
    project = Path(__file__).resolve().parent.parent
    skills_dir = project / "skills_alfworld"
    out_dir = project / "images" / "skill_cards"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'Skill':<42} {'Size':>10} {'VisTok':>7}")
    print("-" * 65)

    total_vis = 0
    cnt = 0
    for sd in sorted(skills_dir.iterdir()):
        if not sd.is_dir():
            continue
        skill = parse_skill_md(sd)
        if skill is None:
            continue
        out = out_dir / f"{sd.name}.png"
        render_skill(skill, out)
        w, h = Image.open(out).size
        vt = (w // 28) * (h // 28)
        total_vis += vt
        cnt += 1
        print(f"  {sd.name:<40} {w}x{h:>4} {vt:>7}")

    print("-" * 65)
    print(f"  Rendered {cnt} skills. Avg vis tokens: {total_vis // cnt if cnt else 0}")


if __name__ == "__main__":
    main()
