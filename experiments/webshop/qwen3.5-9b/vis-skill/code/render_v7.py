"""
Skill card renderer v7 — WebShop
Critical-Salience / Single-Page / Single-Column / Content-Preserving

Goals
-----
1. ONE skill -> ONE PNG.
2. Single-page only.
3. Single-column only.
4. Preserve all semantic text from SKILL.md.
5. Preserve heading hierarchy, lists, code blocks and source code line breaks.
6. Preserve inline Markdown cues:
   - **bold** -> real bold text
   - `inline code` -> monospace visual token
7. Allocate visual capacity by semantic role:
   - Procedure / Workflow: strong
   - Constraint / Verification / Completion: strongest
   - Error / Recovery: strong
   - Example: visually down-weighted, but NEVER removed
8. Never summarize, rewrite, invent, or delete skill content.
9. No global "dense mode" that shrinks an entire long skill.
"""

import os
import re
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# Paths
# ============================================================

FONTS_DIR = Path(__file__).resolve().parent.parent / "fonts"


# ============================================================
# Font loading
# ============================================================

def _load_font(size, mono=False, bold=False):
    if mono and bold:
        candidates = [
            FONTS_DIR / "DejaVuSansMono-Bold.ttf",
            "/usr/share/fonts/dejavu/DejaVuSansMono-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        ]
    elif mono:
        candidates = [
            FONTS_DIR / "DejaVuSansMono.ttf",
            "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        ]
    elif bold:
        candidates = [
            FONTS_DIR / "DejaVuSans-Bold.ttf",
            "/usr/share/fonts/dejavu/DejaSans-Bold.ttf",
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
        f"No usable DejaVu font found: "
        f"size={size}, mono={mono}, bold={bold}. "
        f"Checked: {[str(x) for x in candidates]}"
    )


# ============================================================
# Canvas
# ============================================================

# 1024 gives a clean width and lets us use more horizontal capacity.
CANVAS_W = 1024

MARGIN_X = 40
TOP_PAD = 26
BOTTOM_PAD = 30

CONTENT_W = CANVAS_W - 2 * MARGIN_X  # 944px

# Initial temporary canvas height.
# The renderer automatically retries with a larger canvas if necessary.
INITIAL_CANVAS_H = 7000


# ============================================================
# Typography
# ============================================================

FONT_TITLE = _load_font(42, bold=True)
FONT_DESC = _load_font(25)

FONT_H1 = _load_font(33, bold=True)

FONT_H2_GENERAL = _load_font(30, bold=True)
FONT_H2_PROCEDURE = _load_font(31, bold=True)
FONT_H2_CRITICAL = _load_font(32, bold=True)
FONT_H2_ERROR = _load_font(31, bold=True)
FONT_H2_EXAMPLE = _load_font(28, bold=True)

FONT_H3_GENERAL = _load_font(28, bold=True)
FONT_H3_PROCEDURE = _load_font(29, bold=True)
FONT_H3_CRITICAL = _load_font(29, bold=True)
FONT_H3_ERROR = _load_font(29, bold=True)
FONT_H3_EXAMPLE = _load_font(25, bold=True)

# Semantic body sizes.
FONT_BODY_GENERAL = _load_font(29)
FONT_BODY_GENERAL_BOLD = _load_font(29, bold=True)

FONT_BODY_PROCEDURE = _load_font(30)
FONT_BODY_PROCEDURE_BOLD = _load_font(30, bold=True)

FONT_BODY_CRITICAL = _load_font(31)
FONT_BODY_CRITICAL_BOLD = _load_font(31, bold=True)

FONT_BODY_ERROR = _load_font(30)
FONT_BODY_ERROR_BOLD = _load_font(30, bold=True)

# Example is intentionally smaller.
FONT_BODY_EXAMPLE = _load_font(25)
FONT_BODY_EXAMPLE_BOLD = _load_font(25, bold=True)

# Block code.
FONT_CODE = _load_font(30, mono=True)
FONT_CODE_BOLD = _load_font(30, mono=True, bold=True)

# Example code is smaller but still readable.
FONT_CODE_EXAMPLE = _load_font(25, mono=True)
FONT_CODE_EXAMPLE_BOLD = _load_font(25, mono=True, bold=True)

# Inline code.
FONT_INLINE_CODE = _load_font(28, mono=True)
FONT_INLINE_CODE_EXAMPLE = _load_font(24, mono=True)

FONT_SMALL = _load_font(21)
FONT_SMALL_BOLD = _load_font(21, bold=True)


# ============================================================
# Spacing
# ============================================================

TITLE_BOTTOM_GAP = 12
DESC_BOTTOM_GAP = 17

H1_TOP_GAP = 17
H1_BOTTOM_GAP = 10

H2_TOP_GAP = 15
H2_BOTTOM_GAP = 8

H3_TOP_GAP = 10
H3_BOTTOM_GAP = 6

PARAGRAPH_GAP = 8
LIST_ITEM_GAP = 5
BLANK_GAP = 5

GENERAL_LINE_GAP = 6
PROCEDURE_LINE_GAP = 7
CRITICAL_LINE_GAP = 7
ERROR_LINE_GAP = 7
EXAMPLE_LINE_GAP = 5

CODE_PAD_X = 15
CODE_PAD_Y = 14
CODE_LINE_GAP = 5

EXAMPLE_CODE_PAD_X = 13
EXAMPLE_CODE_PAD_Y = 11
EXAMPLE_CODE_LINE_GAP = 4

INLINE_CODE_PAD_X = 5
INLINE_CODE_PAD_Y = 2

ROLE_RAIL_W = 5
ROLE_RAIL_GAP = 11

STEP_BADGE_MIN_W = 42
STEP_BADGE_H = 38
STEP_BADGE_GAP = 12


# ============================================================
# Colors
# ============================================================

BG = (255, 255, 255)

TEXT = (20, 23, 28)
SUBTEXT = (62, 67, 74)
MUTED = (105, 110, 117)

BORDER = (213, 218, 224)

# General.
GENERAL_ACCENT = (46, 87, 127)
GENERAL_LIGHT = (247, 249, 251)

# Procedure / workflow.
PROCEDURE_ACCENT = (30, 102, 164)
PROCEDURE_LIGHT = (243, 248, 252)

# Verification / requirement / completion.
CRITICAL_ACCENT = (103, 65, 145)
CRITICAL_LIGHT = (248, 245, 251)

# Error / recovery.
ERROR_ACCENT = (180, 55, 48)
ERROR_LIGHT = (253, 246, 245)

# Example: intentionally lower contrast.
EXAMPLE_ACCENT = (125, 105, 70)
EXAMPLE_LIGHT = (250, 249, 246)
EXAMPLE_TEXT = (55, 56, 59)

# Code.
CODE_BG = (244, 246, 248)
CODE_BORDER = (201, 207, 214)
CODE_TEXT = (27, 31, 36)

INLINE_CODE_BG = (239, 242, 245)
INLINE_CODE_BORDER = (207, 212, 218)
INLINE_CODE_TEXT = (31, 36, 42)

# Step badges.
STEP_BG = (231, 240, 248)
STEP_TEXT = (28, 89, 145)

CRITICAL_STEP_BG = (239, 232, 246)
CRITICAL_STEP_TEXT = (94, 56, 135)

ERROR_STEP_BG = (250, 231, 229)
ERROR_STEP_TEXT = (170, 52, 46)


# ============================================================
# Basic measurement helpers
# ============================================================

def _text_size(draw, text, font):
    if text == "":
        text = " "
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def _line_height(draw, font):
    return _text_size(draw, "Ag", font)[1]


def _role_line_gap(role):
    if role == "procedure":
        return PROCEDURE_LINE_GAP
    if role == "critical":
        return CRITICAL_LINE_GAP
    if role == "error":
        return ERROR_LINE_GAP
    if role == "example":
        return EXAMPLE_LINE_GAP
    return GENERAL_LINE_GAP


# ============================================================
# Role classification
# ============================================================

def _section_role(name):
    """
    Determine the visual role of a section from its original heading.

    This changes presentation only.
    It does NOT insert, delete, or rewrite any skill content.
    """
    lo = name.lower()

    # Example first so a heading such as "Example Workflow"
    # remains supporting material.
    if any(k in lo for k in [
        "example",
        "examples",
        "scenario",
        "scenarios",
        "demonstration",
        "sample",
    ]):
        return "example"

    # Exact-success-critical information.
    if any(k in lo for k in [
        "verify",
        "verification",
        "validation",
        "constraint",
        "constraints",
        "requirement",
        "requirements",
        "precondition",
        "pre-condition",
        "postcondition",
        "post-condition",
        "condition",
        "conditions",
        "completion",
        "complete",
        "final check",
        "final verification",
        "success criteria",
        "termination",
        "purchase condition",
        "before purchase",
        "before purchasing",
    ]):
        return "critical"

    # Error / failure handling.
    if any(k in lo for k in [
        "error",
        "errors",
        "recovery",
        "failure",
        "failures",
        "handling",
        "avoid",
        "warning",
        "fallback",
        "retry",
        "exception",
    ]):
        return "error"

    # Main execution logic.
    if any(k in lo for k in [
        "workflow",
        "procedure",
        "execution",
        "instruction",
        "instructions",
        "action",
        "actions",
        "decision",
        "steps",
        "step-by-step",
        "core",
        "process",
    ]):
        return "procedure"

    return "general"


def _role_accent(role):
    if role == "procedure":
        return PROCEDURE_ACCENT, PROCEDURE_LIGHT
    if role == "critical":
        return CRITICAL_ACCENT, CRITICAL_LIGHT
    if role == "error":
        return ERROR_ACCENT, ERROR_LIGHT
    if role == "example":
        return EXAMPLE_ACCENT, EXAMPLE_LIGHT
    return GENERAL_ACCENT, GENERAL_LIGHT


def _body_font(role, bold=False):
    if role == "procedure":
        return FONT_BODY_PROCEDURE_BOLD if bold else FONT_BODY_PROCEDURE

    if role == "critical":
        return FONT_BODY_CRITICAL_BOLD if bold else FONT_BODY_CRITICAL

    if role == "error":
        return FONT_BODY_ERROR_BOLD if bold else FONT_BODY_ERROR

    if role == "example":
        return FONT_BODY_EXAMPLE_BOLD if bold else FONT_BODY_EXAMPLE

    return FONT_BODY_GENERAL_BOLD if bold else FONT_BODY_GENERAL


def _h2_font(role):
    if role == "procedure":
        return FONT_H2_PROCEDURE
    if role == "critical":
        return FONT_H2_CRITICAL
    if role == "error":
        return FONT_H2_ERROR
    if role == "example":
        return FONT_H2_EXAMPLE
    return FONT_H2_GENERAL


def _h3_font(role):
    if role == "procedure":
        return FONT_H3_PROCEDURE
    if role == "critical":
        return FONT_H3_CRITICAL
    if role == "error":
        return FONT_H3_ERROR
    if role == "example":
        return FONT_H3_EXAMPLE
    return FONT_H3_GENERAL


def _block_code_font(role):
    if role == "example":
        return FONT_CODE_EXAMPLE
    return FONT_CODE


def _inline_code_font(role):
    if role == "example":
        return FONT_INLINE_CODE_EXAMPLE
    return FONT_INLINE_CODE


# ============================================================
# Frontmatter
# ============================================================

def _extract_frontmatter(content):
    """
    Extract initial YAML frontmatter.

    The visible name / description are rendered separately at the top.
    Body content itself is otherwise preserved.
    """
    if not content.lstrip().startswith("---"):
        return {}, content

    m = re.match(
        r"^\s*---\s*\n(.*?)\n---\s*\n?",
        content,
        flags=re.S,
    )

    if not m:
        return {}, content

    raw_yaml = m.group(1)
    body = content[m.end():]

    metadata = {}

    try:
        parsed = yaml.safe_load(raw_yaml)
        if isinstance(parsed, dict):
            metadata = parsed
    except Exception:
        metadata = {}

    return metadata, body


# ============================================================
# Markdown block parser
# ============================================================

def _parse_markdown_blocks(body):
    """
    Structural parser.

    Supported:
        h1
        h2
        h3
        paragraph
        numbered
        bullet
        code
        blank

    Fenced code content and code line breaks are preserved exactly.
    """
    lines = body.splitlines()

    blocks = []

    in_code = False
    code_language = ""
    code_lines = []

    paragraph_lines = []

    def flush_paragraph():
        nonlocal paragraph_lines

        if not paragraph_lines:
            return

        # Source Markdown may wrap prose manually across multiple lines.
        # Joining those physical lines does not alter the semantic text.
        text = " ".join(
            line.strip()
            for line in paragraph_lines
        ).strip()

        if text:
            blocks.append({
                "type": "paragraph",
                "text": text,
            })

        paragraph_lines = []

    for raw in lines:
        stripped = raw.strip()

        # ----------------------------------------------------
        # Fenced code
        # ----------------------------------------------------

        if stripped.startswith("```"):
            flush_paragraph()

            if not in_code:
                in_code = True
                code_language = stripped[3:].strip()
                code_lines = []
            else:
                blocks.append({
                    "type": "code",
                    "language": code_language,
                    "lines": code_lines[:],
                })

                in_code = False
                code_language = ""
                code_lines = []

            continue

        if in_code:
            code_lines.append(raw.rstrip("\n"))
            continue

        # ----------------------------------------------------
        # Blank line
        # ----------------------------------------------------

        if stripped == "":
            flush_paragraph()

            if not blocks or blocks[-1]["type"] != "blank":
                blocks.append({"type": "blank"})

            continue

        # ----------------------------------------------------
        # Headings
        # ----------------------------------------------------

        if stripped.startswith("### "):
            flush_paragraph()
            blocks.append({
                "type": "h3",
                "text": stripped[4:].strip(),
            })
            continue

        if stripped.startswith("## "):
            flush_paragraph()
            blocks.append({
                "type": "h2",
                "text": stripped[3:].strip(),
            })
            continue

        if stripped.startswith("# "):
            flush_paragraph()
            blocks.append({
                "type": "h1",
                "text": stripped[2:].strip(),
            })
            continue

        # ----------------------------------------------------
        # Numbered list
        # ----------------------------------------------------

        m = re.match(
            r"^(\s*)(\d+)\.\s+(.*)$",
            raw,
        )

        if m:
            flush_paragraph()

            indent_spaces = len(
                m.group(1).expandtabs(4)
            )

            blocks.append({
                "type": "numbered",
                "number": m.group(2),
                "text": m.group(3),
                "indent": indent_spaces,
            })

            continue

        # ----------------------------------------------------
        # Bullet list
        # ----------------------------------------------------

        m = re.match(
            r"^(\s*)[-*+]\s+(.*)$",
            raw,
        )

        if m:
            flush_paragraph()

            indent_spaces = len(
                m.group(1).expandtabs(4)
            )

            blocks.append({
                "type": "bullet",
                "text": m.group(2),
                "indent": indent_spaces,
            })

            continue

        # ----------------------------------------------------
        # Normal prose
        # ----------------------------------------------------

        paragraph_lines.append(raw)

    flush_paragraph()

    # Preserve an unterminated code block too.
    if in_code:
        blocks.append({
            "type": "code",
            "language": code_language,
            "lines": code_lines[:],
        })

    while blocks and blocks[0]["type"] == "blank":
        blocks.pop(0)

    while blocks and blocks[-1]["type"] == "blank":
        blocks.pop()

    return blocks


def parse_skill_md(skill_dir):
    skill_dir = Path(skill_dir)
    md_path = skill_dir / "SKILL.md"

    if not md_path.exists():
        return None

    content = md_path.read_text(
        encoding="utf-8"
    )

    metadata, body = _extract_frontmatter(
        content
    )

    raw_name = metadata.get(
        "name",
        skill_dir.name,
    )

    raw_description = metadata.get(
        "description",
        "",
    )

    title = (
        str(raw_name)
        .replace("webshop-", "")
        .replace("-", " ")
        .title()
    )

    description = str(
        raw_description
    ).strip()

    blocks = _parse_markdown_blocks(
        body
    )

    return {
        "title": title,
        "description": description,
        "blocks": blocks,
        "raw_path": str(md_path),
    }


# ============================================================
# Inline Markdown parser
# ============================================================

INLINE_PATTERN = re.compile(
    r"(`[^`]+`|\*\*.+?\*\*|__.+?__)"
)


def _parse_inline_spans(text):
    """
    Parse simple inline Markdown into visual spans.

    Examples:
        **must verify** -> bold
        __must verify__ -> bold
        `click[Buy Now]` -> inline code

    The textual content remains unchanged.
    Only Markdown delimiter characters become visual formatting.
    """
    spans = []

    pos = 0

    for m in INLINE_PATTERN.finditer(text):
        if m.start() > pos:
            spans.append({
                "style": "normal",
                "text": text[pos:m.start()],
            })

        token = m.group(0)

        if token.startswith("`") and token.endswith("`"):
            spans.append({
                "style": "code",
                "text": token[1:-1],
            })

        elif (
            token.startswith("**")
            and token.endswith("**")
        ):
            spans.append({
                "style": "bold",
                "text": token[2:-2],
            })

        elif (
            token.startswith("__")
            and token.endswith("__")
        ):
            spans.append({
                "style": "bold",
                "text": token[2:-2],
            })

        else:
            spans.append({
                "style": "normal",
                "text": token,
            })

        pos = m.end()

    if pos < len(text):
        spans.append({
            "style": "normal",
            "text": text[pos:],
        })

    if not spans:
        spans = [{
            "style": "normal",
            "text": text,
        }]

    # Restore common Markdown escape sequences.
    for span in spans:
        span["text"] = (
            span["text"]
            .replace("\\*", "*")
            .replace("\\_", "_")
            .replace("\\`", "`")
        )

    return spans


def _font_for_span(role, style):
    if style == "bold":
        return _body_font(role, bold=True)

    if style == "code":
        return _inline_code_font(role)

    return _body_font(role, bold=False)


# ============================================================
# Rich inline wrapping
# ============================================================

def _span_units(spans):
    """
    Convert spans into wrapping units.

    Normal / bold prose is split into words and whitespace.
    Inline-code spans are kept as one unit whenever possible.
    """
    units = []

    for span in spans:
        style = span["style"]
        text = span["text"]

        if style == "code":
            units.append({
                "style": "code",
                "text": text,
            })
            continue

        pieces = re.findall(
            r"\S+|\s+",
            text,
        )

        for piece in pieces:
            units.append({
                "style": style,
                "text": piece,
            })

    return units


def _unit_width(draw, unit, role):
    font = _font_for_span(
        role,
        unit["style"],
    )

    width, _ = _text_size(
        draw,
        unit["text"],
        font,
    )

    if unit["style"] == "code":
        width += 2 * INLINE_CODE_PAD_X

    return width


def _split_long_unit(draw, unit, role, max_width):
    """
    Character-level fallback for an unusually long word or inline-code span.
    """
    if _unit_width(draw, unit, role) <= max_width:
        return [unit]

    text = unit["text"]

    pieces = []
    cur = ""

    for ch in text:
        test = {
            "style": unit["style"],
            "text": cur + ch,
        }

        if (
            cur
            and _unit_width(draw, test, role) > max_width
        ):
            pieces.append({
                "style": unit["style"],
                "text": cur,
            })
            cur = ch
        else:
            cur += ch

    if cur:
        pieces.append({
            "style": unit["style"],
            "text": cur,
        })

    return pieces


def _layout_rich_text(
    draw,
    text,
    role,
    max_width,
):
    spans = _parse_inline_spans(
        text
    )

    raw_units = _span_units(
        spans
    )

    units = []

    for unit in raw_units:
        units.extend(
            _split_long_unit(
                draw,
                unit,
                role,
                max_width,
            )
        )

    lines = []
    current = []
    current_w = 0

    for unit in units:
        is_space = unit["text"].isspace()

        # Do not begin a new visual line with whitespace.
        if is_space and not current:
            continue

        w = _unit_width(
            draw,
            unit,
            role,
        )

        if (
            current
            and current_w + w > max_width
        ):
            # Remove trailing whitespace.
            while (
                current
                and current[-1]["text"].isspace()
            ):
                current.pop()

            if current:
                lines.append(current)

            current = []
            current_w = 0

            if is_space:
                continue

        current.append(unit)
        current_w += w

    while (
        current
        and current[-1]["text"].isspace()
    ):
        current.pop()

    if current:
        lines.append(current)

    if not lines:
        lines = [[]]

    return lines


def _rich_line_height(draw, line, role):
    if not line:
        return _line_height(
            draw,
            _body_font(role),
        )

    heights = []

    for unit in line:
        font = _font_for_span(
            role,
            unit["style"],
        )

        h = _line_height(
            draw,
            font,
        )

        if unit["style"] == "code":
            h += 2 * INLINE_CODE_PAD_Y

        heights.append(h)

    return max(heights)


def _measure_rich_text(
    draw,
    text,
    role,
    max_width,
):
    lines = _layout_rich_text(
        draw,
        text,
        role,
        max_width,
    )

    gap = _role_line_gap(
        role
    )

    total = 0

    for i, line in enumerate(lines):
        total += _rich_line_height(
            draw,
            line,
            role,
        )

        if i < len(lines) - 1:
            total += gap

    return total


def _draw_rich_text(
    draw,
    x,
    y,
    text,
    role,
    max_width,
    fill=None,
):
    lines = _layout_rich_text(
        draw,
        text,
        role,
        max_width,
    )

    gap = _role_line_gap(
        role
    )

    default_fill = (
        EXAMPLE_TEXT
        if role == "example"
        else TEXT
    )

    if fill is None:
        fill = default_fill

    for line in lines:
        line_h = _rich_line_height(
            draw,
            line,
            role,
        )

        cx = x

        for unit in line:
            style = unit["style"]

            font = _font_for_span(
                role,
                style,
            )

            text_w, text_h = _text_size(
                draw,
                unit["text"],
                font,
            )

            if style == "code":
                box_w = (
                    text_w
                    + 2 * INLINE_CODE_PAD_X
                )

                box_h = (
                    text_h
                    + 2 * INLINE_CODE_PAD_Y
                )

                box_y = (
                    y
                    + max(
                        0,
                        (line_h - box_h) // 2,
                    )
                )

                draw.rounded_rectangle(
                    [
                        cx,
                        box_y,
                        cx + box_w,
                        box_y + box_h,
                    ],
                    radius=4,
                    fill=INLINE_CODE_BG,
                    outline=INLINE_CODE_BORDER,
                    width=1,
                )

                draw.text(
                    (
                        cx + INLINE_CODE_PAD_X,
                        box_y + INLINE_CODE_PAD_Y,
                    ),
                    unit["text"],
                    font=font,
                    fill=INLINE_CODE_TEXT,
                )

                cx += box_w

            else:
                text_y = (
                    y
                    + max(
                        0,
                        (line_h - text_h) // 2,
                    )
                )

                draw.text(
                    (cx, text_y),
                    unit["text"],
                    font=font,
                    fill=fill,
                )

                cx += text_w

        y += line_h + gap

    if lines:
        y -= gap

    return y


# ============================================================
# Heading wrapping
# ============================================================

def _wrap_plain(
    draw,
    text,
    font,
    max_width,
):
    words = text.split()

    if not words:
        return [""]

    lines = []
    cur = ""

    for word in words:
        test = (
            word
            if not cur
            else f"{cur} {word}"
        )

        if (
            _text_size(draw, test, font)[0]
            <= max_width
        ):
            cur = test
            continue

        if cur:
            lines.append(cur)

        # Long-token fallback.
        if (
            _text_size(draw, word, font)[0]
            > max_width
        ):
            chunk = ""

            for ch in word:
                test_chunk = chunk + ch

                if (
                    chunk
                    and _text_size(
                        draw,
                        test_chunk,
                        font,
                    )[0] > max_width
                ):
                    lines.append(chunk)
                    chunk = ch
                else:
                    chunk = test_chunk

            cur = chunk

        else:
            cur = word

    if cur:
        lines.append(cur)

    return lines


# ============================================================
# Section heading drawing
# ============================================================

def _draw_h2(
    draw,
    x,
    y,
    text,
    role,
    width,
):
    font = _h2_font(
        role
    )

    accent, light = _role_accent(
        role
    )

    lines = _wrap_plain(
        draw,
        text,
        font,
        width - 34,
    )

    lh = _line_height(
        draw,
        font,
    )

    line_gap = 4

    box_h = (
        14
        + len(lines) * lh
        + max(
            0,
            len(lines) - 1,
        ) * line_gap
    )

    draw.rounded_rectangle(
        [
            x,
            y,
            x + width,
            y + box_h,
        ],
        radius=7,
        fill=light,
    )

    rail_w = (
        7
        if role == "critical"
        else 6
    )

    draw.rectangle(
        [
            x,
            y,
            x + rail_w,
            y + box_h,
        ],
        fill=accent,
    )

    ty = y + 7

    for line in lines:
        draw.text(
            (x + 18, ty),
            line,
            font=font,
            fill=accent,
        )

        ty += lh + line_gap

    return y + box_h


def _draw_h3(
    draw,
    x,
    y,
    text,
    role,
    width,
):
    font = _h3_font(
        role
    )

    accent, _ = _role_accent(
        role
    )

    lines = _wrap_plain(
        draw,
        text,
        font,
        width - 18,
    )

    lh = _line_height(
        draw,
        font,
    )

    cy = y

    for i, line in enumerate(lines):
        if i == 0:
            draw.ellipse(
                [
                    x,
                    cy + max(5, lh // 3),
                    x + 8,
                    cy + max(5, lh // 3) + 8,
                ],
                fill=accent,
            )

        draw.text(
            (
                x + 17,
                cy,
            ),
            line,
            font=font,
            fill=accent,
        )

        cy += lh + 4

    if lines:
        cy -= 4

    return cy


# ============================================================
# Semantic rail
# ============================================================

def _content_geometry(
    x,
    width,
    role,
):
    """
    Reserve a thin visual rail for action-critical content.

    Example remains visually quiet.
    """
    if role in {
        "procedure",
        "critical",
        "error",
    }:
        return (
            x + ROLE_RAIL_W + ROLE_RAIL_GAP,
            width - ROLE_RAIL_W - ROLE_RAIL_GAP,
        )

    if role == "example":
        return (
            x + 8,
            width - 8,
        )

    return x, width


def _draw_role_rail(
    draw,
    x,
    y0,
    y1,
    role,
):
    if role not in {
        "procedure",
        "critical",
        "error",
    }:
        return

    accent, _ = _role_accent(
        role
    )

    draw.rounded_rectangle(
        [
            x,
            y0,
            x + ROLE_RAIL_W,
            y1,
        ],
        radius=2,
        fill=accent,
    )


# ============================================================
# Paragraph
# ============================================================

def _draw_paragraph(
    draw,
    x,
    y,
    text,
    role,
    width,
):
    content_x, content_w = _content_geometry(
        x,
        width,
        role,
    )

    start_y = y

    y = _draw_rich_text(
        draw,
        content_x,
        y,
        text,
        role,
        content_w,
    )

    if y > start_y:
        _draw_role_rail(
            draw,
            x,
            start_y,
            y,
            role,
        )

    return y


# ============================================================
# Bullet lists
# ============================================================

def _draw_bullet_item(
    draw,
    x,
    y,
    text,
    role,
    width,
    indent_spaces=0,
):
    content_x, content_w = _content_geometry(
        x,
        width,
        role,
    )

    indent_level = min(
        indent_spaces // 2,
        4,
    )

    indent_px = (
        indent_level * 22
    )

    item_x = (
        content_x
        + indent_px
    )

    bullet_font = _body_font(
        role,
        bold=True,
    )

    bullet = "•"

    bullet_w = _text_size(
        draw,
        bullet,
        bullet_font,
    )[0]

    body_x = (
        item_x
        + bullet_w
        + 9
    )

    body_w = (
        content_w
        - indent_px
        - bullet_w
        - 9
    )

    start_y = y

    draw.text(
        (item_x, y),
        bullet,
        font=bullet_font,
        fill=(
            EXAMPLE_ACCENT
            if role == "example"
            else _role_accent(role)[0]
        ),
    )

    y2 = _draw_rich_text(
        draw,
        body_x,
        y,
        text,
        role,
        body_w,
    )

    _draw_role_rail(
        draw,
        x,
        start_y,
        y2,
        role,
    )

    return y2


# ============================================================
# Numbered lists
# ============================================================

def _draw_numbered_item(
    draw,
    x,
    y,
    number,
    text,
    role,
    width,
    indent_spaces=0,
):
    content_x, content_w = _content_geometry(
        x,
        width,
        role,
    )

    indent_level = min(
        indent_spaces // 2,
        4,
    )

    indent_px = (
        indent_level * 22
    )

    item_x = (
        content_x
        + indent_px
    )

    # Procedure / critical / error use strong visual step anchors.
    use_badge = (
        role in {
            "procedure",
            "critical",
            "error",
        }
        and indent_level == 0
    )

    start_y = y

    if use_badge:
        badge_font = FONT_SMALL_BOLD

        number_text = str(
            number
        )

        num_w = _text_size(
            draw,
            number_text,
            badge_font,
        )[0]

        badge_w = max(
            STEP_BADGE_MIN_W,
            num_w + 18,
        )

        if role == "critical":
            badge_bg = CRITICAL_STEP_BG
            badge_text = CRITICAL_STEP_TEXT

        elif role == "error":
            badge_bg = ERROR_STEP_BG
            badge_text = ERROR_STEP_TEXT

        else:
            badge_bg = STEP_BG
            badge_text = STEP_TEXT

        draw.rounded_rectangle(
            [
                item_x,
                y,
                item_x + badge_w,
                y + STEP_BADGE_H,
            ],
            radius=10,
            fill=badge_bg,
        )

        num_h = _line_height(
            draw,
            badge_font,
        )

        draw.text(
            (
                item_x
                + (badge_w - num_w) / 2,
                y
                + (STEP_BADGE_H - num_h) / 2
                - 1,
            ),
            number_text,
            font=badge_font,
            fill=badge_text,
        )

        body_x = (
            item_x
            + badge_w
            + STEP_BADGE_GAP
        )

        body_w = (
            content_w
            - indent_px
            - badge_w
            - STEP_BADGE_GAP
        )

        body_h = _measure_rich_text(
            draw,
            text,
            role,
            body_w,
        )

        # Vertically center single-line items relative to badge.
        body_y = (
            y
            + max(
                0,
                (STEP_BADGE_H - body_h) // 2,
            )
        )

        text_end = _draw_rich_text(
            draw,
            body_x,
            body_y,
            text,
            role,
            body_w,
        )

        item_end = max(
            y + STEP_BADGE_H,
            text_end,
        )

        _draw_role_rail(
            draw,
            x,
            start_y,
            item_end,
            role,
        )

        return item_end

    # General / nested list: standard numbering.
    number_font = _body_font(
        role,
        bold=True,
    )

    prefix = (
        f"{number}."
    )

    prefix_w = _text_size(
        draw,
        prefix,
        number_font,
    )[0]

    draw.text(
        (
            item_x,
            y,
        ),
        prefix,
        font=number_font,
        fill=(
            EXAMPLE_ACCENT
            if role == "example"
            else _role_accent(role)[0]
        ),
    )

    body_x = (
        item_x
        + prefix_w
        + 9
    )

    body_w = (
        content_w
        - indent_px
        - prefix_w
        - 9
    )

    y2 = _draw_rich_text(
        draw,
        body_x,
        y,
        text,
        role,
        body_w,
    )

    _draw_role_rail(
        draw,
        x,
        start_y,
        y2,
        role,
    )

    return y2


# ============================================================
# Code wrapping
# ============================================================

def _wrap_code_line(
    draw,
    text,
    font,
    max_width,
):
    """
    Preserve original code line breaks.

    A source line is additionally wrapped only if it physically
    exceeds the image width.
    """
    if text == "":
        return [""]

    expanded = text.expandtabs(4)

    if (
        _text_size(
            draw,
            expanded,
            font,
        )[0]
        <= max_width
    ):
        return [expanded]

    leading_len = (
        len(expanded)
        - len(expanded.lstrip(" "))
    )

    leading = expanded[
        :leading_len
    ]

    body = expanded[
        leading_len:
    ]

    leading_w = _text_size(
        draw,
        leading,
        font,
    )[0]

    usable = max(
        100,
        max_width - leading_w,
    )

    lines = []
    current = ""

    for ch in body:
        test = current + ch

        if (
            current
            and _text_size(
                draw,
                test,
                font,
            )[0] > usable
        ):
            lines.append(
                leading + current
            )

            current = ch
        else:
            current = test

    if current or not lines:
        lines.append(
            leading + current
        )

    return lines


# ============================================================
# Code block
# ============================================================

def _draw_code_block(
    draw,
    x,
    y,
    block,
    role,
    width,
):
    content_x, content_w = _content_geometry(
        x,
        width,
        role,
    )

    font = _block_code_font(
        role
    )

    if role == "example":
        pad_x = EXAMPLE_CODE_PAD_X
        pad_y = EXAMPLE_CODE_PAD_Y
        line_gap = EXAMPLE_CODE_LINE_GAP
    else:
        pad_x = CODE_PAD_X
        pad_y = CODE_PAD_Y
        line_gap = CODE_LINE_GAP

    inner_w = (
        content_w
        - 2 * pad_x
    )

    rendered_lines = []

    for source_line in block["lines"]:
        rendered_lines.extend(
            _wrap_code_line(
                draw,
                source_line,
                font,
                inner_w,
            )
        )

    if not rendered_lines:
        rendered_lines = [""]

    lh = _line_height(
        draw,
        font,
    )

    language = block.get(
        "language",
        "",
    )

    label_h = 0

    if language:
        label_h = (
            _line_height(
                draw,
                FONT_SMALL,
            )
            + 7
        )

    box_h = (
        2 * pad_y
        + label_h
        + len(rendered_lines) * lh
        + max(
            0,
            len(rendered_lines) - 1,
        ) * line_gap
    )

    if role == "example":
        box_fill = (
            248,
            248,
            247,
        )

        box_border = (
            218,
            216,
            211,
        )

        code_fill = (
            65,
            64,
            61,
        )
    else:
        box_fill = CODE_BG
        box_border = CODE_BORDER
        code_fill = CODE_TEXT

    start_y = y

    draw.rounded_rectangle(
        [
            content_x,
            y,
            content_x + content_w,
            y + box_h,
        ],
        radius=8,
        fill=box_fill,
        outline=box_border,
        width=1,
    )

    cy = (
        y
        + pad_y
    )

    if language:
        draw.text(
            (
                content_x + pad_x,
                cy,
            ),
            language,
            font=FONT_SMALL,
            fill=MUTED,
        )

        cy += (
            _line_height(
                draw,
                FONT_SMALL,
            )
            + 7
        )

    for line in rendered_lines:
        draw.text(
            (
                content_x + pad_x,
                cy,
            ),
            line,
            font=font,
            fill=code_fill,
        )

        cy += (
            lh
            + line_gap
        )

    end_y = (
        y + box_h
    )

    _draw_role_rail(
        draw,
        x,
        start_y,
        end_y,
        role,
    )

    return end_y


# ============================================================
# Description drawing
# ============================================================

def _draw_plain_wrapped(
    draw,
    x,
    y,
    text,
    font,
    fill,
    width,
    line_gap,
):
    lines = _wrap_plain(
        draw,
        text,
        font,
        width,
    )

    lh = _line_height(
        draw,
        font,
    )

    for line in lines:
        draw.text(
            (x, y),
            line,
            font=font,
            fill=fill,
        )

        y += (
            lh + line_gap
        )

    if lines:
        y -= line_gap

    return y


# ============================================================
# Complete skill drawing
# ============================================================

def _draw_skill_on_canvas(
    skill,
    canvas_h,
):
    img = Image.new(
        "RGB",
        (
            CANVAS_W,
            canvas_h,
        ),
        BG,
    )

    draw = ImageDraw.Draw(
        img
    )

    x = MARGIN_X
    y = TOP_PAD

    # --------------------------------------------------------
    # Skill title
    # --------------------------------------------------------

    title_lines = _wrap_plain(
        draw,
        skill["title"],
        FONT_TITLE,
        CONTENT_W,
    )

    title_lh = _line_height(
        draw,
        FONT_TITLE,
    )

    for line in title_lines:
        draw.text(
            (x, y),
            line,
            font=FONT_TITLE,
            fill=GENERAL_ACCENT,
        )

        y += (
            title_lh + 5
        )

    if title_lines:
        y -= 5

    y += 9

    draw.line(
        (
            x,
            y,
            x + CONTENT_W,
            y,
        ),
        fill=GENERAL_ACCENT,
        width=2,
    )

    y += TITLE_BOTTOM_GAP

    # --------------------------------------------------------
    # Metadata description
    # --------------------------------------------------------

    if skill["description"]:
        y = _draw_plain_wrapped(
            draw,
            x,
            y,
            skill["description"],
            FONT_DESC,
            SUBTEXT,
            CONTENT_W,
            5,
        )

        y += DESC_BOTTOM_GAP

    # --------------------------------------------------------
    # Markdown body
    # --------------------------------------------------------

    current_role = "general"

    previous_type = None

    for block in skill["blocks"]:
        typ = block["type"]

        # Safety flag. We still finish drawing and rerender later
        # on a taller canvas if necessary.
        if y > canvas_h - 500:
            pass

        # ----------------------------------------------------
        # Blank
        # ----------------------------------------------------

        if typ == "blank":
            y += BLANK_GAP
            previous_type = typ
            continue

        # ----------------------------------------------------
        # H1
        # ----------------------------------------------------

        if typ == "h1":
            y += H1_TOP_GAP

            lines = _wrap_plain(
                draw,
                block["text"],
                FONT_H1,
                CONTENT_W,
            )

            lh = _line_height(
                draw,
                FONT_H1,
            )

            for line in lines:
                draw.text(
                    (x, y),
                    line,
                    font=FONT_H1,
                    fill=TEXT,
                )

                y += (
                    lh + 4
                )

            if lines:
                y -= 4

            y += 8

            draw.line(
                (
                    x,
                    y,
                    x + CONTENT_W,
                    y,
                ),
                fill=BORDER,
                width=1,
            )

            y += H1_BOTTOM_GAP

        # ----------------------------------------------------
        # H2
        # ----------------------------------------------------

        elif typ == "h2":
            y += H2_TOP_GAP

            current_role = _section_role(
                block["text"]
            )

            y = _draw_h2(
                draw,
                x,
                y,
                block["text"],
                current_role,
                CONTENT_W,
            )

            y += H2_BOTTOM_GAP

        # ----------------------------------------------------
        # H3
        # ----------------------------------------------------

        elif typ == "h3":
            y += H3_TOP_GAP

            local_role = _section_role(
                block["text"]
            )

            # Generic H3 remains under the parent H2 role.
            if local_role == "general":
                local_role = current_role

            y = _draw_h3(
                draw,
                x,
                y,
                block["text"],
                local_role,
                CONTENT_W,
            )

            y += H3_BOTTOM_GAP

        # ----------------------------------------------------
        # Paragraph
        # ----------------------------------------------------

        elif typ == "paragraph":
            y = _draw_paragraph(
                draw,
                x,
                y,
                block["text"],
                current_role,
                CONTENT_W,
            )

            y += PARAGRAPH_GAP

        # ----------------------------------------------------
        # Bullet
        # ----------------------------------------------------

        elif typ == "bullet":
            y = _draw_bullet_item(
                draw,
                x,
                y,
                block["text"],
                current_role,
                CONTENT_W,
                block.get(
                    "indent",
                    0,
                ),
            )

            y += LIST_ITEM_GAP

        # ----------------------------------------------------
        # Numbered
        # ----------------------------------------------------

        elif typ == "numbered":
            y = _draw_numbered_item(
                draw,
                x,
                y,
                block["number"],
                block["text"],
                current_role,
                CONTENT_W,
                block.get(
                    "indent",
                    0,
                ),
            )

            y += LIST_ITEM_GAP

        # ----------------------------------------------------
        # Code
        # ----------------------------------------------------

        elif typ == "code":
            if previous_type != "blank":
                y += 3

            y = _draw_code_block(
                draw,
                x,
                y,
                block,
                current_role,
                CONTENT_W,
            )

            y += PARAGRAPH_GAP + 2

        previous_type = typ

    final_h = int(
        y + BOTTOM_PAD
    )

    return img, final_h


# ============================================================
# Public renderer
# ============================================================

def render_skill(
    skill,
    out_path,
):
    """
    Render exactly one single-page PNG.

    If the temporary canvas is too short, the function transparently
    retries with a taller temporary canvas. The final result is always
    cropped to the exact used height.
    """
    canvas_h = INITIAL_CANVAS_H

    while True:
        img, final_h = _draw_skill_on_canvas(
            skill,
            canvas_h,
        )

        if final_h <= canvas_h:
            break

        canvas_h = max(
            canvas_h * 2,
            final_h + 500,
        )

    img = img.crop(
        (
            0,
            0,
            CANVAS_W,
            final_h,
        )
    )

    img.save(
        out_path,
        optimize=True,
    )

    return img


# ============================================================
# Entry point
# ============================================================

def main():
    project = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )

    skills_dir = (
        project
        / "skills"
    )

    # New directory so v1 / v2 cards are untouched.
    out_dir = (
        project
        / "images"
        / "skill_cards_v7"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"{'Skill':<42} "
        f"{'Size':>14} "
        f"{'Aspect':>8}"
    )

    print(
        "-" * 70
    )

    count = 0
    total_h = 0
    max_h = 0
    max_skill = None

    for skill_dir in sorted(
        skills_dir.iterdir()
    ):
        if not skill_dir.is_dir():
            continue

        skill = parse_skill_md(
            skill_dir
        )

        if skill is None:
            continue

        out_path = (
            out_dir
            / f"{skill_dir.name}.png"
        )

        img = render_skill(
            skill,
            out_path,
        )

        w, h = img.size

        aspect = h / w

        count += 1
        total_h += h

        if h > max_h:
            max_h = h
            max_skill = skill_dir.name

        print(
            f"{skill_dir.name:<42} "
            f"{w:>4}x{h:<8} "
            f"{aspect:>8.2f}"
        )

    print(
        "-" * 70
    )

    if count:
        print(
            f"Rendered: {count}"
        )

        print(
            f"Average height: "
            f"{total_h / count:.1f}px"
        )

        print(
            f"Tallest skill: "
            f"{max_skill} "
            f"({max_h}px)"
        )

        print(
            f"Output directory: "
            f"{out_dir}"
        )

    else:
        print(
            "No skills rendered."
        )


if __name__ == "__main__":
    main()
