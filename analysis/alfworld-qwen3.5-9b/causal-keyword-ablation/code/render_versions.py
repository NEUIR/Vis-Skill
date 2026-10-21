"""
Step 2b — render every counterfactual document version into skill-card PNGs,
in BOTH image styles:

  visual/{state_id}/{version}_k{k}/card{i}_{skill}.png
      structural cards — render_cards.render_card, byte-for-byte the renderer
      the lexical-interference visual condition (= the main experiment's Visual
      Skill) used, so "structural" here means exactly what it meant there.
  rasterized/{state_id}/{version}_k{k}/card{i}_{skill}.png
      plain-text-screenshot cards — render_rasterized.render_rasterized_card,
      the modality-only condition from ../rasterized-interference.

Rendering is per skill part (5 parts -> 5 cards per version document, inside
the servers' image limit), and verification mirrors the two renderers' own
guarantees:
  structural — alphanumeric content drawn vs source, 2% tolerance (markdown
               syntax legitimately disappears into styling);
  rasterized — non-whitespace characters drawn == source, EXACT (anything else
               is a hard failure: the plain render must be lossless).

The three versions (original / counterfactual / matched_control) of one state
and level are rendered with identical geometry settings, so any measured
difference is the one-word manipulation, not the layout.

Usage:
    python render_versions.py --variants_dir ../variants [--styles structural,rasterized]
                              [--workers 8] [--overwrite]
"""
import json
import argparse
import multiprocessing
from pathlib import Path

import render_cards
import render_rasterized as rr

VERSION_GLOB = "*_k*.parts.json"


def _render_one(job):
    style, parts_file, out_root = job
    parts_file, out_root = Path(parts_file), Path(out_root)
    stem = parts_file.name[:-len(".parts.json")]
    sid = parts_file.parent.name
    parts = json.loads(parts_file.read_text(encoding="utf-8"))
    out_dir = out_root / sid / stem
    report = []
    for i, part in enumerate(parts):
        out_path = out_dir / f"card{i}_{part['skill_name']}.png"
        if style == "structural":
            size, _ = render_cards.render_card(
                part["skill_name"], part.get("description", ""),
                part["markdown"], out_path)
            vblocks = render_cards.parse_blocks(part["markdown"])
            src_n = len(render_cards._alnum(render_cards._strip_inline(
                part["skill_name"] + part.get("description", "") + part["markdown"])))
            drawn_n = (len(render_cards._alnum(
                part["skill_name"] + part.get("description", "")))
                + render_cards.drawn_content_chars(vblocks))
            report.append({"path": str(out_path), "style": style, "height": size[1],
                           "drawn_chars": drawn_n, "source_chars": src_n,
                           "lossless_exact": None})
        else:
            size, drawn, source = rr.render_rasterized_card(
                part["skill_name"], part.get("description", ""),
                part["markdown"], out_path)
            report.append({"path": str(out_path), "style": style, "height": size[1],
                           "drawn_chars": drawn, "source_chars": source,
                           "lossless_exact": drawn == source})
    return report


def main(args):
    variants_dir = Path(args.variants_dir)
    text_dir = variants_dir / "text"
    if not text_dir.exists():
        raise SystemExit(f"no text variants at {text_dir}; run "
                         f"construct_counterfactuals.py first")
    styles = [s.strip() for s in args.styles.split(",") if s.strip()]
    out_roots = {"structural": variants_dir / "visual",
                 "rasterized": variants_dir / "rasterized"}

    jobs = []
    for style in styles:
        out_root = out_roots[style]
        for parts_file in sorted(text_dir.glob(f"*/{VERSION_GLOB}")):
            stem = parts_file.name[:-len(".parts.json")]
            sid = parts_file.parent.name
            out_dir = out_root / sid / stem
            existing = sorted(out_dir.glob("card*.png")) if out_dir.exists() else []
            n_parts = len(json.loads(parts_file.read_text(encoding="utf-8")))
            if existing and len(existing) == n_parts and not args.overwrite:
                continue
            jobs.append((style, str(parts_file), str(out_root)))
    print(f"[INFO] {len(jobs)} version documents to render "
          f"(styles: {styles}, workers: {args.workers})")
    if not jobs:
        print("Nothing to do — all cards already present.")
        return

    all_report = []
    if args.workers > 1:
        with multiprocessing.get_context("fork").Pool(args.workers) as pool:
            for rep in pool.imap_unordered(_render_one, jobs, chunksize=1):
                all_report.extend(rep)
    else:
        for j in jobs:
            all_report.extend(_render_one(j))

    # ── verification ──
    bad_raster = [r for r in all_report
                  if r["style"] == "rasterized" and r["lossless_exact"] is False]
    bad_struct = [r for r in all_report
                  if r["style"] == "structural" and r["source_chars"]
                  and abs(r["drawn_chars"] - r["source_chars"]) / r["source_chars"] > 0.02]
    tall = [r for r in all_report if r["height"] > args.warn_height]
    for style in styles:
        out_root = out_roots[style]
        out_root.mkdir(parents=True, exist_ok=True)
        rep = [r for r in all_report if r["style"] == style]
        (out_root / "render_verification.json").write_text(json.dumps(
            {"cards": len(rep),
             "flagged": len(bad_raster if style == "rasterized" else bad_struct),
             "report": rep}, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  {style}: rendered {len(rep)} cards -> {out_root}")

    if tall:
        print(f"[WARN] {len(tall)} cards taller than {args.warn_height}px "
              f"(max {max(r['height'] for r in tall)}px)")
    if bad_struct:
        print(f"[WARN] {len(bad_struct)} structural cards deviate >2% in visible "
              f"characters (same tolerance the lexical-interference render used)")
    if bad_raster:
        print(f"[FAIL] {len(bad_raster)} rasterized cards are NOT lossless — "
              f"fix before probing")
        raise SystemExit(1)
    print("Verification OK.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--variants_dir", required=True)
    p.add_argument("--styles", default="structural,rasterized",
                   help="which card styles to render (the probe's image "
                        "conditions); 'structural' writes to visual/, "
                        "'rasterized' to rasterized/")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--warn_height", type=int, default=6000)
    main(p.parse_args())
