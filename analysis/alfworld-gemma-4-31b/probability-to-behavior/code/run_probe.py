"""
Step 3 — probe every (decision state × k × representation) cell of the
probability-to-behavior bridge.

Same measurement machinery as ../../../alfworld-qwen3.5-9b/lexical-interference/code/run_probe.py and
../../../alfworld-qwen3.5-9b/rasterized-interference/code/run_probe.py (free generation at temperature 0
+ forced decoding of the full candidate set under a byte-identical stem), but
the states come from extract_behavior_states.py, where the "distractor" is a
REAL, currently-executable competing action (competitor_admissible=true).

That changes what the free-generation half of each cell means: when the model
outputs the competitor here, the environment WOULD execute it — so
`chose_distractor` is no longer a hypothetical preference but an actual wrong
behaviour, and the pair

    P(competitor) rises with k   AND   chose-competitor rate rises with k

is the bridge this experiment exists to measure, in all three representations
(text / rasterized / structural). Rollout-level task cost (extra steps /
failure) is the optional next stage, run_rollout.py.

Cells are one flat work queue; `--max_workers` processes pull from it and are
round-robin assigned a vLLM port at pick-up time (true-dynamic scheduling, same
as the two main experiments). Every finished cell is written to its own JSON
file and existing files are skipped, so the run is resume-safe.
"""
import os
import json
import queue
import argparse
import multiprocessing
from pathlib import Path
from tqdm import tqdm

import probe_lib as pl
import skill_lexicon as lex

DEFAULT_MODEL = pl.DEFAULT_MODEL

# representation -> sub-directory of variants_dir holding its card PNGs.
CARD_DIRS = {
    "visual": "visual",          # legacy name of the structural cards
    "structural": "visual",      # same pixels, explicit name
    "rasterized": "rasterized",
}


# ══════════════════════════════════════════════════════════════════════
# One cell
# ══════════════════════════════════════════════════════════════════════

def run_cell(task, model, force_template, max_gen_tokens):
    state = task["state"]
    rep = task["representation"]
    candidates = list(state["candidate_actions"])
    correct = lex.normalize_action(state["correct_action"])
    distractor = lex.normalize_action(state["distractor_action"])

    messages = pl.build_messages(
        state, rep,
        skill_text=task.get("skill_text"),
        card_paths=task.get("card_paths"),
    )

    text, token_lps = pl.generate_action(messages, model=model, max_tokens=max_gen_tokens)
    gen = pl.summarise_generation(text, correct, distractor, candidates)

    scores, prefix_count = pl.score_candidates(
        messages, candidates, model=model, template=force_template)
    stats = pl.distribution_stats(scores, correct, distractor)

    doc = task.get("skill_text") or ""
    return {
        "state_id": state["state_id"],
        "level": task["level"],
        "representation": rep,
        "game_name": state["game_name"],
        "task_type": state["task_type"],
        "step_idx": state["step_idx"],
        "correct_action": correct,
        "correct_action_type": state["correct_action_type"],
        "distractor_action": distractor,
        "distractor_action_type": state["distractor_action_type"],
        "distractor_keyword": state["distractor_keyword"],
        "distractor_phrase": state["distractor_phrase"],
        "competitor_admissible": bool(state.get("competitor_admissible")),
        "candidate_actions": candidates,
        "doc_keyword_count": lex.count_word(doc, state["distractor_keyword"]) if doc else 0,
        "doc_phrase_count": lex.count_phrase(doc, state["distractor_phrase"]) if doc else 0,
        "doc_words": lex.word_count(doc),
        "doc_chars": len(doc),
        "n_cards": len(task.get("card_paths") or []),
        "prefix_tokens": prefix_count,
        "generation": gen,
        "generation_token_logprobs": token_lps if not task.get("drop_token_logprobs") else None,
        "candidate_scores": {a: scores[a] for a in candidates},
        "distribution": stats,
    }


# ══════════════════════════════════════════════════════════════════════
# Workers
# ══════════════════════════════════════════════════════════════════════

def _worker(task_queue, result_queue, ports, counter, model, force_template,
            max_gen_tokens, out_dir):
    while True:
        task = task_queue.get()
        if task is None:
            break
        with counter.get_lock():
            order = counter.value
            counter.value += 1
        pl.set_endpoint(ports[order % len(ports)])
        try:
            result = run_cell(task, model, force_template, max_gen_tokens)
            out_file = Path(out_dir) / task["out_name"]
            out_file.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                encoding="utf-8")
            result_queue.put((task["task_id"], True, task["out_name"]))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            result_queue.put((task["task_id"], False, f"{task['out_name']}: {exc}"))


# ══════════════════════════════════════════════════════════════════════
# Task enumeration
# ══════════════════════════════════════════════════════════════════════

def build_tasks(args):
    states_dir = Path(args.states_dir)
    variants_dir = Path(args.variants_dir)
    out_dir = Path(args.output_dir)
    levels = [int(x) for x in args.levels.split(",")]
    reps = [r.strip() for r in args.representations.split(",") if r.strip()]
    for r in reps:
        if r not in pl.ALL_REPS:
            raise SystemExit(f"unknown representation '{r}' (expected one of {pl.ALL_REPS})")

    manifest = json.loads((states_dir / "manifest.json").read_text(encoding="utf-8"))
    tasks, missing = [], []
    tid = 0
    for info in manifest["states"]:
        state_id = info["state_id"]
        sf = states_dir / f"{state_id}.json"
        if not sf.exists():
            missing.append(f"state file {state_id}")
            continue
        state = json.loads(sf.read_text(encoding="utf-8"))

        cells = [(k, rep) for k in levels for rep in reps if rep != "none"]
        if args.include_no_skill_control:
            cells.append((-1, "none"))

        for level, rep in cells:
            out_name = f"{state_id}__k{level}__{rep}.json"
            if (out_dir / out_name).exists() and not args.overwrite:
                continue

            task = {
                "task_id": tid, "state": state, "level": level,
                "representation": rep, "out_name": out_name,
                "drop_token_logprobs": args.drop_token_logprobs,
            }
            if rep == "text":
                md = variants_dir / "text" / state_id / f"level_{level}.md"
                if not md.exists():
                    missing.append(str(md))
                    continue
                task["skill_text"] = md.read_text(encoding="utf-8")
            elif rep in CARD_DIRS:
                card_dir = variants_dir / CARD_DIRS[rep] / state_id / f"level_{level}"
                cards = sorted(card_dir.glob("card*.png")) if card_dir.exists() else []
                if not cards:
                    missing.append(str(card_dir))
                    continue
                if args.max_images and len(cards) > args.max_images:
                    missing.append(f"{card_dir} has {len(cards)} cards > --max_images")
                    continue
                task["card_paths"] = [str(c) for c in cards]
                # The cards render exactly this document; carry its text so the
                # result's doc_keyword_count/word stats describe what the image
                # cells actually show (build_messages ignores skill_text for
                # image representations).
                md = variants_dir / "text" / state_id / f"level_{level}.md"
                if md.exists():
                    task["skill_text"] = md.read_text(encoding="utf-8")
            tasks.append(task)
            tid += 1

    return tasks, missing, manifest


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks, missing, manifest = build_tasks(args)
    print(f"[INFO] {manifest['total_states']} states; {len(tasks)} cells to run "
          f"(levels={args.levels}, representations={args.representations}, "
          f"no-skill control={args.include_no_skill_control})")
    if missing:
        print(f"[WARN] {len(missing)} inputs missing, e.g. {missing[:3]}")
    if args.limit > 0:
        tasks = tasks[:args.limit]
        print(f"[INFO] --limit -> running only {len(tasks)} cells")
    if args.dry_run:
        n_cand = sum(len(t["state"]["candidate_actions"]) for t in tasks)
        print(f"[DRY RUN] {len(tasks)} cells, {n_cand} forced-decoding targets "
              f"(~{n_cand * 2 + len(tasks)} HTTP calls). Nothing executed.")
        return
    if not tasks:
        print("Nothing to do — all cells already present.")
        return

    ports = [int(p.strip()) for p in args.ports.split(",")]
    workers = max(1, min(args.max_workers, len(tasks)))
    print(f"[INFO] dynamic scheduling: {workers} workers over ports {ports}")

    ctx = multiprocessing.get_context("fork")
    task_queue, result_queue = ctx.Queue(), ctx.Queue()
    counter = ctx.Value("i", 0)
    for t in tasks:
        task_queue.put(t)
    for _ in range(workers):
        task_queue.put(None)

    procs = []
    for _ in range(workers):
        p = ctx.Process(target=_worker,
                        args=(task_queue, result_queue, ports, counter, args.model,
                              args.force_template, args.max_gen_tokens, str(out_dir)),
                        daemon=True)
        p.start()
        procs.append(p)

    ok = fail = 0
    pbar = tqdm(total=len(tasks), desc="Probe cells")
    for _ in range(len(tasks)):
        try:
            _, success, note = result_queue.get(timeout=args.timeout)
        except queue.Empty:
            print(f"\n[WARN] no result for {args.timeout}s; stopping early.")
            break
        ok += int(success)
        fail += int(not success)
        if not success:
            print(f"[FAIL] {note}")
        pbar.update(1)
        pbar.set_postfix({"ok": ok, "fail": fail})
    pbar.close()

    for p in procs:
        p.join(timeout=60)
    for p in procs:
        if p.is_alive():
            p.terminate()

    print(f"\nDone. {ok} cells written, {fail} failed. Results in {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--states_dir", required=True)
    p.add_argument("--variants_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--levels", default="0,1,2,4,8")
    p.add_argument("--representations", default="text,rasterized,structural",
                   help="all three representations are measured here — the "
                        "bridge claim is about their DIFFERENTIAL growth "
                        "('visual' accepted as an alias of structural)")
    p.add_argument("--include_no_skill_control", action="store_true", default=True,
                   help="also probe each state with no skill document (level -1)")
    p.add_argument("--force_template", choices=["thought_action", "action_only"],
                   default="thought_action",
                   help="prefix used when force-decoding candidates (must be constant "
                        "across candidates and match the merged lexical-interference "
                        "run: thought_action)")
    p.add_argument("--max_gen_tokens", type=int, default=256)
    p.add_argument("--max_images", type=int, default=6,
                   help="skip image cells needing more cards than the server allows")
    p.add_argument("--drop_token_logprobs", action="store_true", default=True,
                   help="do not store per-token generation logprobs (keeps files small)")
    p.add_argument("--keep_token_logprobs", dest="drop_token_logprobs",
                   action="store_false")
    p.add_argument("--max_workers", type=int, default=128)
    p.add_argument("--ports", default="8000,8001,8002,8003")
    p.add_argument("--timeout", type=int, default=7200)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int, default=-1)
    p.add_argument("--dry_run", action="store_true")
    main(p.parse_args())
