"""
Step 2 — build the counterfactual skill documents for the causal keyword
ablation.

Question: does the target word in the Skill itself drive the corresponding wrong action?
(Not "does the document change behaviour" — the lexical-interference k-sweep
already shows that — but "is it THIS WORD, at THIS frequency, doing it".)

For every decision state and every high-exposure level k (default 4 and 8),
three document versions are built from the SAME lexical-interference document:

  original         level_{k}.md verbatim — the distractor keyword occurs
                   exactly k times, all inside the k injected sentences.

  counterfactual   every whole-word occurrence of the distractor keyword is
                   replaced by the SAME neutral, same-semantic-level word the
                   base-document scrub uses (microwave -> appliance, fridge ->
                   chiller, ...; skill_lexicon.SCRUB_REPLACEMENT). One word for
                   one word: sentence order, positions, word count and layout
                   are untouched — only the keyword is gone.

  matched_control  the parallel manipulation on a word that has NOTHING to do
                   with the distractor: a control word occurring exactly k
                   times in the document (frequency-matched to the keyword) is
                   replaced, one-for-one, by a neutral filler word of similar
                   length. Control-word preference order:
                     tier 1 — all k occurrences sit inside the k injected
                              sentences (frequency- AND position-matched, the
                              cleanest control: same sentences, same slots,
                              only the word's referent differs);
                     tier 2 — exactly k occurrences anywhere in the document,
                              at least one inside an injected sentence;
                     tier 3 — exactly k occurrences anywhere in the document.
                   The chosen tier is recorded per state; exclusions guarantee
                   the control word is not the keyword, not part of the
                   distractor action / phrase / verb, not in the correct action,
                   not in the task query, not the keyword's neutral replacement,
                   not an environment noun (any receptacle/object — those are
                   arguments of other executable actions), not derived from an
                   action-verb or keyword stem (cleaning, cooling, safety...),
                   and not a template machinery or closed-class function word
                   (replacing those would wreck sentence grammar — see
                   CONTROL_STOPWORDS).

Expected signature if the keyword is causal:
    removing the keyword (original -> counterfactual) drops P(distractor)
    substantially — MORE in Text than in Visual — while removing the
    frequency-matched control word (original -> matched_control) changes
    P(distractor) about as little as measurement noise. The difference of the
    two paired drops (difference-in-differences) is the causal statistic; it
    cannot be explained by "deleting any word changes the probability".

Outputs per state (under --output_dir):
  text/{state_id}/original_k{k}.md            (+ .parts.json for rendering)
  text/{state_id}/counterfactual_k{k}.md      (+ .parts.json)
  text/{state_id}/matched_control_k{k}.md     (+ .parts.json)
  construction_meta.json                      control-word choices, tiers,
                                              measured counts and assertions
"""
import re
import json
import argparse
from pathlib import Path
from collections import Counter

import skill_lexicon as lex

VERSIONS = ("original", "counterfactual", "matched_control")

# Neutral replacement pool for the matched-control word: semantically empty
# document words that name no action, no object and no receptacle. The one
# closest in character length to the control word is used (ties alphabetical),
# preferring a word the document does not already contain, so the replacement
# cannot piggy-back on an existing association.
CONTROL_REPLACEMENTS = [
    "aspect", "detail", "element", "factor", "feature", "matter",
    "respect", "component", "consideration", "particular",
]

# Words that must never be picked as the control word:
#   * machinery words of the injection/filler templates themselves — removing
#     them would dismantle the sentence frame rather than one referent;
#   * English function words — replacing e.g. every "that" with a noun wrecks
#     the grammar of four sentences at once, which is far more invasive than
#     the one-for-one noun→noun counterfactual and would confound Δmc with a
#     fluency shock.
CONTROL_STOPWORDS = {
    # template machinery
    "action", "command", "commands", "step", "steps", "task", "tasks",
    "workflow", "documented", "sequence", "ordering", "skill", "skills",
    "note", "item", "example", "pattern", "reminder", "hint", "summary",
    "reference", "trajectory", "trajectories", "agent", "agents",
    "nothing", "happens", "reissuing", "issuing", "issued", "issue",
    "operational", "checklist", "recovery", "precondition", "frequently",
    "worked", "common", "standard", "recommended", "guidance", "situation",
    "ordering", "prerequisite", "optional", "detail", "manner", "usual",
    "applicable", "described", "noted", "above", "where", "throughout",
    # function words / closed class
    "that", "this", "these", "those", "with", "when", "then", "than",
    "them", "they", "their", "there", "here", "what", "which", "whether",
    "while", "have", "has", "had", "been", "being", "will", "would",
    "can", "could", "should", "shall", "may", "might", "must", "your",
    "our", "its", "from", "into", "onto", "upon", "because", "before",
    "after", "above", "below", "over", "under", "most", "more", "less",
    "such", "some", "any", "many", "much", "also", "only", "just",
    "even", "still", "already", "normally", "usually", "rather", "each",
    "every", "both", "either", "neither", "other", "others", "same",
    "point", "kind", "sort", "thing", "things", "part", "parts", "way",
    "ways", "one", "two", "first", "next", "later", "early", "single",
    "keep", "keeps", "reach", "reaches", "assume", "assumes", "treat",
    "belong", "belongs", "confirm", "complete", "completed", "depend",
    "proceed", "proceeds", "continue", "continuing", "place", "places",
    "show", "shows", "send", "answer", "answers", "putting", "having",
    "doing", "done", "follow", "improvising", "verification", "careful",
    "rereading", "preconditions", "stages", "stage", "remain", "remaining",
    "front", "near", "list", "reading", "plan", "plans", "run", "runs",
    "habit", "advice", "repeated", "why", "how", "who", "whom", "whose",
    # common documentation verbs — replacing a verb slot with a neutral NOUN
    # ("aspect", "detail", ...) breaks grammar outright, which would make the
    # control manipulation more invasive than the noun→noun counterfactual
    "verify", "verified", "verifies", "ensure", "ensures", "identify",
    "identified", "locate", "located", "determine", "determined",
    "maintain", "maintains", "provide", "provides", "provided",
    "avoid", "avoids", "require", "requires", "required", "indicate",
    "indicates", "specify", "specifies", "specified", "obtain", "obtains",
    "select", "selects", "selected", "apply", "applies", "attempt",
    "attempts", "repeat", "repeats", "review", "reviews", "inspect",
    "inspects", "execute", "executes", "executed", "navigate", "navigates",
    "perform", "performs", "performed", "check", "checks", "checked",
    "include", "includes", "including", "consider", "considers",
    "ensure", "ensuring", "refer", "refers", "match", "matches",
}

# Noun-proxy: a candidate is preferred when it occurs at least once right
# after a determiner/preposition — those are the slots where swapping in a
# neutral singular noun ("aspect", "detail", ...) stays grammatical, exactly
# like the keyword→appliance counterfactual does.
_NOUN_PROXY_RE_CACHE = {}


def noun_proxy(text, word):
    import re as _re
    pat = _NOUN_PROXY_RE_CACHE.get(word)
    if pat is None:
        pat = _re.compile(
            r"\b(?:the|a|an|of|in|into|on|for|with|from|several|many|most|some|"
            r"any|its|their|another)\s+" + _re.escape(word) + r"s?\b", _re.I)
        _NOUN_PROXY_RE_CACHE[word] = pat
    return bool(pat.search(text))


def replace_word(text, word, replacement):
    """Whole-word, case- and plural-preserving one-for-one replacement
    (identical semantics to skill_lexicon.scrub_keyword, for an arbitrary
    word/replacement pair)."""
    def _sub(m):
        tok = m.group(0)
        out = replacement + ("s" if tok.lower().endswith("s")
                             and not word.endswith("s") else "")
        return out[:1].upper() + out[1:] if tok[:1].isupper() else out
    return re.sub(rf"\b{re.escape(word)}s?\b", _sub, text, flags=re.I)


def differing_lines(a, b):
    la, lb = a.split("\n"), b.split("\n")
    if len(la) != len(lb):
        return None                      # structure changed — a bug, gated below
    return sum(1 for x, y in zip(la, lb) if x != y)


def injection_sentences(action, keyword, injected_slots):
    pairs = lex.slot_texts(action, keyword=keyword)
    return [pairs[i][0] for i in injected_slots]


def choose_control_word(doc, injections, k, keyword, action, phrase,
                        verb_surface, correct_action, query, neutral_for_keyword):
    """-> (word, tier, candidates_log) or (None, None, candidates_log)."""
    forbidden = set(lex.content_words(
        " ".join([keyword, action, phrase, verb_surface, correct_action,
                  neutral_for_keyword or ""])))
    forbidden |= CONTROL_STOPWORDS
    # The neutral words the base scrub already substituted IN (chiller for
    # fridge, appliance for microwave, ...) refer to the same concepts as the
    # distractor family — ablating one of those would not be an unrelated-word
    # control.
    forbidden |= {v.lower() for v in lex.SCRUB_REPLACEMENT.values()}
    # Every environment noun is the argument of SOME executable action
    # (fridge -> open/cool, cellphone -> take, ...), so ablating one could
    # shift action probabilities through a different route than the distractor
    # under test — not an "unrelated" word. Same for words built on an action
    # verb stem (cleaning/cooling/heating/...) or on the keyword itself
    # (safety for safe): semantically entangled with actions.
    forbidden |= set(lex.RECEPTACLE_NOUNS) | set(lex.OBJECT_NOUNS)
    verb_stems = sorted(({v for v in lex.VERB_SURFACE.values() if " " not in v}
                         | {keyword.lower()}), key=len, reverse=True)

    def verb_entangled(w):
        return any(w.startswith(v) and len(w) > len(v) for v in verb_stems)

    vocab = Counter(lex.content_words(doc))

    def occ_in_injections(w):
        return sum(lex.count_word(s, w) for s in injections)

    tiers = {1: [], 2: [], 3: []}
    for w in sorted(vocab):
        if len(w) < 5 or w in forbidden or verb_entangled(w):
            continue
        if lex.count_word(correct_action, w) or lex.count_word(query, w):
            continue
        c_doc = lex.count_word(doc, w)
        if c_doc != k:                   # frequency must MATCH the keyword's
            continue
        c_inj = occ_in_injections(w)
        if c_inj == k:
            tiers[1].append(w)
        elif c_inj >= 1:
            tiers[2].append(w)
        else:
            tiers[3].append(w)

    proxy = {w: noun_proxy(doc, w) for t in tiers.values() for w in t}
    log = {
        "n_frequency_matched": sum(len(v) for v in tiers.values()),
        "tier1": tiers[1], "tier2": tiers[2], "tier3_n": len(tiers[3]),
        "noun_proxy": {w: proxy[w] for w in proxy},
    }
    # Selection order: noun-like candidates FIRST (a noun→noun swap stays
    # grammatical, exactly like the keyword→appliance counterfactual), then
    # position-matched tier, then length. In practice tier 1 is empty once the
    # template/padding machinery words are excluded, so nothing is lost by
    # letting grammaticality outrank the tier.
    for tier, want_proxy in ([(t, True) for t in (1, 2, 3)]
                             + [(t, False) for t in (1, 2, 3)]):
        pool = [w for w in tiers[tier] if bool(proxy.get(w)) == want_proxy]
        if pool:
            word = sorted(pool, key=lambda w: (-len(w), w))[0]
            log["chosen_tier"] = tier
            log["chosen_noun_proxy"] = want_proxy
            return word, tier, log
    return None, None, log


def choose_control_replacement(doc, control_word):
    """Closest-length neutral replacement, preferring one absent from the doc."""
    absent = [r for r in CONTROL_REPLACEMENTS if lex.count_word(doc, r) == 0]
    pool = absent or CONTROL_REPLACEMENTS
    return sorted(pool, key=lambda r: (abs(len(r) - len(control_word)), r))[0]


def main(args):
    states_dir = Path(args.states_dir)
    li_variants = Path(args.li_variants_dir)
    out_dir = Path(args.output_dir)
    text_dir = out_dir / "text"
    text_dir.mkdir(parents=True, exist_ok=True)

    levels = [int(x) for x in args.levels.split(",")]
    manifest = json.loads((states_dir / "manifest.json").read_text(encoding="utf-8"))

    # injected_slots per (state, level), exactly as construct_variants ran them
    li_meta_file = li_variants / "construction_meta.json"
    li_meta = {}
    if li_meta_file.exists():
        for m in json.loads(li_meta_file.read_text(encoding="utf-8"))["states"]:
            li_meta[m["state_id"]] = m
    else:
        print(f"[WARN] {li_meta_file} missing — recomputing slot assignment "
              f"with the default 'spread' order")

    print(f"[INFO] {manifest['total_states']} states, levels={levels}, "
          f"versions={VERSIONS}")

    all_meta = []
    problems = Counter()

    for info in manifest["states"]:
        state_id = info["state_id"]
        sf = states_dir / f"{state_id}.json"
        if not sf.exists():
            problems["missing_state_file"] += 1
            continue
        state = json.loads(sf.read_text(encoding="utf-8"))

        keyword = state["distractor_keyword"]
        phrase = state["distractor_phrase"]
        action = state["distractor_action"]
        verb_surface = lex.VERB_SURFACE.get(state["distractor_action_type"],
                                            state["distractor_action_type"])
        neutral_kw = lex.SCRUB_REPLACEMENT.get(keyword, "receptacle")
        state_out = text_dir / state_id
        state_out.mkdir(parents=True, exist_ok=True)

        per_level = {}
        for k in levels:
            src_md = li_variants / "text" / state_id / f"level_{k}.md"
            src_parts = li_variants / "text" / state_id / f"level_{k}.parts.json"
            if not src_md.exists() or not src_parts.exists():
                problems["missing_li_variant"] += 1
                continue
            doc = src_md.read_text(encoding="utf-8")
            parts = json.loads(src_parts.read_text(encoding="utf-8"))

            m_li = li_meta.get(state_id, {}).get("levels", {}).get(str(k), {})
            injected_slots = m_li.get("injected_slots")
            if injected_slots is None:
                injected_slots = lex.slot_assignment(k, lex.N_SLOTS, mode="spread")
            injections = injection_sentences(action, keyword, injected_slots)

            rec = {"injected_slots": injected_slots, "versions": {}}

            def write_version(version, doc_v, parts_v):
                (state_out / f"{version}_k{k}.md").write_text(doc_v, encoding="utf-8")
                (state_out / f"{version}_k{k}.parts.json").write_text(
                    json.dumps(parts_v, indent=2, ensure_ascii=False),
                    encoding="utf-8")

            def parts_mapped(fn):
                out = []
                for p in parts:
                    q = dict(p)
                    q["description"] = fn(p.get("description", ""))
                    q["markdown"] = fn(p["markdown"])
                    out.append(q)
                return out

            # ── original: verbatim copy ──
            write_version("original", doc, parts)
            rec["versions"]["original"] = {
                "keyword_count": lex.count_word(doc, keyword),
                "action_count": doc.count(action),
                "words": lex.word_count(doc),
                "chars": len(doc),
                "differing_lines": 0,
            }

            # ── counterfactual: keyword -> its neutral same-level word ──
            cf_doc = lex.scrub_keyword(doc, keyword, replacement=neutral_kw)
            cf_parts = parts_mapped(lambda t: lex.scrub_keyword(t, keyword,
                                                                replacement=neutral_kw))
            write_version("counterfactual", cf_doc, cf_parts)
            dl = differing_lines(doc, cf_doc)
            rec["versions"]["counterfactual"] = {
                "replacement": neutral_kw,
                "keyword_count": lex.count_word(cf_doc, keyword),
                "action_count": cf_doc.count(action),
                "words": lex.word_count(cf_doc),
                "chars": len(cf_doc),
                "differing_lines": dl,
            }
            if lex.count_word(cf_doc, keyword) != 0:
                problems["cf_keyword_not_zero"] += 1
            if lex.word_count(cf_doc) != lex.word_count(doc):
                problems["cf_word_count_changed"] += 1
            if dl != k:
                problems["cf_differing_lines_ne_k"] += 1

            # ── matched control: frequency-matched unrelated word -> neutral ──
            ctrl, tier, clog = choose_control_word(
                doc, injections, k, keyword, action, phrase, verb_surface,
                state.get("correct_action", ""), state.get("query", ""), neutral_kw)
            if ctrl is None:
                problems["no_control_word"] += 1
                rec["versions"]["matched_control"] = {
                    "status": "skipped_no_control_word", "candidates": clog}
            else:
                repl = choose_control_replacement(doc, ctrl)
                mc_doc = replace_word(doc, ctrl, repl)
                mc_parts = parts_mapped(lambda t: replace_word(t, ctrl, repl))
                write_version("matched_control", mc_doc, mc_parts)
                dl = differing_lines(doc, mc_doc)
                rec["versions"]["matched_control"] = {
                    "control_word": ctrl,
                    "control_replacement": repl,
                    "tier": tier,
                    "candidates": clog,
                    "control_count_before": lex.count_word(doc, ctrl),
                    "control_count_after": lex.count_word(mc_doc, ctrl),
                    "keyword_count": lex.count_word(mc_doc, keyword),
                    "action_count": mc_doc.count(action),
                    "words": lex.word_count(mc_doc),
                    "chars": len(mc_doc),
                    "differing_lines": dl,
                }
                if lex.count_word(mc_doc, ctrl) != 0:
                    problems["mc_control_not_zero"] += 1
                if lex.count_word(mc_doc, keyword) != k:
                    problems["mc_keyword_count_changed"] += 1
                if lex.word_count(mc_doc) != lex.word_count(doc):
                    problems["mc_word_count_changed"] += 1
                if dl != k:
                    problems["mc_differing_lines_ne_k"] += 1

            per_level[str(k)] = rec

        all_meta.append({
            "state_id": state_id,
            "distractor_action": action,
            "distractor_keyword": keyword,
            "keyword_neutral_replacement": neutral_kw,
            "levels": per_level,
        })

    meta = {
        "total_states": len(all_meta),
        "levels": levels,
        "versions": list(VERSIONS),
        "li_variants_dir": str(li_variants),
        "problems": dict(problems),
        "states": all_meta,
    }
    (out_dir / "construction_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 68)
    print(f"Constructed counterfactual documents for {len(all_meta)} states "
          f"× levels {levels} -> {text_dir}")
    tiers = Counter()
    for m in all_meta:
        for lvl, rec in m["levels"].items():
            mc = rec["versions"].get("matched_control", {})
            tiers[f"k{lvl}_tier{mc.get('tier', 'none')}"] += 1
    print(f"  control-word tiers: {dict(sorted(tiers.items()))}")
    print(f"  problems: {dict(problems) or 'none'}")
    hard = [p for p in ("cf_keyword_not_zero", "cf_word_count_changed",
                        "mc_control_not_zero", "mc_keyword_count_changed",
                        "mc_word_count_changed", "missing_li_variant")
            if problems.get(p)]
    if hard:
        raise SystemExit(f"[FAIL] hard construction problems: {hard} — see "
                         f"construction_meta.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--states_dir", required=True,
                   help="lexical-interference states dir (symlinked as ./states)")
    p.add_argument("--li_variants_dir", required=True,
                   help="lexical-interference variants dir holding text/ + "
                        "construction_meta.json")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--levels", default="4,8",
                   help="high-exposure levels to ablate (the k where the "
                        "lexical-interference effect is largest)")
    main(p.parse_args())
