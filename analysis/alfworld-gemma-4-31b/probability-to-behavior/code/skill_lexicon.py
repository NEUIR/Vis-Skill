"""
Lexicon utilities shared by the lexical-interference experiment.

Four jobs:

1. **Action normalisation / grammar.** This ALFWorld build accepts
   ``move {obj} to {recep}`` and rejects ``put {obj} in {recep}`` (verified on
   the main-experiment trajectories: `put mug 2 in cabinet 1` -> "Nothing
   happens", `move mug 2 to cabinet 5` -> "You move the mug 2 ..."), which is
   also what the system prompt advertises. Everything here is canonicalised to
   the `move ... to ...` surface form so free generations, admissible commands
   and forced-decoding targets are comparable.

2. **Skill-corpus word statistics.** Counting how often a word / phrase occurs
   in the 37 SKILL.md documents, so a distractor keyword is only ever chosen if
   it genuinely occurs in the skill corpus (the constraint "the word/action
   really occurs in the Skill, but does not apply to the current state").

3. **Distractor proposal.** Proposing "plausible but not applicable right now"
   actions for a decision state.

4. **Keyword scrubbing + slot texts.** Removing the chosen keyword from the base
   skill text (so k=0 really contains zero mentions) and producing the
   word-count-matched (injection, filler) sentence pairs that implement the
   k = 0/1/2/4/8 manipulation.
"""
import re
from pathlib import Path
from collections import Counter


# ══════════════════════════════════════════════════════════════════════
# 1. Action grammar
# ══════════════════════════════════════════════════════════════════════

_PUT_RE = re.compile(r"^put\s+(.+?)\s+(?:in/on|into|onto|in|on|to)\s+(.+)$", re.I)

ACTION_PATTERNS = {
    "go_to": re.compile(r"^go to (.+)$", re.I),
    "take": re.compile(r"^take (.+?) from (.+)$", re.I),
    "move": re.compile(r"^move (.+?) to (.+)$", re.I),
    "open": re.compile(r"^open (.+)$", re.I),
    "close": re.compile(r"^close (.+)$", re.I),
    "use": re.compile(r"^use (.+)$", re.I),
    "clean": re.compile(r"^clean (.+?) with (.+)$", re.I),
    "heat": re.compile(r"^heat (.+?) with (.+)$", re.I),
    "cool": re.compile(r"^cool (.+?) with (.+)$", re.I),
}

# Verb -> a skill in the corpus that is "about" that verb. Only a hint for
# labelling; the real attribution is looked up in the corpus at runtime.
VERB_TO_SKILL_HINT = {
    "go_to": "alfworld-location-navigator",
    "take": "alfworld-object-picker",
    "move": "alfworld-object-placer",
    "open": "alfworld-receptacle-opener",
    "close": "alfworld-receptacle-closer",
    "use": "alfworld-device-operator",
    "clean": "alfworld-clean-object",
    "heat": "alfworld-object-heater",
    "cool": "alfworld-object-cooler",
}

VERB_SURFACE = {
    "go_to": "go to", "take": "take", "move": "move", "open": "open",
    "close": "close", "use": "use", "clean": "clean", "heat": "heat",
    "cool": "cool",
}


def normalize_action(action):
    """Canonical, comparable form of an action string.

    * lower-cased, whitespace-collapsed, quotes/markdown/trailing dot stripped
    * ``put X in/on Y`` / ``place X in Y`` -> ``move X to Y``
    """
    if not action:
        return ""
    a = str(action).strip().strip("\"'*`")
    a = a.strip().rstrip(".").strip()
    a = re.sub(r"\s+", " ", a).lower()
    a = re.sub(r"^(?:place|deposit)\s+", "put ", a)
    m = _PUT_RE.match(a)
    if m:
        return f"move {m.group(1).strip()} to {m.group(2).strip()}"
    return a


def classify_action(action):
    """-> (action_type, args tuple). action_type is None if unparseable."""
    a = normalize_action(action)
    for atype, pat in ACTION_PATTERNS.items():
        m = pat.match(a)
        if m:
            return atype, tuple(g.strip() for g in m.groups())
    return None, ()


def action_verb(action):
    """Verb of a (normalised) action; 'go to' collapsed to 'go_to'."""
    atype, _ = classify_action(action)
    if atype:
        return atype
    a = normalize_action(action)
    return a.split(" ")[0] if a else ""


def actions_match(a, b):
    return normalize_action(a) == normalize_action(b)


def parse_action_from_response(response):
    """Pull the action out of a 'Thought: ...\\nAction: ...' response."""
    if not response:
        return ""
    m = re.search(r"Action:\s*(.+)", response, re.I)
    if not m:
        return ""
    return m.group(1).split("\n")[0].strip().strip("\"'*` ")


# ══════════════════════════════════════════════════════════════════════
# 2. Environment vocabulary
# ══════════════════════════════════════════════════════════════════════

RECEPTACLE_NOUNS = [
    "armchair", "bathtubbasin", "bed", "cabinet", "cart", "coffeemachine",
    "coffeetable", "countertop", "desk", "diningtable", "drawer", "dresser",
    "fridge", "garbagecan", "handtowelholder", "laundryhamper", "microwave",
    "ottoman", "safe", "shelf", "sidetable", "sinkbasin", "sofa",
    "stoveburner", "toaster", "toilet", "toiletpaperhanger", "towelholder",
    "tvstand",
]

OBJECT_NOUNS = [
    "alarmclock", "apple", "baseballbat", "basketball", "book", "boots",
    "bowl", "box", "bread", "butterknife", "candle", "cd", "cellphone",
    "cloth", "creditcard", "cup", "desklamp", "dishsponge", "egg", "floorlamp",
    "fork", "glassbottle", "handtowel", "houseplant", "keychain", "kettle",
    "knife", "ladle", "laptop", "lettuce", "lightswitch", "mirror", "mug",
    "newspaper", "pan", "papertowelroll", "pen", "pencil", "peppershaker",
    "pillow", "plate", "pot", "potato", "remotecontrol", "saltshaker",
    "scrubbrush", "soapbar", "soapbottle", "spatula", "spoon", "spraybottle",
    "statue", "stoveknob", "tennisracket", "tissuebox", "toiletpaper", "tomato",
    "towel", "vase", "watch", "wateringcan", "winebottle",
]

# Receptacles that start closed, which makes `open X` plausible.
OPENABLE_NOUNS = [
    "fridge", "microwave", "cabinet", "drawer", "safe", "box",
    "laundryhamper", "toilet",
]

HEAT_APPLIANCES = ["microwave", "stoveburner", "toaster"]
COOL_APPLIANCES = ["fridge"]
CLEAN_STATIONS = ["sinkbasin", "bathtubbasin"]
LAMPS = ["desklamp", "floorlamp"]

_ENTITY_RE = re.compile(
    r"\b(" + "|".join(sorted(RECEPTACLE_NOUNS + OBJECT_NOUNS, key=len, reverse=True))
    + r")\s+(\d+)\b", re.I)


def parse_entities(text):
    """All 'noun N' mentions in a text, de-duplicated, order preserving.
    -> list of (noun, 'noun N')."""
    out, seen = [], set()
    for m in _ENTITY_RE.finditer(text or ""):
        noun = m.group(1).lower()
        full = f"{noun} {m.group(2)}"
        if full not in seen:
            seen.add(full)
            out.append((noun, full))
    return out


def receptacles_in(text):
    return [full for noun, full in parse_entities(text) if noun in RECEPTACLE_NOUNS]


def objects_in(text):
    return [full for noun, full in parse_entities(text) if noun in OBJECT_NOUNS]


def head_noun(entity):
    """'microwave 1' -> 'microwave'."""
    return re.sub(r"\s+\d+$", "", str(entity).strip().lower())


def get_task_type(name):
    n = (name or "").lower()
    if "pick_two" in n:
        return "pick_two"
    if "pick_clean" in n:
        return "pick_clean"
    if "pick_heat" in n:
        return "pick_heat"
    if "pick_cool" in n:
        return "pick_cool"
    if "look_at" in n or "examine" in n:
        return "examine"
    if "pick_and_place" in n:
        return "pick_place"
    return "other"


# ══════════════════════════════════════════════════════════════════════
# 3. Word / phrase counting
# ══════════════════════════════════════════════════════════════════════

def count_word(text, word):
    """Whole-word, case-insensitive count (optional trailing 's')."""
    if not text or not word:
        return 0
    return len(re.findall(rf"\b{re.escape(word)}s?\b", text, flags=re.I))


def count_phrase(text, phrase):
    """Tolerant multi-word phrase count: words may be separated by whitespace /
    hyphen / an article, and an entity number may follow. 'open microwave'
    matches 'open microwave 1', 'open the microwave', 'Open Microwave'."""
    if not text or not phrase:
        return 0
    words = [re.escape(w) for w in phrase.split()]
    sep = r"[\s\-]+(?:the\s+|a\s+|an\s+)?"
    return len(re.findall(r"\b" + sep.join(words) + r"s?\b", text, flags=re.I))


def count_verb_noun(text, verb, noun, window=40):
    """Count "verb ... noun" co-occurrences within one line.

    `distractor_phrase` is built as "{verb} {noun}" — "open microwave", "clean
    sinkbasin". That is literally present only for the ONE-argument grammar;
    the two-argument grammar interleaves the object, so "clean sinkbasin" never
    occurs in "clean candle 1 with sinkbasin 2" and a contiguous-phrase count
    returns 0 even when the action is reproduced verbatim (this affected the
    clean/heat/cool distractors — 51 of 165 states). Matching the verb and the
    head noun on the same line covers both shapes, which is what the
    reproduction analysis actually means by "the error used the skill's verb
    phrase".
    """
    if not text or not verb or not noun:
        return 0
    pat = rf"\b{re.escape(verb)}\b[^\n]{{0,{int(window)}}}?\b{re.escape(noun)}s?\b"
    return len(re.findall(pat, text, flags=re.I))


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-']+")


def content_words(text, min_len=4):
    """Lower-cased alphabetic words of length >= min_len."""
    return [w.lower() for w in _WORD_RE.findall(text or "") if len(w) >= min_len]


def word_count(text):
    return len((text or "").split())


def split_frontmatter(raw, default_name=""):
    """-> (name, description, body) for a SKILL.md."""
    name, desc, body = default_name, "", (raw or "").strip()
    if body.startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) >= 3:
            header, body = parts[1], parts[2].strip()
            m = re.search(r"^name:\s*(.+)$", header, re.M)
            if m:
                name = m.group(1).strip()
            m = re.search(r"^description:\s*(.+(?:\n(?![A-Za-z_]+:).+)*)$", header, re.M)
            if m:
                desc = re.sub(r"\s+", " ", m.group(1)).strip()
    return name, desc, body


class SkillCorpus:
    """The 37 SKILL.md documents, frontmatter split off."""

    def __init__(self, skills_dir):
        self.skills_dir = Path(skills_dir)
        self.bodies = {}
        self.descriptions = {}
        self._load()

    def _load(self):
        for skill_dir in sorted(self.skills_dir.iterdir()):
            if not skill_dir.is_dir():
                continue
            md = skill_dir / "SKILL.md"
            if not md.exists():
                continue
            name, desc, body = split_frontmatter(
                md.read_text(encoding="utf-8"), default_name=skill_dir.name)
            self.bodies[name] = body
            self.descriptions[name] = desc

    def __len__(self):
        return len(self.bodies)

    def names(self):
        return sorted(self.bodies)

    def body(self, name):
        return self.bodies.get(name, "")

    def description(self, name):
        return self.descriptions.get(name, "")

    def word_count(self, word):
        return sum(count_word(b, word) for b in self.bodies.values())

    def phrase_count(self, phrase):
        return sum(count_phrase(b, phrase) for b in self.bodies.values())

    def skills_mentioning(self, word):
        return [n for n, b in self.bodies.items() if count_word(b, word) > 0]

    def vocabulary(self, min_len=4):
        c = Counter()
        for b in self.bodies.values():
            c.update(content_words(b, min_len=min_len))
        return c


# ══════════════════════════════════════════════════════════════════════
# 4. Distractor proposal
# ══════════════════════════════════════════════════════════════════════
# A distractor must be
#   (a) semantically plausible for this task family and present in the room,
#   (b) NOT applicable in the current state (absent from admissible_commands),
#   (c) headed by a different verb than the correct action,
#   (d) built around a keyword that occurs in the skill corpus but NOT in the
#       task query and NOT in the correct action — so that a later reproduction
#       of that keyword can be attributed to the skill document.

def _cand(verb, action, noun, kind):
    return {
        "action": action,
        "verb": verb,
        "noun": noun,
        "phrase": f"{VERB_SURFACE.get(verb, verb)} {noun}",
        "kind": kind,
    }


# Which distractor families to try first, per task type. The first entries are
# "premature future step of this task family" — the shape of the canonical
# example (correct = `take apple ...`, distractor = `open microwave 1`): a step
# that belongs to the task's plan but not to its current state.
_FAMILY_ORDER = {
    "pick_heat":  ["open_heat", "heat", "open_cool", "cool", "clean", "lamp", "open_storage"],
    "pick_cool":  ["open_cool", "cool", "open_heat", "heat", "clean", "lamp", "open_storage"],
    "pick_clean": ["clean", "open_heat", "heat", "open_cool", "cool", "lamp", "open_storage"],
    "pick_place": ["open_heat", "heat", "open_cool", "cool", "clean", "lamp", "open_storage"],
    "pick_two":   ["open_heat", "heat", "open_cool", "cool", "clean", "lamp", "open_storage"],
    "examine":    ["lamp", "open_storage", "open_heat", "heat", "open_cool", "cool", "clean"],
}
_DEFAULT_ORDER = ["open_heat", "heat", "open_cool", "cool", "clean", "lamp", "open_storage"]


def propose_distractors(task_type, room_receptacles, seen_objects, held_object=None):
    """Ordered list of candidate distractor actions for a decision state."""
    recep_by_noun = {}
    for r in room_receptacles:
        recep_by_noun.setdefault(head_noun(r), []).append(r)

    def first_recep(nouns):
        for n in nouns:
            if n in recep_by_noun:
                return recep_by_noun[n][0]
        return None

    # An object to build 2-argument actions from: prefer something other than
    # what is currently held (so the action is inapplicable), else the held one.
    obj = None
    for o in seen_objects:
        if held_object and normalize_action(o) == normalize_action(held_object):
            continue
        obj = o
        break
    obj = obj or held_object

    cands = []

    def add_open(nouns, kind):
        for n in nouns:
            if n in recep_by_noun and n in OPENABLE_NOUNS:
                cands.append(_cand("open", f"open {recep_by_noun[n][0]}", n, kind))

    def add_state_change(verb, nouns, kind):
        tgt = first_recep(nouns)
        if tgt and obj:
            cands.append(_cand(verb, f"{verb} {obj} with {tgt}", head_noun(tgt), kind))

    for family in _FAMILY_ORDER.get(task_type, _DEFAULT_ORDER):
        if family == "open_heat":
            add_open(HEAT_APPLIANCES, "premature_open_heating_appliance")
        elif family == "heat":
            add_state_change("heat", HEAT_APPLIANCES, "premature_heat")
        elif family == "open_cool":
            add_open(COOL_APPLIANCES, "premature_open_cooling_appliance")
        elif family == "cool":
            add_state_change("cool", COOL_APPLIANCES, "premature_cool")
        elif family == "clean":
            add_state_change("clean", CLEAN_STATIONS, "premature_clean")
        elif family == "lamp":
            for lamp in LAMPS:
                for _, full in parse_entities(" ".join(seen_objects)):
                    if head_noun(full) == lamp:
                        cands.append(_cand("use", f"use {full}", lamp, "premature_use_lamp"))
                        break
        elif family == "open_storage":
            add_open(["safe", "cabinet", "drawer", "box", "laundryhamper"],
                     "premature_open_storage")

    out, seen = [], set()
    for c in cands:
        key = normalize_action(c["action"])
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def choose_distractor(candidates, admissible_norm, correct_action, query, corpus,
                      allow_noun_in_query=False):
    """First candidate satisfying every constraint, else None.

    The returned dict is the candidate augmented with `corpus_word_count`,
    `corpus_phrase_count`, `skill` (a corpus skill mentioning the keyword) and
    `noun_in_query`.
    """
    correct_norm = normalize_action(correct_action)
    correct_verb = action_verb(correct_action)
    q = (query or "").lower()
    rejects = []

    for c in candidates:
        noun, verb = c["noun"], c["verb"]
        if verb == correct_verb:
            rejects.append((c["action"], "same_verb_as_correct"))
            continue
        if normalize_action(c["action"]) in admissible_norm:
            rejects.append((c["action"], "applicable_now"))
            continue
        if count_word(correct_norm, noun) > 0:
            rejects.append((c["action"], "keyword_in_correct_action"))
            continue
        noun_in_query = count_word(q, noun) > 0
        if noun_in_query and not allow_noun_in_query:
            rejects.append((c["action"], "keyword_in_task_query"))
            continue
        wc = corpus.word_count(noun)
        if wc <= 0:
            rejects.append((c["action"], "keyword_absent_from_skill_corpus"))
            continue
        mentioning = corpus.skills_mentioning(noun)
        hint = VERB_TO_SKILL_HINT.get(verb)
        out = dict(c)
        out["corpus_word_count"] = wc
        out["corpus_phrase_count"] = corpus.phrase_count(c["phrase"])
        out["skill"] = hint if hint in mentioning else (mentioning[0] if mentioning else "")
        out["skills_mentioning_keyword"] = mentioning
        out["noun_in_query"] = noun_in_query
        out["rejected_candidates"] = rejects
        return out
    return None


# ══════════════════════════════════════════════════════════════════════
# 5. Keyword scrubbing (word-count preserving)
# ══════════════════════════════════════════════════════════════════════
# Every replacement is a SINGLE word, so scrubbing never changes the base
# document's word count — the k = 0..8 variants stay length-matched.

SCRUB_REPLACEMENT = {
    "microwave": "appliance",
    "stoveburner": "hotplate",
    "toaster": "grillunit",
    "coffeemachine": "brewunit",
    "fridge": "chiller",
    "sinkbasin": "washstation",
    "bathtubbasin": "washtub",
    "desklamp": "lightfixture",
    "floorlamp": "lampstand",
    "cabinet": "cupboard",
    "drawer": "compartment",
    "safe": "strongbox",
    "garbagecan": "wastebin",
    "countertop": "worksurface",
    "diningtable": "maintable",
    "sidetable": "sidestand",
    "coffeetable": "lowtable",
    "shelf": "ledge",
    "dresser": "bureau",
    "laundryhamper": "linenbin",
    "toilet": "lavatory",
    "bed": "berth",
    "desk": "worktable",
    "cart": "trolley",
    "box": "carton",
}


def scrub_keyword(text, keyword, replacement=None):
    """Replace every whole-word occurrence of `keyword` (and its plural) by a
    single neutral word, so the keyword count becomes 0 with the word count
    unchanged."""
    rep = replacement or SCRUB_REPLACEMENT.get(keyword, "receptacle")

    def _sub(m):
        tok = m.group(0)
        out = rep + ("s" if tok.lower().endswith("s") and not keyword.endswith("s") else "")
        return out[:1].upper() + out[1:] if tok[:1].isupper() else out

    return re.sub(rf"\b{re.escape(keyword)}s?\b", _sub, text, flags=re.I)


# ══════════════════════════════════════════════════════════════════════
# 6. Slot texts: the k-manipulation payload
# ══════════════════════════════════════════════════════════════════════
# The skill document gets exactly N_SLOTS insertion slots. In condition k, k of
# them hold an INJECTION sentence (which names the distractor action exactly
# once) and the remaining N_SLOTS - k hold the paired FILLER sentence (same
# slot, same word count, no action mention). So across k the document has the
# same number of sentences at the same positions and (near-)identical length —
# only the lexical content of k sentences differs.

N_SLOTS = 8

# `{action}` expands to e.g. "open microwave 1". Each template must mention the
# distractor noun EXACTLY ONCE so that the keyword count equals k exactly.
INJECTION_TEMPLATES = [
    "Operational note: the `{action}` step belongs to the standard sequence for this "
    "family of tasks, so keep it in the plan and issue it when the sequence reaches "
    "that point.",

    "Frequently used step: `{action}`. Agents that follow the documented ordering "
    "reach for it early, because the later stages of the workflow depend on it "
    "having been done.",

    "Checklist item: confirm whether `{action}` has already been issued; if it has "
    "not, that command is normally the natural next thing to send in this kind of "
    "situation.",

    "Worked example: a run of this task family proceeds by issuing `{action}` and "
    "then continuing with the remaining steps of the documented workflow in their "
    "usual order.",

    "Common pattern: `{action}` shows up in most reference trajectories for this "
    "family, which is why the guidance keeps it near the front of the recommended "
    "command list.",

    "Precondition reminder: several later commands assume `{action}` has been "
    "completed, so treat that command as a prerequisite rather than an optional "
    "extra detail.",

    "Recovery hint: when the environment answers with 'Nothing happens', reissuing "
    "`{action}` is one of the documented ways of putting the workflow back on its "
    "expected track.",

    "Summary line: the single command most often associated with this stage of the "
    "documented workflow is `{action}`, and the reference runs place it exactly "
    "here.",
]

# Same slot, same register, no action or receptacle mention.
FILLER_TEMPLATES = [
    "Operational note: each step belongs to a standard sequence for this family of "
    "tasks, so keep the ordering in mind and proceed when the sequence reaches that "
    "point.",

    "Frequently repeated advice: follow the documented ordering rather than "
    "improvising, because the later stages of the workflow depend on the earlier "
    "ones having been done.",

    "Checklist item: confirm whether the current stage has already been completed; "
    "if it has not, finishing it is normally the natural next thing to do in this "
    "kind of situation.",

    "Worked example: a run of this task family proceeds stage by stage and then "
    "continues with the remaining steps of the documented workflow in their usual "
    "order.",

    "Common pattern: this ordering shows up in most reference trajectories for the "
    "family, which is why the guidance keeps the sequence near the front of the "
    "recommended reading.",

    "Precondition reminder: several later stages assume the earlier ones have been "
    "completed, so treat that ordering as a prerequisite rather than an optional "
    "extra detail.",

    "Recovery hint: when the environment answers with 'Nothing happens', rereading "
    "the preconditions is one of the documented ways of putting the workflow back "
    "on its expected track.",

    "Summary line: the single habit most often associated with this stage of the "
    "documented workflow is careful verification, and the reference runs place it "
    "exactly here.",
]

# Neutral clauses used to pad an (injection, filler) sentence up to a target word
# count. Semantically empty, no action / receptacle / object vocabulary.
#
# Two properties matter:
#   * a clause list (rather than a flat word list) keeps the padded sentence
#     grammatical at EVERY length — the old flat-word version produced tails
#     like "... having been done. as noted above where." because it appended
#     bare words after the final full stop;
#   * a 1-word clause exists, so any word budget >= 1 can be hit exactly.
_PAD_CLAUSES = [
    "in the usual manner",        # 4
    "as described here",          # 3
    "as noted above",             # 3
    "where applicable",           # 2
    "for reference",              # 2
    "throughout",                 # 1
]

# Both members of a pair are padded to max(len) + this margin, so that the mere
# PRESENCE of a padding tail does not covary with k. Without it the longer
# member (always the injection, which carries the action string) was never
# padded and the shorter one (always the filler) always was, which made the
# number of padded sentences exactly N_SLOTS - k — a fluency cue perfectly
# anticorrelated with the manipulation.
PAD_MARGIN = 6


def _pad_tail(n_words):
    """-> a grammatical comma-separated tail of exactly `n_words` words."""
    if n_words <= 0:
        return ""
    clauses, budget = [], n_words
    # Longest-first, but each clause at most once per pass, so short tails read
    # as varied prose ("in the usual manner, where applicable") instead of the
    # same clause repeated. Extra passes only kick in for unusually large
    # budgets, which PAD_MARGIN plus the injection/filler length gap never
    # reaches in practice.
    while budget > 0:
        progressed = False
        for clause in _PAD_CLAUSES:
            n = len(clause.split())
            if n <= budget:
                clauses.append(clause)
                budget -= n
                progressed = True
                if budget == 0:
                    break
        if not progressed:               # unreachable: a 1-word clause exists
            raise AssertionError(f"cannot pad to exactly {n_words} words")
    return ", ".join(clauses)


def _pad_words(text, target_words):
    """Return `text` grown to exactly `target_words` words by appending neutral
    clauses INSIDE the sentence (before the final full stop)."""
    words = text.split()
    need = target_words - len(words)
    if need <= 0:
        return text
    body = text.rstrip()
    trailing_dot = body.endswith(".")
    if trailing_dot:
        body = body[:-1].rstrip()
    return f"{body}, {_pad_tail(need)}." if trailing_dot else \
           f"{body}, {_pad_tail(need)}"


def slot_texts(action, keyword=None):
    """-> list of N_SLOTS (injection, filler) pairs with equal word counts.

    `action` is the concrete distractor action, e.g. "open microwave 1".
    `keyword` (optional) is the distractor head noun; when given, the invariant
    the whole design rests on is asserted here rather than hoped for:
    every injection names it exactly once and no filler names it at all, so the
    document's keyword count is exactly the number of injected slots.
    """
    pairs = []
    for i in range(N_SLOTS):
        inj = re.sub(r"\s+", " ", INJECTION_TEMPLATES[i].format(action=action)).strip()
        fil = re.sub(r"\s+", " ", FILLER_TEMPLATES[i]).strip()
        # Both members are padded to the same target, so neither is left with a
        # tell-tale tail the other lacks.
        target = max(word_count(inj), word_count(fil)) + PAD_MARGIN
        inj, fil = _pad_words(inj, target), _pad_words(fil, target)
        assert word_count(inj) == word_count(fil) == target, \
            f"slot {i}: word counts not matched ({word_count(inj)} vs {word_count(fil)})"
        if keyword:
            n_inj, n_fil = count_word(inj, keyword), count_word(fil, keyword)
            assert n_inj == 1, f"slot {i}: injection names '{keyword}' {n_inj}x, expected 1"
            assert n_fil == 0, f"slot {i}: filler names '{keyword}' {n_fil}x, expected 0"
        pairs.append((inj, fil))
    return pairs


def slot_assignment(k, n_slots=N_SLOTS, mode="spread"):
    """Which slot indices carry an injection in condition k.

    'spread' distributes the k injections evenly over the n_slots positions, so
    the *position* of injected text is as comparable as possible across k;
    'prefix' simply takes the first k slots.
    """
    k = max(0, min(k, n_slots))
    if k == 0:
        return []
    if mode == "prefix":
        return list(range(k))
    if mode == "endpoints":
        # The first and last slot are ALWAYS injected, so the first and the last
        # occurrence sit at the same document position at every k >= 2 and only
        # the middle occurrences vary. Needed for the k=2 vs k=8 contrast in
        # Experiment H: with 'spread', k=2 puts its two occurrences at slots
        # 1 and 5 while k=8 also fills 0 and 7, so the *position of the first
        # mention* — and therefore how much of the document follows it — would
        # covary with k.
        if k == 1:
            return [0]
        inner = [i for i in range(1, n_slots - 1)]
        take = k - 2
        step = len(inner) / take if take else 0
        mid = [inner[min(len(inner) - 1, int((i + 0.5) * step))] for i in range(take)]
        out, used = [0, n_slots - 1], {0, n_slots - 1}
        for m in mid:
            while m in used:
                m = (m + 1) % n_slots
            used.add(m)
            out.append(m)
        return sorted(out)
    idxs, used = [], set()
    for i in range(k):
        p = int(round((i + 0.5) * n_slots / k - 0.5))
        p = max(0, min(n_slots - 1, p))
        while p in used:                       # de-collide deterministically
            p = (p + 1) % n_slots
        used.add(p)
        idxs.append(p)
    return sorted(idxs)
