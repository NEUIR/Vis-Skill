"""
Patch spacy.load to avoid downloading en_core_web_sm model.

WebShop's reward function uses spacy only for POS tagging in title matching
(goal.py get_type_reward). Since we use human_goals mode, this is only used
in reward computation, not goal generation. A simple heuristic tagger suffices.
"""
import spacy
import re as _re

_STOP_WORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
    'has', 'have', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'can', 'shall', 'not', 'no', 'if', 'then',
    'than', 'that', 'this', 'these', 'those', 'it', 'its', 'i', 'my',
    'your', 'his', 'her', 'our', 'their', 'we', 'you', 'he', 'she', 'they',
    'me', 'him', 'us', 'them', 'who', 'what', 'which', 'where', 'when',
    'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
    'some', 'such', 'only', 'very', 'so', 'too', 'also', 'just',
}


class _Token:
    def __init__(self, text, pos):
        self.text = text
        self.pos_ = pos


class _SimpleNlp:
    def __call__(self, text):
        tokens = _re.findall(r'\b\w+\b', text)
        result = []
        for t in tokens:
            lower = t.lower()
            if lower in _STOP_WORDS:
                pos = 'DET'
            elif t[0].isupper() and len(t) > 1:
                pos = 'PROPN'
            else:
                pos = 'NOUN'
            result.append(_Token(t, pos))
        return result


def patch_spacy():
    _original_load = spacy.load

    def _patched_load(name, **kwargs):
        if 'en_core_web' in name:
            return _SimpleNlp()
        return _original_load(name, **kwargs)

    spacy.load = _patched_load
