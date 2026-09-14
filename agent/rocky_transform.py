"""Post-process any LLM output into Rocky's speech pattern before it reaches TTS."""

import re

ARTICLES = {"a", "an", "the"}

CONTRACTIONS = {
    "don't": "no", "doesn't": "no", "didn't": "no", "won't": "no", "can't": "no",
    "isn't": "no", "aren't": "no", "wasn't": "no", "weren't": "no", "haven't": "no",
    "hasn't": "no", "hadn't": "no", "wouldn't": "no", "couldn't": "no", "shouldn't": "no",
    "i'm": "I", "you're": "you", "he's": "he", "she's": "she", "it's": "it",
    "we're": "we", "they're": "they", "that's": "that", "there's": "there",
    "i've": "I", "you've": "you", "we've": "we", "they've": "they",
    "i'll": "I", "you'll": "you", "he'll": "he", "she'll": "she",
    "we'll": "we", "they'll": "they", "it'll": "it",
    "i'd": "I", "you'd": "you", "he'd": "he", "she'd": "she", "we'd": "we", "they'd": "they",
    "let's": "let",
}

EMPHASIS_WORDS = {
    "amazing": "amaze", "incredible": "amaze", "awesome": "amaze", "wonderful": "amaze",
    "terrible": "bad", "horrible": "bad", "awful": "bad", "dreadful": "bad",
    "huge": "big", "enormous": "big", "massive": "big", "gigantic": "big",
    "tiny": "small", "minuscule": "small", "microscopic": "small",
    "great": "good", "fantastic": "good", "excellent": "good", "perfect": "good",
    "beautiful": "pretty", "gorgeous": "pretty", "stunning": "pretty",
    "furious": "angry", "livid": "angry", "enraged": "angry",
    "exhausted": "tired", "starving": "hungry", "terrified": "scared",
}

_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_TRAILING_PUNCT_RE = re.compile(r"[.!?]+$")


def _apply_case(original: str, replacement: str) -> str:
    if original.isupper() and len(original) > 1:
        return replacement.upper()
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _transform_word(match: re.Match) -> str:
    word = match.group(0)
    lower = word.lower()
    if lower in CONTRACTIONS:
        return _apply_case(word, CONTRACTIONS[lower])
    if lower in EMPHASIS_WORDS:
        base = EMPHASIS_WORDS[lower]
        return _apply_case(word, f"{base} {base} {base}")
    if lower in ARTICLES:
        return ""
    return word


def _transform_sentence(raw: str) -> str:
    punct_match = _TRAILING_PUNCT_RE.search(raw)
    punct = punct_match.group(0) if punct_match else ""
    body = raw[: -len(punct)] if punct else raw
    is_question = "?" in punct

    body = _WORD_RE.sub(_transform_word, body)
    body = re.sub(r"\s{2,}", " ", body).strip()
    if not body:
        return ""
    body = body[0].upper() + body[1:]

    if is_question:
        return f"{body}, question?"
    return f"{body}{punct}"


def rocky_transform(text: str) -> str:
    sentences = re.findall(r"[^.!?]+[.!?]*", text)
    transformed = [_transform_sentence(s.strip()) for s in sentences if s.strip()]
    return " ".join(s for s in transformed if s)
