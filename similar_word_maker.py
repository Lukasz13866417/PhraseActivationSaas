from phonemizer import phonemize
from phonemizer.separator import Separator

try:
    # If available, this speeds up distance a lot
    from Levenshtein import distance as _levenshtein
except Exception:
    # Tiny pure-Python fallback
    def _levenshtein(a: str, b: str) -> int:
        if a == b:
            return 0
        if not a:
            return len(b)
        if not b:
            return len(a)
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                ins = cur[-1] + 1
                dele = prev[j] + 1
                sub = prev[j - 1] + (ca != cb)
                cur.append(min(ins, dele, sub))
            prev = cur
        return prev[-1]

def _normalize_language(lang: str) -> tuple[str, str]:
    """
    Map a loose language code to:
      - wordfreq language code (e.g., 'en', 'fr')
      - phonemizer language code (e.g., 'en-us', 'fr-fr')
    """
    lang = (lang or "en").lower().replace("_", "-")
    # If caller already gave a phonemizer style code, keep it
    if "-" in lang:
        wf = lang.split("-")[0]
        pm = lang
    else:
        wf = lang
        mapping = {
            "en": "en-us",
            "fr": "fr-fr",
            "de": "de-de",
            "es": "es-es",
            "pt": "pt-br",   # tweak to taste
            "it": "it-it",
            "pl": "pl-pl",
            "nl": "nl-nl",
            "ru": "ru-ru",
            "sv": "sv-se",
            "tr": "tr-tr",
        }
        pm = mapping.get(lang, "en-us")
    return wf, pm

def _phonemes(words: list[str], pm_lang: str) -> list[str]:
    """
    Batch phonemize words. Returns one phoneme string per input word.
    """
    # Keep phones separated by spaces, strip stress/tones/word separators.
    sep = Separator(phone=" ", syllable=None, word=None)
    return phonemize(
        words,
        language=pm_lang,
        backend="espeak",
        separator=sep,
        strip=True,
        preserve_punctuation=False,
        njobs=4,
    )

def similar_sounding_words(
    word: str,
    count: int = 10,
    language: str = "en",
    *,
    vocabulary: list[str] | None = None,
    max_candidates: int = 20000,
    return_scores: bool = False,
) -> list[str] | list[tuple[str, float]]:
    """
    Find 'count' words that sound similar to `word`, using phoneme edit distance.

    Args:
        word: The query word.
        count: How many similar words to return.
        language: Language code. Loose codes like 'en', 'fr' work.
                  Phonemizer-style codes like 'en-us', 'fr-fr' also work.
        vocabulary: Optional list of candidate words to search over.
                    If not provided, uses top-N words from `wordfreq`.
        max_candidates: Size of the candidate pool (if using wordfreq).
        return_scores: If True, return (word, similarity) with 0.0 = identical
                       and 1.0 = maximally different (normalized edit distance).

    Returns:
        Either a list of words, or a list of (word, score) pairs ranked by similarity.
    """
    if not isinstance(word, str) or not word:
        raise ValueError("`word` must be a non-empty string")
    wf_lang, pm_lang = _normalize_language(language)

    # Build candidate vocabulary
    if vocabulary is None:
        try:
            from wordfreq import top_n_list
        except ImportError as e:
            raise RuntimeError(
                "wordfreq is required when `vocabulary` is not provided. "
                "Install with `pip install wordfreq`, or pass your own candidate list."
            ) from e
        vocab = top_n_list(wf_lang, n=max_candidates)
    else:
        vocab = list(vocabulary)

    base = word.strip()
    base_lower = base.lower()

    # Simple cleanup & filtering
    # - unify case
    # - drop duplicates
    # - exclude the query word itself
    # - keep tokens with letters only (adjust if you want compounds)
    seen = set()
    candidates = []
    for w in vocab:
        lw = w.lower()
        if lw == base_lower:
            continue
        if not lw.isalpha():
            continue
        if lw in seen:
            continue
        seen.add(lw)
        candidates.append(lw)

    if not candidates:
        return [] if not return_scores else []

    # Phonemize the base word and all candidates
    base_ph = _phonemes([base_lower], pm_lang)[0]
    cand_ph_list = _phonemes(candidates, pm_lang)

    # Compute normalized phoneme edit distance
    scored = []
    for w, ph in zip(candidates, cand_ph_list):
        if not ph or not base_ph:
            # If phonemizer failed or gave empty, skip conservatively
            continue
        d = _levenshtein(base_ph, ph)
        norm = d / max(len(base_ph), len(ph))
        scored.append((w, norm))

    # Sort by increasing distance (more similar first)
    scored.sort(key=lambda t: t[1])

    top = scored[:max(0, count)]
    if return_scores:
        return top
    return [w for w, _ in top]


# --- tiny demo ---
if __name__ == "__main__":
    # Example: similar words to "data" in English
    out = similar_sounding_words("hello", 100, "en", return_scores=True)
    for w, s in out:
        print(f"{w:15s}  distance={s:.3f}")
