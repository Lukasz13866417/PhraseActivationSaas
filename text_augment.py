from __future__ import annotations
import random
import re
from typing import Sequence


def augment_punct(
    phrase: str,
    rng: random.Random,
    *,
    p_replace_space: float = 0.6,    # chance to replace a space with a pause-like symbol
    p_extra_end: float = 0.4,        # chance to add end punctuation
    candidates_space: Sequence[str] = (",", ";", ":", "—", "-"),
    candidates_end: Sequence[str] = ("", ".", "!", "?", "…", "!!", "?!"),
) -> str:
    """
    Light-weight punctuation augmentation for short wake phrases.
    - Randomly replaces some spaces with pause-inducing punctuation
    - Optionally appends end punctuation (including expressive variants)
    """
    words = re.split(r"\s+", phrase.strip())
    if not words:
        return phrase

    if len(words) == 1:
        out = words[0]
    else:
        out_parts = [words[0]]
        for w in words[1:]:
            sep = rng.choice(candidates_space) if rng.random() < p_replace_space else " "
            out_parts.append(sep)
            out_parts.append(w)
        out = "".join(out_parts)

    if rng.random() < p_extra_end:
        out += rng.choice(candidates_end)
    return out


