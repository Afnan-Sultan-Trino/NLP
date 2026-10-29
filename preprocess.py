# src/preprocess.py
"""Lightweight Bangla text normalisation.

Kept deliberately simple: no stemming / no external dictionaries,
so the whole project stays reproducible and dependency-free.
"""
import re
import unicodedata

# Bangla digits -> ASCII digits (so "৫০০" and "500" are the same token)
_BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

_URL_RE = re.compile(r"http\S+|www\.\S+")
# keep: Bangla block (U+0980-U+09FF), latin letters, digits, whitespace
_KEEP_RE = re.compile(r"[^\u0980-\u09FFa-zA-Z0-9\s]")
_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Normalise a raw Bangla string into a clean, model-ready string."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)   # canonical Bangla encoding
    text = text.translate(_BN_DIGITS)
    text = _URL_RE.sub(" ", text)
    text = _KEEP_RE.sub(" ", text)              # drop punctuation / emoji
    text = _WS_RE.sub(" ", text).strip()
    return text


if __name__ == "__main__":  # quick sanity check
    sample = "দাম ৫০০ টাকা! খুব ভালো 😍 https://example.com"
    print(repr(normalize(sample)))