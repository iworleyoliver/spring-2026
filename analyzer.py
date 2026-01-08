"""
Simple file and PDF analyzer utilities.
- Extracts text from PDFs using PyPDF2
- Reads plain text files
- Computes basic stats: chars, words, lines, top words
No heavy external deps.
"""
from typing import Dict, Any
import re

try:
    from PyPDF2 import PdfReader
except Exception:  # pragma: no cover - optional dependency
    PdfReader = None  # type: ignore

DEFAULT_STOPWORDS = {
    "the", "and", "to", "of", "a", "in", "is", "it", "that", "for",
    "on", "with", "as", "are", "this", "by", "an", "be", "or", "from",
}

word_re = re.compile(r"\b[0-9A-Za-z']+\b")


def extract_text_from_pdf(path: str) -> Dict[str, Any]:
    if PdfReader is None:
        raise RuntimeError("PyPDF2 is not installed. Add it to requirements.txt and install.")
    reader = PdfReader(path)
    texts = []
    for p in reader.pages:
        try:
            texts.append(p.extract_text() or "")
        except Exception:
            texts.append("")
    return {"text": "\n".join(texts), "n_pages": len(reader.pages)}


def read_text_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        t = f.read()
    return {"text": t}


def compute_stats(text: str, top_n: int = 10, stopwords: set | None = None) -> Dict[str, Any]:
    if stopwords is None:
        stopwords = DEFAULT_STOPWORDS
    chars = len(text)
    lines = text.count("\n") + (1 if text and not text.endswith("\n") else 0)
    words = word_re.findall(text.lower())
    word_count = len(words)
    freqs: Dict[str, int] = {}
    for w in words:
        if w in stopwords:
            continue
        freqs[w] = freqs.get(w, 0) + 1
    top = sorted(freqs.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return {"chars": chars, "lines": lines, "words": word_count, "top_words": top}


def analyze_file(path: str, top_n: int = 10) -> Dict[str, Any]:
    path_lower = path.lower()
    if path_lower.endswith(".pdf"):
        pdf_data = extract_text_from_pdf(path)
        stats = compute_stats(pdf_data["text"], top_n=top_n)
        stats.update({"n_pages": pdf_data.get("n_pages", 0)})
        return stats
    else:
        data = read_text_file(path)
        stats = compute_stats(data["text"], top_n=top_n)
        return stats
