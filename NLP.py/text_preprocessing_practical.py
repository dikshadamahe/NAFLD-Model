import re
import unicodedata
S = {"a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it", "of", "on", "that", "the", "to", "was", "were", "with", "or", "but", "not"}


def script_ok(t, allow="LATIN"):
    k = {unicodedata.name(c, "").split(" ", 1)[0] for c in t if c.isalpha()}
    return len(k) == 1 and allow in k


def stem(t):
    for s, r in (("ingly", ""), ("edly", ""), ("ing", ""), ("ed", ""), ("ies", "y"), ("sses", "ss"), ("s", "")):
        if len(t) > len(s) + 2 and t.endswith(s):
            return t[:-len(s)] + r
    return t


def preprocess(x):
    a = re.findall(r"\b\w+\b", x, flags=re.UNICODE)
    b = [t.lower() for t in a if len(t) >= 2 and not t.isnumeric() and any(c.isalpha() for c in t)]
    c = [t for t in b if script_ok(t)]
    d = [t for t in c if t not in S]
    e = [stem(t) for t in d]
    return a, b, c, d, e


if __name__ == "__main__":
    x = "The patients were diagnosed with non-alcoholic fatty liver disease, and findings were progressing in 2026."
    for i, y in enumerate(preprocess(x), 1):
        print(("1) Tokenization", "2) Filtration", "3) Script Validation", "4) Stop-word Removal", "5) Stemming")[i - 1], y)
