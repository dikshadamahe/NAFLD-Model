import re

P = ("un", "re", "dis", "pre", "mis")
S = ("ingly", "edly", "ing", "ed", "ly", "ness", "ment", "s")


def morph(w):
    w = w.lower()
    p = next((x for x in P if w.startswith(x) and len(w) > len(x) + 2), "")
    b = w[len(p):]
    s = next((x for x in S if b.endswith(x) and len(b) > len(x) + 2), "")
    return {"word": w, "prefix": p, "root": b[:-len(s)] if s else b, "suffix": s}


if __name__ == "__main__":
    text = "Replaying unhealthy movements disconnected runners"
    words = re.findall(r"[A-Za-z]+", text)
    for w in words:
        print(morph(w))
