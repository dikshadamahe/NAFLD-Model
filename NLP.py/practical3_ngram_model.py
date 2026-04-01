import re
from collections import Counter


def ngrams(text, n=2):
    w = re.findall(r"[A-Za-z]+", text.lower())
    return Counter(tuple(w[i:i+n]) for i in range(len(w) - n + 1))


if __name__ == "__main__":
    t = "fatty liver disease is common and liver disease needs early care"
    for n in (1, 2, 3):
        print(f"{n}-gram", dict(ngrams(t, n)))
