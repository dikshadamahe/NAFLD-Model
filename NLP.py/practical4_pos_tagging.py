import re


def tag(w):
    if w in {"a", "an", "the"}:
        return "DET"
    if w in {"is", "am", "are", "was", "were", "be"}:
        return "VERB"
    if w.endswith("ing") or w.endswith("ed"):
        return "VERB"
    if w.endswith("ly"):
        return "ADV"
    if w.endswith(("ous", "ful", "able", "al", "ive")):
        return "ADJ"
    if w.endswith(("ness", "ment", "tion", "ity", "er", "or")):
        return "NOUN"
    return "NOUN"


if __name__ == "__main__":
    t = "The liver is slowly improving and fatty tissue was reduced"
    ws = re.findall(r"[A-Za-z]+", t.lower())
    print([(w, tag(w)) for w in ws])
