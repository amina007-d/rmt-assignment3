"""Three detection approaches under one interface:  detect(text) -> list of spans
   span = {"start", "end", "label"}   labels follow resources.LABELS (+ "MISC" from NER)

- RuleDetector   : regular expressions + checksums (IIN, IBAN) + gazetteers (names, cities, orgs)
- NERDetector    : spaCy multilingual NER (xx_ent_wiki_sm), one model for all languages
- HybridDetector : rules for structured identifiers + union of rule/NER spans for the rest
"""
import re, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from resources import (gaz_first_names, gaz_surname_stems, gaz_cities, gaz_orgs, STRUCTURED)
from generate_prompts import iin_check_digit

CYR_UP = "А-ЯЁӘҒҚҢӨҰҮҺІ"
WORD = re.compile(r"[\w’'\-]+")

# ------------------------------------------------------------- validators
def valid_iin(s):
    if not re.fullmatch(r"\d{12}", s):
        return False
    # Only the control digit is checked. A birth-date / century-digit check is deliberately NOT
    # applied: IIN formation rules changed in 2013 and newer numbers are not required to encode
    # the birth date, so such a check could reject genuine IINs (lower recall).
    cd = iin_check_digit(s[:11])
    return cd is not None and cd == int(s[11])

def valid_iban(s):
    s = s.replace(" ", "")
    if not re.fullmatch(r"KZ\d{2}[0-9A-Z]{16}", s):
        return False
    if not 2 <= int(s[2:4]) <= 98:          # ISO 13616: check digits 00, 01, 99 are never valid
        return False
    r = s[4:] + s[:4]
    return int("".join(str(int(ch, 36)) for ch in r)) % 97 == 1

# ------------------------------------------------------------- patterns
P_IIN = re.compile(r"(?<!\d)\d{12}(?!\d)")
P_IBAN = re.compile(r"\bKZ\d{2}(?: ?[0-9A-Z]{4}){4}\b")
P_PHONE = re.compile(r"(?<![\d+])(?:\+7|8)[\s\-]?\(?7\d{2}\)?[\s\-]?\d{3}[\s\-]?(?:\d{2}[\s\-]?\d{2}|\d{4})(?!\d)")
P_EMAIL = re.compile(r"\b[\w.+\-]+@[\w\-]+(?:\.[\w\-]+)+\b")
P_ADDR = [
    re.compile(r"\b(?:ул\.|улица|пр\.|проспект)\s*[" + CYR_UP + r"A-Z][\w\- ]*?,\s*(?:д\.\s*)?\d+\w?(?:,\s*кв\.\s*\d+)?"),
    re.compile(r"[" + CYR_UP + r"][\w\-]*(?: [\w\-]+)?\s(?:көшесі|даңғылы),\s*\d+-үй(?:,\s*\d+-пәтер)?"),
    re.compile(r"\b\d+\s+[A-Z][\w\-]*(?: [A-Z][\w\-]*)?\s(?:Street|St\.|Avenue|Ave\.)(?:,\s*apt\.?\s*\d+)?"),
]
P_ORG_LEGAL = [
    re.compile(r"\b(?:ТОО|АО|ЖШС|АҚ)\s*«[^»]+»"),
    re.compile(r"«[^»]+»\s*(?:ЖШС|АҚ|ТОО|АО)\b"),
    re.compile(r"\b[A-Z][\w\-]*(?: [A-Z][\w\-]*){0,3} (?:LLP|JSC|LLC)\b"),
]
PERSON_SUFFIX = re.compile(r"(?:ов|ев|ин|ова|ева|ина|ұлы|қызы|ov|ev|in|ova|eva|ina|uly|kyzy)$", re.I)

def _stem(w):
    """Crude stem for matching inflected forms (Астана -> Астан, matches Астане/Астанада)."""
    return w[:-1] if len(w) > 4 and w[-1].lower() in "аеыіоөуұүяиaeiouy" else w

class RuleDetector:
    name = "rule"

    def __init__(self):
        self.first = {w.lower() for w in gaz_first_names()}
        self.surn = sorted({w.lower() for w in gaz_surname_stems()}, key=len, reverse=True)
        self.cities = sorted({_stem(w).lower() for w in gaz_cities()}, key=len, reverse=True)
        self.orgs = sorted(gaz_orgs(), key=len, reverse=True)

    # -- unstructured helpers
    def _is_cap(self, tok):
        return tok[:1].isupper()

    def _is_first(self, tok):
        return self._is_cap(tok) and tok.lower() in self.first

    def _is_surname(self, tok):
        t = tok.lower()
        return self._is_cap(tok) and any(t.startswith(s) and len(t) - len(s) <= 2 for s in self.surn)

    def _is_city(self, tok):
        t = tok.lower()
        return self._is_cap(tok) and any(t.startswith(c) and len(t) - len(c) <= 4 for c in self.cities)

    def detect(self, text):
        spans = []
        def add(s, e, lab):
            if not any(s < x["end"] and x["start"] < e for x in spans):   # first match wins
                spans.append({"start": s, "end": e, "label": lab})
        # 1) structured identifiers (validated)
        for m in P_IBAN.finditer(text):
            if valid_iban(m.group()):
                add(m.start(), m.end(), "IBAN")
        for m in P_EMAIL.finditer(text):
            add(m.start(), m.end(), "EMAIL")
        for m in P_IIN.finditer(text):
            if valid_iin(m.group()):
                add(m.start(), m.end(), "IIN")
        for m in P_PHONE.finditer(text):
            add(m.start(), m.end(), "PHONE")
        # 2) addresses and organisations
        for p in P_ADDR:
            for m in p.finditer(text):
                add(m.start(), m.end(), "LOC")
        for p in P_ORG_LEGAL:
            for m in p.finditer(text):
                add(m.start(), m.end(), "ORG")
        for o in self.orgs:
            for m in re.finditer(re.escape(o), text):
                add(m.start(), m.end(), "ORG")
        # 3) persons and cities from gazetteers (token level)
        toks = list(WORD.finditer(text))
        i = 0
        while i < len(toks):
            t = toks[i].group()
            if self._is_first(t) or self._is_surname(t):
                s, e, j = toks[i].start(), toks[i].end(), i + 1
                # extend to an adjacent capitalised name-like token (e.g. unknown surname)
                if j < len(toks) and text[e:toks[j].start()] == " ":
                    nt = toks[j].group()
                    if self._is_first(nt) or self._is_surname(nt) or (self._is_cap(nt) and PERSON_SUFFIX.search(nt)):
                        e, j = toks[j].end(), j + 1
                # extend backwards: "Unknown Knownsurname" / "Knownfirst" preceded by name-like token
                if i > 0 and text[toks[i - 1].end():s] == " ":
                    pt = toks[i - 1].group()
                    if self._is_cap(pt) and (PERSON_SUFFIX.search(pt) or self._is_first(pt)):
                        s = toks[i - 1].start()
                add(s, e, "PERSON")
                i = j
                continue
            if self._is_city(t):
                add(toks[i].start(), toks[i].end(), "LOC")
            i += 1
        return sorted(spans, key=lambda x: x["start"])


NER_MAP = {"PER": "PERSON", "LOC": "LOC", "ORG": "ORG", "MISC": "MISC"}

class NERDetector:
    name = "ner"

    def __init__(self, model="xx_ent_wiki_sm"):
        import spacy
        self.nlp = spacy.load(model)
        self.model = model

    def detect(self, text):
        return [{"start": e.start_char, "end": e.end_char, "label": NER_MAP.get(e.label_, e.label_)}
                for e in self.nlp(text).ents]


class HybridDetector:
    """Structured identifiers come only from validated rules (NER spans that overlap
    them are discarded). For names/places/orgs, rule and NER spans are merged:
    overlapping spans are joined into their union, label taken from the rule if present."""
    name = "hybrid"

    def __init__(self, rule=None, ner=None):
        self.rule = rule or RuleDetector()
        self.ner = ner or NERDetector()

    def detect(self, text):
        r = self.rule.detect(text)
        n = self.ner.detect(text)
        fixed = [s for s in r if s["label"] in STRUCTURED]
        def hits_fixed(s):
            return any(s["start"] < f["end"] and f["start"] < s["end"] for f in fixed)
        pool = [dict(s, src="rule") for s in r if s["label"] not in STRUCTURED]
        pool += [dict(s, src="ner") for s in n if not hits_fixed(s)]
        pool.sort(key=lambda x: (x["start"], -x["end"]))
        merged = []
        for s in pool:
            if merged and s["start"] < merged[-1]["end"]:
                m = merged[-1]
                m["end"] = max(m["end"], s["end"])
                if s["src"] == "rule" and m["src"] != "rule":
                    m["label"], m["src"] = s["label"], "rule"
            else:
                merged.append(dict(s))
        out = fixed + [{"start": m["start"], "end": m["end"], "label": m["label"]} for m in merged]
        return sorted(out, key=lambda x: x["start"])
