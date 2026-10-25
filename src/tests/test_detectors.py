"""Unit tests for validators and detectors. Run: python -m pytest -q tests  (or python tests/test_detectors.py)"""
import os, sys, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from generate_prompts import make_iin, make_iban, make_invalid_12
from detectors import valid_iin, valid_iban, RuleDetector, HybridDetector

def test_generated_iins_are_valid():
    rng = random.Random(1)
    assert all(valid_iin(make_iin(rng, g)) for g in "mf" * 200)

def test_invalid_12_digit_numbers_rejected():
    rng = random.Random(2)
    assert not any(valid_iin(make_invalid_12(rng)) for _ in range(400))

def test_iban_checksum():
    rng = random.Random(3)
    assert all(valid_iban(make_iban(rng)) for _ in range(200))
    assert not valid_iban("KZ00 1234 5678 9012 3456")       # wrong check digits

def test_rule_detector_structured():
    r = RuleDetector()
    rng = random.Random(4)
    iin, iban = make_iin(rng, "f"), make_iban(rng)
    text = f"ИИН {iin}, счёт {iban}, тел. +7 701 123 45 67, a.b@mail.ru, договор № {make_invalid_12(rng)}"
    labels = sorted(s["label"] for s in r.detect(text))
    assert labels == ["EMAIL", "IBAN", "IIN", "PHONE"]

def test_rule_detector_kazakh_city_inflection():
    spans = RuleDetector().detect("Ол Шымкентте тұрады.")
    assert [s["label"] for s in spans] == ["LOC"]

def test_hybrid_keeps_structured_boundaries():
    h = HybridDetector()
    rng = random.Random(5)
    iin = make_iin(rng, "m")
    text = f"ИИН {iin}"
    spans = h.detect(text)
    assert {"start": 4, "end": 16, "label": "IIN"} in spans

if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
