# PII detection in Kazakh, Russian and code-switched LLM prompts — pilot evaluation

Research Methods, Assignment 3 (Core Practical Component). Team: Nazerke Abzhamal, Amina Dossan (SE-2422).

**RQ:** How do rule-based, NER-based and hybrid approaches compare in recall and processing time when detecting
personal data in Kazakh, Russian and code-switched Kazakh–Russian prompts intended for third-party LLM chat
services, relative to English prompts?

This repository contains the first working version of the evaluation pipeline: a synthetic annotated prompt set,
three detectors behind one interface, an entity-level scorer with latency measurement, hypothesis tests (H1–H2)
and figures. All numbers in `results/` come from running the code below; nothing is entered by hand.

## Structure
```
src/resources.py         entity taxonomy, name/place/org pools, gazetteers (subset of pools)
src/generate_prompts.py  template-based generator; gold spans recorded at generation time
src/detectors.py         RuleDetector, NERDetector (spaCy xx_ent_wiki_sm), HybridDetector
src/evaluate.py          predictions, latency (5 runs, median), metrics, bootstrap CIs, H1–H2 tests
src/plots.py             figures/
tests/test_detectors.py  unit tests (IIN/IBAN checksums, detectors)
data/prompts.jsonl       240 prompts (60 per condition), 988 gold entities
results/                 CSV/JSON outputs, run log, environment record
```

## Run
```
pip install -r requirements.txt
pip install https://github.com/explosion/spacy-models/releases/download/xx_ent_wiki_sm-3.8.0/xx_ent_wiki_sm-3.8.0-py3-none-any.whl
python tests/test_detectors.py
python src/generate_prompts.py --n 60 --seed 42 --out data/prompts.jsonl
python src/evaluate.py --data data/prompts.jsonl --out results --runs 5
python src/plots.py
```
Detection metrics are deterministic for a fixed seed. Latency depends on hardware; the reported pilot run used
a Windows 11 machine with Python 3.12.10 and 16 logical cores (see `results/environment.json`).

## Data and privacy
All prompts are synthetic. IINs and IBANs are random numbers that only satisfy the checksum rules; names are
common first names and surnames combined at random. No real personal data is used, and no prompt is sent to any
external service — all detection runs locally.

## Known limitations of this pilot
- Prompts are template-generated (8 templates per condition) and the rules were written for the same
  identifier formats, so rule-based results are optimistic. The full study adds hand-written prompts.
- The NER model is a CNN trained on WikiNER (no Kazakh training data). A transformer NER model
  (e.g. XLM-R fine-tuned for NER, or a KazNERD model) is planned for the full study.
- Lower-cased names are missed by all three pipelines (see `results/person_recall_by_form.csv`).