"""Runs the three detectors on the annotated prompt set and writes metrics.

Primary metric (privacy view): an entity counts as DETECTED if any predicted span
overlaps it, whatever label was predicted (a masked identifier is protected even if
mislabelled, cf. TAB recall). A predicted span is a false positive if it overlaps
no gold entity. Secondary metric: STRICT = exact boundaries and the same label.

Usage: python src/evaluate.py --data data/prompts.jsonl --out results --runs 5
"""
import argparse, json, os, sys, time, platform, random, statistics
from collections import defaultdict, Counter
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy import stats
from detectors import RuleDetector, NERDetector, HybridDetector
from resources import STRUCTURED, UNSTRUCTURED, LANGS, LABELS

def overlap(a, b):
    return a["start"] < b["end"] and b["start"] < a["end"]

def score_prompt(gold, pred):
    g_hit = [any(overlap(g, p) for p in pred) for g in gold]
    g_strict = [any(g["start"] == p["start"] and g["end"] == p["end"] and g["label"] == p["label"]
                    for p in pred) for g in gold]
    p_tp = [any(overlap(p, g) for g in gold) for p in pred]
    p_strict = [any(g["start"] == p["start"] and g["end"] == p["end"] and g["label"] == p["label"]
                    for g in gold) for p in pred]
    return g_hit, g_strict, p_tp, p_strict

def prf(tp_g, n_g, tp_p, n_p):
    r = tp_g / n_g if n_g else float("nan")
    p = tp_p / n_p if n_p else float("nan")
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f

def bootstrap_recall(per_prompt, iters=2000, seed=0):
    """per_prompt: list of (hits, total) for each prompt -> 95% percentile CI."""
    rng = np.random.default_rng(seed)
    arr = np.array(per_prompt, dtype=float)
    if arr.sum(axis=0)[1] == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, len(arr), size=(iters, len(arr)))
    s = arr[idx].sum(axis=1)
    vals = s[:, 0] / np.where(s[:, 1] == 0, np.nan, s[:, 1])
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return f"{lo:.3f}-{hi:.3f}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/prompts.jsonl")
    ap.add_argument("--out", default="results")
    ap.add_argument("--runs", type=int, default=5)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = [json.loads(l) for l in open(a.data, encoding="utf-8")]

    rule, ner = RuleDetector(), NERDetector()
    dets = {"rule": rule, "ner": ner, "hybrid": HybridDetector(rule, ner)}

    for d in dets.values():
        for r in rows[:20]:
            d.detect(r["text"])
    preds = {k: {} for k in dets}
    lat = {k: defaultdict(list) for k in dets}
    order = list(dets)
    for run in range(a.runs):
        random.Random(run).shuffle(order)
        for k in order:
            for r in rows:
                t0 = time.perf_counter()
                out = dets[k].detect(r["text"])
                lat[k][r["id"]].append((time.perf_counter() - t0) * 1000)
                preds[k][r["id"]] = out
    lat_med = {k: {pid: statistics.median(v) for pid, v in lat[k].items()} for k in dets}
    lang_of = {r["id"]: r["lang"] for r in rows}
    with open(f"{a.out}/latency_per_prompt.csv", "w") as f:
        f.write("pipeline,id,lang,latency_ms\n")
        for k in dets:
            for pid, v in lat_med[k].items():
                f.write(f"{k},{pid},{lang_of[pid]},{v:.4f}\n")

    with open(f"{a.out}/predictions.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"id": r["id"], **{k: preds[k][r["id"]] for k in dets}}, ensure_ascii=False) + "\n")

    ent_rows, fp_rows = [], []
    for r in rows:
        for k in dets:
            p = preds[k][r["id"]]
            gh, gs, pt, ps = score_prompt(r["entities"], p)
            for g, h, s in zip(r["entities"], gh, gs):
                ent_rows.append({"id": r["id"], "lang": r["lang"], "pipeline": k, "label": g["label"],
                                 "group": "structured" if g["label"] in STRUCTURED else "unstructured",
                                 "text": g["text"], "form": g.get("meta_form"),
                                 "in_gaz": g.get("meta_in_gazetteer"), "hit": h, "strict": s})
            for sp, t in zip(p, pt):
                if not t:
                    fp_rows.append({"id": r["id"], "lang": r["lang"], "pipeline": k,
                                    "text": r["text"][sp["start"]:sp["end"]], "label": sp["label"]})
    import pandas as pd
    E = pd.DataFrame(ent_rows); FP = pd.DataFrame(fp_rows)
    E.to_csv(f"{a.out}/entity_level.csv", index=False)
    FP.to_csv(f"{a.out}/false_positives.csv", index=False)

    npred = {k: {lang: sum(len(preds[k][r["id"]]) for r in rows if r["lang"] == lang) for lang in LANGS} for k in dets}
    nfp = FP.groupby(["pipeline", "lang"]).size().to_dict() if len(FP) else {}
    strict_tp_pred = defaultdict(int)
    for r in rows:
        for k in dets:
            _, _, _, ps = score_prompt(r["entities"], preds[k][r["id"]])
            strict_tp_pred[(k, r["lang"])] += sum(ps)

    summary = []
    for k in dets:
        for lang in LANGS:
            sub = E[(E.pipeline == k) & (E.lang == lang)]
            n_p = npred[k][lang]; fp = nfp.get((k, lang), 0)
            p, rcl, f1 = prf(sub.hit.sum(), len(sub), n_p - fp, n_p)
            sp, sr, sf = prf(sub.strict.sum(), len(sub), strict_tp_pred[(k, lang)], n_p)
            row = {"pipeline": k, "lang": lang, "n_gold": len(sub), "n_pred": n_p, "fp": fp,
                   "precision": round(p, 3), "recall": round(rcl, 3), "f1": round(f1, 3),
                   "strict_precision": round(sp, 3), "strict_recall": round(sr, 3), "strict_f1": round(sf, 3)}
            for grp in ["structured", "unstructured"]:
                g = sub[sub.group == grp]
                row[f"recall_{grp}"] = round(g.hit.mean(), 3)
                per_prompt = g.groupby("id").hit.agg(["sum", "count"]).values.tolist()
                row[f"recall_{grp}_ci"] = bootstrap_recall(per_prompt)
            lats = [lat_med[k][r["id"]] for r in rows if r["lang"] == lang]
            row["latency_median_ms"] = round(statistics.median(lats), 3)
            row["latency_p95_ms"] = round(float(np.percentile(lats, 95)), 3)
            summary.append(row)
    S = pd.DataFrame(summary)
    S.to_csv(f"{a.out}/summary_by_language.csv", index=False)

    T = E.groupby(["pipeline", "lang", "label"]).agg(n=("hit", "size"), recall=("hit", "mean"),
                                                    strict_recall=("strict", "mean")).round(3).reset_index()
    T.to_csv(f"{a.out}/recall_by_type.csv", index=False)
    P = E[E.label == "PERSON"].groupby(["pipeline", "form"]).hit.agg(["size", "mean"]).round(3).reset_index()
    P.to_csv(f"{a.out}/person_recall_by_form.csv", index=False)
    G = E[E.label.isin(["PERSON", "LOC", "ORG"]) & E.in_gaz.notna()].groupby(["pipeline", "label", "in_gaz"]).hit.agg(["size", "mean"]).round(3).reset_index()
    G.to_csv(f"{a.out}/recall_by_gazetteer_coverage.csv", index=False)

    import re as _re
    hn = []
    for r in rows:
        gold_spans = [(g["start"], g["end"]) for g in r["entities"]]
        for m in _re.finditer(r"(?<!\d)\d{12}(?!\d)", r["text"]):
            if not any(m.start() < e and s_ < m.end() for s_, e in gold_spans):
                for k in dets:
                    hn.append({"pipeline": k, "flagged": any(m.start() < p["end"] and p["start"] < m.end() for p in preds[k][r["id"]])})
    HN = pd.DataFrame(hn).groupby("pipeline").flagged.agg(["size", "sum"]).reset_index()
    HN.to_csv(f"{a.out}/hard_negatives_12digit.csv", index=False)

    tests = {}
    a_ = E[(E.pipeline == "hybrid") & (E.group == "unstructured")].sort_values(["id", "text"]).hit.values
    b_ = E[(E.pipeline == "ner") & (E.group == "unstructured")].sort_values(["id", "text"]).hit.values
    b01 = int(((a_ == True) & (b_ == False)).sum()); b10 = int(((a_ == False) & (b_ == True)).sum())
    pv = stats.binomtest(b01, b01 + b10, 0.5, alternative="greater").pvalue if b01 + b10 else 1.0
    tests["H1_mcnemar_hybrid_vs_ner_unstructured"] = {
        "n_entities": int(len(a_)), "hybrid_recall": round(float(a_.mean()), 3), "ner_recall": round(float(b_.mean()), 3),
        "hybrid_only": b01, "ner_only": b10, "p_one_sided": pv,
        "by_condition": {lang: {p_: round(float(E[(E.pipeline == p_) & (E.lang == lang) & (E.group == "unstructured")].hit.mean()), 3)
                                for p_ in ["hybrid", "ner"]} for lang in LANGS}}
    ids = [r["id"] for r in rows]
    h2 = {}
    for other in ["ner", "hybrid"]:
        x = np.array([lat_med["rule"][i] for i in ids]); y = np.array([lat_med[other][i] for i in ids])
        w = stats.wilcoxon(x, y, alternative="less")
        h2[f"rule_vs_{other}"] = {"rule_median_ms": round(float(np.median(x)), 3),
                                  f"{other}_median_ms": round(float(np.median(y)), 3), "p_one_sided": w.pvalue}
    tests["H2_wilcoxon_latency"] = h2
    json.dump(tests, open(f"{a.out}/hypothesis_tests.json", "w"), indent=2, default=float)

    import spacy, scipy, pandas
    env = {"python": platform.python_version(), "platform": platform.platform(),
           "processor": platform.processor() or platform.machine(), "cpu_count": os.cpu_count(),
           "spacy": spacy.__version__, "ner_model": ner.model,
           "ner_model_version": ner.nlp.meta.get("version"), "numpy": np.__version__,
           "scipy": scipy.__version__, "pandas": pandas.__version__, "runs": a.runs,
           "n_prompts": len(rows), "n_gold": int(len(E) / len(dets))}
    json.dump(env, open(f"{a.out}/environment.json", "w"), indent=2)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(S[["pipeline", "lang", "n_gold", "n_pred", "fp", "precision", "recall", "f1", "strict_f1",
             "recall_structured", "recall_unstructured", "latency_median_ms", "latency_p95_ms"]].to_string(index=False))
    print(json.dumps(tests, indent=1, default=float))

if __name__ == "__main__":
    main()
