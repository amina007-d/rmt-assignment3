"""Re-generates the prompt set with 5 different seeds and re-runs detection (1 latency run)
to show how much the pilot metrics vary with the random draw. Output: results/seed_robustness.csv"""
import subprocess, json, os, sys, pandas as pd
rows = []
for seed in [1, 2, 3, 4, 5]:
    d = f"results/seeds/s{seed}"; os.makedirs(d, exist_ok=True)
    subprocess.run([sys.executable, "src/generate_prompts.py", "--seed", str(seed), "--out", f"{d}/prompts.jsonl"], check=True, capture_output=True)
    subprocess.run([sys.executable, "src/evaluate.py", "--data", f"{d}/prompts.jsonl", "--out", d, "--runs", "1"], check=True, capture_output=True)
    S = pd.read_csv(f"{d}/summary_by_language.csv"); S["seed"] = seed
    h1 = json.load(open(f"{d}/hypothesis_tests.json"))["H1_mcnemar_hybrid_vs_ner_unstructured"]
    S["H1_hybrid_only"], S["H1_ner_only"], S["H1_p"] = h1["hybrid_only"], h1["ner_only"], h1["p_one_sided"]
    rows.append(S)
A = pd.concat(rows)
A.to_csv("results/seed_robustness.csv", index=False)
agg = A.groupby(["pipeline", "lang"]).agg(recall_min=("recall", "min"), recall_max=("recall", "max"),
      unstr_min=("recall_unstructured", "min"), unstr_max=("recall_unstructured", "max"),
      prec_min=("precision", "min"), prec_max=("precision", "max")).round(3)
agg.to_csv("results/seed_robustness_ranges.csv")
print(agg.to_string())
print(A.groupby("seed")[["H1_hybrid_only", "H1_ner_only", "H1_p"]].first().to_string())
