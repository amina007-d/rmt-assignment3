"""Figures for the report (static PNG, light theme for print). Run after evaluate.py."""
import os, sys, json
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES, FIG = "results", "figures"
os.makedirs(FIG, exist_ok=True)
COL = {"rule": "#2a78d6", "ner": "#eb6834", "hybrid": "#1baf7a"}     # validated slots 1-3
NAME = {"rule": "Rule-based", "ner": "NER-based", "hybrid": "Hybrid"}
LANGN = {"en": "English", "kk": "Kazakh", "ru": "Russian", "cs": "Code-switched"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False})

S = pd.read_csv(f"{RES}/summary_by_language.csv")
E = pd.read_csv(f"{RES}/entity_level.csv")
pipes, langs = ["rule", "ner", "hybrid"], ["en", "kk", "ru", "cs"]

def ci(s):
    lo, hi = map(float, s.split("-"))
    return lo, hi

# ---- Figure 1: recall by entity group x language (with bootstrap 95% CI)
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2), sharey=True)
for ax, grp in zip(axes, ["structured", "unstructured"]):
    x = np.arange(len(langs)); w = 0.26
    for i, p in enumerate(pipes):
        sub = S[S.pipeline == p].set_index("lang").loc[langs]
        v = sub[f"recall_{grp}"].values
        lo = np.array([ci(c)[0] for c in sub[f"recall_{grp}_ci"]]); hi = np.array([ci(c)[1] for c in sub[f"recall_{grp}_ci"]])
        ax.bar(x + (i - 1) * w, v, w - 0.03, color=COL[p], label=NAME[p], zorder=2)
        ax.errorbar(x + (i - 1) * w, v, yerr=[v - lo, hi - v], fmt="none", ecolor=INK, elinewidth=0.8, capsize=2, zorder=3)
    ax.set_xticks(x, [LANGN[l] for l in langs])
    ax.set_title("Structured identifiers (IIN, IBAN, phone, e-mail)" if grp == "structured" else "Names, places, organisations",
                 fontsize=9, color=INK, loc="left")
    ax.set_ylim(0, 1.05); ax.yaxis.grid(True, color=GRID, zorder=0)
axes[0].set_ylabel("Recall (overlap, type-agnostic)")
axes[1].legend(frameon=False, loc="lower right", fontsize=8)
fig.tight_layout(); fig.savefig(f"{FIG}/fig1_recall_by_group.png", dpi=200); plt.close(fig)

# ---- Figure 2: latency per prompt (median of 5 runs), log scale
fig, ax = plt.subplots(figsize=(8.6, 2.8))
lat = pd.read_csv(f"{RES}/latency_per_prompt.csv")
for i, p in enumerate(pipes):
    for j, l in enumerate(langs):
        v = lat[(lat.pipeline == p) & (lat.lang == l)].latency_ms.values
        xpos = j * 4 + i
        bp = ax.boxplot(v, positions=[xpos], widths=0.7, patch_artist=True, showfliers=False,
                        medianprops=dict(color=INK, linewidth=1.2), whiskerprops=dict(color=INK2),
                        capprops=dict(color=INK2), boxprops=dict(edgecolor=INK2))
        bp["boxes"][0].set_facecolor(COL[p])
ax.set_yscale("log"); ax.set_ylabel("ms per prompt (log)")
ax.set_xticks([j * 4 + 1 for j in range(4)], [LANGN[l] for l in langs])
ax.yaxis.grid(True, color=GRID, which="both", linewidth=0.5)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=COL[p], label=NAME[p]) for p in pipes], frameon=False, fontsize=8, loc="center right", bbox_to_anchor=(1.0, 0.5))
fig.tight_layout(); fig.savefig(f"{FIG}/fig2_latency.png", dpi=200); plt.close(fig)

# ---- Figure 3: person recall by surface form
P = pd.read_csv(f"{RES}/person_recall_by_form.csv")
forms = ["first_last", "last_first", "first_only", "last_only", "lowercase"]
fl = {"first_last": "First Last", "last_first": "Last First", "first_only": "First only",
      "last_only": "Last only", "lowercase": "all lowercase"}
fig, ax = plt.subplots(figsize=(8.6, 2.8)); x = np.arange(len(forms)); w = 0.26
for i, p in enumerate(pipes):
    v = P[P.pipeline == p].set_index("form").loc[forms]["mean"].values
    n = P[P.pipeline == p].set_index("form").loc[forms]["size"].values
    ax.bar(x + (i - 1) * w, v, w - 0.03, color=COL[p], label=NAME[p], zorder=2)
    for xi, vi in zip(x + (i - 1) * w, v):
        ax.text(xi, vi + 0.02, f"{vi:.2f}", ha="center", va="bottom", fontsize=6.5, color=INK2)
ax.set_xticks(x, [f"{fl[f]}\n(n={m})" for f, m in zip(forms, n)])
ax.set_ylim(0, 1.12); ax.set_ylabel("Recall, PERSON"); ax.yaxis.grid(True, color=GRID, zorder=0)
ax.legend(frameon=False, fontsize=8, loc="upper right")
fig.tight_layout(); fig.savefig(f"{FIG}/fig3_person_forms.png", dpi=200); plt.close(fig)
print("figures written")

# ---- Figure 4: pipeline / experiment flow diagram
from matplotlib.patches import FancyBboxPatch
fig, ax = plt.subplots(figsize=(8.6, 3.0)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 40)
def box(x, y, w, h, t, fc="#f3f2ef"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=INK2, lw=0.8))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=7.5, color=INK)
def arr(x1, y1, x2, y2):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8))
box(1, 16, 15, 9, "Generator\n(templates, seed=42)\n4 language conditions")
box(21, 16, 14, 9, "prompts.jsonl\n240 prompts\n988 gold spans")
box(41, 29, 17, 8, "Rule-based\nregex + checksums\n+ gazetteers", "#dbe8f8")
box(41, 16, 17, 8, "NER-based\nspaCy xx_ent_wiki_sm", "#fbe0d5")
box(41, 3, 17, 8, "Hybrid\nrules for identifiers\n+ merged NER spans", "#d4f0e5")
box(64, 16, 15, 9, "Scoring\noverlap & strict\nP / R / F1, latency")
box(84, 16, 15, 9, "Outputs\nCSV tables, tests\n(H1-H2), figures")
arr(16.8, 20.5, 20.2, 20.5)
for y in (33, 20, 7): arr(35.8, 20.5, 40.2, y)
for y in (33, 20, 7): arr(58.8, y, 63.2, 20.5)
arr(79.8, 20.5, 83.2, 20.5)
fig.tight_layout(); fig.savefig(f"{FIG}/fig4_pipeline.png", dpi=200); plt.close(fig)
