"""
Supplementary Table 1: Vision DNN test scores by rating target.

For each vision model, reports the test-set Pearson r (best CV-selected layer)
for predicting each behavioral rating, alongside the model's name and family
from data/processed/grouped_models.csv.
"""

import pandas as pd

from src.encoding import get_top_sota_scores
from src.config import RATING_OF_INTEREST
from src.plottings import change_name

targets = RATING_OF_INTEREST

COLUMN_ORDER = [
    "model name",
    "model family",
    "spatial expanse",
    "interagent distance",
    "agents facing",
    "communicative interaction",
    "physical interaction",
]

OUT_PATH = "results/supp_table_1.csv"

# ── 1. Best test score per model per target (layer selected via CV) ──────────

sota_scores = get_top_sota_scores(targets=targets, collect="target")
sota_scores["y"] = sota_scores["y"].apply(change_name)

# ── 2. Attach model name / family ─────────────────────────────────────────────

grouped = pd.read_csv("data/processed/grouped_models.csv")[
    ["Model UID", "Model Name", "model_family"]
]
df = sota_scores.merge(grouped, left_on="model_name", right_on="Model UID", how="left")

# ── 3. Pivot targets into columns ─────────────────────────────────────────────

table = df.pivot_table(index="Model UID", columns="y", values="score_r", aggfunc="first")
table = table.join(
    df.drop_duplicates("Model UID").set_index("Model UID")[["Model Name", "model_family"]]
)
table = table.rename(columns={"Model Name": "model name", "model_family": "model family"})
table = table.reset_index(drop=True)[COLUMN_ORDER]

score_columns = [c for c in COLUMN_ORDER if c not in ("model name", "model family")]
table = table.loc[table[score_columns].mean(axis=1).sort_values(ascending=False).index]
table[score_columns] = table[score_columns].round(4)

table.to_csv(OUT_PATH, index=False)
print(f"[LOGGING] Saved supplementary table to {OUT_PATH} ({len(table)} models)")
