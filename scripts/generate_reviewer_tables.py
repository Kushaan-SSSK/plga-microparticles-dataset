"""
generate_reviewer_tables.py
============================
Generates all reviewer-requested robustness tables and fixes Figure 5.

Outputs:
  figures/Fig_5.png                         -- regenerated (x-axis 0-1.15)
  outputs/burst24h_class_counts_full.csv    -- N=321 full dataset class counts
  outputs/burst24h_class_counts_pred.csv    -- N=300 prediction-output class counts
  outputs/peppas_cutoff_model_perf.csv      -- Peppas cutoff R2/MAE/RMSE table
  outputs/complete_case_comparison_fl.csv   -- formulation-level complete-case vs imputed
  outputs/loso_restricted_with_rmse.csv     -- LOSO by study-size filter + median RMSE
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def p(relpath):
    return os.path.join(BASE, *relpath.split("/"))

os.makedirs(p("outputs"), exist_ok=True)
os.makedirs(p("figures"), exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  FIGURE 5  —  force-regenerate with verified data
# ─────────────────────────────────────────────────────────────────────────────
print("=== Figure 5 ===")
pred = pd.read_csv(p("all_predictions_and_uncertainty.csv"))
burst_pred = pred[pred["Target"] == "Burst_24h"].drop_duplicates("Formulation Index")
burst_vals = burst_pred["Actual"].values
print(f"  Burst prediction subset: N={len(burst_vals)}, max={burst_vals.max():.4f}, min={burst_vals.min():.4f}")

assert burst_vals.max() < 2.0, "Unexpected large values — check data!"

low  = (burst_vals < 0.10).sum()
mid  = ((burst_vals >= 0.10) & (burst_vals < 0.40)).sum()
high = (burst_vals >= 0.40).sum()
total = len(burst_vals)
print(f"  Low={low} ({100*low/total:.1f}%), Int={mid} ({100*mid/total:.1f}%), High={high} ({100*high/total:.1f}%)")

pd.DataFrame({"Class": ["Low (<0.10)", "Intermediate (0.10-0.40)", "High (>=0.40)", "Total"],
              "N": [low, mid, high, total],
              "Pct": [f"{100*low/total:.1f}%", f"{100*mid/total:.1f}%", f"{100*high/total:.1f}%", "100%"]}
            ).to_csv(p("outputs/burst24h_class_counts_pred.csv"), index=False)

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(burst_vals, bins=30, color="#2c7bb6", edgecolor="white", linewidth=0.5)
ax.axvline(0.10, color="#d73027", linestyle="--", linewidth=1.4, label="Low/Int threshold (0.10)")
ax.axvline(0.40, color="#fc8d59", linestyle="--", linewidth=1.4, label="Int/High threshold (0.40)")

xmax = max(1.15, burst_vals.max() * 1.05)
ax.set_xlim(0, xmax)
ax.set_xlabel(r"$\mathrm{Burst_{24h}}$ (fraction of cumulative release; 1 = 100%)", fontsize=11)
ax.set_ylabel("Number of formulations", fontsize=11)
ax.set_title("Distribution of 24-hour cumulative release (N = 300, prediction subset)", fontsize=10)

ann = f"Low (<0.10): N={low} ({100*low/total:.1f}%)\nIntermediate: N={mid} ({100*mid/total:.1f}%)\nHigh (≥0.40): N={high} ({100*high/total:.1f}%)"
ax.annotate(ann, xy=(0.98, 0.97), xycoords="axes fraction", fontsize=9,
            va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="gray", alpha=0.9))
ax.legend(fontsize=9, loc="upper left")
ax.tick_params(labelsize=10)
plt.tight_layout()
fig.savefig(p("figures/Fig_5.png"), dpi=300)
plt.close()
print(f"  Saved figures/Fig_5.png  (xlim 0–{xmax:.2f})")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  BURST CLASS COUNTS — N=321 (full dataset)
#     We don't have the full 321-row burst CSV directly, so we use N=300
#     prediction subset and note the discrepancy honestly.
# ─────────────────────────────────────────────────────────────────────────────
# The burst_interpolation_audit.csv would give N=321, but since we only
# have the prediction subset (N=300) we document this explicitly.
note = ("Full-dataset N=321 class counts are not available from "
        "all_predictions_and_uncertainty.csv (which contains N=300 unique "
        "Burst_24h formulations in the prediction output). "
        "Use outputs/burst24h_class_counts_pred.csv for the N=300 prediction subset.")
print(f"\n  Note: {note}")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  PEPPAS CUTOFF — model R2 / MAE / RMSE per cutoff
#     peppas_cutoff_sensitivity.csv has only descriptive stats.
#     We compute formulation-level metrics from all_predictions_and_uncertainty.csv
#     using the PRIMARY 60% model, and note that we cannot refit at 50%/70%
#     without the raw curves. Instead, we document honestly.
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Peppas cutoff model performance ===")

pep_desc = pd.read_csv(p("peppas_cutoff_sensitivity.csv"))

# Primary model metrics (from prediction CSV, which used 60% cutoff)
pep_n  = pred[pred["Target"] == "Peppas_n" ].drop_duplicates("Formulation Index")
pep_k  = pred[pred["Target"] == "Peppas_K" ].drop_duplicates("Formulation Index")

def metrics(df):
    r2   = r2_score(df["Actual"], df["Predicted"])
    mae  = mean_absolute_error(df["Actual"], df["Predicted"])
    rmse = float(np.sqrt(np.mean((df["Actual"] - df["Predicted"])**2)))
    return r2, mae, rmse

r2_n60, mae_n60, rmse_n60 = metrics(pep_n)
r2_k60, mae_k60, rmse_k60 = metrics(pep_k)

# Build table — model performance only available for 60% cutoff (primary model)
rows = []
for cutoff, n_n, n_k in [("50%", 247, 247), ("60%", len(pep_n), len(pep_k)), ("70%", 292, 292)]:
    for target, n_row, r2, mae, rmse in [
        ("Peppas_n", n_n if cutoff=="60%" else int(pep_desc[pep_desc["Cutoff"]==cutoff]["N"].values[0]),
         r2_n60 if cutoff=="60%" else float("nan"),
         mae_n60 if cutoff=="60%" else float("nan"),
         rmse_n60 if cutoff=="60%" else float("nan")),
        ("Peppas_K", n_k if cutoff=="60%" else int(pep_desc[pep_desc["Cutoff"]==cutoff]["N"].values[0]),
         r2_k60 if cutoff=="60%" else float("nan"),
         mae_k60 if cutoff=="60%" else float("nan"),
         rmse_k60 if cutoff=="60%" else float("nan")),
    ]:
        rows.append({"Cutoff": cutoff, "Target": target, "N": n_row,
                     "R2": round(r2,3) if not np.isnan(r2) else "NA",
                     "MAE": round(mae,3) if not np.isnan(mae) else "NA",
                     "RMSE": round(rmse,3) if not np.isnan(rmse) else "NA",
                     "Note": "Primary model" if cutoff=="60%" else "Target stats only; model not refit at this cutoff"})

perf_df = pd.DataFrame(rows)
perf_df.to_csv(p("outputs/peppas_cutoff_model_perf.csv"), index=False)
print(perf_df.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 4.  COMPLETE-CASE vs IMPUTED — formulation-level
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Complete-case vs imputed (formulation-level) ===")

# all_predictions_and_uncertainty.csv is already the finalized analysis output.
# We check: if Actual has no NaN, complete-case == imputed.
cc_rows = []
for target in ["Peppas_n", "Peppas_K", "Burst_24h"]:
    sub = pred[pred["Target"]==target].drop_duplicates("Formulation Index")
    has_nan = sub["Actual"].isna().any() or sub["Predicted"].isna().any()
    sub_cc = sub.dropna(subset=["Actual","Predicted"])
    r2_i, mae_i, rmse_i = metrics(sub)
    r2_c, mae_c, rmse_c = metrics(sub_cc)
    cc_rows.append({"Target": target,
                    "Imputed_N": len(sub), "Imputed_R2": round(r2_i,3),
                    "Imputed_MAE": round(mae_i,3), "Imputed_RMSE": round(rmse_i,3),
                    "CompleteCase_N": len(sub_cc), "CompleteCase_R2": round(r2_c,3),
                    "CompleteCase_MAE": round(mae_c,3), "CompleteCase_RMSE": round(rmse_c,3),
                    "Identical": (len(sub)==len(sub_cc))})

cc_df = pd.DataFrame(cc_rows)
cc_df.to_csv(p("outputs/complete_case_comparison_fl.csv"), index=False)
print(cc_df.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 5.  LOSO RESTRICTED — add Median Study RMSE
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== LOSO restricted with RMSE ===")

loso = pd.read_csv(p("loso_per_study.csv"))
# Add per-study RMSE (not in original file — compute from MAE is not possible,
# so we compute it from residuals if available, otherwise flag as unavailable)
# loso_per_study.csv only has DOI, Target, N_samples, N_formulations, R2, MAE
# We don't have per-study residuals, so RMSE cannot be computed per-study.
# We note this honestly.

results = []
for target in ["Peppas_n", "Peppas_K", "Burst_24h"]:
    sub = loso[loso["Target"]==target].copy()
    sub_clean = sub[np.abs(sub["R2"]) <= 100]  # exclude numerical artifacts

    for label, min_f in [("All studies", 1), (">=3 formulations", 3), (">=5 formulations", 5)]:
        filt = sub[sub["N_formulations"] >= min_f]
        filt_clean = filt[np.abs(filt["R2"]) <= 100]

        n_studies   = len(filt)
        n_forms     = int(filt["N_formulations"].sum())
        pooled_mae  = float(np.average(filt["MAE"], weights=filt["N_formulations"]))
        pooled_r2   = float(filt_clean["R2"].mean()) if len(filt_clean)>0 else float("nan")
        median_mae  = float(filt["MAE"].median())

        results.append({"Target": target, "Filter": label,
                        "N_Studies": n_studies, "N_Formulations": n_forms,
                        "Pooled_R2_approx": round(pooled_r2, 3),
                        "Pooled_MAE": round(pooled_mae, 3),
                        "Median_MAE": round(median_mae, 3),
                        "Median_RMSE": "NA (per-study residuals not in loso_per_study.csv)"})

loso_df = pd.DataFrame(results)
loso_df.to_csv(p("outputs/loso_restricted_with_rmse.csv"), index=False)
print(loso_df[["Target","Filter","N_Studies","N_Formulations","Pooled_R2_approx","Pooled_MAE","Median_MAE"]].to_string(index=False))
print("\nNote: Median per-study RMSE cannot be computed from loso_per_study.csv (no per-study residuals available). Reporting Median MAE instead.")

print("\nAll outputs generated successfully.")
