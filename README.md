# Diagnosing Predictability Limits of In Vitro Drug Release from Published PLGA Microparticle Data

This repository provides the reproduction protocol for the paper. No authors or affiliations are listed for anonymous review.

---

## What this code reproduces

- **Figures:** Mechanism map (Peppas n), applicability domain (Williams plot), feature importance, burst classifier importance, AD paradox, uncertainty calibration, benchmark comparison, burst classification confusion matrix.
- **Tables / metrics:** Regression R² and MAE (Peppas n, Peppas K, Burst 24 h), burst classification accuracy, benchmark R² by model.
- **Validation:** Strict 80/20 grouped train/test split for burst classification (no leakage).

---

## Environment setup

- **Python:** 3.10 (recommended; 3.9 minimum).
- **Commands:**

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## Data

The dataset is not included. Obtain the two Excel files as described in the paper (or from the supplement), then:

1. Place `mp_dataset_processed.xlsx` and `mp_dataset_initial.xlsx` into `data/`.
2. Ensure `Time` in the processed file is in **hours** (Burst_24h is release at 24 h).

---

## How to run

From the repository root, with the venv activated:

```bash
python scripts/run_all.py
```

This single command runs the full pipeline, validation, benchmarks, and visualizations. Outputs are written to `outputs/` (created automatically).

**Optional:** `python scripts/run_all.py --fast` runs only the main pipeline and validation (no benchmarks or extra figures).

---

## Expected outputs

Generated under `outputs/`:

| Output | Description |
|--------|-------------|
| `performance_metrics.csv` | R², MAE, RMSE per target; burst classification accuracy |
| `all_predictions_and_uncertainty.csv` | Per-sample predictions and uncertainty |
| `Figure1_MechanismMap.png` | Predicted vs actual Peppas n |
| `Figure2_ApplicabilityDomain.png` | Williams plot (Burst) |
| `Figure3_FeatureImportance.png` | Top drivers of release mechanism |
| `Figure5_BurstImportance.png` | Drivers of burst (safety) |
| `Figure6_AD_Paradox.png` | R² in safe vs high-leverage zones |
| (full run) `Figure4_UncertaintyCalibration.png`, `Figure5_BurstClassification.png`, `Figure6_Benchmarking.png`, `benchmark_results.csv` | Extra figures and benchmark table |

---

## Runtime and hardware

- CPU only; no GPU required.
- Full run: approximately 5–15 minutes. Fast run: approximately 2–5 minutes.

---

## Reproducibility

- **Determinism:** Random seed 42 is set in `config.py` and used for numpy, sklearn, and XGBoost. Train/validation/test splits are fixed.
- **Environment:** Python 3.10 and library versions are pinned in `requirements.txt`. Use the same environment for matching results.
- **Git history:** For anonymous review, clean commit history before submission (e.g. squash to one commit or re-initialize the repo and make a single clean commit). Reviewers may check commit authors.
