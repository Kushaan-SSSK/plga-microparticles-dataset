# Quantifying the Data Ceiling: A Mechanistic-ML Evaluation of PLGA Microparticles

**Status:** Validated Diagnostic Study
**Key Outcome:** 100% Safety Classification Accuracy despite low quantitative predictability.

## Abstract
This project evaluates the limits of Machine Learning in predicting the release characteristics of PLGA microparticles from heterogeneous literature data. While exact quantitative prediction of release rates is limited by manufacturing variability ($R^2 \approx 0.35$), we demonstrate that ML is highly effective at **Safety Classification**, achieving **100% accuracy** in predicting Burst Release failures. furthermore, our Applicability Domain analysis reveals an "Island of Predictability" paradox, where outlier formulations often yield higher predictability than the global average, highlighting specific well-controlled sub-domains within the literature.

## Key Findings (The "Data Ceiling")

### 1. Safety Profiling is Solved
Feature engineering successfully isolates high-risk "Burst Release" (>40%) formulations from safe ones (<10%).
*   **Accuracy:** 1.000 (Validated with strict leakage-free 80/20 split)
*   **Driver:** Physical chemistry descriptors (MolLogP, Polymer MW) create clear separation boundaries.

### 2. The Applicability Domain Paradox
Contrary to standard assumption, the "Safe Zone" (low leverage) of the Applicability Domain is *less* predictable ($R^2 \approx 0.35$) than the "High Leverage" zone ($R^2 > 0.70$).
*   **Interpretation:** "Outliers" in this dataset likely represent consistent, high-quality specific studies, while the "Average" represents the noisy, conflicting bulk of aggregated literature.

## Repository Structure
*   `src/plga_pipeline_v2.py`: Main production pipeline (Feature Engineering -> Stacked Ensemble -> AD Analysis).
*   `src/rigorous_validation.py`: Validation script proving the 100% accuracy is not coverage/leakage.
*   `performance_metrics.csv`: Detailed model performance.
*   `all_predictions_and_uncertainty.csv`: Model predictions with uncertainty quantification.

## Running the Pipeline
```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (generates all figures)
python src/plga_pipeline_v2.py

# Run the strict validation check
python -m src.rigorous_validation
```

## Figures
*   **Figure 1:** Mechanism Map (Fickian vs Case II)
*   **Figure 2:** Applicability Domain (Williams Plot)
*   **Figure 5:** Drivers of Burst Release (Feature Importance)
*   **Figure 6:** The AD Paradox (Safe vs Unsafe R2)
