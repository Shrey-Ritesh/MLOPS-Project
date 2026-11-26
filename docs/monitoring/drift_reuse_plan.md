# Drift Code Reuse and Windowing Adaptation Plan

**Role:** Person-2 (ML: Drift & Retraining)  
**Topic:** Transitioning from Offline Two-CSV Drift Evaluation to Sliding-Window Production Monitoring  
**Reference Existing Code:** `src/track_b/drift.py`, `src/track_b/evaluate_drift.py`, `configs/track_b/drift_experiment.yaml`

---

## 1. Background & Context

In Phase 1 and Track B, drift evaluation was performed offline by comparing two static CSV datasets:
- **Reference dataset:** `data/synthetic/learner_interactions_v1.csv` (`baseline_v1`)
- **Current dataset:** `data/synthetic/learner_interactions_shifted_v1.csv` (`shifted_v1`)

The output was a single static report (`reports/track_b/drift_report.json`) computing:
- Population Stability Index (PSI) on numerical features (10 quantile bins derived on the fly)
- Categorical PSI over aligned unique categories
- Kolmogorov-Smirnov (KS) two-sample test statistic and p-value

In Phase 3, the model is serving real-time predictions inside the deployed app. We must transition from static batch comparison to **sliding time-window monitoring** over streaming inference logs.

---

## 2. Key Differences: Static vs. Windowed Monitoring

| Aspect | Phase 1 Offline (`evaluate_drift.py`) | Phase 3 Production (`drift_window.py`) |
| :--- | :--- | :--- |
| **Input Reference** | Full static CSV loaded every run | Precomputed `reference_profile.json` (tied to champion's `dataset_sha256`) |
| **Input Current** | Full shifted CSV | A dynamic slice/window of live prediction logs ($N$ predictions over a time window $\Delta t$) |
| **Bin Edges for Numerical PSI** | Computed dynamically from reference series per run | **Fixed precomputed quantile edges** loaded from `reference_profile.json` to ensure comparability across windows |
| **Prediction Drift** | Not computed directly on outputs | Evaluates `prediction_psi` comparing window score distribution against reference score distribution |
| **Performance Tracking** | Static F1/ROC-AUC | Evaluates window F1, ROC-AUC, and Brier score *only when* delayed ground truth labels join |
| **Sample Size Safeguards** | None (datasets were ~20k rows) | If $N < N_{\min}$ (e.g., $N < 50$), returns status `insufficient_data` instead of false "no drift" |

---

## 3. Mathematical Principles & Code Reuse

The core statistical formulas from `src/track_b/drift.py` are preserved:

### 3.1 Numerical Feature PSI
$$\text{PSI} = \sum_{i=1}^{B} (P_i - Q_i) \times \ln\left(\frac{P_i}{Q_i}\right)$$
- $Q_i$: Reference proportion in bin $i$ (precomputed).
- $P_i$: Current window proportion falling within the fixed reference bin edges $[e_{i}, e_{i+1})$.
- Epsilon smoothing ($\epsilon = 10^{-4}$) is applied to prevent division by zero or log of zero.

### 3.2 Categorical Feature PSI
- Categories unioned from reference and observed window.
- Unobserved categories in the window are zero-filled and clipped with $\epsilon$.

### 3.3 Two-Sample Kolmogorov-Smirnov (KS) Test
- Performed between the current window's numerical series and a representative sample of the reference distribution.
- Significant drift flagged when $p\text{-value} < \alpha$ (where $\alpha = 0.01$).

### 3.4 Prediction Score PSI
- Assesses distribution of model probabilities $\hat{p} = P(Y=1 \mid X)$ across the window.
- Uses fixed reference probability quantiles.

---

## 4. Contract Conformance (Section 4.2)

The new window evaluation module `src/monitoring/drift_window.py` outputs a standard dictionary consumed by Person 1's scheduled runner (`src/monitoring/run_window.py`):

```json
{
  "window_id": "win_20261128_1200",
  "start": "2026-11-28T11:00:00Z",
  "end": "2026-11-28T12:00:00Z",
  "n_predictions": 120,
  "n_labelled": 85,
  "drift": {
    "recent_accuracy": {"psi": 0.004, "ks_stat": 0.021, "ks_p": 0.42, "status": "OK"},
    "hint_count": {"psi": 0.245, "ks_stat": 0.128, "ks_p": 1.4e-12, "status": "ALERT"},
    "topic": {"psi": 0.032, "status": "OK"}
  },
  "prediction_psi": 0.182,
  "performance": {
    "f1": 0.542,
    "roc_auc": 0.628,
    "brier": 0.221
  },
  "alerts": [
    {
      "type": "FEATURE_DRIFT",
      "feature": "hint_count",
      "metric": "psi",
      "value": 0.245,
      "threshold": 0.20,
      "severity": "ALERT"
    }
  ],
  "reference_dataset_sha256": "52876f92add9058271b7d42bccb7c83e197815a19be7184642ab8e8e24fbafc0",
  "champion_version": "candidate_hgb_shifted_v2"
}
```

This guarantees complete backward compatibility and clean decoupling between Person 1 (orchestration & storage) and Person 2 (drift mathematics & triggers).
