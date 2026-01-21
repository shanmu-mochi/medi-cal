# ED Intensity & System Strain Regression Results

## Overview
This analysis tests whether high Medi-Cal share → system strain (ED utilization) → worse outcomes (PQI).

## Mechanism Step 1: ED Intensity as Outcome

| Model | DV | Predictor | Weighted | Coef | SE | P-value | R² | N |
|-------|-----|-----------|----------|------|-----|---------|-----|---|
| ED1a | ed_visits_per_1k | medi_cal_share | No | -428.65 | 331.17 | 0.1955 | 0.385 | 55 |
| ED1a_W | ed_visits_per_1k | medi_cal_share | Yes | -452.58 | 258.23 | 0.0797 | 0.517 | 55 |
| ED1b | ed_visits_per_1k | medi_cal_ge_30 | No | -7.62 | 37.40 | 0.8386 | 0.356 | 55 |
| ED1b_W | ed_visits_per_1k | medi_cal_ge_30 | Yes | -2.51 | 19.84 | 0.8993 | 0.474 | 55 |
| ED2 | log_ed_visits_per_1k | medi_cal_share | No | -0.92 | 0.88 | 0.2961 | 0.405 | 55 |
| ED2_W | log_ed_visits_per_1k | medi_cal_share | Yes | -1.08 | 0.78 | 0.1653 | 0.536 | 55 |


**Key Finding:** Higher Medi-Cal share is NOT significantly associated with higher ED utilization.

## Mechanism Step 2: PQI with ED Control

| Model | Specification | Weighted | MC Coef | MC P-val | ED Coef | ED P-val | R² | N |
|-------|---------------|----------|---------|----------|---------|----------|-----|---|
| PQI1 | Baseline | No | 207.36 | 0.1392 | N/A | N/A | 0.324 | 55 |
| PQI1_W | Baseline | Yes | 88.77 | 0.4186 | N/A | N/A | 0.406 | 55 |
| PQI2 | With ED | No | 194.92 | 0.2272 | -0.029 | 0.7634 | 0.326 | 55 |
| PQI2_W | With ED | Yes | 101.30 | 0.3966 | 0.028 | 0.6630 | 0.407 | 55 |
| PQI3 | Interaction | No | 198.88 | 0.4316 | -0.026 | 0.8816 | 0.326 | 55 |
| PQI3_W | Interaction | Yes | 173.44 | 0.2734 | 0.126 | 0.3704 | 0.410 | 55 |
| PQI_THR | >=30% + ED | No | 58.75 | 0.0161 | -0.045 | 0.5782 | 0.392 | 55 |


**Key Findings:**
1. Baseline MC coef = 207.36
2. With ED, MC coef = 194.92 (-6.0% change)
3. Interaction term p-value: 0.9810

---
*Generated: 2026-01-19*
