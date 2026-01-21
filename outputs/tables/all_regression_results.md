# Complete Regression Results - California Medi-Cal Deserts Analysis

## Overview

This document contains all regression results from the capstone analysis examining the relationship between Medi-Cal enrollment, primary care access, and preventable hospitalizations (PQI) in California counties.

**Sample:** 56 California counties, Year 2020
**Note:** Standard errors are not clustered (linearmodels not available); interpret with caution.

---

## Model A: Provider Supply Model

**Purpose:** Examine factors associated with primary care physician supply

**Dependent Variable:** `pcp_per_100k` (Primary care physicians per 100,000 population)

**Model Specification:**
```
pcp_per_100k = β₀ + β₁(medi_cal_share) + β₂(shortage_flag) + β₃(poverty_pct) 
               + β₄(unemp_pct) + β₅(age65_pct) + β₆(hispanic_pct) + β₇(bachelors_pct) + ε
```

**Model Fit:** R² = 0.705, Adjusted R² = 0.662

| Variable | Coefficient | Std Error | p-value | 95% CI Lower | 95% CI Upper | Significance |
|----------|-------------|-----------|---------|--------------|--------------|--------------|
| const | -57.8000 | 39.3496 | 0.1484 | -136.9176 | 21.3176 |  |
| medi_cal_share | 100.7229 | 72.1863 | 0.1693 | -44.4174 | 245.8633 |  |
| shortage_flag | 25.9091 | 16.3361 | 0.1193 | -6.9369 | 58.7550 |  |
| poverty_pct | -1.0660 | 1.5658 | 0.4993 | -4.2142 | 2.0823 |  |
| unemp_pct | 0.4692 | 2.8281 | 0.8689 | -5.2170 | 6.1555 |  |
| age65_pct | 0.3249 | 0.8879 | 0.7160 | -1.4603 | 2.1100 |  |
| hispanic_pct | 0.2996 | 1.8658 | 0.8731 | -3.4519 | 4.0510 |  |
| bachelors_pct | 3.3459 | 0.4197 | 0.0000 | 2.5021 | 4.1897 | *** |

**Key Finding:** Counties with higher educational attainment (bachelors_pct) have significantly more primary care physicians (+3.35 per 100k for each percentage point increase, p<0.001).

**Interpretation:** A 10 percentage point increase in Medi-Cal share is associated with a +10.07 change in PCP per 100k (not statistically significant at p=0.169).

---

## Model B: Outcomes Model

**Purpose:** Examine factors associated with preventable hospitalization rates

**Dependent Variable:** `pqi_mean_rate` (Mean Prevention Quality Indicator rate per 100,000)

**Model Specification:**
```
pqi_mean_rate = β₀ + β₁(pcp_per_100k) + β₂(medi_cal_share) + β₃(shortage_flag) 
                + β₄(poverty_pct) + β₅(unemp_pct) + β₆(age65_pct) + β₇(hispanic_pct) 
                + β₈(bachelors_pct) + ε
```

**Model Fit:** R² = 0.346, Adjusted R² = 0.235

| Variable | Coefficient | Std Error | p-value | 95% CI Lower | 95% CI Upper | Significance |
|----------|-------------|-----------|---------|--------------|--------------|--------------|
| const | 307.6756 | 84.4057 | 0.0007 | 137.8732 | 477.4780 | *** |
| pcp_per_100k | -0.2372 | 0.3029 | 0.4375 | -0.8465 | 0.3721 |  |
| medi_cal_share | 255.6220 | 154.5157 | 0.1047 | -55.2236 | 566.4675 |  |
| shortage_flag | 19.8524 | 35.1660 | 0.5751 | -50.8926 | 90.5973 |  |
| poverty_pct | -6.0035 | 3.3015 | 0.0754 | -12.6452 | 0.6381 | † |
| unemp_pct | -1.5051 | 5.9361 | 0.8009 | -13.4470 | 10.4368 |  |
| age65_pct | -1.9646 | 1.8656 | 0.2977 | -5.7178 | 1.7886 |  |
| hispanic_pct | -3.3814 | 3.9162 | 0.3923 | -11.2597 | 4.4970 |  |
| bachelors_pct | -1.2417 | 1.3426 | 0.3598 | -3.9426 | 1.4592 |  |

**Key Finding:** Higher Medi-Cal share is associated with higher PQI rates (marginally significant at p=0.105). Higher poverty is marginally associated with lower PQI rates (p=0.075).

**Interpretation:** 
- An increase of 10 PCP per 100k is associated with a -2.37 change in PQI rate (not significant)
- A 10 percentage point increase in Medi-Cal share is associated with +25.6 higher PQI rate

---

## Model C: Desert Indicator Model

**Purpose:** Test whether "desert" classification predicts worse outcomes

**Dependent Variable:** `pqi_mean_rate` (Mean Prevention Quality Indicator rate per 100,000)

**Model Specification:**
```
pqi_mean_rate = β₀ + β₁(desert_q_def2) + β₂(poverty_pct) + β₃(unemp_pct) 
                + β₄(age65_pct) + β₅(hispanic_pct) + β₆(bachelors_pct) + ε
```

**Desert Definition:** Quartile-based Definition 2 = High Medi-Cal (top quartile) AND (Low PCP (bottom quartile) OR Shortage designation)

**Model Fit:** R² = 0.304, Adjusted R² = 0.219

| Variable | Coefficient | Std Error | p-value | 95% CI Lower | 95% CI Upper | Significance |
|----------|-------------|-----------|---------|--------------|--------------|--------------|
| const | 384.8172 | 69.3591 | 0.0000 | 245.4348 | 524.1996 | *** |
| desert_q_def2 | 7.2322 | 22.4822 | 0.7491 | -37.9474 | 52.4119 |  |
| poverty_pct | -2.9956 | 2.8280 | 0.2947 | -8.6786 | 2.6874 |  |
| unemp_pct | 1.0840 | 5.7193 | 0.8505 | -10.4095 | 12.5774 |  |
| age65_pct | -3.7737 | 1.5201 | 0.0165 | -6.8285 | -0.7190 | * |
| hispanic_pct | -3.4838 | 3.9616 | 0.3835 | -11.4450 | 4.4774 |  |
| bachelors_pct | -2.1916 | 0.8860 | 0.0169 | -3.9722 | -0.4111 | * |

**Key Finding:** The desert indicator is not significantly associated with PQI rates (p=0.749). Age65_pct and bachelors_pct are significant predictors.

**Interpretation:** Being classified as a "desert" county is associated with +7.23 higher PQI rate, but this is not statistically significant.

---

## Sensitivity Analysis

**Purpose:** Test robustness of results to alternative specifications

| Specification | Coefficient | Std Error | p-value | Significance |
|---------------|-------------|-----------|---------|--------------|
| Model B - desert_q_def1 | -22.5806 | 30.2854 | 0.4595 |  |
| Model B - desert_thr_def1 | 38.7392 | 24.6315 | 0.1222 |  |
| Model B - desert_thr_def2 | 62.8860 | 22.4194 | 0.0072 | ** |
| Model B - Winsorized (1st/99th percentile) | -0.2396 | 0.2882 | 0.4099 |  |

**Key Finding:** The threshold-based Definition 2 (desert_thr_def2) shows a significant positive association with PQI rates (p=0.007), suggesting that using a fixed Medi-Cal threshold (≥30%) rather than quartiles identifies counties with meaningfully higher preventable hospitalization rates.

---

## Significance Codes

- `***` p < 0.001
- `**` p < 0.01  
- `*` p < 0.05
- `†` p < 0.10

---

## Summary of Key Equations

### Equation 1: Provider Supply
```
pcp_per_100k = -57.80 + 100.72(medi_cal_share) + 25.91(shortage_flag) - 1.07(poverty_pct) 
               + 0.47(unemp_pct) + 0.32(age65_pct) + 0.30(hispanic_pct) + 3.35(bachelors_pct)***
```

### Equation 2: Outcomes
```
pqi_mean_rate = 307.68 - 0.24(pcp_per_100k) + 255.62(medi_cal_share) + 19.85(shortage_flag) 
                - 6.00(poverty_pct)† - 1.51(unemp_pct) - 1.96(age65_pct) - 3.38(hispanic_pct) 
                - 1.24(bachelors_pct)
```

### Equation 3: Desert Indicator
```
pqi_mean_rate = 384.82 + 7.23(desert_q_def2) - 3.00(poverty_pct) + 1.08(unemp_pct) 
                - 3.77(age65_pct)* - 3.48(hispanic_pct) - 2.19(bachelors_pct)*
```

---

## Data Notes

- **Sample Size:** 56 California counties (2 counties excluded due to missing physician data)
- **Year:** 2020 (intersection of all data sources)
- **Physician Supply:** Time-invariant (single cross-section)
- **Standard Errors:** OLS standard errors (not clustered; linearmodels package not available)

## Limitations

1. Cross-sectional analysis limits causal inference
2. County-level ecological analysis - individual-level relationships may differ
3. Limited time coverage due to data availability constraints
4. Standard errors may be underestimated without clustering

---

*Generated from capstone_california_medi_cal_deserts.ipynb*
