# California Medi-Cal Deserts Capstone Project - Results Summary

## Data Sources
- County crosswalk: county name.xlsx
- Medi-Cal enrollment: medi-cal-enrollment-dashboard-data.csv
- Population: E4 estiamtes.xlsx
- Physician supply: physicians-actively-working-by-specialty-and-patient-care-hours.xlsx
- Shortage designation: Primary CAre Shortage .csv
- PQI outcomes: PQI.csv
- ACS controls: demoACS.csv, educACS.csv, EconACS.csv

---

## Final Dataset
- **Master Panel Year range:** 2020 (single year due to data intersection)
- **N counties:** 56
- **N county-years:** 56
- **Desert definition (main results):** Quartile-based Definition 2 (high Medi-Cal & (low PCP OR shortage))

### Data Availability by Source
| Dataset | Years Available |
|---------|-----------------|
| PQI Outcomes | 2005-2024 (20 years) |
| Medi-Cal Enrollment | 2019-2022 (4 years) |
| Population (E4) | 2010-2023 (14 years) |
| Physician Supply | Cross-sectional only |
| Shortage Designation | Time-varying |
| ACS Controls | Cross-sectional only |

---

## Key Descriptive Findings

### Medi-Cal Intensity
- Mean Medi-Cal share: 29.7%
- Median Medi-Cal share: 27.1%
- Range: 13.5% - 53.1%
- **Counties >= 30% Medi-Cal:** 27 (48% of counties)

### Primary Care Supply
- Mean PCP per 100k: 92.96
- Median PCP per 100k: 90.96
- Range: 27.37 - 280.66
- **Time-varying:** No (time-invariant)
- **Outlier:** San Francisco (280.66 PCP/100k)

### Preventable Hospitalizations (PQI)
- Mean PQI rate: 205.25 per 100,000
- Median PQI rate: 202.37 per 100,000
- Range: 51.36 - 394.60
- **Top 3 counties:** Colusa (394.6), Butte (347.5), Kings (333.9)
- **Bottom 3 counties:** Trinity (51.4), Del Norte (121.0), Placer (135.4)

### Shortage Designations
- Counties with shortage designation: 53/56 (95%)
- **Note:** Near-universal shortage designation limits statistical utility

### Desert Classifications
| Definition | N Counties | % of Total |
|------------|------------|------------|
| Quartile Def1 (high MC & low PCP) | 5 | 9% |
| Quartile Def2 (high MC & (low PCP OR shortage)) | 14 | 25% |
| Threshold Def1 (>=30% MC & low PCP) | 9 | 16% |
| Threshold Def2 (>=30% MC & (low PCP OR shortage)) | 27 | 48% |

---

## Time Series Analysis

### 20-Year PQI Trends (2005-2024)
- **Pre-ACA (2005-2013) mean PQI:** 289.4 per 100k
- **Post-ACA (2015-2019) mean PQI:** 234.9 per 100k
- **Change:** -18.8% reduction in preventable hospitalizations
- **45/56 counties improved** post-ACA expansion

### COVID-19 Impact (2020)
- **2019 baseline:** 234.9 per 100k
- **2020 COVID year:** 203.9 per 100k
- **Change:** -13.2% drop (likely due to care avoidance)
- **Post-COVID recovery:** 2021-2022 rates returning toward baseline

### Medi-Cal Enrollment Growth (2019-2022)
- **2019:** 12.8 million enrollees
- **2022:** 15.0 million enrollees
- **Growth:** +17.5% increase over 3 years

---

## Key Regression Findings

### Model A: Provider Supply Model
**Dependent Variable:** pcp_per_100k

| Variable | Coefficient | SE | P-value |
|----------|-------------|-----|---------|
| Medi-Cal Share | 100.72 | 72.19 | 0.170 |
| Shortage Flag | 25.91 | 16.34 | 0.120 |

**Interpretation:** A +0.10 increase in medi_cal_share is associated with +10.07 pcp_per_100k (not significant).

### Model B: Outcomes Model
**Dependent Variable:** pqi_mean_rate

| Variable | Coefficient | SE | P-value |
|----------|-------------|-----|---------|
| PCP per 100k | -0.24 | 0.30 | 0.438 |
| Medi-Cal Share | 255.62 | 154.52 | 0.105 |
| Shortage Flag | 19.85 | 35.17 | 0.576 |

**Interpretation:** An increase of +10 pcp_per_100k is associated with -2.37 pqi_mean_rate (not significant).

### Model C: Desert Indicator Model
**Dependent Variable:** pqi_mean_rate

| Variable | Coefficient | SE | P-value |
|----------|-------------|-----|---------|
| Desert (Quartile Def2) | 7.23 | 22.48 | 0.749 |

---

## >= 30% Medi-Cal Threshold Analysis

### Main Finding
Counties with >=30% Medi-Cal share have **64.52 higher PQI rate** (p=0.007).

This is equivalent to **37.6% higher preventable hospitalizations** (log model: coef=0.319, p=0.011).

---

## Comprehensive Robustness Analysis

### 1. Population-Weighted Models

| Model | Coefficient | P-value | Significant |
|-------|-------------|---------|-------------|
| Unweighted | 64.52 | 0.007 | Yes |
| **Population-weighted** | **14.37** | **0.265** | **No** |

**CRITICAL FINDING:** The effect **DISAPPEARS** when weighted by population.
- Result is driven by **small rural counties**, NOT large population centers
- Los Angeles, San Diego, Orange County do NOT show the same pattern
- **Policy implication:** Statewide claims should be made cautiously

### 2. Region Fixed Effects

| Model | Coefficient | P-value |
|-------|-------------|---------|
| Without regions | 64.52 | 0.007 |
| With region FE | 68.74 | 0.005 |

**Finding:** Effect PERSISTS with region fixed effects - Not purely geographic clustering

### 3. Threshold Grid Search

| Threshold | Coefficient | P-value | N Counties Above |
|-----------|-------------|---------|------------------|
| >=20% | 7.95 | 0.740 | 45 |
| >=25% | 37.53 | 0.074 | 33 |
| **>=30%** | **64.52** | **0.007** | **27** |
| >=35% | 14.38 | 0.554 | 20 |
| >=40% | -12.96 | 0.679 | 7 |

**Finding:** 30% shows peak effect; neighboring thresholds weaker.

### 4. Nonlinearity Tests

**Quadratic Model:**
- Linear term: 792.69 (p=0.078)
- Quadratic term: -913.23 (p=0.198)
- Vertex at 43.4% - Inverted U-shape

**Piecewise Spline (knot at 30%):**
- Slope below 30%: 394.66 (p=0.062)
- Slope above 30%: 81.10 (p=0.727)
- Effect levels off past 30% threshold

### 5. Interaction Effect: PCP x Medi-Cal

| Variable | Coefficient | P-value |
|----------|-------------|---------|
| PCP per 100k | -0.36 | 0.609 |
| Medi-Cal share | 218.01 | 0.382 |
| PCP x Medi-Cal | 0.46 | 0.845 |

**Finding:** No significant interaction. Extra PCP supply is NOT especially protective in high Medi-Cal counties.

### 6. Influence Diagnostics

**High-Influence Counties (Cook's D > 0.087):**
1. Inyo (Cook's D = 0.464)
2. Trinity (0.157)
3. Butte (0.146)
4. San Francisco (0.139)
5. Del Norte (0.134)
6. Imperial (0.128)
7. Mariposa (0.096)
8. Colusa (0.093)

**Sensitivity to Exclusions:**

| Excluded | Coefficient | P-value |
|----------|-------------|---------|
| None (baseline) | 64.52 | 0.007 |
| All 8 high-influence | 52.52 | 0.003 |

**Finding:** Result remains significant even excluding all high-influence counties.

### 7. Leave-One-Out Analysis
- **Base coefficient:** 64.52
- **Range when excluding individual counties:** 52.72 - 73.28
- **All 56 exclusions remain significant at p<0.05**

### 8. Nested Models (Stability Check)

| Controls | Coefficient | P-value | R-squared |
|----------|-------------|---------|-----------|
| None | 63.01 | 0.0001 | 0.257 |
| + PCP | 58.67 | 0.0005 | 0.265 |
| + Poverty, Unemployment | 78.74 | 0.0002 | 0.333 |
| Full controls | 64.52 | 0.0069 | 0.409 |

**Finding:** Coefficient is STABLE across different control specifications.

### 9. Mediation Decomposition

| Path | Estimate |
|------|----------|
| Total effect (MC to PQI) | 62.89 |
| Direct effect (MC to PQI given PCP) | 64.47 |
| Indirect via PCP | -1.58 |
| **% mediated by PCP** | **-2.5%** |

**Finding:** Virtually NONE of the Medi-Cal effect operates through PCP supply differences.

### 10. Alternative Outcome Models

| Model | Coefficient | P-value | Interpretation |
|-------|-------------|---------|----------------|
| OLS (levels) | 64.52 | 0.007 | +64.5 PQI rate |
| Log(PQI) | 0.319 | 0.011 | +37.6% higher |
| Median regression | 31.26 | 0.155 | Not significant |
| Winsorized (1%/99%) | 60.97 | 0.007 | Robust to outliers |

---

## Summary: Robustness Assessment

### PASSES
- Region fixed effects
- Nested model stability
- Leave-one-out (all remain significant)
- Log transformation
- Winsorization
- Excluding high-influence counties

### CONCERNS
- **Population weighting eliminates effect** (driven by small counties)
- Threshold choice appears data-driven (30% is "magic number")
- Median regression not significant (influenced by outliers)
- No PCP x Medi-Cal interaction (mechanism unclear)

---

## Limitations

1. **Single Year Master Panel:** Due to data intersection, master panel only has 2020.

2. **Population Weighting:** Main finding driven by small rural counties, not large urban centers.

3. **ACS Controls:** Single-year (baseline), treated as time-invariant.

4. **Physician Supply:** Time-invariant, limits within-county analysis.

5. **Shortage Designation:** 95% of counties designated, limiting statistical utility.

6. **Ecological Inference:** County-level results may not apply to individuals.

7. **Causality:** Observational associations only.

---

## Policy Implications

1. **Targeting:** Focus resources on **small/rural counties** with high Medi-Cal share.

2. **Beyond Physician Supply:** The Medi-Cal effect is NOT operating through PCP supply differences.

3. **30% Threshold:** Counties crossing ~30% Medi-Cal share warrant additional monitoring.

4. **Central Valley Focus:** Region shows concentration of high Medi-Cal, high PQI counties.

---

## Output Files Generated

### Data Files (outputs/data/)
- county_crosswalk_clean.csv
- pop_e4_clean.csv
- medical_enrollment_clean.csv
- physician_supply_clean.csv
- shortage_clean.csv
- acs_controls_clean.csv
- pqi_long_clean.csv
- pqi_county_year_clean.csv
- ca_master_panel.csv
- ca_variable_dictionary.csv

### Figures (outputs/figures/)

#### Descriptive
- hist_medi_cal_share.png
- hist_pcp_per_100k.png
- scatter_medi_cal_vs_pcp.png
- scatter_pcp_vs_pqi.png
- bar_top_bottom_pqi.png
- desert_comparison.png
- correlation_heatmap.png
- county_pqi_ranking.png
- scatter_desert_highlight.png

#### Time Series
- time_series_statewide.png
- pqi_trend_2005_2024.png
- medi_cal_trend_2019_2022.png
- ts_pqi_statewide_20yr.png
- ts_pqi_by_measure.png
- ts_medi_cal_enrollment.png
- ts_pqi_medi_cal_combined.png
- ts_pqi_by_county.png
- ts_pre_post_aca.png
- ts_covid_impact.png

#### Robustness
- robustness_threshold_grid.png
- robustness_quadratic.png
- robustness_cooks_d.png
- robustness_leave_one_out.png
- robustness_summary_all.png
- coefficient_plot_model_b.png

### Tables (outputs/tables/)

#### Main Results
- primary_care_specialty_definition.csv
- reg_model_a_supply.csv
- reg_model_b_outcomes.csv
- reg_model_c_desert.csv
- regression_summary.md
- all_regression_results.csv
- all_regression_results.md
- regression_equations.md

#### Robustness
- sensitivity_checks.csv
- robustness_summary.csv
- threshold_grid.csv
- leave_one_out.csv
- nested_models.csv
- influence_exclusions.csv

#### Diagnostics
- missingness_table.csv
- correlation_matrix.csv
- county_coverage_by_year.csv
- desert_counties_list.csv

### Reports
- outputs/results_summary.md (this file)
- outputs/FINDINGS_REVIEW.md
- outputs/ROBUSTNESS_REPORT.md
- outputs/README.md

---

*Generated: 2026-01-19*

---

## Part 10: Hospital Cost Analysis (2012-2023)

### Data Integration
- **Source:** California OSHPD/HCAI Hospital Financial Data (2012-2023)
- **Files:** 12 annual cost files covering 380+ hospitals per year
- **Counties matched:** 54 of 56 counties

### Cost Summary Statistics (2020)

| Metric | Mean | Median | Range |
|--------|------|--------|-------|
| Total Operating Expenses | $2.33B | $0.61B | $18M - $49.8B |
| Cost per Patient Day | $9,153 | $6,919 | Varies |
| Medi-Cal Revenue % | 28.7% | 25.4% | Varies |

### Cost Trends (2014-2023)

| Year | Total Expenses | Cost/Day | Medi-Cal % |
|------|----------------|----------|------------|
| 2014 | $90.3B | $5,632 | 20.7% |
| 2017 | $107.9B | $6,456 | 24.9% |
| 2020 | $125.7B | $7,579 | 25.1% |
| 2023 | $168.6B | $9,618 | 23.3% |

**Key Trends:**
- Hospital operating expenses grew 87% from 2014 to 2023
- Cost per patient day increased 71% ($5,632 → $9,618)
- Medi-Cal revenue share fluctuated around 24-28%

### Correlations with PQI Rate

| Variable | Correlation |
|----------|-------------|
| Total Operating Expenses | 0.019 |
| Cost per Patient Day | -0.212 |
| Medi-Cal Revenue % | 0.002 |
| Medi-Cal Share (enrollment) | 0.426 |

**Finding:** Hospital costs show weak negative correlation with PQI (-0.212), suggesting higher-cost hospitals may have slightly lower preventable hospitalization rates.

### Cost-Outcome Regression Models

**Model 1: Cost Variables Only**
- Cost per Patient Day: -0.0013 (p=0.120) - Not significant
- Medi-Cal Revenue %: -0.70 (p=0.268) - Not significant
- R² = 0.363

**Model 2: Full Model with Cost + PCP + Medi-Cal**
- Cost per Patient Day: -0.0012 (p=0.164)
- Medi-Cal Share: 181.0 (p=0.305)
- PCP per 100k: -0.12 (p=0.714)
- R² = 0.378

**Model 3: ≥30% Threshold with Cost Controls**
- **≥30% MC Threshold: 56.32 (p=0.029)** ✓ Significant
- Cost per Patient Day: -0.0011 (p=0.181)
- R² = 0.429

**Key Finding:** The ≥30% Medi-Cal threshold effect SURVIVES controlling for hospital costs.

### Desert vs Non-Desert Cost Comparison

| Metric | Non-Desert | Desert |
|--------|------------|--------|
| Cost per Patient Day | Higher | Lower |
| Medi-Cal Revenue % | Lower | Higher |

### New Output Files

**Data:**
- outputs/data/hospital_costs_county_year.csv
- outputs/data/ca_master_panel_with_costs.csv

**Figures:**
- outputs/figures/cost_analysis_overview.png
- outputs/figures/cost_desert_comparison.png

**Tables:**
- outputs/tables/cost_regression_results.csv

---

*Updated: 2026-01-19 with hospital cost analysis*
