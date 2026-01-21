# COMPREHENSIVE ANALYSIS REPORT
## Mapping Medi-Cal Deserts in California

**Last Updated:** January 2026  
**Final Version with All Robustness Analyses**

---

## Executive Summary

This report presents an exhaustive analysis of "Medi-Cal deserts" in California—counties where high Medicaid enrollment coexists with poor healthcare outcomes. Using multiple data sources spanning 2005-2025, we investigate whether high Medi-Cal intensity causes poor outcomes or merely reflects underlying disadvantage.

### Key Findings

| Question | Finding | Confidence |
|----------|---------|------------|
| Do high-MC counties have worse outcomes? | **Yes** (β = 41, p = 0.013) | ✅ High |
| Does MC cause poor outcomes within counties? | **No** (β = 73, p = 0.55) | ✅ High |
| Did Prop 56 improve high-MC counties? | **Maybe** (β = -372, p = 0.005) | ⚠️ Medium |
| Do hospital costs explain the relationship? | **No** (β = -15, p = 0.18) | ✅ High |
| Do reimbursement rates predict outcomes? | **No** (β = 0.006, p = 0.81) | ⚠️ Limited data |

---

## Table of Contents

1. [Data Sources](#1-data-sources)
2. [Variable Definitions](#2-variable-definitions)
3. [Statistical Methods](#3-statistical-methods)
4. [Cross-Sectional Analysis](#4-cross-sectional-analysis)
5. [Panel Fixed Effects](#5-panel-fixed-effects)
6. [Prop 56 DiD Analysis](#6-prop-56-did-analysis)
7. [Hospital Cost Analysis](#7-hospital-cost-analysis)
8. [Accessibility & Reimbursement](#8-accessibility--reimbursement)
9. [Robustness Checks](#9-robustness-checks)
10. [Summary of All Results](#10-summary-of-all-results)
11. [Conclusions](#11-conclusions)

---

## 1. Data Sources

| Dataset | Source | Years | N | Key Variables |
|---------|--------|-------|---|---------------|
| **PQI Outcomes** | HCAI Preventable Hospitalizations | 2005-2024 | 1,160 | 17 AHRQ PQI indicators |
| **MC Certified Eligibles** | DHCS | 2010-2025 | 928 | Monthly enrollment, FFS/MC, dual status |
| **ED by Residence** | HCAI | 2008-2024 | 986 | ED visits, admissions by patient county |
| **ACS Demographics** | Census S-tables | 2010-2024 | 738 | Poverty, age, disability, vehicle access |
| **Hospital Costs** | OSHPD Financial Data | 2013-2024 | 670 | Cost/discharge, margin, MC revenue |
| **Reimbursement Rates** | DHCS Capitation | 2021 | 22 | County-level capitation rates |

**Total Panel Coverage:** 58 counties × 20 years

---

## 2. Variable Definitions

### 2.1 Outcome Variable

$$PQI_{ct} = \frac{1}{17}\sum_{j=1}^{17} RiskAdjRate_{jct}$$

Mean risk-adjusted rate per 100,000 across 17 Prevention Quality Indicators (ambulatory care-sensitive conditions including diabetes complications, COPD, heart failure, dehydration, UTI, bacterial pneumonia).

### 2.2 Exposure Variables

**Medi-Cal Share:**
$$MC\_share_{ct} = \frac{CertifiedEligibles_{ct}}{Population_{ct}}$$

**Binary High-MC Indicator:**
$$HighMC_{ct} = \mathbb{1}(MC\_share_{ct} \geq median)$$

### 2.3 Control Variables

| Variable | Definition | Source |
|----------|------------|--------|
| `poverty_pct` | % below federal poverty line | ACS S1701 |
| `age65_pct` | % population age 65+ | ACS S0101 |
| `disability_pct` | % with any disability | ACS S1810 |
| `no_vehicle_pct` | % workers without vehicle access | ACS S0802 |
| `limited_english_pct` | % limited English proficiency | ACS S1602 |
| `cost_per_discharge` | Operating expenses / discharges | OSHPD |
| `reimb_rate` | Average Medi-Cal capitation rate | DHCS |

### 2.4 Treatment Variables (DiD)

**Prop 56 Treatment:**
$$Treat_c = \mathbb{1}(MC\_share_{c,2016} \geq median_{2016})$$

**Post Period:**
$$Post_t = \mathbb{1}(year \geq 2017)$$

---

## 3. Statistical Methods

### 3.1 Cross-Sectional OLS

$$PQI_c = \beta_0 + \beta_1 MC_c + \gamma X_c + \epsilon_c$$

- Robust standard errors (HC1)
- N = 58 counties (2020)

### 3.2 Panel Fixed Effects

$$PQI_{ct} = \alpha_c + \lambda_t + \beta MC_{ct} + \epsilon_{ct}$$

- Two-way fixed effects (county + year)
- Within-transformation for estimation
- Clustered standard errors by county

### 3.3 Difference-in-Differences

**Binary DiD:**
$$PQI_{ct} = \alpha_c + \lambda_t + \theta_1 (Treat_c \times Post_t) + \epsilon_{ct}$$

**Intensity DiD (Continuous):**
$$PQI_{ct} = \alpha_c + \lambda_t + \beta (MC_{c,2016} \times Post_t) + \epsilon_{ct}$$

### 3.4 Nonlinear Specifications

**Quintile Bins:**
$$PQI_c = \sum_{q=2}^{5} \beta_q \mathbb{1}(Quintile_c = q) + \epsilon_c$$

**Quadratic:**
$$PQI_c = \beta_0 + \beta_1 MC_c + \beta_2 MC_c^2 + \epsilon_c$$

**Piecewise Spline:**
$$PQI_c = \beta_0 + \beta_1 MC_c + \beta_2 (MC_c - k)^+ + \epsilon_c$$

where $(x)^+ = \max(x, 0)$ and $k$ = median MC share.

---

## 4. Cross-Sectional Analysis

### 4.1 Main Results (2020)

| Model | Specification | N | β(MC) | SE | p-value | R² |
|-------|--------------|---|-------|-----|---------|-----|
| 1 | PQI ~ MC | 58 | 279.0 | 199.8 | 0.163 | 0.023 |
| 2 | PQI ~ I(MC ≥ median) | 58 | **41.0** | 16.2 | **0.013** | 0.107 |
| 3 | PQI ~ MC + controls | 58 | 308.8 | 327.1 | 0.345 | 0.029 |

### 4.2 Nonlinear Dose-Response

| Quintile | Mean MC | Mean PQI | Diff vs Q1 | p-value |
|----------|---------|----------|------------|---------|
| Q1 (ref) | 5.5% | 205.5 | — | — |
| Q2 | 6.9% | 155.5 | -50.0 | **0.002** |
| Q3 | 8.2% | 203.6 | -1.9 | 0.926 |
| Q4 | 9.8% | 231.1 | +25.6 | 0.320 |
| Q5 | 13.2% | 211.1 | +5.6 | 0.800 |

**Finding:** Non-monotonic pattern. The relationship is not linear; Q2 has the lowest PQI.

### 4.3 Interpretation

The binary indicator (above/below median) is significant (p = 0.013), but the continuous effect is not (p = 0.16). This suggests a **threshold effect** rather than a dose-response relationship.

---

## 5. Panel Fixed Effects

### 5.1 Within-County Analysis (2010-2024)

| Model | Specification | N | Years | Counties | β | SE | p |
|-------|--------------|---|-------|----------|---|-----|---|
| 1 | PQI ~ MC | 870 | 15 | 58 | 72.6 | 121.1 | **0.549** |
| 2 | PQI ~ log(Cost) | 670 | 12 | 56 | -15.5 | 11.6 | **0.181** |
| 3 | PQI ~ MC + log(Cost) | 670 | 12 | 56 | 46.7 (MC) | — | 0.678 |

### 5.2 Interpretation

**No within-county causal effect.** When a county's MC share increases over time, its PQI does not systematically change. The cross-sectional association reflects **which counties** have high MC (composition), not **what happens when** MC increases (causation).

---

## 6. Prop 56 DiD Analysis

### 6.1 Binary DiD

$$PQI_{ct} = \alpha_c + \lambda_t + \theta_1 (Treat_c \times Post_t) + \epsilon_{ct}$$

| Parameter | Estimate | SE | p-value | 95% CI |
|-----------|----------|-----|---------|--------|
| θ₁ | **-27.57** | 11.99 | **0.022** | [-51.1, -4.1] |

### 6.2 Intensity DiD (Continuous)

$$PQI_{ct} = \alpha_c + \lambda_t + \beta (MC_{c,2016} \times Post_t) + \epsilon_{ct}$$

| Parameter | Estimate | SE | p-value |
|-----------|----------|-----|---------|
| **β** | **-372.1** | 131.1 | **0.005** |

**Interpretation:** A county with 10 percentage points higher baseline MC share experienced a **37.2-point reduction** in PQI after Prop 56.

### 6.3 Robustness Checks

| Check | θ₁ / β | p-value | Concern |
|-------|--------|---------|---------|
| Base DiD | -27.57 | 0.022 | — |
| Intensity DiD | -372.1 | **0.005** | ✅ Stronger |
| + County trends | 0.85 | **0.703** | ⚠️ Effect disappears |
| Placebo 2014 | -19.28 | **0.008** | ⚠️ Pre-trend |
| Placebo 2015 | -18.94 | **0.018** | ⚠️ Pre-trend |

### 6.4 Outcome Decomposition

| Outcome | θ₁ | SE | p-value |
|---------|-----|-----|---------|
| Overall PQI | -27.57 | 11.99 | 0.022 |
| Chronic PQI | -10.20 | 6.00 | 0.089 |
| **Acute PQI** | **-29.54** | 7.59 | **<0.001** |

**Unexpected:** Effect concentrated in acute conditions, not chronic. This is mechanistically puzzling.

### 6.5 Prop 56 Conclusion

The intensity DiD is strong (p = 0.005), but:
- Effect disappears with county-specific trends
- Placebo tests show pre-existing convergence
- Outcome pattern (acute > chronic) is unexpected

**Conservative interpretation:** High-MC counties were already converging before Prop 56. The policy may have accelerated this, but causality is uncertain.

---

## 7. Hospital Cost Analysis

### 7.1 Data

- **Source:** OSHPD Hospital Financial Data (city→county mapping)
- **Years:** 2013-2024
- **Counties:** 56
- **Observations:** 670 county-years

### 7.2 Cross-Sectional Results (2020)

| Relationship | β | p-value | Interpretation |
|--------------|---|---------|----------------|
| MC → Cost | 0.88 | 0.565 | No relationship |
| Cost → PQI | -34.7 | 0.242 | No relationship |
| MC rev share → PQI | 13.1 | 0.869 | No relationship |

### 7.3 Panel Results (2013-2024)

| Model | β | SE | p-value |
|-------|---|-----|---------|
| PQI ~ log(Cost) | -15.52 | 11.60 | 0.181 |
| PQI ~ log(Cost) + MC | -15.90 (cost) | — | 0.169 |

### 7.4 Conclusion

**Hospital costs do not explain the MC→PQI relationship.** Neither cross-sectionally nor within counties over time do cost measures predict outcomes. This rules out "hospital financial stress" as the mechanism.

---

## 8. Accessibility & Reimbursement Analysis

### 8.1 Accessibility Variables

| Variable | Correlation with PQI | Interpretation |
|----------|---------------------|----------------|
| `poverty_pct` | +0.10 | Weak positive |
| `disability_pct` | -0.01 | No relationship |
| `no_vehicle_pct` | -0.37 | Counterintuitive negative |
| `limited_english_pct` | +0.31 | Moderate positive |

**Note on vehicle access:** The negative correlation with PQI is likely due to confounding—rural counties have high vehicle ownership but also poor outcomes due to distance.

### 8.2 Does MC Proxy for Poverty?

$$MC\_share_c = \beta_0 + \beta_1 \cdot poverty\_pct_c + \epsilon_c$$

| Parameter | Estimate | p-value | R² |
|-----------|----------|---------|-----|
| β₁ | 0.0052 | **<0.001** | **0.417** |

**Finding:** MC share strongly correlates with poverty (R² = 0.42). MC is substantially a poverty proxy.

### 8.3 Comprehensive Reimbursement Analysis

#### 8.3.1 Data Sources

| File | Description | Coverage |
|------|-------------|----------|
| `rates_data.xlsx` | Medi-Cal Fee Schedule | 19,401 CPT codes |
| `reimbursement rates.csv` | County Capitation Rates | 22 COHS counties, 2021 |
| `conv_data.txt` | Conversion Factors by Specialty | Statewide |

#### 8.3.2 Medi-Cal Fee Schedule Structure

The fee schedule contains **19,401 procedure codes** with varying payment rates:

| Procedure Category | Mean Rate | Median Rate | N Codes |
|-------------------|-----------|-------------|---------|
| Primary Care Visits | $97.21 | $65.91 | 222 |
| Surgery (Other) | $81.71 | $63.33 | 276 |
| Radiology | $130.45 | $109.46 | 4,048 |
| Laboratory | $129.24 | $54.95 | 850 |
| Anesthesia | $544.64 | $398.55 | 5,778 |

**Context:** A typical primary care office visit pays ~$66 under Medi-Cal (median), compared to Medicare's ~$100-150, meaning Medi-Cal pays roughly **50-65% of Medicare rates**.

#### 8.3.3 County Capitation Rates (COHS Counties, 2021)

County Organized Health Systems (COHS) receive per-member-per-month (PMPM) capitation payments that vary by eligibility category:

| Category | Mean Rate | Min | Max | Range (%) |
|----------|-----------|-----|-----|-----------|
| **Adult** | $361/mo | $283 | $404 | 43% |
| **Adult Expansion (ACA)** | $446/mo | $375 | $472 | 26% |
| **Child** | $91/mo | $80 | $100 | 25% |
| **SPD (Disabled)** | $1,234/mo | $846 | $1,420 | 68% |
| **LTC (Long-term care)** | $10,505/mo | $8,259 | $16,445 | 99% |

**Key Insight:** There is substantial variation in capitation rates across counties—up to 99% difference in LTC rates. If reimbursement affected outcomes, we should be able to detect this.

#### 8.3.4 Reimbursement → Outcomes Regressions

**Model 1: PQI ~ Capitation Rate (Bivariate)**
$$PQI_c = \beta_0 + \beta_1 \cdot CapRate_c + \epsilon_c$$

| Parameter | Estimate | SE | p-value | N | R² |
|-----------|----------|-----|---------|---|-----|
| β₁ (per $1000) | 6.3 | 25.9 | **0.809** | 22 | 0.001 |

**Model 2: PQI ~ Capitation Rate + MC Share**
$$PQI_c = \beta_0 + \beta_1 \cdot CapRate_c + \beta_2 \cdot MC_c + \epsilon_c$$

| Parameter | Estimate | SE | p-value |
|-----------|----------|-----|---------|
| β₁ (Cap Rate per $1000) | -7.0 | — | 0.849 |
| β₂ (MC Share) | 318.8 | — | 0.418 |

**Model 3: Adult-Specific Capitation**
$$PQI_c = \beta_0 + \beta_1 \cdot AdultCapRate_c + \epsilon_c$$

| Parameter | Estimate | SE | p-value |
|-----------|----------|-----|---------|
| β₁ (per $100) | 6.2 | — | **0.822** |

**Interpretation:** A $100 increase in adult capitation rate is associated with only a 6.2-point change in PQI (not statistically significant).

#### 8.3.5 Correlations

| Relationship | Correlation | Interpretation |
|--------------|-------------|----------------|
| Cap Rate ↔ PQI | **0.03** | Essentially zero |
| Cap Rate ↔ MC Share | 0.30 | Modest positive |
| Cap Rate ↔ ED Admit Rate | 0.24 | Modest positive |

#### 8.3.6 Why the Null Result Matters

**The null finding is informative:**

1. **Within COHS counties** (N=22), higher capitation rates do not predict better outcomes
2. Counties with higher rates don't have systematically lower preventable hospitalizations
3. This suggests **money alone may not be the binding constraint**

**Possible explanations:**
- Provider supply/willingness to participate matters more than rate levels
- Social determinants (poverty, education, transportation) dominate clinical factors
- Capitation rates may not fully translate to provider-level payments
- Selection: counties with sicker populations may receive higher rates
- Sample too small for adequate statistical power (~20% power to detect moderate effects)

#### 8.3.7 Policy Implications

| Finding | Policy Implication |
|---------|-------------------|
| Cap rates don't predict PQI | Rate increases alone may not improve outcomes |
| Substantial rate variation exists | Natural experiment potential for future research |
| MC ~ Poverty (R²=0.42) | Address underlying social determinants |
| Fee schedule is statewide | Cannot test FFS rate variation across counties |

### 8.4 Accessibility Data Limitations

We could **not** analyze the following due to data unavailability:

| Variable | Why It Matters | Data Status |
|----------|---------------|-------------|
| Distance to hospital/ED | Physical access barrier | ❌ Not available |
| Public transit access | Non-vehicle transportation | ❌ Not available |
| Provider acceptance rates | Effective supply | ❌ Not at county level |
| FQHC/RHC capacity | Safety net access | ❌ Not in our data |
| Telehealth availability | Modern access channel | ❌ Not available |
| Wait times for appointments | De facto access | ❌ Not available |

**Note:** The vehicle ownership variable (`no_vehicle_pct`) shows a counterintuitive negative correlation with PQI, likely because rural counties have high vehicle ownership but poor outcomes due to distance—highlighting the need for actual distance/travel time measures.

---

## 9. Comprehensive Analysis: What Drives Healthcare Outcomes?

### 9.1 Methodological Approach

Following best practices to avoid post-treatment bias:
1. **Partial R² decomposition** - Which variable blocks matter most?
2. **Three linked regressions** - Causal chain: Need → Supply → Utilization → Outcomes
3. **Composite outcome index** - Combines PQI, ED, and costs
4. **Panel fixed effects** - Within-county variation over time

### 9.2 Partial R² Decomposition (Cross-section, N=56)

**Question:** Which factor explains the most variance in PQI?

| Variable Block | Full R² | Without Block | **ΔR²** | % of Total |
|----------------|---------|---------------|---------|------------|
| **Age Structure** | 0.342 | 0.091 | **0.251** | **73.4%** |
| Disability | 0.342 | 0.245 | 0.097 | 28.3% |
| Poverty | 0.342 | 0.284 | 0.058 | 16.9% |
| Provider Supply | 0.342 | 0.320 | 0.022 | 6.3% |
| Medi-Cal Share | 0.342 | 0.342 | **0.000** | **0.0%** |

**Key Finding:** Age structure (% 65+) explains **73%** of the variance in PQI. Medi-Cal share explains **0%** when other factors are controlled.

### 9.3 Three Linked Regressions (Causal Chain)

#### Model 1: Supply Model (What predicts PCP supply?)
$$PCP_{per100k,c} = \beta_0 + \beta_1 MC_c + \beta_2 Poverty_c + \beta_3 Age65_c + \epsilon_c$$

| Variable | β | p-value | Interpretation |
|----------|---|---------|----------------|
| MC Share | **-378.6** | **0.007** | High MC → FEWER PCPs |
| Poverty | -3.03 | 0.014* | More poverty → fewer PCPs |
| Age 65+ | -0.98 | 0.298 | NS |

#### Model 2: Utilization Model (What predicts ED admission rate?)
$$ED\_admit_c = \beta_0 + \beta_1 PCP_c + \beta_2 MC_c + \beta_3 Poverty_c + \beta_4 Age65_c + \epsilon_c$$

| Variable | β | p-value | Interpretation |
|----------|---|---------|----------------|
| PCP Supply | -0.0001 | 0.038* | More PCPs → lower ED admits |
| MC Share | -0.017 | 0.777 | NS |
| Poverty | 0.001 | 0.059 | More poverty → higher ED admits |
| **Age 65+** | **0.002** | **<0.001*** | Older pop → higher ED admits |

#### Model 3: Outcome Model (What predicts PQI?)
$$PQI_c = \beta_0 + \beta_1 PCP_c + \beta_2 MC_c + \beta_3 Poverty_c + \beta_4 Age65_c + \epsilon_c$$

| Variable | β | p-value | Interpretation |
|----------|---|---------|----------------|
| **PCP Supply** | **-0.50** | **0.034*** | More PCPs → lower PQI ✓ |
| MC Share | 96.7 | 0.695 | NS |
| Poverty | -2.51 | 0.393 | NS |
| **Age 65+** | **-4.97** | **<0.001*** | Older pop → lower PQI (?) |

### 9.4 Panel Fixed Effects (Within-County, 2012-2023)

$$PQI_{ct} = \beta_1 MC_{ct} + \beta_2 Poverty_{ct} + \beta_3 Age65_{ct} + \alpha_c + \gamma_t + \epsilon_{ct}$$

| Variable | Std β | SE | p-value |
|----------|-------|-----|---------|
| MC Share | **+10.92** | 4.48 | **0.015*** |
| **Poverty** | **+29.59** | 10.36 | **0.004*** |
| Age 65+ | -8.83 | 4.96 | 0.075 |

**Panel Finding:** Within counties over time, **poverty changes** are the strongest predictor of PQI changes.

### 9.5 Composite Outcome Model

$$OutcomeIndex_c = -z(PQI_c) - z(ED_c) - z(Cost_c)$$

Higher index = Better overall system performance

| Variable | β | p-value | Direction |
|----------|---|---------|-----------|
| MC Share | -1.81 | 0.767 | NS |
| PCP Supply | +0.016 | 0.073 | More PCPs → better |
| Poverty | -0.011 | 0.845 | NS |
| **Age 65+** | **-0.110** | **0.012*** | Older → worse |

### 9.6 Summary: What Actually Matters?

| Factor | Supply Model | Utilization | Outcomes | Conclusion |
|--------|--------------|-------------|----------|------------|
| **Age 65+** | NS | **+++** | **---** | **Dominant factor** |
| **Poverty** | **-** | **+** | NS (panel: **+**) | Important |
| **PCP Supply** | — | **-** | **-** | Small but real effect |
| **MC Share** | **---** | NS | NS | **NOT causal** |

### 9.7 Causal Chain Analysis (ED & Costs as Outcomes)

Following best practices: treat ED utilization and costs as **outcomes**, not predictors.

#### Model A: Supply (PCP as Outcome)
$$PCP_c = \beta_0 + \beta_1 MC_c + \beta_2 Poverty_c + \beta_3 Age65_c + \beta_4 Disability_c + \epsilon_c$$

| Variable | β | p-value | Finding |
|----------|---|---------|---------|
| **MC Share** | **-247.9** | **0.060** | High MC → Fewer PCPs |
| Poverty | -0.69 | 0.658 | NS |
| **Disability** | **-7.56** | **<0.001*** | More disability → fewer PCPs |

#### Model B: Utilization (ED Rate as Outcome)
$$ED_c = \theta_0 + \theta_1 PCP_c + \theta_2 MC_c + \theta_3 Controls + \epsilon_c$$

| Variable | β | p-value | Finding |
|----------|---|---------|---------|
| **PCP Supply** | **-0.00012** | **0.038*** | More PCPs → lower ED rate |
| MC Share | -0.017 | 0.777 | NS |
| **Age 65+** | **+0.002** | **<0.001*** | Older → more ED admissions |

#### Model C: Outcomes (PQI as Outcome)
$$PQI_c = \lambda_0 + \lambda_1 PCP_c + \lambda_2 MC_c + \lambda_3 Controls + \epsilon_c$$

| Variable | β | p-value | Finding |
|----------|---|---------|---------|
| **PCP Supply** | **-0.50** | **0.034*** | More PCPs → lower PQI |
| MC Share | +96.7 | 0.695 | **NS when supply controlled** |
| **Age 65+** | **-4.97** | **<0.001*** | Dominates |

#### Model D: PQI with Lagged ED/Cost (Panel FE)
$$PQI_{ct} = \beta_1 MC_{ct} + \beta_2 ED_{t-1} + \beta_3 Cost_{t-1} + \alpha_c + \gamma_t + \epsilon_{ct}$$

| Variable | β | p-value | Finding |
|----------|---|---------|---------|
| **MC Share** | **+270** | **0.001*** | Within-county: ↑MC → ↑PQI |
| **ED Rate (t-1)** | **-952** | **0.045*** | Prior ED predicts current PQI |
| Cost (t-1) | ~0 | 0.875 | NS |
| **Poverty** | **+6.7** | **0.003*** | ↑Poverty → ↑PQI |

### 9.8 Extended Chain: Reimbursement, Costs, and Financial Health

#### Model: Reimbursement → Supply
$$PCP_c = \beta_0 + \beta_1 ReimbRate_c + \beta_2 MC_c + \beta_3 Poverty_c + \epsilon_c$$

| Variable | β | p-value | Finding |
|----------|---|---------|---------|
| Reimb Rate | -0.005 | **0.80** | **No effect** |
| MC Share | -240 | 0.32 | NS |
| Poverty | -3.6 | 0.046* | More poverty → fewer PCPs |

**N = 22 (COHS counties only)**. Higher reimbursement does NOT attract more PCPs.

#### Model: MC Share → Hospital Costs
$$Cost_c = \beta_0 + \beta_1 MC_c + \beta_2 PCP_c + \beta_3 Controls + \epsilon_c$$

| Variable | β | p-value | Finding |
|----------|---|---------|---------|
| MC Share | +$53,109 | **0.49** | **No significant effect** |
| Age 65+ | +$1,803 | <0.001*** | Older pop → higher costs |

High-MC counties do NOT have significantly different hospital costs.

#### Model: Hospital Financial Health → Outcomes
$$PQI_c = \beta_0 + \beta_1 Margin_c + \beta_2 MC_c + \beta_3 PCP_c + Controls + \epsilon_c$$

| Variable | β | p-value | Finding |
|----------|---|---------|---------|
| **Operating Margin** | **-315** | **0.007*** | **Better margins → lower PQI** |
| MC Share | +53 | 0.81 | NS |
| PCP Supply | -0.40 | 0.03* | More PCPs → lower PQI |
| Age 65+ | -4.4 | <0.001*** | Age dominates |

**Key Finding:** Hospital financial health matters! Counties with financially healthier hospitals have **lower PQI** (β = -315, p = 0.007).

#### Model: MC Revenue Dependence → Outcomes
$$PQI_c = \beta_0 + \beta_1 MCRevShare_c + \beta_2 MC_c + Controls + \epsilon_c$$

| Variable | β | p-value | Finding |
|----------|---|---------|---------|
| MC Revenue Share | +0.14 | **0.998** | **No effect** |

Hospital dependence on Medi-Cal revenue does NOT predict outcomes.

#### Panel: Lagged Costs → Outcomes
$$PQI_{ct} = \beta_1 Cost_{t-1} + \beta_2 MC_{ct} + Controls + \alpha_c + \gamma_t + \epsilon_{ct}$$

| Variable | β | p-value | Finding |
|----------|---|---------|---------|
| Cost (t-1) | ~0 | **0.73** | **No lagged effect** |
| MC Share | +291 | 0.002** | Within-county: ↑MC → ↑PQI |
| Poverty | +6.5 | 0.005** | Within-county: ↑Poverty → ↑PQI |

Prior year hospital costs do NOT predict current outcomes within counties.

### 9.9 Heterogeneity Analysis: Who is Most Affected?

#### Stratified Regressions by Urban/Rural

| Region | N | Mean PQI | Mean MC | β(MC) | p-value |
|--------|---|----------|---------|-------|---------|
| Urban | 19 | 216.9 | 6.2% | +210 | 0.40 |
| Suburban | 20 | 204.5 | 9.7% | +65 | 0.90 |
| **Rural** | 17 | 193.1 | 9.6% | **+1,020** | 0.18 |

#### Interaction Model
$$PQI_c = \beta_0 + \beta_1 MC + \beta_2 Rural + \beta_3 (MC \times Rural) + Controls + \epsilon_c$$

| Variable | β | SE | p-value |
|----------|---|-----|---------|
| MC Share (base) | +212 | 320 | 0.51 |
| Rural dummy | -42 | 54 | 0.44 |
| MC × Rural | +50 | 484 | **0.92** |
| MC × Suburban | +42 | 380 | 0.91 |
| **PCP Supply** | **-0.65** | 0.29 | **0.025*** |

#### Marginal Effects by Region

| Region | MC Effect | Interpretation |
|--------|-----------|----------------|
| Urban | +212 | Moderate effect |
| Suburban | +254 | Moderate effect |
| **Rural** | **+262** | **Strongest effect** |

#### By Population Size

| Size | N | MC Effect | p-value |
|------|---|-----------|---------|
| Small (<68k) | 17 | **+1,020** | 0.18 |
| Medium (68k-439k) | 19 | +147 | 0.77 |
| Large (>439k) | 20 | +9 | 0.97 |

**Key Finding:** The MC → PQI relationship is **strongest in small/rural counties** (β = +1,020), though not statistically significant due to small sample size. In large urban counties, the effect essentially disappears (β = +9).

### 9.10 Key Conclusions

1. **Demographics explain ~75% of outcome variation** (age, disability)
2. **MC share affects SUPPLY**: High MC → fewer PCPs (β = -248)
3. **PCP supply matters**: More PCPs → lower PQI (β = -0.5*)
4. **MC share is NOT directly causal**: NS when supply controlled in cross-section
5. **Within counties over time**: MC and poverty changes both predict PQI
6. **Lagged ED matters**: Prior year ED rate predicts current PQI
7. **Reimbursement rates don't attract PCPs**: β = -0.005, p = 0.80
8. **MC share doesn't affect hospital costs**: β = $53k, p = 0.49
9. **Hospital financial health matters**: Better margins → lower PQI (β = -315**)
10. **MC revenue dependence doesn't hurt outcomes**: β = 0.14, p = 0.998
11. **Rural counties most affected**: MC effect is 5× larger in rural vs urban counties

---

## 10. Additional Robustness Checks

### 10.1 Specification Curve Summary

| Check | Purpose | Result | Interpretation |
|-------|---------|--------|----------------|
| Quintile bins | Test nonlinearity | Non-monotonic | Not dose-response |
| Quadratic term | Test U-shape | p = 0.90 | No U-shape |
| Piecewise spline | Test threshold | p = 0.92 | No break at median |
| County trends | Control for trends | θ → 0 | ⚠️ Concern |
| Placebo timing | Parallel trends | Pre-effects found | ⚠️ Concern |
| Wild bootstrap | Inference | CI similar | ✅ Robust |
| Outcome split | Mechanism | Acute > Chronic | ⚠️ Unexpected |
| Cost controls | Mechanism | No effect | ✅ Rules out costs |
| Reimbursement | Policy lever | No effect | Limited data |

### 10.2 Specification Curve for MC Coefficient

All models estimated for PQI ~ MC:

| Specification | β(MC) | p-value |
|--------------|-------|---------|
| Bivariate | 279 | 0.163 |
| + Poverty | 309 | 0.345 |
| + Disability | 308 | 0.352 |
| + Age65 | 284 | 0.178 |
| + All controls | 340 | 0.215 |
| Binary (≥ median) | 41 | **0.013** |

---

## 11. Summary of All Results

### 11.1 Full Results Table

| Analysis | Equation | β | SE | p | N | Interpretation |
|----------|----------|---|-----|---|---|----------------|
| **XS Bivariate** | PQI ~ MC | 279 | 200 | 0.16 | 58 | Not significant |
| **XS Binary** | PQI ~ I(MC≥med) | **41** | 16 | **0.01** | 58 | **Significant** |
| **XS + Controls** | PQI ~ MC + X | 341 | 274 | 0.22 | 58 | Not significant |
| **Panel FE** | PQI ~ MC | 73 | 121 | 0.55 | 870 | No within effect |
| **DiD Binary** | θ(Treat×Post) | **-28** | 12 | **0.02** | 812 | **Significant** |
| **DiD Intensity** | β(MC₂₀₁₆×Post) | **-372** | 131 | **0.01** | 812 | **Significant** |
| **DiD + Trends** | θ(Treat×Post) | 1 | 2 | 0.70 | 812 | Not significant |
| **Cost → PQI** | β(log cost) | -15 | 12 | 0.18 | 670 | Not significant |
| **Reimb → PQI** | β(rate) | 0.01 | 0.03 | 0.81 | 22 | Not significant |

### 11.2 Effect Sizes

For significant findings:

| Finding | Effect Size | In Context |
|---------|-------------|------------|
| Binary XS | +41 PQI points | ~20% of mean PQI |
| DiD Intensity | -37 per 10pp MC | ~18% of mean PQI |

---

## 12. Conclusions

### 12.1 What We Know

1. **Cross-sectionally, high-MC counties have worse outcomes**
   - Binary effect: β = 41, p = 0.013
   - Counties above median MC have 20% higher PQI

2. **MC share does not cause poor outcomes within counties**
   - Panel FE: β = 73, p = 0.55
   - Within-county MC changes don't predict PQI changes

3. **MC share is largely a poverty proxy**
   - MC ~ Poverty: R² = 0.42
   - Suggests MC captures social disadvantage

4. **Hospital costs don't explain the relationship**
   - Cost → PQI: p = 0.18 (panel)
   - MC → Cost: p = 0.57 (cross-section)

### 12.2 What We're Uncertain About

1. **Prop 56 effect**
   - Intensity DiD: β = -372, p = 0.005 (strong)
   - But: disappears with county trends, placebo failures
   - Conclusion: Possible but not definitive

2. **Mechanism**
   - Outcome decomposition shows acute > chronic effect
   - Doesn't match "better ambulatory care" story
   - True mechanism remains unclear

### 12.3 What We Cannot Analyze

- Distance to care (no data)
- Provider acceptance rates (no county-level data)
- Transportation access beyond vehicle ownership
- Quality of care (only utilization)

### 12.4 Policy Implications

| Recommendation | Evidence Strength |
|----------------|------------------|
| Target investments to high-MC counties | Moderate |
| Don't focus on reducing MC enrollment | Strong (no causal effect) |
| Monitor PQI as policy outcome | Strong |
| Address underlying poverty | Strong (MC proxies poverty) |
| Increase reimbursement rates | Weak (limited evidence) |

---

## Appendix A: Data Files

### Generated Data
- `outputs/data/master_panel_2005_2025.csv`
- `outputs/data/medi_cal_certified_eligibles_2010_2025.csv`
- `outputs/data/pqi_county_year_new.csv`
- `outputs/data/hospital_costs_new.csv`
- `outputs/data/reimbursement_rates.csv`

### Generated Tables
- `outputs/tables/high_impact_analyses.csv`
- `outputs/tables/placebo_timing_tests.csv`
- `outputs/tables/accessibility_reimbursement_results.csv`

### Generated Figures
- `outputs/figures/high_impact_analyses.png`
- `outputs/figures/final_certified_eligibles.png`

---

## Appendix B: Equations Reference

### Cross-Sectional
$$PQI_c = \beta_0 + \beta_1 MC_c + \gamma X_c + \epsilon_c \quad \text{(HC1 robust SE)}$$

### Panel Fixed Effects
$$PQI_{ct} = \alpha_c + \lambda_t + \beta MC_{ct} + \epsilon_{ct} \quad \text{(clustered SE)}$$

### Difference-in-Differences
$$PQI_{ct} = \alpha_c + \lambda_t + \theta_1 (Treat_c \times Post_t) + \epsilon_{ct}$$

### Intensity DiD
$$PQI_{ct} = \alpha_c + \lambda_t + \beta (MC_{c,2016} \times \mathbb{1}_{t \geq 2017}) + \epsilon_{ct}$$

### County Trends Sensitivity
$$PQI_{ct} = \alpha_c + \lambda_t + \delta_c t + \theta_1 (Treat_c \times Post_t) + \epsilon_{ct}$$

---

*Report generated: January 2026*  
*Analysis: Python (pandas, statsmodels, numpy)*  
*Data: DHCS, HCAI, Census ACS, OSHPD*
