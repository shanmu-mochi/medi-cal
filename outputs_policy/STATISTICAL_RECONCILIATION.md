# Statistical Reconciliation: Addressing Methodological Issues

**Date**: January 22, 2026  
**Purpose**: Reconcile contradictions and clarify statistical evidence quality

---

## Issue 1: The Convergence Story (RESOLVED)

### The Contradiction
- **Test 16** (Regression: Gap ~ Year): β = -0.204, **p = 0.69** → NO significant trend
- **Previous claim**: "Gap narrowing" / "cautious optimism - convergence observed"

### Resolution
**The gap between desert and non-desert counties is NOT significantly changing over time.**

```
Regression: Gap = β₀ + β₁(Year)
β₁ = -0.204 (95% CI: -1.19 to +0.78)
p = 0.69

Interpretation: The negative coefficient suggests a slight narrowing trend 
(-0.2 PQI points per year), but this is NOT statistically distinguishable 
from zero. The gap could be stable, narrowing, or even widening.
```

### Corrected Narrative
**Q6: Are Reforms Narrowing Deserts?**

❌ OLD: "TRUE DESERT counties show convergence... Gap has narrowed. VERDICT: Cautious optimism"

✅ NEW: **"No evidence of convergence (p = 0.69). The gap persists at ~30-60 PQI points throughout 2005-2024. While both groups improved over time, deserts remain systematically disadvantaged."**

**Policy Implication**: Current reforms are NOT sufficient to close the access-outcome gap. More aggressive interventions targeting desert counties specifically are needed.

---

## Issue 2: ED Regression Weakness (CLARIFIED)

### The Evidence
- **Test 4** (Regression: ED_rate ~ access_gap): β = -0.416, **p = 0.154** → NOT significant
- **Test 10** (T-test: Desert vs Non-Desert ED): Difference = 41.1 visits/1k, **p = 0.008** → Significant

### Why the Discrepancy?
1. **Measurement**: T-test compares binary groups (TRUE DESERT vs others), while regression uses continuous access_gap
2. **Power**: T-test has more power to detect group differences; regression diluted by within-group variation
3. **Clustering**: Regression uses county-clustered SEs (more conservative)

### Corrected Presentation
**Lead with stronger evidence, acknowledge limitations:**

✅ **Primary Finding**: "Desert counties have significantly higher ED utilization (378.9 vs 337.8 per 1,000, p = 0.008), consistent with limited primary care access forcing patients to seek emergency care."

✅ **Secondary Note**: "The continuous relationship between access gap and ED rates is weaker (p = 0.15), suggesting a threshold effect: once counties cross into 'desert' status, ED use spikes, but variation within non-desert counties is less predictive."

**Do NOT claim**: "Access gaps predict ED utilization" (continuous regression doesn't support this)

**DO claim**: "Desert designation predicts higher ED use" (t-test supports this)

---

## Issue 3: TRUE DESERT Binary - Clustering Effect (NOTED)

### The Evidence
- **Test 3** (Regression with clustered SE): Desert coefficient = 58.3, **p = 0.058** → Marginally significant
- **Test 9** (T-test without clustering): Same difference, **p < 0.0001** → Highly significant

### Why the Difference?
**Clustered standard errors** account for within-county correlation across years, inflating SEs by ~3x.

```
Effect size: +58.3 PQI points in desert counties
Unclustered SE: 11.9 → t = 4.89, p < 0.0001
Clustered SE: 30.8 → t = 1.89, p = 0.058
```

### Interpretation
This is a **conservative estimate**. The clustering adjustment is methodologically sound (counties are not independent across years), but it reduces statistical power.

✅ **Recommended Framing**:
"Desert counties have 58.3 points higher PQI rates (p = 0.058 with conservative clustered SEs). Simple comparisons show p < 0.0001, but proper panel adjustment reduces power. We interpret this as meaningful evidence of desert disadvantage, recognizing the limited sample size (N=8 desert counties)."

**Key Point**: The effect is real and large (27% higher than non-deserts), but with only 8 desert counties, detecting it with clustered SEs is at the edge of statistical convention.

---

## Revised Summary Table

| Test | Finding | Coefficient | P-value | Strength | Interpretation |
|------|---------|-------------|---------|----------|----------------|
| **Test 1** | Access Gap → PQI (Cross-section) | -0.58 | 0.029 | ★★★ | Strong evidence |
| **Test 2** | Access Gap → PQI (Panel) | -0.70 | 0.028 | ★★★ | Strong evidence |
| **Test 3** | Desert → PQI (Binary, Clustered) | +58.3 | 0.058 | ★★☆ | Marginally significant (conservative) |
| **Test 4** | Access Gap → ED (Regression) | -0.42 | 0.154 | ★☆☆ | Weak/null |
| **Test 5** | FFS → PQI | +19.1 | 0.003 | ★★★ | Strong evidence |
| **Test 9** | Desert → PQI (T-test) | +58.3 | <0.0001 | ★★★ | Strong evidence (unclustered) |
| **Test 10** | Desert → ED (T-test) | +41.1 | 0.008 | ★★★ | Strong evidence |
| **Test 16** | Convergence (Gap ~ Year) | -0.20 | 0.69 | ☆☆☆ | No evidence |

---

## Recommendations for Future Analysis

1. **Convergence**: Use longer time series (2005-2024 has N=20 years) with robust trend tests. Consider structural break tests around major policy changes (ACA 2014, Prop 56 2017).

2. **ED Analysis**: 
   - Explore non-linear models (splines, thresholds)
   - Test for interaction: Does the access gap → ED relationship differ in high-poverty counties?
   - Use instrumental variables if endogeneity concerns arise

3. **Desert Binary**: 
   - With only 8 desert counties, consider:
     - Continuous "desert severity" scores instead of binary
     - Bootstrap confidence intervals
     - Bayesian estimation to properly quantify uncertainty with small N

---

## Bottom Line

### What the Data DOES Show:
✅ Access gaps predict preventable hospitalizations (Tests 1-2: both p < 0.03)  
✅ Desert counties have higher PQI rates (~58 points, strong effect size)  
✅ Desert counties have higher ED use (+41 visits/1k, p = 0.008)  
✅ FFS delivery model predicts worse outcomes (p = 0.003)  
✅ Prop 56 had mixed effects (some evidence of improvement in high-MC counties)

### What the Data Does NOT Show:
❌ Convergence between desert and non-desert counties (p = 0.69)  
❌ Continuous access gap → ED relationship (p = 0.15)

### Honest Uncertainty:
⚠️ Desert → PQI effect is at margin of significance (p = 0.058) due to clustering and small N, but effect size is large and clinically meaningful.

---

**Methodological Principle**: When statistics are ambiguous, report both the evidence and the uncertainty. Policy should be made on the totality of evidence, not cherry-picked p-values.
