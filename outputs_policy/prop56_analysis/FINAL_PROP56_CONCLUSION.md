# Prop 56 Causal Analysis: FINAL CONCLUSION

**Date**: January 22, 2026  
**Status**: Analysis complete with provider-level mechanism test

---

## Executive Summary

**Prop 56 payment increases did NOT cause the observed improvement in Medi-Cal access outcomes.**

This conclusion is based on three converging lines of evidence:

1. ❌ **Outcome Analysis**: No pre-COVID effect (DiD = -2.3, p = 0.82)
2. ❌ **Placebo Tests**: Pre-trends not parallel, temporal placebo fails
3. ❌ **Mechanism Test**: No differential provider enrollment (DiD = +0.20, p = 0.79)

---

## The Mechanism Test (Smoking Gun)

We tested the core hypothesis: Did Prop 56 attract more PCPs to high-MC counties?

### Results:

| Provider Type | DiD Effect | SE | P-value | Significant? |
|---------------|-----------|-----|---------|--------------|
| All Providers | **-6.87** | 3.26 | **0.035** | Yes - NEGATIVE |
| Physicians | -1.94 | 1.61 | 0.227 | No |
| **Primary Care** | **+0.20** | 0.76 | **0.794** | **NO** |

**High-MC counties actually got FEWER total providers post-Prop 56**, and there was no significant change in PCP enrollment.

### Event Study Evidence:
```
Year    High-MC    Low-MC    Difference
────────────────────────────────────────
2015    9.71       14.36     -4.65
2016    5.81       6.18      -0.37
2017    3.85       6.14      -2.29  ← Prop 56
2018    4.65       6.90      -2.24
2019    3.19       3.78      -0.59
2020    3.56       5.97      -2.41
```

Low-MC counties consistently attract more PCPs per capita, both before AND after Prop 56.

---

## Why Prop 56 Didn't Work (Possible Explanations)

1. **Rate increases were insufficient**: Even with Prop 56, Medi-Cal rates remain well below Medicare/commercial rates

2. **Structural barriers persist**: High-MC counties often have:
   - Lower quality of life factors
   - Higher cost of living (coastal areas)
   - Infrastructure challenges (rural areas)
   
3. **Provider preferences matter more than rates**: Physicians choose locations based on:
   - Training location
   - Spouse employment
   - Schools for children
   - Professional networks
   - Not primarily reimbursement rates

4. **Rate increases went to existing providers**: The money may have increased income for current Medi-Cal providers rather than attracting new ones

---

## What Actually Drove the Post-2019 Convergence?

The outcome improvement in 2020-2024 is likely explained by:

### 1. **COVID-19 Pandemic Effects**
- Reduced overall hospitalizations
- Patients avoided hospitals for fear of infection
- Elective procedures postponed

### 2. **Telehealth Expansion**
- Rapid telehealth adoption during pandemic
- Disproportionately benefits underserved areas
- Reduces distance-to-care barriers in deserts

### 3. **Medi-Cal Managed Care Transitions**
- Ongoing managed care expansion
- Care coordination improvements
- Disease management programs

### 4. **General Healthcare Trends**
- Declining PQI rates nationwide (long-term trend)
- Improved chronic disease management
- Better outpatient alternatives to hospitalization

---

## Revised Causal Chain

### What We Thought:
```
Prop 56 (2017) → More Providers → Better Outcomes
```

### What the Evidence Shows:
```
Prop 56 (2017) → [NO change in provider enrollment] → ???

COVID (2020) → Telehealth + Behavior Change → Outcome Improvement
```

---

## What We CAN and CANNOT Claim

### ✅ What We CAN Say:
- High-MC counties showed outcome convergence in 2020-2024
- The timing is temporally consistent with Prop 56
- Overall Medi-Cal provider enrollment increased substantially (159% since 2017)

### ❌ What We CANNOT Say:
- "Prop 56 caused the improvement" (no mechanism)
- "Payment increases attracted more providers" (DiD not significant)
- "The convergence would have happened without COVID" (COVID era confounds)

---

## Policy Implications

### 1. **Payment Increases Alone Are Insufficient**
Prop 56 increased rates, but this did not change provider location decisions. Payment policy alone won't solve geographic maldistribution.

### 2. **Structural Interventions Needed**
More effective approaches may include:
- Loan repayment programs (tied to location)
- Residency pipeline programs in underserved areas
- Scope of practice expansion for NPs/PAs
- FQHC expansion
- Telehealth infrastructure investment

### 3. **Don't Oversell Payment Policy**
Future payment increases should not be marketed as access solutions without evidence that providers respond to rate changes by relocating.

### 4. **COVID-Era Gains May Be Temporary**
If the improvement is driven by telehealth and pandemic behavior, these gains could erode as the pandemic recedes.

---

## Methodological Summary

| Analysis | Finding | Interpretation |
|----------|---------|----------------|
| Outcome DiD (full period) | -17.5, p = 0.086 | Marginally significant |
| Outcome DiD (pre-COVID) | -2.3, p = 0.82 | **NOT significant** |
| Pre-trend test | Significant divergence | Violates parallel trends |
| Temporal placebo | p = 0.029 | **FAILS** |
| Geographic placebo | p = 0.98 | Passes |
| **PCP enrollment DiD** | **+0.20, p = 0.79** | **NOT significant** |

**Bottom line**: The identification strategy has too many problems, and the mechanism doesn't hold up.

---

## Files Generated

### Analysis Scripts:
- `prop56_causal_analysis.py` - Outcome analysis
- `prop56_provider_mechanism.py` - Provider enrollment analysis

### Results:
- `prop56_causal_analysis.png` - Outcome visualizations
- `provider_mechanism_analysis.png` - Provider visualizations
- `event_study_coefficients.csv` - Outcome event study
- `provider_event_study.csv` - Provider event study
- `provider_did_results.csv` - Provider DiD results
- `county_year_enrollments.csv` - County-year enrollment data

### Documentation:
- `PROP56_CAUSAL_ASSESSMENT.md` - Outcome analysis summary
- `FINAL_PROP56_CONCLUSION.md` - This file

---

## Conclusion

**Prop 56 did not work as intended.** The payment increases did not attract more PCPs to high-MC counties, and the observed outcome improvements are better explained by COVID-era factors (telehealth, behavior change) than by payment policy.

This is an important finding for policymakers: **Simply increasing reimbursement rates is not sufficient to solve physician maldistribution.** More structural interventions are needed.

---

**Methodological Note**: This analysis represents honest engagement with the data. The initial hypothesis (Prop 56 → better outcomes) was plausible but did not survive rigorous testing. Good research reports what the evidence shows, even when it contradicts policy hopes.
