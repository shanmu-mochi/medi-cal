# Prop 56 Causal Assessment: Honest Evaluation

**Date**: January 22, 2026  
**Method**: Event Study, Placebo Tests, COVID Adjustment, Heterogeneity Analysis

---

## Summary: The Evidence is WEAKER Than Initially Suggested

Your instinct about the "diverge then converge" pattern around 2017 was reasonable, but the rigorous tests reveal problems with causal identification.

---

## Key Findings

### 1. Event Study: Pre-Trends Are NOT Parallel ⚠️

```
Year   Event_t   Coefficient   P-value   Status
────────────────────────────────────────────────
2012    -5        +28.7        0.018     SIGNIFICANT ⚠️
2013    -4        +27.6        0.025     SIGNIFICANT ⚠️
2014    -3        +10.2        0.266     OK
2015    -2        +27.0        0.000     SIGNIFICANT ⚠️
2016    -1        -12.0        0.136     OK
2017     0          0.0        (ref)     Reference
2018    +1         +5.9        0.505     Implementation lag ✓
2019    +2         -4.0        0.652     OK
2020    +3        -19.8        0.047     SIGNIFICANT (COVID?)
2022    +5        -17.6        0.039     SIGNIFICANT (COVID?)
```

**Problem**: Significant pre-trend coefficients in 2012, 2013, 2015 mean high-MC and low-MC counties were NOT on parallel trajectories before Prop 56. This violates the key DiD assumption.

**The 2018 lag (+5.9) IS consistent with your implementation lag theory**, but the pre-trend violation undermines the causal interpretation.

---

### 2. Placebo Tests: Mixed Results ⚠️

| Test | Result | P-value | Interpretation |
|------|--------|---------|----------------|
| **Temporal Placebo (2014)** | FAIL ⚠️ | 0.029 | Finding "effect" where none should exist |
| **Geographic Placebo** | PASS ✓ | 0.978 | No spurious effect in control group |

**Temporal placebo failure** is concerning: We detect a "fake" Prop 56 effect in 2014 (DiD = -27.8, p = 0.03). This suggests the model is picking up secular trends, not policy effects.

---

### 3. Pre-COVID Analysis: NO Effect ❌

**Critical finding**: When we restrict to 2014-2019 (before COVID confounds):

```
Pre-COVID DiD = -2.3
P-value = 0.82 (NOT significant)
```

**The Prop 56 effect DOES NOT HOLD pre-COVID.**

This strongly suggests the post-2019 convergence is driven by:
- COVID pandemic effects
- Telehealth expansion
- Managed care changes

...NOT Prop 56 payment increases.

---

### 4. Heterogeneity: Unexpected Pattern

| Subgroup | DiD | P-value | Interpretation |
|----------|-----|---------|----------------|
| Urban | -7.8 | 0.63 | No effect |
| **Rural** | **+42.0** | **0.03** | Worse outcomes in high-MC rural! |
| High baseline PQI | +25.1 | 0.13 | Not significant |
| Low baseline PQI | +8.0 | 0.60 | No effect |

**Troubling**: Rural high-MC counties show WORSE outcomes relative to rural low-MC counties post-2017. This is the opposite of what Prop 56 should produce.

---

## Revised Interpretation of the 2018 Peak

### What We Initially Thought:
> "Gap widened through 2018, then narrowed → Prop 56 worked with implementation lag"

### What the Evidence Actually Shows:
1. **Pre-trends were not parallel** - High-MC counties were already diverging/fluctuating relative to low-MC counties
2. **Temporal placebo fails** - We find "effects" of fake policies
3. **No pre-COVID effect** - The improvement is 2020+ only
4. **Rural counties got worse** - Opposite of prediction

### Most Likely Explanation:
The post-2019 convergence is **COVID-driven**, not Prop 56-driven:
- Both groups improved dramatically in 2020+ (lower overall PQI rates)
- High-MC counties improved faster, but this coincides with:
  - Telehealth expansion (benefits underserved areas)
  - Medi-Cal managed care transitions
  - Reduced elective care during pandemic

---

## What Would Change This Assessment?

### 1. Provider-Level Data (The Smoking Gun You Identified)
If you could show:
```
New_MC_Providers_it = β(Post_2017 × High_HPSA) + ...
```
And β > 0 and significant, that would demonstrate the MECHANISM (Prop 56 → More providers → Better outcomes).

### 2. Dose-Response Evidence
If counties with larger rate increases showed bigger improvements, that would support causality.

### 3. Different Comparison Group
Synthetic control matching on pre-trends might yield cleaner identification.

---

## Updated Evidence Summary

| Evidence Type | For Prop 56 | Against Prop 56 |
|---------------|-------------|-----------------|
| Timing | 2018 lag fits theory | Improvement is 2020+ (COVID era) |
| Pre-trends | | NOT parallel (violates DiD) |
| Temporal placebo | | FAILS (p = 0.03) |
| Geographic placebo | PASSES | |
| Pre-COVID effect | | NONE (p = 0.82) |
| Heterogeneity | | Rural counties WORSE |
| Mechanism | No provider data | Can't verify |

---

## Honest Bottom Line

### What We Can Say:
✅ "High-MC counties showed convergence toward low-MC counties in 2020-2024"  
✅ "The 2018 peak is consistent with an implementation lag"  
✅ "The pattern is temporally consistent with Prop 56"

### What We CANNOT Say:
❌ "Prop 56 caused the improvement" (pre-trends violated, no pre-COVID effect)  
❌ "Payment increases attracted more providers" (no provider data)  
❌ "The convergence would have happened without COVID" (COVID era confounds)

### Appropriate Framing:
> "The timing of outcomes improvement in high-MC counties is temporally consistent with Prop 56, but causal identification is weak. Pre-trend violations and the lack of pre-COVID effects suggest the post-2019 convergence may reflect COVID-era changes (telehealth, managed care) rather than payment policy. **Provider-level enrollment data would be required to establish a causal link.**"

---

## Policy Recommendation

**Don't oversell Prop 56 based on this evidence.**

The analysis cannot rule out that Prop 56 had some effect, but it also cannot confirm it. The most honest interpretation is:

1. Prop 56 **may have contributed** to improvement
2. But **COVID-era factors likely played a larger role**
3. **Without provider data, causality is speculative**

If you want to strengthen the Prop 56 story, the **provider enrollment data from DHCS** is essential. That's the intermediate step that would complete the causal chain.

---

## Files Generated

- `event_study_coefficients.csv` - Full event study results
- `causal_tests_summary.csv` - Summary of all tests
- `prop56_causal_analysis.png` - Visualization
- `ed_disposition_aggregated.csv` - ED data for further analysis

---

**Methodological Integrity Note**: Good research reports what the evidence shows, even when it contradicts expectations. The initial "diverge then converge" intuition was reasonable but doesn't survive rigorous testing.
