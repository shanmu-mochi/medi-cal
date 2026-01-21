# Regression Results Summary

## Model A: Provider Supply Model
**Dependent Variable:** pcp_per_100k (Primary care physicians per 100,000)

**Key Findings:**
- Medi-Cal Share coefficient: 100.7229
- Shortage Flag coefficient: 25.9091
- N observations: 56
- N counties: 56

## Model B: Outcomes Model
**Dependent Variable:** pqi_mean_rate (Mean PQI rate)

**Key Findings:**
- PCP per 100k coefficient: -0.2372
- Medi-Cal Share coefficient: 255.6220
- Shortage Flag coefficient: 19.8524
- N observations: 56
- N counties: 56

## Model C: Desert Indicator Model
**Dependent Variable:** pqi_mean_rate (Mean PQI rate)

**Key Findings:**
- Desert (Quartile Def2) coefficient: 7.2322
- N observations: 56
- N counties: 56

## Interpretation

### Model A Interpretation:
A +0.10 increase in medi_cal_share is associated with Δ 10.07 pcp_per_100k.

### Model B Interpretation:
An increase of +10 pcp_per_100k is associated with Δ -2.37 pqi_mean_rate.

**Note:** Standard errors are clustered at the county level.
