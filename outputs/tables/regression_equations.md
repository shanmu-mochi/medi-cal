# Regression Equations - California Medi-Cal Deserts Analysis

## Model A: Provider Supply Model
**Dependent Variable:** pcp_per_100k (Primary Care Physicians per 100,000 population)
**R² = 0.705**

pcp_per_100k = -57.80 + 100.72×(medi_cal_share) + 25.91×(shortage_flag) 
               - 1.07×(poverty_pct) + 0.47×(unemp_pct) + 0.32×(age65_pct) 
               + 0.30×(hispanic_pct) + 3.35×(bachelors_pct)***

---

## Model B: Outcomes Model  
**Dependent Variable:** pqi_mean_rate (Preventable Hospitalizations per 100,000)
**R² = 0.346**

pqi_mean_rate = 307.68 - 0.24×(pcp_per_100k) + 255.62×(medi_cal_share) 
                + 19.85×(shortage_flag) - 6.00×(poverty_pct)† - 1.51×(unemp_pct) 
                - 1.96×(age65_pct) - 3.38×(hispanic_pct) - 1.24×(bachelors_pct)

---

## Model C: Desert Indicator Model
**Dependent Variable:** pqi_mean_rate (Preventable Hospitalizations per 100,000)
**R² = 0.304**

pqi_mean_rate = 384.82 + 7.23×(desert_q_def2) - 3.00×(poverty_pct) 
                + 1.08×(unemp_pct) - 3.77×(age65_pct)* - 3.48×(hispanic_pct) 
                - 2.19×(bachelors_pct)*

---

## Significance Codes
*** p < 0.001  |  ** p < 0.01  |  * p < 0.05  |  † p < 0.10

## Variable Definitions
- medi_cal_share: Medi-Cal enrollment / county population (proportion, 0-1)
- shortage_flag: Primary care shortage area designation (0/1)
- pcp_per_100k: Primary care physicians per 100,000 population
- pqi_mean_rate: Mean PQI rate (preventable hospitalizations per 100,000)
- desert_q_def2: Desert indicator (high Medi-Cal AND (low PCP OR shortage))
- poverty_pct: Percent below poverty line
- unemp_pct: Unemployment rate
- age65_pct: Percent age 65+
- hispanic_pct: Percent Hispanic
- bachelors_pct: Percent with bachelor's degree or higher

## Sample
56 California counties, Year 2020
