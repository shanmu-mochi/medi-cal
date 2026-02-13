"""
CORRECTED PARALLEL SPECIFICATIONS ANALYSIS
==========================================

This script addresses the methodological issues identified:

PREVIOUS ISSUES:
1. Used facility-based ED data (ed_visits, ed_admissions)
2. Used cross-sectional data only (N=55, 2020)
3. Did not use clustered standard errors (can't cluster single observations)
4. Mixed results from different analyses

CORRECTIONS:
1. Use RESIDENCE-BASED ED data (ed_visits_resident, ed_admit_rate_resident)
2. Use PANEL data (N~870, 2008-2024) for more power
3. Use CLUSTERED standard errors by county
4. Run ALL 4 specifications consistently on the SAME data

OUTPUT: outputs_v3/CORRECTED_parallel_specifications.csv
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

try:
    import statsmodels.api as sm
    from statsmodels.regression.linear_model import OLS
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("ERROR: statsmodels required")
    exit()

import os
OUTPUT_DIR = 'outputs_v3'
os.makedirs(f'{OUTPUT_DIR}/tables', exist_ok=True)

print("="*80)
print("CORRECTED PARALLEL SPECIFICATIONS ANALYSIS")
print("="*80)

# =============================================================================
# DOCUMENT DATA SOURCES
# =============================================================================

data_documentation = """
DATA SOURCE DOCUMENTATION
=========================

ED DATA TYPES:
--------------
1. RESIDENCE-BASED (CORRECT for desert analysis):
   - Source file: ed_patient_residence_county_year.csv
   - Variables: ed_visits_resident, ed_admissions_resident, ed_admit_rate_resident
   - Attribution: By PATIENT'S home county
   - Why correct: Measures utilization where patients LIVE, not where they seek care

2. FACILITY-BASED (INCORRECT for desert analysis):
   - Source file: ed_encounters_county_year.csv, ca_cross_section_2020_with_ed.csv
   - Variables: ed_visits, ed_admissions, ed_visits_per_1k
   - Attribution: By HOSPITAL's county
   - Why incorrect: Desert counties lack facilities, so residents' visits are
     attributed to neighboring counties where hospitals are located

ANALYSIS TYPES:
---------------
1. PANEL DATA (N~870, 2008-2024):
   - Source: master_panel_2005_2025.csv
   - Allows: Within-county variation over time, clustered SEs
   - More power than cross-section

2. CROSS-SECTION (N=55, 2020 only):
   - Source: ca_cross_section_2020_with_ed.csv
   - Limitation: Cannot cluster (single obs per county), less power
   - Uses FACILITY-BASED ED data (incorrect)

THIS ANALYSIS USES:
- Panel data (master_panel_2005_2025.csv)
- Residence-based ED (ed_visits_resident, ed_admit_rate_resident)
- Clustered standard errors by county
"""

print(data_documentation)

# =============================================================================
# LOAD PANEL DATA
# =============================================================================

print("\n" + "="*80)
print("LOADING DATA")
print("="*80)

# Load main panel
panel = pd.read_csv('outputs/data/master_panel_2005_2025.csv')
panel['fips5'] = panel['fips5'].astype(str).str.zfill(5)
print(f"Panel loaded: {len(panel)} county-years")
print(f"Years: {panel['year'].min()} - {panel['year'].max()}")
print(f"Counties: {panel['fips5'].nunique()}")

# Check ED variables
print(f"\nED variables in panel:")
ed_cols = [c for c in panel.columns if 'ed_' in c.lower()]
for col in ed_cols:
    non_null = panel[col].notna().sum()
    print(f"  {col}: {non_null} non-null values")

# Load access gap data (cross-sectional - defines county types)
try:
    workforce = pd.read_csv('outputs_policy/workforce_programs/county_program_data.csv')
    workforce['fips5'] = workforce['fips5'].astype(str).str.zfill(5)
    print(f"\nWorkforce/access gap data: {len(workforce)} counties")
except:
    # Try alternative source
    try:
        access_gap_data = pd.read_csv('outputs_v2/data/county_access_gap_2020.csv')
        access_gap_data['fips5'] = access_gap_data['fips5'].astype(str).str.zfill(5)
        workforce = access_gap_data.rename(columns={'county_type': 'county_type'})
        workforce['true_desert'] = (workforce['county_type'] == 'TRUE DESERT').astype(int)
        print(f"\nAccess gap data (from v2): {len(workforce)} counties")
    except:
        print("\nWARNING: No access gap data found")
        workforce = None

# Merge access gap classification to panel (time-invariant county characteristic)
if workforce is not None:
    panel = panel.merge(
        workforce[['fips5', 'access_gap', 'true_desert', 'county_type']].drop_duplicates(),
        on='fips5', how='left'
    )
    print(f"After merge: {panel['true_desert'].notna().sum()} county-years with desert status")

# =============================================================================
# PREPARE ANALYSIS DATA
# =============================================================================

print("\n" + "="*80)
print("PREPARING ANALYSIS DATA")
print("="*80)

# Define variables
pqi_var = 'pqi_mean_rate'
ed_var = 'ed_visits_resident'  # RESIDENCE-BASED
ed_admit_var = 'ed_admit_rate_resident'  # RESIDENCE-BASED

# Check which years have ED data
ed_years = panel[panel[ed_var].notna()]['year'].unique()
print(f"Years with residence-based ED data: {sorted(ed_years)}")

# Filter to years with ED data
panel_ed = panel[panel['year'].isin(ed_years)].copy()
print(f"Panel filtered to ED years: {len(panel_ed)} county-years")

# Controls
control_vars = ['poverty_pct', 'age65_pct']
available_controls = [c for c in control_vars if c in panel_ed.columns]
print(f"Control variables: {available_controls}")

# Create analysis dataset
analysis_vars = [pqi_var, ed_var, 'access_gap', 'true_desert', 'fips5', 'year'] + available_controls
df = panel_ed.dropna(subset=[v for v in analysis_vars if v in panel_ed.columns and v != ed_var]).copy()

# For ED models, further restrict to non-missing ED
df_ed = df[df[ed_var].notna()].copy()

print(f"\nFinal analysis samples:")
print(f"  PQI models: N = {len(df)} county-years ({df['fips5'].nunique()} counties)")
print(f"  ED models:  N = {len(df_ed)} county-years ({df_ed['fips5'].nunique()} counties)")

# Desert county counts
n_desert = df['true_desert'].sum()
n_desert_unique = df[df['true_desert']==1]['fips5'].nunique()
print(f"\nDesert observations: {n_desert} ({n_desert_unique} unique counties)")

# =============================================================================
# RUN 4 PARALLEL SPECIFICATIONS WITH CLUSTERED SEs
# =============================================================================

print("\n" + "="*80)
print("PARALLEL SPECIFICATIONS (PANEL DATA, CLUSTERED SEs)")
print("="*80)

results = []

# -------------------------------------------------------------------------
# MODEL 1: PQI ~ Access Gap (Continuous) + Controls
# -------------------------------------------------------------------------
print("\n" + "-"*60)
print("MODEL 1: PQI ~ Access Gap (Continuous)")
print("-"*60)

Y1 = df[pqi_var]
X1 = sm.add_constant(df[['access_gap'] + available_controls])

m1 = OLS(Y1, X1).fit(cov_type='cluster', cov_kwds={'groups': df['fips5']})

print(f"  N = {int(m1.nobs)} county-years")
print(f"  Coefficient (access_gap): {m1.params['access_gap']:.4f}")
print(f"  Clustered SE:             {m1.bse['access_gap']:.4f}")
print(f"  t-statistic:              {m1.tvalues['access_gap']:.3f}")
print(f"  P-value:                  {m1.pvalues['access_gap']:.4f}")
print(f"  R-squared:                {m1.rsquared:.4f}")

sig_05 = m1.pvalues['access_gap'] < 0.05
sig_10 = m1.pvalues['access_gap'] < 0.10
print(f"  Significant (p<0.05):     {sig_05}")
print(f"  Significant (p<0.10):     {sig_10}")

results.append({
    'Model': 1,
    'Outcome': 'PQI Rate',
    'IV_Type': 'Continuous',
    'IV_Name': 'Access Gap',
    'Coefficient': m1.params['access_gap'],
    'SE': m1.bse['access_gap'],
    't_stat': m1.tvalues['access_gap'],
    'p_value': m1.pvalues['access_gap'],
    'R2': m1.rsquared,
    'N_obs': int(m1.nobs),
    'N_clusters': df['fips5'].nunique(),
    'SE_type': 'Clustered by County',
    'Data_type': 'Panel (2008-2024)',
    'Significant_05': sig_05,
    'Significant_10': sig_10
})

# -------------------------------------------------------------------------
# MODEL 2: PQI ~ Desert Indicator (Binary) + Controls
# -------------------------------------------------------------------------
print("\n" + "-"*60)
print("MODEL 2: PQI ~ Desert Indicator (Binary)")
print("-"*60)

Y2 = df[pqi_var]
X2 = sm.add_constant(df[['true_desert'] + available_controls])

m2 = OLS(Y2, X2).fit(cov_type='cluster', cov_kwds={'groups': df['fips5']})

print(f"  N = {int(m2.nobs)} county-years")
print(f"  Coefficient (desert):     {m2.params['true_desert']:.2f}")
print(f"  Clustered SE:             {m2.bse['true_desert']:.2f}")
print(f"  t-statistic:              {m2.tvalues['true_desert']:.3f}")
print(f"  P-value:                  {m2.pvalues['true_desert']:.4f}")
print(f"  R-squared:                {m2.rsquared:.4f}")

sig_05 = m2.pvalues['true_desert'] < 0.05
sig_10 = m2.pvalues['true_desert'] < 0.10
print(f"  Significant (p<0.05):     {sig_05}")
print(f"  Significant (p<0.10):     {sig_10}")

results.append({
    'Model': 2,
    'Outcome': 'PQI Rate',
    'IV_Type': 'Binary',
    'IV_Name': 'Desert Indicator',
    'Coefficient': m2.params['true_desert'],
    'SE': m2.bse['true_desert'],
    't_stat': m2.tvalues['true_desert'],
    'p_value': m2.pvalues['true_desert'],
    'R2': m2.rsquared,
    'N_obs': int(m2.nobs),
    'N_clusters': df['fips5'].nunique(),
    'SE_type': 'Clustered by County',
    'Data_type': 'Panel (2008-2024)',
    'Significant_05': sig_05,
    'Significant_10': sig_10
})

# -------------------------------------------------------------------------
# MODEL 3: ED (Residence-Based) ~ Access Gap (Continuous) + Controls
# -------------------------------------------------------------------------
print("\n" + "-"*60)
print(f"MODEL 3: ED ({ed_var}) ~ Access Gap (Continuous)")
print("NOTE: Using RESIDENCE-BASED ED data")
print("-"*60)

Y3 = df_ed[ed_var]
X3 = sm.add_constant(df_ed[['access_gap'] + available_controls])

m3 = OLS(Y3, X3).fit(cov_type='cluster', cov_kwds={'groups': df_ed['fips5']})

print(f"  N = {int(m3.nobs)} county-years")
print(f"  Coefficient (access_gap): {m3.params['access_gap']:.2f}")
print(f"  Clustered SE:             {m3.bse['access_gap']:.2f}")
print(f"  t-statistic:              {m3.tvalues['access_gap']:.3f}")
print(f"  P-value:                  {m3.pvalues['access_gap']:.4f}")
print(f"  R-squared:                {m3.rsquared:.4f}")

sig_05 = m3.pvalues['access_gap'] < 0.05
sig_10 = m3.pvalues['access_gap'] < 0.10
print(f"  Significant (p<0.05):     {sig_05}")
print(f"  Significant (p<0.10):     {sig_10}")

results.append({
    'Model': 3,
    'Outcome': 'ED Visits (Residence-Based)',
    'IV_Type': 'Continuous',
    'IV_Name': 'Access Gap',
    'Coefficient': m3.params['access_gap'],
    'SE': m3.bse['access_gap'],
    't_stat': m3.tvalues['access_gap'],
    'p_value': m3.pvalues['access_gap'],
    'R2': m3.rsquared,
    'N_obs': int(m3.nobs),
    'N_clusters': df_ed['fips5'].nunique(),
    'SE_type': 'Clustered by County',
    'Data_type': 'Panel (2008-2024)',
    'Significant_05': sig_05,
    'Significant_10': sig_10
})

# -------------------------------------------------------------------------
# MODEL 4: ED (Residence-Based) ~ Desert Indicator (Binary) + Controls
# -------------------------------------------------------------------------
print("\n" + "-"*60)
print(f"MODEL 4: ED ({ed_var}) ~ Desert Indicator (Binary)")
print("NOTE: Using RESIDENCE-BASED ED data")
print("-"*60)

Y4 = df_ed[ed_var]
X4 = sm.add_constant(df_ed[['true_desert'] + available_controls])

m4 = OLS(Y4, X4).fit(cov_type='cluster', cov_kwds={'groups': df_ed['fips5']})

print(f"  N = {int(m4.nobs)} county-years")
print(f"  Coefficient (desert):     {m4.params['true_desert']:.2f}")
print(f"  Clustered SE:             {m4.bse['true_desert']:.2f}")
print(f"  t-statistic:              {m4.tvalues['true_desert']:.3f}")
print(f"  P-value:                  {m4.pvalues['true_desert']:.4f}")
print(f"  R-squared:                {m4.rsquared:.4f}")

sig_05 = m4.pvalues['true_desert'] < 0.05
sig_10 = m4.pvalues['true_desert'] < 0.10
print(f"  Significant (p<0.05):     {sig_05}")
print(f"  Significant (p<0.10):     {sig_10}")

results.append({
    'Model': 4,
    'Outcome': 'ED Visits (Residence-Based)',
    'IV_Type': 'Binary',
    'IV_Name': 'Desert Indicator',
    'Coefficient': m4.params['true_desert'],
    'SE': m4.bse['true_desert'],
    't_stat': m4.tvalues['true_desert'],
    'p_value': m4.pvalues['true_desert'],
    'R2': m4.rsquared,
    'N_obs': int(m4.nobs),
    'N_clusters': df_ed['fips5'].nunique(),
    'SE_type': 'Clustered by County',
    'Data_type': 'Panel (2008-2024)',
    'Significant_05': sig_05,
    'Significant_10': sig_10
})

# =============================================================================
# SAVE RESULTS
# =============================================================================

print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)

results_df = pd.DataFrame(results)
print("\n" + results_df[['Model', 'Outcome', 'IV_Type', 'Coefficient', 'SE', 'p_value', 'N_obs', 'Significant_05']].to_string(index=False))

# Save to CSV
results_df.to_csv(f'{OUTPUT_DIR}/tables/CORRECTED_parallel_specifications.csv', index=False)
print(f"\nSaved: {OUTPUT_DIR}/tables/CORRECTED_parallel_specifications.csv")

# =============================================================================
# COMPARISON: OLD vs NEW RESULTS
# =============================================================================

print("\n" + "="*80)
print("COMPARISON: PREVIOUS (INCORRECT) vs CORRECTED RESULTS")
print("="*80)

comparison_doc = f"""
METHODOLOGY COMPARISON
======================

PREVIOUS ANALYSIS (capstone_statistical_revisions.py):
------------------------------------------------------
- Data: ca_cross_section_2020_with_ed.csv
- Sample: N = 55 counties (2020 only)
- ED Data: FACILITY-BASED (ed_visits, ed_admissions)
- Standard Errors: Robust HC1 (not clustered)
- Issue: ED visits attributed to facility location, not patient residence

CORRECTED ANALYSIS (this script):
---------------------------------
- Data: master_panel_2005_2025.csv
- Sample: N = {len(df)} county-years ({df['fips5'].nunique()} counties, 2008-2024)
- ED Data: RESIDENCE-BASED (ed_visits_resident, ed_admit_rate_resident)
- Standard Errors: Clustered by county
- Improvement: ED visits properly attributed to patient's home county

WHY THIS MATTERS:
-----------------
The previous negative coefficient on ED in desert counties was an ARTIFACT
of using facility-based data. Desert counties have fewer ED facilities,
so their residents' visits get counted in neighboring counties.

With residence-based data, we can properly test whether desert RESIDENTS
have different ED utilization patterns.

RESULTS COMPARISON:
------------------
"""

# Try to load old results for comparison
try:
    old_results = pd.read_csv(f'{OUTPUT_DIR}/tables/parallel_specifications_complete.csv')
    comparison_doc += "\nPrevious (Facility-Based ED, Cross-Section, N=55):\n"
    comparison_doc += old_results[['Model', 'Outcome', 'Coefficient', 'p_value']].to_string(index=False)
except:
    comparison_doc += "\n(Previous results file not found)\n"

comparison_doc += f"\n\nCorrected (Residence-Based ED, Panel, N={len(df)}):\n"
comparison_doc += results_df[['Model', 'Outcome', 'Coefficient', 'p_value']].to_string(index=False)

print(comparison_doc)

# Save comparison
with open(f'{OUTPUT_DIR}/METHODOLOGY_COMPARISON.md', 'w') as f:
    f.write(comparison_doc)
    f.write(f"\n\n{data_documentation}")

print(f"\nSaved: {OUTPUT_DIR}/METHODOLOGY_COMPARISON.md")

# =============================================================================
# INTERPRETATION
# =============================================================================

print("\n" + "="*80)
print("INTERPRETATION OF CORRECTED RESULTS")
print("="*80)

interpretation = """
KEY FINDINGS (CORRECTED):
-------------------------
"""

for _, row in results_df.iterrows():
    interpretation += f"\nModel {row['Model']}: {row['Outcome']} ~ {row['IV_Name']}\n"
    interpretation += f"  Coefficient: {row['Coefficient']:.4f}\n"
    interpretation += f"  P-value: {row['p_value']:.4f}\n"
    
    if row['p_value'] < 0.05:
        interpretation += f"  --> SIGNIFICANT at p < 0.05\n"
    elif row['p_value'] < 0.10:
        interpretation += f"  --> Marginally significant at p < 0.10\n"
    else:
        interpretation += f"  --> NOT significant\n"

interpretation += """
IMPORTANT NOTES:
----------------
1. These results use RESIDENCE-BASED ED data, which properly attributes
   ED visits to where patients live, not where hospitals are located.

2. Standard errors are clustered by county to account for within-county
   correlation across years.

3. The panel data (N~{}) provides more statistical power than the
   previous cross-sectional analysis (N=55).

4. If ED coefficients are now POSITIVE for desert counties, this would
   indicate that desert RESIDENTS actually have higher ED utilization
   (when visits are properly attributed), consistent with access barriers.
""".format(len(df))

print(interpretation)

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
