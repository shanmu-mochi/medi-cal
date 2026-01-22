#!/usr/bin/env python3
"""
Prop 56 Provider Mechanism Analysis
====================================

THE SMOKING GUN: Did Prop 56 increase provider enrollment in Medi-Cal?

Key Test:
New_MC_Providers_it = β₀ + β₁(Post_July2017) + β₂(High_MC) + β₃(Post × High_MC) + ε

If β₃ > 0 and significant: Prop 56 attracted MORE providers to high-MC counties
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import os
os.makedirs('outputs_policy/prop56_analysis', exist_ok=True)

print("="*80)
print("PROP 56 MECHANISM TEST: Did Payment Increases Attract Providers?")
print("="*80)

# ============================================================================
# LOAD PROVIDER DATA
# ============================================================================

print("\n--- Loading Provider Data ---")

providers = pd.read_csv('MedicalFFS_providers.csv', encoding='utf-8-sig')
print(f"Total provider records: {len(providers):,}")

# Parse enrollment date
providers['enroll_date'] = pd.to_datetime(providers['Enroll_Status_Eff_DT'], format='%m/%d/%Y', errors='coerce')
providers['enroll_year'] = providers['enroll_date'].dt.year
providers['enroll_month'] = providers['enroll_date'].dt.month

# Filter to valid dates and California
providers = providers[providers['enroll_date'].notna()]
providers = providers[providers['State'] == 'CA']
print(f"CA providers with valid dates: {len(providers):,}")

# Year range
print(f"Enrollment years: {providers['enroll_year'].min():.0f} - {providers['enroll_year'].max():.0f}")

# ============================================================================
# IDENTIFY PRIMARY CARE PROVIDERS
# ============================================================================

print("\n--- Identifying Primary Care Providers ---")

# Method 1: By ANC_Provider_Type
primary_care_anc = ['Adult Primary Care', 'Pediatric Primary Care']
providers['is_primary_care_anc'] = providers['ANC_Provider_Type'].isin(primary_care_anc)

# Method 2: By specialty
primary_care_specialties = ['Family Practice', 'Internal Medicine', 'Pediatrics', 'General Practice']
providers['is_primary_care_spec'] = providers['FI_Provider_Specialty'].isin(primary_care_specialties)

# Method 3: By provider type = Physicians with primary care taxonomy
providers['is_physician'] = providers['FI_Provider_Type'] == 'PHYSICIANS'

# Combined: Primary care providers
providers['is_pcp'] = (providers['is_primary_care_anc'] | providers['is_primary_care_spec']) & providers['is_physician']

print(f"Primary Care (ANC): {providers['is_primary_care_anc'].sum():,}")
print(f"Primary Care (Specialty): {providers['is_primary_care_spec'].sum():,}")
print(f"Physicians: {providers['is_physician'].sum():,}")
print(f"PCPs (combined): {providers['is_pcp'].sum():,}")

# ============================================================================
# AGGREGATE BY COUNTY-YEAR
# ============================================================================

print("\n--- Aggregating by County-Year ---")

# Clean county FIPS - convert float to int string
providers['fips5'] = providers['FIPS_County_CD'].fillna(0).astype(int).astype(str).str.zfill(5)
# Remove invalid FIPS (00000)
providers = providers[providers['fips5'] != '00000']

# Filter to 2010-2024 for analysis
providers_analysis = providers[(providers['enroll_year'] >= 2010) & (providers['enroll_year'] <= 2024)].copy()

# Count new enrollments by county-year
enrollments = providers_analysis.groupby(['fips5', 'enroll_year']).agg(
    new_providers=('NPI', 'count'),
    new_physicians=('is_physician', 'sum'),
    new_pcps=('is_pcp', 'sum')
).reset_index()
enrollments.columns = ['fips5', 'year', 'new_providers', 'new_physicians', 'new_pcps']

print(f"County-year observations: {len(enrollments):,}")
print(f"Counties: {enrollments['fips5'].nunique()}")
print(f"Years: {enrollments['year'].min()} - {enrollments['year'].max()}")

# ============================================================================
# MERGE WITH COUNTY CHARACTERISTICS
# ============================================================================

print("\n--- Merging with County Characteristics ---")

# Load access gap data for MC share
access_gap = pd.read_csv('outputs_v2/data/county_access_gap_2020.csv')
access_gap['fips5'] = access_gap['fips5'].astype(int).astype(str).str.zfill(5)

# Load panel for population and MC share
panel = pd.read_csv('outputs/data/master_panel_2005_2025.csv')
panel['fips5'] = panel['fips5'].astype(int).astype(str).str.zfill(5)

# Get county characteristics
county_chars = panel.groupby('fips5').agg({
    'population': 'mean',
    'medi_cal_share': 'mean'
}).reset_index()

# Merge
enrollments = enrollments.merge(county_chars, on='fips5', how='left')
enrollments = enrollments.merge(access_gap[['fips5', 'county_type', 'access_gap']], on='fips5', how='left')

# Create treatment indicators
mc_median = enrollments['medi_cal_share'].median()
enrollments['high_mc'] = (enrollments['medi_cal_share'] > mc_median).astype(int)

# Post Prop 56 (July 2017 - use 2018+ as full post period)
enrollments['post_prop56'] = (enrollments['year'] >= 2018).astype(int)

# Interaction
enrollments['treat_x_post'] = enrollments['high_mc'] * enrollments['post_prop56']

# Per capita rates (per 100k population)
enrollments['new_providers_per100k'] = enrollments['new_providers'] / enrollments['population'] * 100000
enrollments['new_physicians_per100k'] = enrollments['new_physicians'] / enrollments['population'] * 100000
enrollments['new_pcps_per100k'] = enrollments['new_pcps'] / enrollments['population'] * 100000

# Drop missing
enrollments_clean = enrollments.dropna(subset=['new_providers_per100k', 'high_mc', 'population'])
print(f"Analysis sample: {len(enrollments_clean):,} county-years")

# ============================================================================
# DESCRIPTIVE ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("DESCRIPTIVE ANALYSIS: Provider Enrollment Trends")
print("="*80)

# Overall trends
yearly_trends = providers_analysis.groupby('enroll_year').agg(
    total_new=('NPI', 'count'),
    new_physicians=('is_physician', 'sum'),
    new_pcps=('is_pcp', 'sum')
)

print("\n--- Yearly Enrollment Trends ---")
print(yearly_trends.to_string())

# Pre vs Post comparison
pre_prop56 = providers_analysis[providers_analysis['enroll_year'] < 2017]
post_prop56 = providers_analysis[providers_analysis['enroll_year'] >= 2018]

pre_annual = len(pre_prop56) / (2017 - 2010)
post_annual = len(post_prop56) / (2024 - 2018 + 1)

print(f"\n--- Pre vs Post Prop 56 ---")
print(f"Pre (2010-2016): {len(pre_prop56):,} providers ({pre_annual:.0f}/year)")
print(f"Post (2018-2024): {len(post_prop56):,} providers ({post_annual:.0f}/year)")
print(f"Change: {(post_annual/pre_annual - 1)*100:+.1f}%")

# By county type
if 'county_type' in providers_analysis.columns:
    print("\n--- By County Type (Post 2018) ---")
    post_by_type = post_prop56.merge(access_gap[['fips5', 'county_type']], on='fips5', how='left')
    type_counts = post_by_type.groupby('county_type').size()
    print(type_counts)

# ============================================================================
# THE KEY TEST: DiD for Provider Enrollment
# ============================================================================

print("\n" + "="*80)
print("KEY TEST: Did Prop 56 Increase Provider Enrollment in High-MC Counties?")
print("="*80)

print("""
Model: New_Providers_it = β₀ + β₁(Post_2018) + β₂(High_MC) + β₃(Post × High_MC) + ε

If β₃ > 0 and significant: Prop 56 attracted MORE providers to high-MC counties
""")

# Test 1: All providers
print("\n--- Test 1: All New Providers ---")
X1 = sm.add_constant(enrollments_clean[['post_prop56', 'high_mc', 'treat_x_post']])
Y1 = enrollments_clean['new_providers_per100k']
m1 = OLS(Y1, X1).fit(cov_type='cluster', cov_kwds={'groups': enrollments_clean['fips5']})

print(f"DiD (β₃): {m1.params['treat_x_post']:.2f} providers per 100k")
print(f"SE: {m1.bse['treat_x_post']:.2f}")
print(f"p-value: {m1.pvalues['treat_x_post']:.4f}")
sig1 = '***' if m1.pvalues['treat_x_post'] < 0.01 else '**' if m1.pvalues['treat_x_post'] < 0.05 else '*' if m1.pvalues['treat_x_post'] < 0.10 else ''
print(f"Significance: {sig1}")

# Test 2: Physicians only
print("\n--- Test 2: New Physicians ---")
X2 = sm.add_constant(enrollments_clean[['post_prop56', 'high_mc', 'treat_x_post']])
Y2 = enrollments_clean['new_physicians_per100k']
m2 = OLS(Y2, X2).fit(cov_type='cluster', cov_kwds={'groups': enrollments_clean['fips5']})

print(f"DiD (β₃): {m2.params['treat_x_post']:.2f} physicians per 100k")
print(f"SE: {m2.bse['treat_x_post']:.2f}")
print(f"p-value: {m2.pvalues['treat_x_post']:.4f}")
sig2 = '***' if m2.pvalues['treat_x_post'] < 0.01 else '**' if m2.pvalues['treat_x_post'] < 0.05 else '*' if m2.pvalues['treat_x_post'] < 0.10 else ''
print(f"Significance: {sig2}")

# Test 3: PCPs only (most relevant)
print("\n--- Test 3: New Primary Care Providers (KEY TEST) ---")
X3 = sm.add_constant(enrollments_clean[['post_prop56', 'high_mc', 'treat_x_post']])
Y3 = enrollments_clean['new_pcps_per100k']
m3 = OLS(Y3, X3).fit(cov_type='cluster', cov_kwds={'groups': enrollments_clean['fips5']})

print(f"DiD (β₃): {m3.params['treat_x_post']:.2f} PCPs per 100k")
print(f"SE: {m3.bse['treat_x_post']:.2f}")
print(f"p-value: {m3.pvalues['treat_x_post']:.4f}")
sig3 = '***' if m3.pvalues['treat_x_post'] < 0.01 else '**' if m3.pvalues['treat_x_post'] < 0.05 else '*' if m3.pvalues['treat_x_post'] < 0.10 else ''
print(f"Significance: {sig3}")

# ============================================================================
# ADDITIONAL TESTS
# ============================================================================

print("\n" + "="*80)
print("ADDITIONAL MECHANISM TESTS")
print("="*80)

# Test 4: By desert status
print("\n--- Test 4: DiD by Desert Status ---")
enrollments_clean['true_desert'] = (enrollments_clean['county_type'] == 'TRUE DESERT').astype(int)
enrollments_clean['desert_x_post'] = enrollments_clean['true_desert'] * enrollments_clean['post_prop56']

X4 = sm.add_constant(enrollments_clean[['post_prop56', 'true_desert', 'desert_x_post']])
Y4 = enrollments_clean['new_pcps_per100k']
m4 = OLS(Y4, X4).fit(cov_type='cluster', cov_kwds={'groups': enrollments_clean['fips5']})

print(f"DiD (Desert × Post): {m4.params['desert_x_post']:.2f} PCPs per 100k")
print(f"p-value: {m4.pvalues['desert_x_post']:.4f}")

# Test 5: Event study for enrollments
print("\n--- Test 5: Event Study - PCP Enrollment ---")
enrollments_clean['event_time'] = enrollments_clean['year'] - 2017

es_results = []
for year in range(2012, 2024):
    subset = enrollments_clean[enrollments_clean['year'] == year]
    if len(subset) > 20:
        high_mc_mean = subset[subset['high_mc'] == 1]['new_pcps_per100k'].mean()
        low_mc_mean = subset[subset['high_mc'] == 0]['new_pcps_per100k'].mean()
        diff = high_mc_mean - low_mc_mean
        es_results.append({'year': year, 'event_time': year - 2017, 'diff': diff,
                          'high_mc': high_mc_mean, 'low_mc': low_mc_mean})

es_enroll = pd.DataFrame(es_results)
print(es_enroll.to_string(index=False))

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

fig = plt.figure(figsize=(16, 12))
fig.suptitle('Prop 56 Mechanism Analysis: Provider Enrollment', fontsize=16, fontweight='bold')

gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# 1. Overall enrollment trends
ax1 = fig.add_subplot(gs[0, 0])
ax1.bar(yearly_trends.index, yearly_trends['new_pcps'], color='steelblue', alpha=0.7, label='New PCPs')
ax1.axvline(x=2017, color='red', linestyle='--', linewidth=2, label='Prop 56')
ax1.set_xlabel('Year')
ax1.set_ylabel('New PCP Enrollments')
ax1.set_title('1. Statewide PCP Enrollment in Medi-Cal FFS')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. High-MC vs Low-MC trends
ax2 = fig.add_subplot(gs[0, 1])
if len(es_enroll) > 0:
    ax2.plot(es_enroll['year'], es_enroll['high_mc'], 'r-o', label='High-MC Counties', linewidth=2)
    ax2.plot(es_enroll['year'], es_enroll['low_mc'], 'b-o', label='Low-MC Counties', linewidth=2)
    ax2.axvline(x=2017, color='gray', linestyle='--', linewidth=2, label='Prop 56')
ax2.set_xlabel('Year')
ax2.set_ylabel('New PCPs per 100k Population')
ax2.set_title('2. PCP Enrollment by MC Share')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Event study - difference
ax3 = fig.add_subplot(gs[1, 0])
if len(es_enroll) > 0:
    colors = ['green' if et < 0 else 'red' for et in es_enroll['event_time']]
    ax3.bar(es_enroll['event_time'], es_enroll['diff'], color=colors, alpha=0.7)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax3.set_xlabel('Years Relative to Prop 56')
ax3.set_ylabel('High-MC - Low-MC (PCPs per 100k)')
ax3.set_title('3. Event Study: Differential PCP Enrollment')
ax3.grid(True, alpha=0.3)

# 4. Summary results
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis('off')

pcp_effect = "POSITIVE" if m3.params['treat_x_post'] > 0 else "NEGATIVE"
pcp_sig = "YES" if m3.pvalues['treat_x_post'] < 0.10 else "NO"

summary_text = f"""
PROP 56 MECHANISM TEST RESULTS
═══════════════════════════════════════════════════

DiD: Did High-MC Counties Get More Providers Post-2018?

Provider Type         DiD Effect    P-value   Significant?
─────────────────────────────────────────────────────────
All Providers         {m1.params['treat_x_post']:>+8.1f}     {m1.pvalues['treat_x_post']:.4f}    {sig1 if sig1 else 'No'}
Physicians            {m2.params['treat_x_post']:>+8.1f}     {m2.pvalues['treat_x_post']:.4f}    {sig2 if sig2 else 'No'}
PRIMARY CARE (KEY)    {m3.params['treat_x_post']:>+8.1f}     {m3.pvalues['treat_x_post']:.4f}    {sig3 if sig3 else 'No'}

═══════════════════════════════════════════════════

INTERPRETATION:
The DiD effect for PCPs is {pcp_effect} ({m3.params['treat_x_post']:+.1f} per 100k)
Statistically significant: {pcp_sig}

{'✓ MECHANISM SUPPORTED: Prop 56 DID attract more PCPs to high-MC counties!' if m3.params['treat_x_post'] > 0 and m3.pvalues['treat_x_post'] < 0.10 else '✗ MECHANISM NOT SUPPORTED: No significant differential enrollment in high-MC counties'}

This {'strengthens' if m3.params['treat_x_post'] > 0 and m3.pvalues['treat_x_post'] < 0.10 else 'does NOT strengthen'} the causal case for Prop 56.
"""

ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', 
                   facecolor='lightgreen' if (m3.params['treat_x_post'] > 0 and m3.pvalues['treat_x_post'] < 0.10) else 'lightyellow',
                   alpha=0.8))
ax4.set_title('4. Summary Assessment')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('outputs_policy/prop56_analysis/provider_mechanism_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Saved: outputs_policy/prop56_analysis/provider_mechanism_analysis.png")

# ============================================================================
# SAVE RESULTS
# ============================================================================

# Save enrollment data
enrollments_clean.to_csv('outputs_policy/prop56_analysis/county_year_enrollments.csv', index=False)
print("✓ Saved: county_year_enrollments.csv")

# Save DiD results
did_results = pd.DataFrame([
    {'outcome': 'All Providers', 'did_coef': m1.params['treat_x_post'], 
     'se': m1.bse['treat_x_post'], 'pval': m1.pvalues['treat_x_post']},
    {'outcome': 'Physicians', 'did_coef': m2.params['treat_x_post'],
     'se': m2.bse['treat_x_post'], 'pval': m2.pvalues['treat_x_post']},
    {'outcome': 'Primary Care', 'did_coef': m3.params['treat_x_post'],
     'se': m3.bse['treat_x_post'], 'pval': m3.pvalues['treat_x_post']},
])
did_results.to_csv('outputs_policy/prop56_analysis/provider_did_results.csv', index=False)
print("✓ Saved: provider_did_results.csv")

# Save event study
es_enroll.to_csv('outputs_policy/prop56_analysis/provider_event_study.csv', index=False)
print("✓ Saved: provider_event_study.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

# Final assessment
print(f"""
FINAL MECHANISM ASSESSMENT
══════════════════════════════════════════════════════════════════════════

Question: Did Prop 56 payment increases attract more PCPs to high-MC counties?

Result: DiD = {m3.params['treat_x_post']:+.2f} PCPs per 100k, p = {m3.pvalues['treat_x_post']:.4f}

{'=' * 70}
{'MECHANISM SUPPORTED!' if m3.params['treat_x_post'] > 0 and m3.pvalues['treat_x_post'] < 0.10 else 'MECHANISM NOT SUPPORTED'}
{'=' * 70}

{
'This STRENGTHENS the causal case for Prop 56:' + chr(10) +
'  1. Payment increases led to differential provider enrollment' + chr(10) +
'  2. High-MC counties gained ' + f"{m3.params['treat_x_post']:.1f}" + ' more PCPs/100k' + chr(10) +
'  3. Combined with outcome improvement, this supports the causal chain:' + chr(10) +
'     Prop 56 → More Providers → Better Outcomes'
if m3.params['treat_x_post'] > 0 and m3.pvalues['treat_x_post'] < 0.10 
else 
'This WEAKENS the causal case for Prop 56:' + chr(10) +
'  - No significant differential enrollment in high-MC counties' + chr(10) +
'  - The payment increases did NOT attract more providers' + chr(10) +
'  - Outcome improvements (if any) likely due to other factors'
}
""")

plt.show()
