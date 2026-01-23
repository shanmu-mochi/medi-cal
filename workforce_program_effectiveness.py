#!/usr/bin/env python3
"""
Workforce Program Effectiveness Analysis
=========================================

Test which programs are most effective at:
1. Redistributing providers to desert counties
2. Decreasing PQI (preventable hospitalizations)
3. Reducing healthcare costs

Programs tested:
- NHSC (National Health Service Corps)
- FQHCs (Federally Qualified Health Centers)
- Rural Health Clinics (RHCs)
- Community Mental Health Centers (CMHCs)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import os

os.makedirs('outputs_policy/workforce_programs', exist_ok=True)

print("="*80)
print("WORKFORCE PROGRAM EFFECTIVENESS ANALYSIS")
print("="*80)

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n--- Loading Data ---")

# Load AHRF health facilities data (has program counts)
ahrf_hf = pd.read_csv('newdata/AHRF 2023-2024 CSV/CSV Files by Categories/ahrf2024hf.csv')
ahrf_hf['fips5'] = ahrf_hf['fips_st_cnty'].astype(str).str.zfill(5)

# Filter to California
ahrf_ca = ahrf_hf[ahrf_hf['fips5'].str.startswith('06')].copy()
print(f"California counties: {len(ahrf_ca)}")

# Load our panel and access gap data
panel = pd.read_csv('outputs/data/master_panel_2005_2025.csv')
panel['fips5'] = panel['fips5'].astype(int).astype(str).str.zfill(5)

access_gap = pd.read_csv('outputs_v2/data/county_access_gap_2020.csv')
access_gap['fips5'] = access_gap['fips5'].astype(int).astype(str).str.zfill(5)

# Load AHRF health professionals data for PCP counts
ahrf_hp = pd.read_csv('newdata/AHRF 2023-2024 CSV/CSV Files by Categories/ahrf2024hp.csv')
ahrf_hp['fips5'] = ahrf_hp['fips_st_cnty'].astype(str).str.zfill(5)
ahrf_hp_ca = ahrf_hp[ahrf_hp['fips5'].str.startswith('06')].copy()

# Load cost data if available
try:
    costs = pd.read_csv('outputs/data/hospital_costs_new.csv')
    costs['fips5'] = costs['fips5'].astype(str).str.zfill(5)
    has_costs = True
    print("✓ Cost data loaded")
except:
    has_costs = False
    print("✗ Cost data not available")

# ============================================================================
# EXTRACT PROGRAM VARIABLES
# ============================================================================

print("\n--- Extracting Program Variables ---")

# NHSC variables
nhsc_cols = [col for col in ahrf_ca.columns if 'nhsc' in col.lower()]
print(f"NHSC columns: {nhsc_cols}")

# Use most recent year
ahrf_ca['nhsc_sites'] = pd.to_numeric(ahrf_ca.get('nhsc_activ_sites_24', ahrf_ca.get('nhsc_activ_sites_23', 0)), errors='coerce').fillna(0)
ahrf_ca['nhsc_pcp_sites'] = pd.to_numeric(ahrf_ca.get('nhsc_prim_care_sites_24', ahrf_ca.get('nhsc_prim_care_sites_23', 0)), errors='coerce').fillna(0)
ahrf_ca['nhsc_fte'] = pd.to_numeric(ahrf_ca.get('nhsc_fte_provdrs_24', ahrf_ca.get('nhsc_fte_provdrs_23', 0)), errors='coerce').fillna(0)
ahrf_ca['nhsc_pcp_fte'] = pd.to_numeric(ahrf_ca.get('nhsc_fte_prim_care_provdrs_24', ahrf_ca.get('nhsc_fte_prim_care_provdrs_23', 0)), errors='coerce').fillna(0)

# FQHC
ahrf_ca['fqhc_count'] = pd.to_numeric(ahrf_ca.get('fedly_qualfd_hlth_ctr_23', ahrf_ca.get('fedly_qualfd_hlth_ctr_22', 0)), errors='coerce').fillna(0)

# Rural Health Clinics
ahrf_ca['rhc_count'] = pd.to_numeric(ahrf_ca.get('rural_hlth_clincs_23', ahrf_ca.get('rural_hlth_clincs_22', 0)), errors='coerce').fillna(0)

# Community Mental Health Centers
ahrf_ca['cmhc_count'] = pd.to_numeric(ahrf_ca.get('comn_mentl_hlth_ctr_23', ahrf_ca.get('comn_mentl_hlth_ctr_22', 0)), errors='coerce').fillna(0)

# Community Health Centers (grants)
ahrf_ca['chc_grants'] = pd.to_numeric(ahrf_ca.get('comn_hlth_ctr_grants_only_24', ahrf_ca.get('comn_hlth_ctr_grants_only_23', 0)), errors='coerce').fillna(0)

print(f"NHSC sites range: {ahrf_ca['nhsc_sites'].min():.0f} - {ahrf_ca['nhsc_sites'].max():.0f}")
print(f"FQHC count range: {ahrf_ca['fqhc_count'].min():.0f} - {ahrf_ca['fqhc_count'].max():.0f}")
print(f"RHC count range: {ahrf_ca['rhc_count'].min():.0f} - {ahrf_ca['rhc_count'].max():.0f}")

# ============================================================================
# BUILD ANALYSIS DATASET
# ============================================================================

print("\n--- Building Analysis Dataset ---")

# Get population from panel (most recent year)
pop_recent = panel[panel['year'] == panel['year'].max()][['fips5', 'population']].drop_duplicates()

# Get outcomes (PQI) from panel
outcomes = panel[panel['year'] >= 2020].groupby('fips5').agg({
    'pqi_mean_rate': 'mean',
    'population': 'mean'
}).reset_index()

# Merge everything
analysis = ahrf_ca[['fips5', 'nhsc_sites', 'nhsc_pcp_sites', 'nhsc_fte', 'nhsc_pcp_fte', 
                    'fqhc_count', 'rhc_count', 'cmhc_count', 'chc_grants']].copy()
analysis = analysis.merge(access_gap[['fips5', 'county_type', 'access_gap', 'pcp_per_100k']], on='fips5', how='left')
analysis = analysis.merge(outcomes, on='fips5', how='left')
analysis = analysis.merge(ahrf_hp_ca[['fips5', 'phys_nf_prim_care_pc_exc_rsdt_22']], on='fips5', how='left')

# Rename PCP column
analysis.rename(columns={'phys_nf_prim_care_pc_exc_rsdt_22': 'ahrf_pcp_count'}, inplace=True)

# Calculate per capita rates
analysis['nhsc_per_100k'] = analysis['nhsc_fte'] / analysis['population'] * 100000
analysis['fqhc_per_100k'] = analysis['fqhc_count'] / analysis['population'] * 100000
analysis['rhc_per_100k'] = analysis['rhc_count'] / analysis['population'] * 100000
analysis['cmhc_per_100k'] = analysis['cmhc_count'] / analysis['population'] * 100000
analysis['pcp_per_100k_ahrf'] = analysis['ahrf_pcp_count'] / analysis['population'] * 100000

# Create desert indicator
analysis['true_desert'] = (analysis['county_type'] == 'TRUE DESERT').astype(int)

# Merge costs if available
if has_costs:
    cost_recent = costs[costs['year'] >= 2020].groupby('fips5')['cost_per_discharge'].mean().reset_index()
    analysis = analysis.merge(cost_recent, on='fips5', how='left')

print(f"Analysis dataset: {len(analysis)} counties")
print(f"TRUE DESERT counties: {analysis['true_desert'].sum()}")

# ============================================================================
# ANALYSIS 1: WHICH PROGRAMS REDISTRIBUTE PROVIDERS TO DESERTS?
# ============================================================================

print("\n" + "="*80)
print("ANALYSIS 1: PROVIDER REDISTRIBUTION TO DESERT COUNTIES")
print("="*80)
print("""
Question: Do desert counties have MORE program resources (per capita)?
If yes → Program is targeting underserved areas (good!)
If no → Program is NOT redistributive
""")

results_redistribution = []

programs = {
    'NHSC (All Sites)': 'nhsc_per_100k',
    'NHSC (PCP FTE)': 'nhsc_pcp_fte',
    'FQHCs': 'fqhc_per_100k',
    'Rural Health Clinics': 'rhc_per_100k',
    'Community MH Centers': 'cmhc_per_100k'
}

print(f"\n{'Program':<25} {'Desert Mean':<12} {'Non-Desert':<12} {'Diff':<10} {'T-stat':<10} {'P-value':<10}")
print("-"*80)

for name, col in programs.items():
    if col in analysis.columns:
        desert = analysis[analysis['true_desert'] == 1][col].dropna()
        non_desert = analysis[analysis['true_desert'] == 0][col].dropna()
        
        if len(desert) > 0 and len(non_desert) > 0:
            from scipy import stats
            t_stat, p_val = stats.ttest_ind(desert, non_desert)
            diff = desert.mean() - non_desert.mean()
            
            results_redistribution.append({
                'Program': name,
                'Desert_Mean': desert.mean(),
                'NonDesert_Mean': non_desert.mean(),
                'Difference': diff,
                'T_stat': t_stat,
                'P_value': p_val,
                'Redistributive': 'YES ✓' if diff > 0 and p_val < 0.10 else 'NO'
            })
            
            sig = '*' if p_val < 0.10 else ''
            print(f"{name:<25} {desert.mean():>10.2f}  {non_desert.mean():>10.2f}  {diff:>+8.2f}  {t_stat:>8.2f}  {p_val:>8.4f} {sig}")

# Regression approach
print("\n--- Regression: Program Resources ~ Desert Status ---")
for name, col in programs.items():
    if col in analysis.columns:
        reg_data = analysis.dropna(subset=[col, 'true_desert'])
        if len(reg_data) > 20:
            X = sm.add_constant(reg_data[['true_desert']])
            Y = reg_data[col]
            m = OLS(Y, X).fit(cov_type='HC1')
            sig = '***' if m.pvalues['true_desert'] < 0.01 else '**' if m.pvalues['true_desert'] < 0.05 else '*' if m.pvalues['true_desert'] < 0.10 else ''
            direction = '↑ MORE in deserts' if m.params['true_desert'] > 0 else '↓ LESS in deserts'
            print(f"  {name}: β = {m.params['true_desert']:+.2f}, p = {m.pvalues['true_desert']:.4f} {sig} → {direction}")

# ============================================================================
# ANALYSIS 2: WHICH PROGRAMS REDUCE PQI?
# ============================================================================

print("\n" + "="*80)
print("ANALYSIS 2: EFFECT ON PREVENTABLE HOSPITALIZATIONS (PQI)")
print("="*80)
print("""
Question: Do counties with more program resources have lower PQI?
Negative coefficient → Program improves outcomes
""")

results_pqi = []

# Single program regressions
print(f"\n{'Program':<25} {'Coefficient':<12} {'SE':<10} {'P-value':<10} {'Effect'}")
print("-"*70)

for name, col in programs.items():
    if col in analysis.columns:
        reg_data = analysis.dropna(subset=[col, 'pqi_mean_rate'])
        if len(reg_data) > 20:
            X = sm.add_constant(reg_data[[col]])
            Y = reg_data['pqi_mean_rate']
            m = OLS(Y, X).fit(cov_type='HC1')
            
            results_pqi.append({
                'Program': name,
                'Coefficient': m.params[col],
                'SE': m.bse[col],
                'P_value': m.pvalues[col],
                'Effective': 'YES ✓' if m.params[col] < 0 and m.pvalues[col] < 0.10 else 'NO'
            })
            
            sig = '***' if m.pvalues[col] < 0.01 else '**' if m.pvalues[col] < 0.05 else '*' if m.pvalues[col] < 0.10 else ''
            effect = 'REDUCES PQI ✓' if m.params[col] < 0 else 'Increases PQI'
            print(f"{name:<25} {m.params[col]:>10.2f}  {m.bse[col]:>8.2f}  {m.pvalues[col]:>8.4f} {sig:<4} {effect}")

# Combined model with controls
print("\n--- Combined Model: PQI ~ Programs + Controls ---")
prog_cols = ['nhsc_per_100k', 'fqhc_per_100k', 'rhc_per_100k']
avail_cols = [c for c in prog_cols if c in analysis.columns and analysis[c].notna().sum() > 20]

if len(avail_cols) >= 2:
    reg_data = analysis.dropna(subset=['pqi_mean_rate'] + avail_cols + ['access_gap'])
    if len(reg_data) > 20:
        X = sm.add_constant(reg_data[avail_cols + ['access_gap']])
        Y = reg_data['pqi_mean_rate']
        m_combined = OLS(Y, X).fit(cov_type='HC1')
        
        print(f"\nCombined Model Results (N = {len(reg_data)}):")
        print(f"{'Variable':<20} {'Coef':<10} {'P-value':<10} {'Interpretation'}")
        print("-"*60)
        for var in avail_cols + ['access_gap']:
            sig = '***' if m_combined.pvalues[var] < 0.01 else '**' if m_combined.pvalues[var] < 0.05 else '*' if m_combined.pvalues[var] < 0.10 else ''
            interp = 'Reduces PQI' if m_combined.params[var] < 0 else 'Increases PQI'
            print(f"{var:<20} {m_combined.params[var]:>8.2f}  {m_combined.pvalues[var]:>8.4f} {sig:<4} {interp}")
        print(f"\nR² = {m_combined.rsquared:.3f}")

# ============================================================================
# ANALYSIS 3: WHICH PROGRAMS REDUCE COSTS?
# ============================================================================

print("\n" + "="*80)
print("ANALYSIS 3: EFFECT ON HEALTHCARE COSTS")
print("="*80)

if has_costs and 'cost_per_discharge' in analysis.columns:
    print("""
Question: Do counties with more program resources have lower costs?
Negative coefficient → Program reduces costs
""")
    
    results_costs = []
    
    print(f"\n{'Program':<25} {'Coefficient':<12} {'SE':<10} {'P-value':<10} {'Effect'}")
    print("-"*70)
    
    for name, col in programs.items():
        if col in analysis.columns:
            reg_data = analysis.dropna(subset=[col, 'cost_per_discharge'])
            if len(reg_data) > 15:
                X = sm.add_constant(reg_data[[col]])
                Y = reg_data['cost_per_discharge']
                m = OLS(Y, X).fit(cov_type='HC1')
                
                results_costs.append({
                    'Program': name,
                    'Coefficient': m.params[col],
                    'SE': m.bse[col],
                    'P_value': m.pvalues[col],
                    'Effective': 'YES ✓' if m.params[col] < 0 and m.pvalues[col] < 0.10 else 'NO'
                })
                
                sig = '***' if m.pvalues[col] < 0.01 else '**' if m.pvalues[col] < 0.05 else '*' if m.pvalues[col] < 0.10 else ''
                effect = 'REDUCES COSTS ✓' if m.params[col] < 0 else 'Increases costs'
                print(f"{name:<25} {m.params[col]:>10.0f}  {m.bse[col]:>8.0f}  {m.pvalues[col]:>8.4f} {sig:<4} {effect}")
else:
    print("Cost data not available for this analysis.")
    print("Using PQI as proxy for avoidable costs (PQI hospitalizations are expensive)")
    
    # Calculate implied cost savings
    print("\n--- Implied Cost Savings from PQI Reduction ---")
    avg_pqi_cost = 15000  # Average cost per PQI hospitalization
    
    for name, col in programs.items():
        if col in analysis.columns:
            reg_data = analysis.dropna(subset=[col, 'pqi_mean_rate', 'population'])
            if len(reg_data) > 20:
                X = sm.add_constant(reg_data[[col]])
                Y = reg_data['pqi_mean_rate']
                m = OLS(Y, X).fit()
                
                if m.params[col] < 0:
                    # Calculate: if program increases by 1 unit per 100k, PQI drops by β
                    # Cost savings = β × (pop/100k) × $15k per hospitalization
                    avg_pop = reg_data['population'].mean()
                    implied_savings = abs(m.params[col]) * (avg_pop / 100000) * avg_pqi_cost
                    print(f"  {name}: {m.params[col]:.2f} PQI/100k → ${implied_savings:,.0f} savings per county")

# ============================================================================
# ANALYSIS 4: DESERT-SPECIFIC EFFECTS
# ============================================================================

print("\n" + "="*80)
print("ANALYSIS 4: PROGRAM EFFECTS IN DESERT COUNTIES SPECIFICALLY")
print("="*80)
print("""
Question: Do programs have STRONGER effects in desert counties?
(Interaction: Program × Desert)
""")

for name, col in programs.items():
    if col in analysis.columns:
        reg_data = analysis.dropna(subset=[col, 'pqi_mean_rate', 'true_desert'])
        if len(reg_data) > 20:
            reg_data['prog_x_desert'] = reg_data[col] * reg_data['true_desert']
            X = sm.add_constant(reg_data[[col, 'true_desert', 'prog_x_desert']])
            Y = reg_data['pqi_mean_rate']
            m = OLS(Y, X).fit(cov_type='HC1')
            
            interact_coef = m.params['prog_x_desert']
            interact_p = m.pvalues['prog_x_desert']
            sig = '***' if interact_p < 0.01 else '**' if interact_p < 0.05 else '*' if interact_p < 0.10 else ''
            
            effect = 'STRONGER in deserts' if interact_coef < 0 else 'Weaker in deserts'
            print(f"  {name}: Interaction = {interact_coef:.2f}, p = {interact_p:.4f} {sig} → {effect}")

# ============================================================================
# SUMMARY AND RANKING
# ============================================================================

print("\n" + "="*80)
print("PROGRAM EFFECTIVENESS RANKING")
print("="*80)

# Build summary table
summary_data = []
for name, col in programs.items():
    if col in analysis.columns:
        row = {'Program': name}
        
        # Redistribution
        desert = analysis[analysis['true_desert'] == 1][col].dropna()
        non_desert = analysis[analysis['true_desert'] == 0][col].dropna()
        if len(desert) > 0 and len(non_desert) > 0:
            from scipy import stats
            diff = desert.mean() - non_desert.mean()
            t_stat, p_val = stats.ttest_ind(desert, non_desert)
            row['Redistribution'] = '✓' if diff > 0 else '✗'
            row['Redist_Score'] = 1 if diff > 0 and p_val < 0.10 else 0
        
        # PQI effect
        reg_data = analysis.dropna(subset=[col, 'pqi_mean_rate'])
        if len(reg_data) > 20:
            X = sm.add_constant(reg_data[[col]])
            Y = reg_data['pqi_mean_rate']
            m = OLS(Y, X).fit()
            row['PQI_Effect'] = '✓' if m.params[col] < 0 else '✗'
            row['PQI_Score'] = 1 if m.params[col] < 0 and m.pvalues[col] < 0.10 else 0
            row['PQI_Coef'] = m.params[col]
            row['PQI_P'] = m.pvalues[col]
        
        # Total score
        row['Total_Score'] = row.get('Redist_Score', 0) + row.get('PQI_Score', 0)
        summary_data.append(row)

summary_df = pd.DataFrame(summary_data)
summary_df = summary_df.sort_values('Total_Score', ascending=False)

print("\n" + "-"*80)
print(f"{'Rank':<5} {'Program':<25} {'Targets Deserts?':<18} {'Reduces PQI?':<15} {'Score'}")
print("-"*80)

for i, (_, row) in enumerate(summary_df.iterrows(), 1):
    print(f"{i:<5} {row['Program']:<25} {row.get('Redistribution', '?'):<18} {row.get('PQI_Effect', '?'):<15} {row.get('Total_Score', 0)}")

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\n--- Creating Visualization ---")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Workforce Program Effectiveness Analysis', fontsize=16, fontweight='bold')

# 1. Program presence in Desert vs Non-Desert
ax1 = axes[0, 0]
prog_means_desert = []
prog_means_nondesert = []
prog_names = []

for name, col in programs.items():
    if col in analysis.columns:
        desert_mean = analysis[analysis['true_desert'] == 1][col].mean()
        nondesert_mean = analysis[analysis['true_desert'] == 0][col].mean()
        if not np.isnan(desert_mean) and not np.isnan(nondesert_mean):
            prog_names.append(name.replace(' ', '\n'))
            prog_means_desert.append(desert_mean)
            prog_means_nondesert.append(nondesert_mean)

x = np.arange(len(prog_names))
width = 0.35
ax1.bar(x - width/2, prog_means_desert, width, label='Desert Counties', color='coral')
ax1.bar(x + width/2, prog_means_nondesert, width, label='Non-Desert', color='steelblue')
ax1.set_ylabel('Program Resources (per 100k)')
ax1.set_title('1. Program Presence: Desert vs Non-Desert')
ax1.set_xticks(x)
ax1.set_xticklabels(prog_names, fontsize=9)
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Program Effect on PQI
ax2 = axes[0, 1]
pqi_effects = []
pqi_pvals = []
pqi_names = []

for name, col in programs.items():
    if col in analysis.columns:
        reg_data = analysis.dropna(subset=[col, 'pqi_mean_rate'])
        if len(reg_data) > 20:
            X = sm.add_constant(reg_data[[col]])
            Y = reg_data['pqi_mean_rate']
            m = OLS(Y, X).fit()
            pqi_names.append(name.replace(' ', '\n'))
            pqi_effects.append(m.params[col])
            pqi_pvals.append(m.pvalues[col])

colors = ['green' if e < 0 else 'red' for e in pqi_effects]
bars = ax2.bar(pqi_names, pqi_effects, color=colors, alpha=0.7)
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax2.set_ylabel('Effect on PQI Rate')
ax2.set_title('2. Program Effect on PQI (Negative = Better)')
ax2.grid(True, alpha=0.3)

# Add significance stars
for i, (bar, pval) in enumerate(zip(bars, pqi_pvals)):
    if pval < 0.01:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), '***', ha='center', va='bottom')
    elif pval < 0.05:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), '**', ha='center', va='bottom')
    elif pval < 0.10:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), '*', ha='center', va='bottom')

# 3. FQHC vs PQI scatter
ax3 = axes[1, 0]
scatter_data = analysis.dropna(subset=['fqhc_per_100k', 'pqi_mean_rate'])
colors_scatter = ['red' if d == 1 else 'blue' for d in scatter_data['true_desert']]
ax3.scatter(scatter_data['fqhc_per_100k'], scatter_data['pqi_mean_rate'], 
            c=colors_scatter, alpha=0.6, s=60)
ax3.set_xlabel('FQHCs per 100k Population')
ax3.set_ylabel('PQI Rate')
ax3.set_title('3. FQHCs vs PQI (Red = Desert)')

# Add trend line
z = np.polyfit(scatter_data['fqhc_per_100k'], scatter_data['pqi_mean_rate'], 1)
p = np.poly1d(z)
ax3.plot(scatter_data['fqhc_per_100k'].sort_values(), 
         p(scatter_data['fqhc_per_100k'].sort_values()), 
         'g--', linewidth=2, label=f'Trend: slope={z[0]:.1f}')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Summary text
ax4 = axes[1, 1]
ax4.axis('off')

# Find best program
if len(summary_df) > 0:
    best_prog = summary_df.iloc[0]['Program']
    best_score = summary_df.iloc[0]['Total_Score']
else:
    best_prog = "Unknown"
    best_score = 0

summary_text = f"""
WORKFORCE PROGRAM EFFECTIVENESS SUMMARY
═══════════════════════════════════════════════════════════════

KEY FINDINGS:

1. REDISTRIBUTION TO DESERTS:
   {"None of the programs significantly target desert counties" 
    if summary_df['Redist_Score'].sum() == 0 
    else f"Programs targeting deserts: {', '.join(summary_df[summary_df['Redist_Score']==1]['Program'].tolist())}"}

2. PQI REDUCTION:
   {"None of the programs significantly reduce PQI" 
    if summary_df['PQI_Score'].sum() == 0 
    else f"Programs reducing PQI: {', '.join(summary_df[summary_df['PQI_Score']==1]['Program'].tolist())}"}

3. BEST OVERALL PROGRAM:
   {best_prog} (Score: {best_score}/2)

═══════════════════════════════════════════════════════════════

POLICY IMPLICATION:

{"Current workforce programs are NOT effectively targeting " +
 "desert counties OR reducing preventable hospitalizations. " +
 "Need to redesign program targeting and incentives."
 if best_score == 0 
 else f"{best_prog} shows promise and should be expanded in desert counties."}

═══════════════════════════════════════════════════════════════
"""

ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
ax4.set_title('4. Summary')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('outputs_policy/workforce_programs/program_effectiveness.png', dpi=150, bbox_inches='tight')
print("✓ Saved: outputs_policy/workforce_programs/program_effectiveness.png")

# Save results
summary_df.to_csv('outputs_policy/workforce_programs/program_ranking.csv', index=False)
print("✓ Saved: program_ranking.csv")

analysis.to_csv('outputs_policy/workforce_programs/county_program_data.csv', index=False)
print("✓ Saved: county_program_data.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
