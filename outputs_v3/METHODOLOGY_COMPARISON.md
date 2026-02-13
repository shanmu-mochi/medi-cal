
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
- Sample: N = 714 county-years (56 counties, 2008-2024)
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

Previous (Facility-Based ED, Cross-Section, N=55):
 Model  Outcome  Coefficient  p_value
     1 PQI Rate    -0.390611 0.103941
     2 PQI Rate    34.394849 0.293483
     3  ED Rate     0.158483 0.034176
     4  ED Rate   -15.953970 0.031654

Corrected (Residence-Based ED, Panel, N=714):
 Model                     Outcome    Coefficient  p_value
     1                    PQI Rate      -0.507455 0.022997
     2                    PQI Rate      20.972932 0.438943
     3 ED Visits (Residence-Based)    1410.833426 0.104541
     4 ED Visits (Residence-Based) -180203.145712 0.054462


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
