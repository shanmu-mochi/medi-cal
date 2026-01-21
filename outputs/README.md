# California Medi-Cal Deserts Capstone Project

## How to Run

1. Ensure all required data files are in the same directory as this notebook:
   - county name.xlsx
   - medi-cal-enrollment-dashboard-data.csv
   - E4 estiamtes.xlsx
   - physicians-actively-working-by-specialty-and-patient-care-hours.xlsx
   - Primary CAre Shortage .csv
   - PQI.csv
   - demoACS.csv
   - educACS.csv
   - EconACS.csv

2. Optional files (will be loaded if present):
   - pqi-physicians-specialities-list.xlsx
   - Other physician activity files
   - County shapefile (for maps)

3. Run all cells in order (Cell > Run All)

4. Outputs will be saved to:
   - outputs/data/ - Cleaned datasets
   - outputs/figures/ - All figures
   - outputs/tables/ - Regression results and diagnostic tables

## Dependencies

Required:
- pandas
- numpy
- matplotlib
- openpyxl (for Excel files)

Optional but recommended:
- linearmodels (for PanelOLS)
- statsmodels (fallback if linearmodels not available)
- geopandas (for maps, if shapefile available)

## Project Structure

The notebook follows this structure:
0. Setup & Reproducibility
1. Load Raw Data
2. Build Clean County Crosswalk
3. Clean Population
4. Clean Medi-Cal Enrollment
5. Clean Physician Supply
6. Clean Shortage File
7. Clean ACS Controls
8. Clean PQI and Build Outcomes
9. Build Master Panel
10. Descriptive Figures
11. Regression Analyses
12. Robustness / Sensitivity
13. Appendix Diagnostics
14. Final Outputs

## Key Outputs

- **ca_master_panel.csv**: Final county-year panel dataset
- **results_summary.md**: Comprehensive results summary
- **regression_summary.md**: Regression results interpretation
- All figures and tables in outputs/ subdirectories

## Notes

- The notebook automatically detects whether physician supply and shortage designations are time-varying
- Models adapt accordingly (panel vs cross-sectional)
- Standard errors are clustered at the county level
- Desert definitions use both quartile-based and threshold-based approaches
