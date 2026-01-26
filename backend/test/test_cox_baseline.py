"""
Test script for Cox proportional hazards baseline model.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path to import analytics module
sys.path.insert(0, str(Path(__file__).parent.parent))

from analytics.survival import build_covariate_table, fit_cox_baseline, validate_cox_model
from app.db import run_query_internal

# Load transactions from SQLite
sql = """
SELECT customer_id, invoice_no, invoice_date, revenue, stock_code, country
FROM transactions
WHERE customer_id IS NOT NULL
"""
rows, _ = run_query_internal(sql, max_rows=2_000_000)
df = pd.DataFrame(rows)

# Build covariate table
cov_table = build_covariate_table(
    transactions=df,
    cutoff_date="2011-12-09",
    inactivity_days=90,
)
cov = cov_table.df

print("\n" + "="*80)
print("COX PROPORTIONAL HAZARDS BASELINE MODEL")
print("="*80)

# Fit Cox model
result = fit_cox_baseline(
    covariates=cov,
    covariate_cols=['n_orders', 'log_product_diversity'],
    train_frac=0.8,
    random_state=42,
    penalizer=0.1,
)

print(f"\n=== DATA PREPARATION ===")
print(f"Original customers: {len(cov)}")
print(f"After dropping missing: {result['n_train'] + result['n_validation']}")
print(f"Dropped due to missing values: {result['n_dropped']}")
print(f"Training set: {result['n_train']} customers")
print(f"Validation set: {result['n_validation']} customers")

print(f"\n=== MODEL SUMMARY ===")
summary = result['summary']
print(summary[['covariate', 'coef', 'hazard_ratio', 'se(coef)', 'p']].to_string(index=False))

print(f"\n=== COEFFICIENT INTERPRETATION ===")
for cov, interp in result['interpretation'].items():
    print(f"\n{cov}:")
    print(f"  Sign: {interp['sign']}")
    print(f"  Meaning: {interp['meaning']}")
    print(f"  Hazard Ratio: {interp['hazard_ratio']:.4f}")
    print(f"  Effect: {interp['effect']}")

print(f"\n=== FLAGS & WARNINGS ===")

if result['flags']['unexpected_signs']:
    print("\n⚠️  UNEXPECTED COEFFICIENT SIGNS:")
    for flag in result['flags']['unexpected_signs']:
        print(f"  - {flag['covariate']}: Expected {flag['expected']}, got {flag['actual']} (coef={flag['coef']:.4f})")
else:
    print("✓ No unexpected coefficient signs")

if result['flags']['large_se']:
    print("\n⚠️  LARGE STANDARD ERRORS:")
    for flag in result['flags']['large_se']:
        print(f"  - {flag['covariate']}: SE={flag['se']:.4f}, Coef={flag['coef']:.4f}, Ratio={flag['ratio']:.2f}")
else:
    print("✓ No large standard errors")

if result['flags']['non_significant']:
    print("\n⚠️  NON-SIGNIFICANT VARIABLES (p > 0.05):")
    for flag in result['flags']['non_significant']:
        print(f"  - {flag['covariate']}: p={flag['p']:.4f}, coef={flag['coef']:.4f}")
else:
    print("✓ All variables are significant (p <= 0.05)")

print("\n" + "="*80)
print("MODEL TRUSTWORTHINESS ASSESSMENT")
print("="*80)

# Overall assessment
n_warnings = (
    len(result['flags']['unexpected_signs']) +
    len(result['flags']['large_se']) +
    len(result['flags']['non_significant'])
)

if n_warnings == 0:
    print("✓ Model appears trustworthy - no major issues detected")
elif n_warnings <= 2:
    print("⚠️  Model has some warnings - review flags above")
else:
    print("⚠️  Model has multiple warnings - investigate before use")

print(f"\nTotal warnings: {n_warnings}")

print("\n" + "="*80)
print("MODEL VALIDATION")
print("="*80)

# Validate the model
validation_result = validate_cox_model(
    model=result['model'],
    train_df=result['train_df'],
    validation_df=result['validation_df'],
)

print(f"\n=== PROPORTIONAL HAZARDS ASSUMPTION TESTS ===")
ph_tests = validation_result['ph_tests']

# Display all columns including notes if present
display_cols = ['covariate', 'test_statistic', 'p_value', 'violates_ph']
if 'note' in ph_tests.columns:
    display_cols.append('note')

print(ph_tests[display_cols].to_string(index=False))

interp = validation_result['interpretation']

if interp['ph_test_failed']:
    print(f"\n⚠️  PH TEST COMPUTATION FAILED")
    if interp['ph_test_error']:
        print(f"  Error: {interp['ph_test_error']}")
    print("  PH assumptions cannot be evaluated - results are not trustworthy")
elif not interp['ph_evaluable']:
    print(f"\n⚠️  PH TESTS NOT EVALUABLE")
    print("  PH assumptions cannot be evaluated")
elif validation_result['ph_violations']:
    print(f"\n⚠️  PH ASSUMPTION VIOLATIONS DETECTED:")
    for cov in validation_result['ph_violations']:
        print(f"  - {cov} (p < 0.05)")
    print("  Model violates proportional hazards assumption")
else:
    print("\n✓ PH assumption holds: All covariates have p >= 0.05")

print(f"\n=== PREDICTIVE PERFORMANCE (VALIDATION SET) ===")
c_index = validation_result['c_index']
if np.isnan(c_index):
    print(f"C-index: NaN (computation failed)")
else:
    print(f"C-index: {c_index:.4f}")

print(f"\n=== VALIDATION INTERPRETATION ===")
print(f"PH assumption holds: {'Yes' if interp['ph_assumption_holds'] else 'No'}")
if not interp['ph_evaluable']:
    print(f"PH assumption evaluable: No (tests failed or not computable)")
else:
    print(f"PH assumption evaluable: Yes")

print(f"\nPredictive performance:")
print(f"  Acceptable: {'Yes' if interp['acceptable_performance'] else 'No'}")
print(f"  Interpretation: {interp['c_index_interpretation']}")

if interp['c_index_failed']:
    print(f"  ⚠️  C-index computation failed")

print("\n" + "="*80)

