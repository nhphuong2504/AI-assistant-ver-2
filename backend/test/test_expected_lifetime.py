"""
Test script for expected remaining lifetime computation using fitted Cox model.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path to import analytics module
sys.path.insert(0, str(Path(__file__).parent.parent))

from analytics.survival import (
    build_covariate_table,
    fit_cox_baseline,
    expected_remaining_lifetime,
)
from app.db import run_query_internal

# Load transactions from SQLite
print("Loading data...")
sql = """
SELECT customer_id, invoice_no, invoice_date, revenue, stock_code, country
FROM transactions
WHERE customer_id IS NOT NULL
"""
rows, _ = run_query_internal(sql, max_rows=2_000_000)
df = pd.DataFrame(rows)

print(f"Total transactions: {len(df)}")

# Step 1: Fit the final Cox model with specified covariates
print("\n" + "="*80)
print("FITTING FINAL COX MODEL")
print("="*80)

# Build covariate table
cov_table = build_covariate_table(
    transactions=df,
    cutoff_date="2011-12-09",
    inactivity_days=90,
)
cov = cov_table.df

print(f"Total customers: {len(cov)}")
print(f"Churn rate: {cov['event'].mean():.3f}")
print(f"Active customers (event=0): {(cov['event'] == 0).sum()}")
print(f"Churned customers (event=1): {(cov['event'] == 1).sum()}")

# Fit model with final covariates: n_orders, log_monetary_value, product_diversity
print("\nFitting Cox model with covariates: n_orders, log_monetary_value, product_diversity")
result = fit_cox_baseline(
    covariates=cov,
    covariate_cols=['n_orders', 'log_monetary_value', 'product_diversity'],
    train_frac=0.8,
    random_state=42,
    penalizer=0.1,
)

print(f"Training set: {result['n_train']} customers")
print(f"Validation set: {result['n_validation']} customers")

# Get the fitted model
cox_model = result['model']

print("\nModel Summary:")
print(result['summary'][['covariate', 'coef', 'hazard_ratio', 'p']].to_string(index=False))

# Step 2: Compute expected remaining lifetime for active customers
print("\n" + "="*80)
print("COMPUTING EXPECTED REMAINING LIFETIME")
print("="*80)

H_days = 365
print(f"\nHorizon: {H_days} days")

# Compute expected remaining lifetime
expected_lifetime = expected_remaining_lifetime(
    model=cox_model,
    covariates_df=cov,
    H_days=H_days,
)

print(f"\nActive customers with expected lifetime: {len(expected_lifetime)}")

# Verify output columns
expected_cols = ['customer_id', 't0', 'H_days', 'expected_remaining_life_days']
assert all(col in expected_lifetime.columns for col in expected_cols), \
    f"Missing expected columns. Got: {list(expected_lifetime.columns)}"

# Verify H_days is correct
assert (expected_lifetime['H_days'] == H_days).all(), f"All H_days should be {H_days}"

# Verify t0 > 0
assert (expected_lifetime['t0'] > 0).all(), "All t0 (duration/tenure_days) should be > 0"

# Validation: 0 ≤ expected_remaining_life_days ≤ H_days
assert (expected_lifetime['expected_remaining_life_days'] >= 0).all(), \
    "Expected remaining lifetime must be >= 0"
assert (expected_lifetime['expected_remaining_life_days'] <= H_days).all(), \
    f"Expected remaining lifetime must be <= {H_days}"

print(f"\nExpected Remaining Lifetime Summary (horizon: {H_days} days):")
print(expected_lifetime['expected_remaining_life_days'].describe())

# Get risk scores for comparison
from analytics.survival import score_customers
scored = score_customers(
    model=cox_model,
    transactions=df,
    cutoff_date="2011-12-09",
)

# Merge expected lifetime with risk scores and covariates
merged = expected_lifetime.merge(
    scored[['customer_id', 'risk_score']],
    on='customer_id',
    how='inner'
).merge(
    cov[['customer_id', 'n_orders', 'product_diversity']],
    on='customer_id',
    how='inner'
)

# Display sample rows with requested columns
print("\n" + "="*80)
print("SAMPLE CUSTOMERS (customer_id, risk_score, expected_remaining_life_days, n_orders, product_diversity, t0)")
print("="*80)
sample_size = min(20, len(merged))
sample = merged.sample(n=sample_size, random_state=42)
display_cols = ['customer_id', 'risk_score', 'expected_remaining_life_days', 'n_orders', 'product_diversity', 't0']
print(sample[display_cols].to_string(index=False))

# Display specific customers
print("\n" + "="*80)
print("SPECIFIC CUSTOMERS (customer_id, risk_score, expected_remaining_life_days, n_orders, product_diversity, t0)")
print("="*80)
specific_customer_ids = [13860, 15822, 14646]
specific_customers = merged[merged['customer_id'].isin(specific_customer_ids)]
if len(specific_customers) > 0:
    print(specific_customers[display_cols].to_string(index=False))
else:
    print(f"None of the requested customers ({specific_customer_ids}) found in the merged data.")

print("\n" + "="*80)
print("TEST COMPLETED")
print("="*80)
print("\nThe expected_remaining_lifetime function correctly computes restricted")
print("expected remaining lifetime for active customers using the Cox model.")

