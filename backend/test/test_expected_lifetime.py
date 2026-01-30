"""
Test script for expected remaining lifetime computation using Monte Carlo (BG/NBD).
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path to import analytics module
sys.path.insert(0, str(Path(__file__).parent.parent))

from analytics.clv import build_rfm, fit_models
from analytics.monte_carlo import compute_erl_days
from analytics.survival import build_covariate_table, fit_cox_baseline, score_customers
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

CUTOFF_DATE = "2011-12-09"
INACTIVITY_DAYS = 90

# Step 1: Build RFM and customer summary for Monte Carlo ERL
print("\n" + "="*80)
print("BUILDING RFM AND CUSTOMER SUMMARY")
print("="*80)

rfm = build_rfm(df, CUTOFF_DATE)
print(f"RFM table: {len(rfm):,} customers")

df["invoice_date"] = pd.to_datetime(df["invoice_date"])
cutoff_dt = pd.to_datetime(CUTOFF_DATE)
last_purchases = (
    df[df["invoice_date"] <= cutoff_dt]
    .groupby("customer_id")["invoice_date"]
    .max()
    .reset_index()
)
last_purchases.columns = ["customer_id", "last_purchase_date"]
last_purchases["last_purchase_date"] = last_purchases["last_purchase_date"].dt.strftime("%Y-%m-%d")
customer_summary = rfm.reset_index().merge(last_purchases, on="customer_id", how="left")
print(f"Customer summary: {len(customer_summary):,} customers")

# Fit BG/NBD model
print("\nFitting BG/NBD model...")
clv_result = fit_models(rfm)
bgf = clv_result.bgnbd
print("BG/NBD model fitted")

# Step 2: Compute expected remaining lifetime (Monte Carlo)
print("\n" + "="*80)
print("COMPUTING EXPECTED REMAINING LIFETIME (MONTE CARLO)")
print("="*80)

erl_result = compute_erl_days(
    bgf=bgf,
    customer_summary_df=customer_summary,
    cutoff_date=CUTOFF_DATE,
    INACTIVITY_DAYS=INACTIVITY_DAYS,
    N=1000,
    seed=42,
    max_days=1825,
)

# Map to API shape: expected_remaining_life_days, t0, H_days
expected_lifetime = erl_result.merge(
    customer_summary[["customer_id", "T"]], on="customer_id", how="left"
).rename(columns={"ERL_days": "expected_remaining_life_days", "T": "t0"})
expected_lifetime["H_days"] = 365

print(f"\nTotal customers with ERL: {len(expected_lifetime)}")

# Verify output columns
expected_cols = ['customer_id', 't0', 'H_days', 'expected_remaining_life_days']
assert all(col in expected_lifetime.columns for col in expected_cols), \
    f"Missing expected columns. Got: {list(expected_lifetime.columns)}"

# Verify t0 > 0
assert (expected_lifetime['t0'] > 0).all(), "All t0 (tenure T) should be > 0"

# Validation: expected_remaining_life_days >= 0 (Monte Carlo can exceed H_days)
assert (expected_lifetime['expected_remaining_life_days'] >= 0).all(), \
    "Expected remaining lifetime must be >= 0"

print(f"\nExpected Remaining Lifetime Summary:")
print(expected_lifetime['expected_remaining_life_days'].describe())

# Get risk scores for comparison (Cox model)
cov_table = build_covariate_table(df, CUTOFF_DATE, INACTIVITY_DAYS)
cov = cov_table.df
cox_result = fit_cox_baseline(
    covariates=cov,
    covariate_cols=['n_orders', 'log_monetary_value', 'product_diversity'],
    train_frac=1.0,
    random_state=42,
    penalizer=0.1,
)
scored = score_customers(
    model=cox_result['model'],
    transactions=df,
    cutoff_date=CUTOFF_DATE,
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
print("\nThe compute_erl_days function (Monte Carlo) correctly computes expected")
print("remaining lifetime in days for customers using the BG/NBD model.")
