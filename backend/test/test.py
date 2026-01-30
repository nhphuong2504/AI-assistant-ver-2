"""
Test script: run expected_remaining_lifetime and print results.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from app.db import run_query_internal
from analytics.survival import (
    build_covariate_table,
    fit_cox_baseline,
    expected_remaining_lifetime,
)

# Load data, build cov, fit Cox
print("Loading data...")
sql = """SELECT customer_id, invoice_no, invoice_date, revenue, stock_code, country
         FROM transactions WHERE customer_id IS NOT NULL"""
rows, _ = run_query_internal(sql, max_rows=2_000_000)
df = pd.DataFrame(rows)
cov = build_covariate_table(df, cutoff_date="2011-12-09", inactivity_days=90).df

cox = fit_cox_baseline(
    covariates=cov,
    covariate_cols=["n_orders", "log_monetary_value", "product_diversity"],
    train_frac=1.0,
)
model = cox["model"]
H_days = 365

# Compute expected remaining lifetime
print("\n" + "=" * 60)
print("EXPECTED REMAINING LIFETIME — RESULTS")
print("=" * 60)

erl = expected_remaining_lifetime(
    model=model,
    covariates_df=cov,
    H_days=H_days,
)

print(f"\nHorizon H_days: {H_days}")
print(f"Active customers with ERL: {len(erl)}")

# Beyond horizon: expected_remaining_life_days == -1 (VIP / low risk)
beyond = erl[erl["expected_remaining_life_days"] == -1.0]
in_range = erl[erl["expected_remaining_life_days"] >= 0]

print(f"  - Beyond model horizon (ERL = -1): {len(beyond)}")
print(f"  - In-range (ERL in [0, H]):        {len(in_range)}")

if len(in_range) > 0:
    print("\nSummary (in-range only):")
    print(in_range["expected_remaining_life_days"].describe().to_string())

print("\n" + "-" * 60)
print("Top 10 by expected_remaining_life_days (in-range):")
top = in_range.head(10)[["customer_id", "t0", "H_days", "expected_remaining_life_days"]]
print(top.to_string(index=False))

print("\n" + "-" * 60)
print("Bottom 10 by expected_remaining_life_days (in-range):")
bottom = in_range.tail(10)[["customer_id", "t0", "H_days", "expected_remaining_life_days"]]
print(bottom.to_string(index=False))

if len(beyond) > 0:
    print("\n" + "-" * 60)
    print("Beyond model horizon (ERL = -1, sample up to 15):")
    sample_beyond = beyond.head(15)[["customer_id", "t0", "H_days", "expected_remaining_life_days"]]
    print(sample_beyond.to_string(index=False))

print("\n" + "=" * 60)
print("DONE")
