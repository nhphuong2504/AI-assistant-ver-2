"""
Test script for customer scoring and ranking using fitted Cox model.
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
    score_customers,
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

# Step 2: Score customers using the fitted model
print("\n" + "="*80)
print("SCORING CUSTOMERS (LEAKAGE-FREE)")
print("="*80)

# Score all customers at cutoff date
scored_customers = score_customers(
    model=cox_model,
    transactions=df,
    cutoff_date="2011-12-09",
)

print(f"\nScored customers: {len(scored_customers)}")
print(f"\nRisk Score Summary:")
print(scored_customers['risk_score'].describe())

print(f"\nRisk Bucket Distribution:")
print(scored_customers['risk_bucket'].value_counts().sort_index())

# Display top 20 highest risk customers
print("\n" + "="*80)
print("TOP 20 HIGHEST RISK CUSTOMERS")
print("="*80)
top_risk = scored_customers.head(20)
print(top_risk[['customer_id', 'n_orders', 'log_monetary_value', 'product_diversity', 
                'risk_score', 'risk_rank', 'risk_percentile', 'risk_bucket']].to_string(index=False))

# Display sample from each risk bucket
print("\n" + "="*80)
print("SAMPLE CUSTOMERS BY RISK BUCKET")
print("="*80)

for bucket in ['High', 'Medium', 'Low']:
    bucket_customers = scored_customers[scored_customers['risk_bucket'] == bucket]
    if len(bucket_customers) > 0:
        sample = bucket_customers.sample(min(5, len(bucket_customers)), random_state=42)
        print(f"\n{bucket} Risk (sample of {len(sample)}):")
        print(sample[['customer_id', 'n_orders', 'log_monetary_value', 'product_diversity',
                     'risk_score', 'risk_percentile']].to_string(index=False))

print("\n" + "="*80)
print("NOTE")
print("="*80)
print("Risk scores represent relative churn risk and are intended for prioritization,")
print("not probability estimation.")

