"""
Test script for conditional churn probability prediction using fitted Cox model.
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
    predict_churn_probability,
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

# Step 2: Predict churn probabilities for active customers
print("\n" + "="*80)
print("PREDICTING CHURN PROBABILITIES (CONDITIONAL)")
print("="*80)

# Test with different prediction horizons
for X_days in [30, 60, 90, 180]:
    print(f"\n{'='*80}")
    print(f"Prediction horizon: {X_days} days")
    print(f"{'='*80}")
    
    predictions = predict_churn_probability(
        model=cox_model,
        transactions=df,
        cutoff_date="2011-12-09",
        X_days=X_days,
        inactivity_days=90,
    )
    
    print(f"\nActive customers with predictions: {len(predictions)}")
    
    # Verify output columns
    expected_cols = ['customer_id', 't0', 'X_days', 'churn_probability', 
                     'survival_at_t0', 'survival_at_t0_plus_X']
    assert all(col in predictions.columns for col in expected_cols), \
        f"Missing expected columns. Got: {list(predictions.columns)}"
    
    # Verify probability ranges
    assert (predictions['churn_probability'] >= 0).all(), "Churn probabilities must be >= 0"
    assert (predictions['churn_probability'] <= 1).all(), "Churn probabilities must be <= 1"
    
    # Verify survival probabilities
    assert (predictions['survival_at_t0'] >= 0).all(), "Survival at t0 must be >= 0"
    assert (predictions['survival_at_t0'] <= 1).all(), "Survival at t0 must be <= 1"
    assert (predictions['survival_at_t0_plus_X'] >= 0).all(), "Survival at t0+X must be >= 0"
    assert (predictions['survival_at_t0_plus_X'] <= 1).all(), "Survival at t0+X must be <= 1"
    
    # Verify survival_at_t0 >= survival_at_t0_plus_X (survival decreases over time)
    assert (predictions['survival_at_t0'] >= predictions['survival_at_t0_plus_X']).all(), \
        "Survival at t0 should be >= survival at t0+X"
    
    # Verify X_days is correct
    assert (predictions['X_days'] == X_days).all(), f"All X_days should be {X_days}"
    
    # Verify t0 > 0
    assert (predictions['t0'] > 0).all(), "All t0 (tenure_days) should be > 0"
    
    print(f"\nChurn Probability Summary (next {X_days} days):")
    print(predictions['churn_probability'].describe())
    
    print(f"\nSurvival at t0 Summary:")
    print(predictions['survival_at_t0'].describe())
    
    print(f"\nSurvival at t0+X Summary:")
    print(predictions['survival_at_t0_plus_X'].describe())
    
    # Display randomly selected customers
    print(f"\n{'='*80}")
    print(f"RANDOM SAMPLE OF CUSTOMERS (next {X_days} days)")
    print(f"{'='*80}")
    sample_size = min(20, len(predictions))
    random_sample = predictions.sample(n=sample_size, random_state=42)
    print(random_sample[['customer_id', 't0', 'X_days', 'churn_probability', 
                    'survival_at_t0', 'survival_at_t0_plus_X']].to_string(index=False))
    
    # Verify conditional probability formula
    print(f"\n{'='*80}")
    print("VERIFYING CONDITIONAL PROBABILITY FORMULA")
    print(f"{'='*80}")
    sample = predictions.sample(n=min(10, len(predictions)), random_state=42)
    for _, row in sample.iterrows():
        computed = row['churn_probability']
        s_t0 = row['survival_at_t0']
        s_t1 = row['survival_at_t0_plus_X']
        expected = 1.0 - (s_t1 / s_t0) if s_t0 > 0 else 1.0
        diff = abs(computed - expected)
        assert diff < 1e-6, f"Formula mismatch for customer {row['customer_id']}: computed={computed}, expected={expected}"
        print(f"Customer {row['customer_id']}: P(churn) = {computed:.4f}, "
              f"S(t0) = {s_t0:.4f}, S(t0+X) = {s_t1:.4f}, "
              f"1 - S(t0+X)/S(t0) = {expected:.4f} ✓")

# Test edge cases
print("\n" + "="*80)
print("TESTING EDGE CASES")
print("="*80)

# Test with different cutoff date
print("\nTesting with different cutoff date (2011-11-09):")
predictions_early = predict_churn_probability(
    model=cox_model,
    transactions=df,
    cutoff_date="2011-11-09",
    X_days=90,
    inactivity_days=90,
)
print(f"Active customers at earlier cutoff: {len(predictions_early)}")

# Test with very short horizon
print("\nTesting with very short horizon (7 days):")
predictions_short = predict_churn_probability(
    model=cox_model,
    transactions=df,
    cutoff_date="2011-12-09",
    X_days=7,
    inactivity_days=90,
)
print(f"Active customers: {len(predictions_short)}")
print(f"Mean churn probability (7 days): {predictions_short['churn_probability'].mean():.4f}")

# Test with long horizon
print("\nTesting with long horizon (365 days):")
predictions_long = predict_churn_probability(
    model=cox_model,
    transactions=df,
    cutoff_date="2011-12-09",
    X_days=365,
    inactivity_days=90,
)
print(f"Active customers: {len(predictions_long)}")
print(f"Mean churn probability (365 days): {predictions_long['churn_probability'].mean():.4f}")

# Verify that longer horizons have higher probabilities
print("\nVerifying that longer horizons yield higher probabilities...")
sample_customers = predictions[predictions['customer_id'].isin(predictions_short['customer_id'].head(100))]
sample_customers_short = predictions_short[predictions_short['customer_id'].isin(sample_customers['customer_id'])]
sample_customers_long = predictions_long[predictions_long['customer_id'].isin(sample_customers['customer_id'])]

merged = sample_customers.merge(
    sample_customers_short[['customer_id', 'churn_probability']], 
    on='customer_id', 
    suffixes=('_90', '_7')
).merge(
    sample_customers_long[['customer_id', 'churn_probability']], 
    on='customer_id'
).rename(columns={'churn_probability': 'churn_probability_365'})

# Generally, longer horizons should have higher probabilities
assert (merged['churn_probability_365'] >= merged['churn_probability_90']).all(), \
    "365-day probabilities should be >= 90-day probabilities"
assert (merged['churn_probability_90'] >= merged['churn_probability_7']).all(), \
    "90-day probabilities should be >= 7-day probabilities"

print("✓ Longer horizons yield higher probabilities (as expected)")

print("\n" + "="*80)
print("ALL TESTS PASSED")
print("="*80)
print("\nThe predict_churn_probability function correctly computes conditional")
print("churn probabilities for active customers using the Cox model.")

