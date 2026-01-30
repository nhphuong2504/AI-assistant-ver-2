"""
Test script for prioritize_retention_targets.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from app.db import run_query_internal
from analytics.survival import (
    build_covariate_table,
    fit_cox_baseline,
    prioritize_retention_targets,
)
from analytics.clv import build_rfm, fit_models, predict_clv

# Load transactions
sql = """
SELECT customer_id, invoice_no, invoice_date, revenue, stock_code, country
FROM transactions
WHERE customer_id IS NOT NULL
"""
rows, _ = run_query_internal(sql, max_rows=2_000_000)
df = pd.DataFrame(rows)

# Cox model
cov_table = build_covariate_table(
    transactions=df,
    cutoff_date="2011-12-09",
    inactivity_days=90,
)
cov = cov_table.df
cox_result = fit_cox_baseline(
    covariates=cov,
    covariate_cols=["n_orders", "log_monetary_value", "product_diversity"],
    train_frac=1.0,
    random_state=42,
    penalizer=0.1,
)

# CLV
rfm = build_rfm(df, cutoff_date="2011-12-09")
clv_models = fit_models(rfm)
clv_df = predict_clv(clv_models, horizon_days=90, aov_fallback="global_mean")

# Run prioritize_retention_targets
result = prioritize_retention_targets(
    model=cox_result["model"],
    transactions=df,
    clv_df=clv_df,
    prediction_horizon=90,
)
print(result.head(20))
