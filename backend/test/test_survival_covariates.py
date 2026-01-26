import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path to import analytics module
sys.path.insert(0, str(Path(__file__).parent.parent))

from analytics.survival import build_covariate_table
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
cov = build_covariate_table(
    transactions=df,
    cutoff_date="2011-12-09",
    inactivity_days=90,
).df

# --------------------
# SANITY CHECK OUTPUT
# --------------------
print("\n=== BASIC CHECKS ===")
print("Customers:", len(cov))
print("Churn rate:", round(cov["event"].mean(), 3))

print("\n=== TOP FREQUENCY CUSTOMERS ===")
print(
    cov.sort_values("n_orders", ascending=False)[
        ["customer_id", "n_orders", "tenure_days"]
    ].head(5)
)

print("\n=== FEATURE SUMMARY ===")
print(
    cov[
        [
            "n_orders",
            "recency_from_cutoff",
            "tenure_days",
            "frequency_rate",
            "product_diversity",
            "monetary_value",
        ]
    ].describe()
)

print("\n=== RANDOM 20 ROWS ===")
print(cov.sample(n=min(20, len(cov))))