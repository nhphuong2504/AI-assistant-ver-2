"""
Print results of build_segmentation_table using the project dataset.
"""
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from analytics.survival import (
    build_covariate_table,
    fit_cox_baseline,
    build_segmentation_table,
    CUTOFF_DATE,
    INACTIVITY_DAYS,
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
print(f"Total transactions: {len(df):,}")

# Build covariate table and fit Cox model (for risk labels)
print("\nBuilding covariate table and fitting Cox model...")
cov_table = build_covariate_table(df, CUTOFF_DATE, INACTIVITY_DAYS)
cov = cov_table.df
cox_result = fit_cox_baseline(
    covariates=cov,
    covariate_cols=["n_orders", "log_monetary_value", "product_diversity"],
    train_frac=1.0,
    random_state=42,
    penalizer=0.1,
)
model = cox_result["model"]

# Build segmentation table (active customers only; ERL from Monte Carlo)
print("Building segmentation table...")
segmentation_df, cutoffs = build_segmentation_table(
    model=model,
    transactions=df,
    covariates_df=cov,
    cutoff_date=CUTOFF_DATE,
)

# Print cutoffs (ERL bucket boundaries in days)
print("\n" + "=" * 60)
print("CUTOFFS (ERL bucket boundaries)")
print("=" * 60)
for k, v in cutoffs.items():
    print(f"  {k}: {v}")

# Print segment counts
print("\n" + "=" * 60)
print("SEGMENT COUNTS")
print("=" * 60)
print(segmentation_df["segment"].value_counts().sort_index().to_string())

# Print risk_label and life_bucket counts
print("\n" + "=" * 60)
print("RISK LABEL COUNTS")
print("=" * 60)
print(segmentation_df["risk_label"].value_counts().to_string())
print("\nLIFE BUCKET COUNTS")
print("=" * 60)
print(segmentation_df["life_bucket"].value_counts().to_string())

# Print summary stats
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Total active customers in segmentation: {len(segmentation_df):,}")
print(f"  erl_days mean: {segmentation_df['erl_days'].mean():.2f}")
print(f"  erl_days median: {segmentation_df['erl_days'].median():.2f}")

# Sample of results
print("\n" + "=" * 60)
print("SAMPLE ROWS (first 15)")
print("=" * 60)
cols = ["customer_id", "risk_label", "erl_days", "life_bucket", "segment", "action_tag"]
print(segmentation_df[cols].head(15).to_string(index=False))

# Look up specific customer_id(s) to see which segment they are in
CUSTOMER_IDS_TO_LOOKUP = [16000]  # Add or change IDs to test

print("\n" + "=" * 60)
print("LOOKUP: SEGMENT BY CUSTOMER_ID")
print("=" * 60)
display_cols = [
    "customer_id",
    "risk_label",
    "erl_days",
    "life_bucket",
    "segment",
    "action_tag",
    "recommended_action",
]
for cid in CUSTOMER_IDS_TO_LOOKUP:
    match = segmentation_df[segmentation_df["customer_id"] == cid]
    if len(match) > 0:
        row = match.iloc[0]
        print(f"\n  customer_id: {cid}")
        print(f"    risk_label:      {row['risk_label']}")
        print(f"    erl_days:        {row['erl_days']:.2f}")
        print(f"    life_bucket:     {row['life_bucket']}")
        print(f"    segment:         {row['segment']}")
        print(f"    action_tag:      {row['action_tag']}")
        print(f"    recommended:     {row['recommended_action']}")
    else:
        print(f"\n  customer_id: {cid}  ->  NOT FOUND (not in active segmentation; may be churned or not in data)")

# Optional: print all lookups as a table when at least one is found
found = segmentation_df[segmentation_df["customer_id"].isin(CUSTOMER_IDS_TO_LOOKUP)]
if len(found) > 0:
    print("\n  Table of found customers:")
    print(found[display_cols].to_string(index=False))
else:
    print("\n  No requested customer_ids were found in the segmentation table.")

print("\nDone.")
