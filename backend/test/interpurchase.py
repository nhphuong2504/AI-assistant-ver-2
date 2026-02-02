"""
Read transactions from SQLite, group by CustomerID, compute inter-purchase intervals
(days between consecutive InvoiceDates) for customers with ≥2 purchases,
then report the average of per-customer mean intervals.
"""
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.db import run_query_internal


def mean_interpurchase_days(group: pd.DataFrame) -> float | None:
    """Mean days between consecutive invoice dates for one customer."""
    dates = group["invoice_date"].drop_duplicates().sort_values().values
    if len(dates) < 2:
        return None
    diffs = pd.Series(dates).diff().dt.total_seconds() / 86400
    return float(diffs.dropna().mean())


def compute_avg_interpurchase_days() -> float:
    """Load transactions, compute inter-purchase intervals, return average of customer means."""
    sql = """
    SELECT customer_id, invoice_date
    FROM transactions
    WHERE customer_id IS NOT NULL
    """
    rows, _ = run_query_internal(sql)
    df = pd.DataFrame(rows)
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])

    # One row per (customer_id, invoice_date) - multiple line items per invoice collapse
    invoices = df.drop_duplicates(subset=["customer_id", "invoice_date"])

    # Per-customer mean inter-purchase interval (days)
    customer_means = (
        invoices.groupby("customer_id", group_keys=False)
        .apply(mean_interpurchase_days, include_groups=False)
        .dropna()
    )

    return float(customer_means.mean())


if __name__ == "__main__":
    avg = compute_avg_interpurchase_days()
    print(f"Average inter-purchase interval (days): {avg:.2f}")
