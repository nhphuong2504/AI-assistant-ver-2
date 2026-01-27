"""
Test for expected_remaining_lifetime: compute for all active customers and sort ascending.
Also: ask LLM agent to find expected remaining life of customer 16446.

Requires: database at backend/data/retail.sqlite (run etl/load_online_retail.py),
         and deps from requirements.txt (lifelines, pandas, etc.).
         LLM test requires OPENAI_API_KEY and network.

Run from backend: python test/test.py
Or from project root: python backend/test/test.py
"""
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analytics.survival import (
    build_covariate_table,
    fit_cox_baseline,
    expected_remaining_lifetime,
)
from app.db import run_query_internal
from app.llm_langchain import ask_question


def _load_transactions():
    sql = """
    SELECT customer_id, invoice_no, invoice_date, revenue, stock_code, country
    FROM transactions
    WHERE customer_id IS NOT NULL
    """
    rows, _ = run_query_internal(sql, max_rows=2_000_000)
    return pd.DataFrame(rows)


def _get_model_and_cov():
    df = _load_transactions()
    cov_table = build_covariate_table(
        transactions=df,
        cutoff_date="2011-12-09",
        inactivity_days=90,
    )
    cov = cov_table.df
    result = fit_cox_baseline(
        covariates=cov,
        covariate_cols=["n_orders", "log_monetary_value", "product_diversity"],
        train_frac=1.0,
        random_state=42,
        penalizer=0.1,
    )
    return result["model"], cov


if __name__ == "__main__":
    print("Loading data and fitting Cox model...")
    model, cov = _get_model_and_cov()

    H_days = 365
    print(f"Computing expected_remaining_lifetime for all active customers (H_days={H_days})...")

    out = expected_remaining_lifetime(
        model=model,
        covariates_df=cov,
        H_days=H_days,
    )

    # Sort ascending by expected_remaining_life_days
    sorted_asc = out.sort_values("expected_remaining_life_days", ascending=True).reset_index(drop=True)

    print(f"\nTotal active customers: {len(sorted_asc)}")
    print(f"\nExpected remaining life (days) - summary:")
    print(sorted_asc["expected_remaining_life_days"].describe())

    print("\n" + "=" * 80)
    print("FIRST 20 CUSTOMERS, SORTED ASCENDING by expected_remaining_life_days")
    print("=" * 80)
    print(sorted_asc[["customer_id", "t0", "H_days", "expected_remaining_life_days"]].to_string(index=False))

    print("\n" + "=" * 80)
    print("LLM: expected remaining life of customer 16446")
    print("=" * 80)
    answer = ask_question("What is the expected remaining life of customer 16446?", use_memory=False)
    print(answer)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
