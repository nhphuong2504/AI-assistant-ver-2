import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path to import analytics module
sys.path.insert(0, str(Path(__file__).parent.parent))

from analytics.monte_carlo import compute_erl_days
from analytics.clv import build_rfm, fit_models
from app.db import run_query_internal


def test_monte_carlo_basic():
    """Test compute_erl_days with basic synthetic data."""
    from lifetimes import BetaGeoFitter
    
    # Create synthetic customer data
    customer_summary = pd.DataFrame({
        "customer_id": ["C1", "C2", "C3", "C4", "C5"],
        "frequency": [0.0, 1.0, 2.0, 3.0, 5.0],
        "recency": [0.0, 10.0, 30.0, 60.0, 90.0],
        "T": [30.0, 40.0, 60.0, 90.0, 120.0],
        "last_purchase_date": ["2023-02-15", "2023-02-25", "2023-03-15", "2023-04-15", "2023-05-15"]
    })
    
    # Fit a simple BG/NBD model
    bgf = BetaGeoFitter(penalizer_coef=0.05)
    bgf.fit(customer_summary["frequency"], customer_summary["recency"], customer_summary["T"])
    
    # Test compute_erl_days
    cutoff_date = "2023-06-01"
    INACTIVITY_DAYS = 90
    
    result = compute_erl_days(
        bgf=bgf,
        customer_summary_df=customer_summary,
        cutoff_date=cutoff_date,
        INACTIVITY_DAYS=INACTIVITY_DAYS,
        N=100,  # Smaller N for faster testing
        seed=42
    )
    
    # Check structure
    assert isinstance(result, pd.DataFrame)
    assert "customer_id" in result.columns
    assert "ERL_days" in result.columns
    assert "prob_alive" in result.columns
    assert len(result) == len(customer_summary)
    
    # Check that ERL_days is non-negative
    assert (result["ERL_days"] >= 0).all()
    
    print("✓ test_monte_carlo_basic passed")
    print(f"  Result shape: {result.shape}")
    print(f"  ERL_days range: {result['ERL_days'].min():.2f} - {result['ERL_days'].max():.2f}")
    print(f"  Mean ERL_days: {result['ERL_days'].mean():.2f}")
    print("\nSample results:")
    print(result.head())


def test_monte_carlo_real_data():
    """Test compute_erl_days with real data from database."""
    print("\n" + "="*60)
    print("TESTING MONTE CARLO ON REAL DATA")
    print("="*60)
    
    # Load transactions from SQLite
    sql = """
    SELECT customer_id, invoice_no, invoice_date, revenue
    FROM transactions
    WHERE customer_id IS NOT NULL
    """
    rows, _ = run_query_internal(sql, max_rows=2_000_000)
    df = pd.DataFrame(rows)
    
    print(f"\nLoaded {len(df):,} transactions")
    print(f"Date range: {df['invoice_date'].min()} to {df['invoice_date'].max()}")
    
    # Use a reasonable cutoff date
    cutoff_date = "2011-12-09"
    INACTIVITY_DAYS = 90
    
    # Build RFM table
    print(f"\n--- Building RFM table with cutoff_date={cutoff_date} ---")
    rfm = build_rfm(df, cutoff_date)
    print(f"✓ RFM table created: {len(rfm):,} customers")
    
    # Fit BG/NBD model
    print(f"\n--- Fitting BG/NBD model ---")
    clv_result = fit_models(rfm)
    bgf = clv_result.bgnbd
    print(f"✓ BG/NBD model fitted")
    print(f"  Model parameters: {list(bgf.params_.keys())}")
    
    # Prepare customer summary with last_purchase_date
    # We need to get last_purchase_date from the original transactions
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    cutoff_dt = pd.to_datetime(cutoff_date)
    
    # Get last purchase date per customer (before cutoff)
    last_purchases = (
        df[df["invoice_date"] <= cutoff_dt]
        .groupby("customer_id")["invoice_date"]
        .max()
        .reset_index()
    )
    last_purchases.columns = ["customer_id", "last_purchase_date"]
    
    # Merge with RFM data
    customer_summary = rfm.reset_index().merge(last_purchases, on="customer_id", how="left")
    
    # Convert last_purchase_date to string format
    customer_summary["last_purchase_date"] = customer_summary["last_purchase_date"].dt.strftime("%Y-%m-%d")
    
    print(f"\n--- Customer summary prepared: {len(customer_summary):,} customers ---")
    print(f"  Customers with last_purchase_date: {customer_summary['last_purchase_date'].notna().sum():,}")
    
    # Run Monte Carlo simulation
    print(f"\n--- Running Monte Carlo simulation (INACTIVITY_DAYS={INACTIVITY_DAYS}) ---")
    result = compute_erl_days(
        bgf=bgf,
        customer_summary_df=customer_summary,
        cutoff_date=cutoff_date,
        INACTIVITY_DAYS=INACTIVITY_DAYS,
        N=1000,  # Number of simulations
        seed=42,
        max_days=1825  # 5 years max
    )
    
    print(f"✓ Monte Carlo simulation completed")
    print(f"\n--- Results Summary ---")
    print(f"Total customers: {len(result):,}")
    print(f"Customers with ERL_days > 0: {(result['ERL_days'] > 0).sum():,}")
    print(f"Customers already churned (ERL_days = 0): {(result['ERL_days'] == 0).sum():,}")
    
    print(f"\nERL_days Statistics:")
    print(result["ERL_days"].describe())
    
    print(f"\nProb_alive Statistics:")
    print(result["prob_alive"].describe())
    
    # Active customers only (not already churned by inactivity rule)
    active_customers = result[result["ERL_days"] > 0].copy()
    print(f"\n--- Active Customers Statistics (ERL_days > 0) ---")
    print(f"Active customers: {len(active_customers):,}")
    print(f"\nERL_days Statistics (Active Customers Only):")
    erl_stats = active_customers["ERL_days"].describe(percentiles=[.25, .5, .75, .9, .95, .99])
    print(erl_stats)
    print(f"\nProb_alive Statistics (Active Customers Only):")
    prob_stats = active_customers["prob_alive"].describe(percentiles=[.25, .5, .75, .9, .95, .99])
    print(prob_stats)
    
    # Distribution counts (binned) instead of plots
    def print_binned_counts(series, name, bins_edges):
        """Print count in each bin for a series."""
        bins = pd.cut(series, bins=bins_edges, include_lowest=True)
        counts = bins.value_counts().sort_index()
        print(f"\n--- {name} (n={len(series):,}) ---")
        for interval, count in counts.items():
            print(f"  {interval}: {count:,}")
        print(f"  Total: {counts.sum():,}")

    # ERL_days bins: 0, 0-90, 90-180, ..., 900-1825 (and cap at 1825 if needed)
    erl_bins = [0, 90, 180, 270, 360, 450, 540, 630, 720, 900, 1825]
    print_binned_counts(result["ERL_days"], "ERL_days Distribution - All Customers", erl_bins)
    if len(active_customers) > 0:
        print_binned_counts(active_customers["ERL_days"], "ERL_days Distribution - Active Customers Only", erl_bins)

    # Prob_alive bins: 0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0
    prob_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    print_binned_counts(result["prob_alive"], "Prob_alive Distribution - All Customers", prob_bins)
    if len(active_customers) > 0:
        print_binned_counts(active_customers["prob_alive"], "Prob_alive Distribution - Active Customers Only", prob_bins)
    
    # Show top 10 customers by ERL_days
    print(f"\n--- Top 10 Customers by ERL_days ---")
    top_10 = result.nlargest(10, "ERL_days")[
        ["customer_id", "ERL_days", "prob_alive", "last_purchase_age_days"]
    ]
    print(top_10.to_string(index=False))
    
    # Show bottom 10 active customers by ERL_days (only customers with ERL_days > 0)
    print(f"\n--- Bottom 10 Active Customers by ERL_days ---")
    if len(active_customers) > 0:
        bottom_10 = active_customers.nsmallest(10, "ERL_days")[
            ["customer_id", "ERL_days", "prob_alive", "last_purchase_age_days"]
        ]
        print(bottom_10.to_string(index=False))
    else:
        print("No active customers found.")
    
    # Show random 10 customers
    print(f"\n--- Random 10 Customers ---")
    random_10 = result.sample(n=min(10, len(result)), random_state=42)[
        ["customer_id", "ERL_days", "prob_alive", "last_purchase_age_days"]
    ]
    print(random_10.to_string(index=False))
    
    print("\n✅ Real data test completed successfully!")


if __name__ == "__main__":
    # Run basic test
    # test_monte_carlo_basic()
    
    # Run real data test
    test_monte_carlo_real_data()

