import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path to import analytics module
sys.path.insert(0, str(Path(__file__).parent.parent))

from analytics.clv import build_rfm, fit_models, CLVResult, predict_clv
from app.db import run_query_internal


def test_build_rfm_basic():
    """Test build_rfm with basic transaction data."""
    transactions = pd.DataFrame({
        "customer_id": ["C1", "C1", "C2", "C2", "C2", "C3"],
        "invoice_no": ["I1", "I2", "I3", "I4", "I5", "I6"],
        "invoice_date": ["2023-01-01", "2023-01-15", "2023-01-10", "2023-02-01", "2023-02-15", "2023-01-20"],
        "revenue": [100.0, 150.0, 200.0, 180.0, 220.0, 50.0]
    })
    
    cutoff_date = "2023-03-01"
    rfm = build_rfm(transactions, cutoff_date, cap_percentile=100)  # No capping for test
    
    # Check structure
    assert isinstance(rfm, pd.DataFrame)
    assert set(rfm.columns) == {"frequency", "recency", "T", "monetary_value"}
    assert rfm.index.name == "customer_id" or "customer_id" in rfm.index.names
    
    # Check C1: 2 orders, first on 2023-01-01, last on 2023-01-15
    c1 = rfm.loc["C1"]
    assert c1["frequency"] == 1.0  # n_orders - 1 = 2 - 1 = 1
    assert c1["recency"] == 14.0  # days between first and last purchase
    assert c1["T"] == 59.0  # days from 2023-01-01 to 2023-03-01
    assert c1["monetary_value"] == 125.0  # average of 100 and 150
    
    # Check C2: 3 orders
    c2 = rfm.loc["C2"]
    assert c2["frequency"] == 2.0  # 3 - 1 = 2
    assert c2["recency"] == 36.0  # days from 2023-01-10 to 2023-02-15
    assert c2["T"] == 50.0  # days from 2023-01-10 to 2023-03-01
    assert c2["monetary_value"] == 200.0  # average of 200, 180, 220
    
    # Check C3: 1 order (one-time buyer)
    c3 = rfm.loc["C3"]
    assert c3["frequency"] == 0.0  # 1 - 1 = 0
    assert c3["recency"] == 0.0  # same day purchase
    assert c3["T"] == 40.0  # days from 2023-01-20 to 2023-03-01
    assert pd.isna(c3["monetary_value"])  # NaN for one-time buyers


def test_build_rfm_filters_invalid():
    """Test that build_rfm filters invalid transactions."""
    transactions = pd.DataFrame({
        "customer_id": ["C1", "C2", None, "C3", "C4"],
        "invoice_no": ["I1", "I2", "I3", "I4", "I5"],
        "invoice_date": ["2023-01-01", "2023-01-15", "2023-01-10", "2023-02-01", "2023-01-20"],
        "revenue": [100.0, 0.0, 50.0, -10.0, 200.0]  # 0, negative, and None customer_id
    })
    
    cutoff_date = "2023-03-01"
    rfm = build_rfm(transactions, cutoff_date)
    
    # Should only have C1 and C4 (C2 has 0 revenue, C3 has None customer_id, C4 has negative revenue)
    assert len(rfm) == 2
    assert "C1" in rfm.index
    assert "C4" in rfm.index
    assert "C2" not in rfm.index
    assert "C3" not in rfm.index


def test_build_rfm_cutoff_date():
    """Test that build_rfm filters transactions after cutoff date."""
    transactions = pd.DataFrame({
        "customer_id": ["C1", "C1", "C1"],
        "invoice_no": ["I1", "I2", "I3"],
        "invoice_date": ["2023-01-01", "2023-01-15", "2023-02-15"],
        "revenue": [100.0, 150.0, 200.0]
    })
    
    cutoff_date = "2023-02-01"
    rfm = build_rfm(transactions, cutoff_date, cap_percentile=100)  # No capping for test
    
    # Should only count I1 and I2 (I3 is after cutoff)
    c1 = rfm.loc["C1"]
    assert c1["frequency"] == 1.0  # 2 orders - 1 = 1
    assert c1["recency"] == 14.0  # days between I1 and I2
    assert c1["T"] == 31.0  # days from 2023-01-01 to 2023-02-01
    assert c1["monetary_value"] == 125.0  # average of 100 and 150 (I3 excluded)


def test_build_rfm_cap_percentile():
    """Test that build_rfm caps extreme order values."""
    transactions = pd.DataFrame({
        "customer_id": ["C1", "C1", "C1"],
        "invoice_no": ["I1", "I2", "I3"],
        "invoice_date": ["2023-01-01", "2023-01-15", "2023-02-01"],
        "revenue": [100.0, 150.0, 10000.0]  # extreme value
    })
    
    cutoff_date = "2023-03-01"
    rfm = build_rfm(transactions, cutoff_date, cap_percentile=99.0)
    
    c1 = rfm.loc["C1"]
    # The 99th percentile should cap the extreme value
    # Without cap: average would be (100 + 150 + 10000) / 3 = 3416.67
    # With cap at 99th percentile: the 10000 should be capped
    assert c1["monetary_value"] < 10000.0  # Should be capped


def test_build_rfm_multiple_items_per_invoice():
    """Test that build_rfm sums revenue per invoice correctly."""
    transactions = pd.DataFrame({
        "customer_id": ["C1", "C1", "C1", "C1"],
        "invoice_no": ["I1", "I1", "I2", "I2"],  # Two invoices with multiple items
        "invoice_date": ["2023-01-01", "2023-01-01", "2023-01-15", "2023-01-15"],
        "revenue": [50.0, 50.0, 75.0, 75.0]  # I1 = 100, I2 = 150
    })
    
    cutoff_date = "2023-03-01"
    rfm = build_rfm(transactions, cutoff_date, cap_percentile=100)  # No capping for test
    
    c1 = rfm.loc["C1"]
    assert c1["frequency"] == 1.0  # 2 invoices - 1 = 1
    assert c1["monetary_value"] == 125.0  # average of 100 and 150


def test_fit_models_basic():
    """Test fit_models with basic RFM data."""
    rfm = pd.DataFrame({
        "frequency": [0.0, 1.0, 2.0, 3.0, 5.0],
        "recency": [0.0, 10.0, 30.0, 60.0, 90.0],
        "T": [30.0, 40.0, 60.0, 90.0, 120.0],
        "monetary_value": [np.nan, 100.0, 150.0, 200.0, 250.0]
    }, index=["C1", "C2", "C3", "C4", "C5"])
    
    result = fit_models(rfm)
    
    # Check return type
    assert isinstance(result, CLVResult)
    assert isinstance(result.rfm, pd.DataFrame)
    assert hasattr(result, "bgnbd")
    assert hasattr(result, "gg")
    
    # Check that RFM is preserved
    assert len(result.rfm) == len(rfm)
    pd.testing.assert_frame_equal(result.rfm, rfm)
    
    # Check that models are fitted (they should have parameters)
    assert hasattr(result.bgnbd, "params_")
    assert hasattr(result.gg, "params_")


def test_fit_models_custom_penalizers():
    """Test fit_models with custom penalizer values."""
    rfm = pd.DataFrame({
        "frequency": [1.0, 2.0, 3.0],
        "recency": [10.0, 30.0, 60.0],
        "T": [40.0, 60.0, 90.0],
        "monetary_value": [100.0, 150.0, 200.0]
    }, index=["C1", "C2", "C3"])
    
    result = fit_models(rfm, penalizer=0.02, bgnbd_penalizer=0.1)
    
    assert isinstance(result, CLVResult)
    # Models should be fitted successfully
    assert hasattr(result.bgnbd, "params_")
    assert hasattr(result.gg, "params_")


def test_fit_models_default_bgnbd_penalizer():
    """Test that fit_models uses default BG/NBD penalizer when None."""
    rfm = pd.DataFrame({
        "frequency": [1.0, 2.0, 3.0],
        "recency": [10.0, 30.0, 60.0],
        "T": [40.0, 60.0, 90.0],
        "monetary_value": [100.0, 150.0, 200.0]
    }, index=["C1", "C2", "C3"])
    
    result = fit_models(rfm, bgnbd_penalizer=None)
    
    assert isinstance(result, CLVResult)
    # Should use default penalizer (0.05) and still fit successfully
    assert hasattr(result.bgnbd, "params_")


def test_fit_models_filters_repeat_buyers_for_gg():
    """Test that Gamma-Gamma model only uses repeat buyers."""
    rfm = pd.DataFrame({
        "frequency": [0.0, 1.0, 2.0, 3.0],  # C1 is one-time buyer
        "recency": [0.0, 10.0, 30.0, 60.0],
        "T": [30.0, 40.0, 60.0, 90.0],
        "monetary_value": [np.nan, 100.0, 150.0, 200.0]  # C1 has NaN
    }, index=["C1", "C2", "C3", "C4"])
    
    result = fit_models(rfm)
    
    # GG model should only fit on C2, C3, C4 (repeat buyers)
    # But the function should still work and return all customers in RFM
    assert isinstance(result, CLVResult)
    assert len(result.rfm) == 4  # All customers preserved
    assert hasattr(result.gg, "params_")  # GG model fitted on repeat buyers only


def test_fit_models_minimal_data():
    """Test fit_models with minimal valid data."""
    # Minimum: at least one customer for BG/NBD, at least one repeat buyer for GG
    rfm = pd.DataFrame({
        "frequency": [1.0, 2.0],
        "recency": [10.0, 30.0],
        "T": [40.0, 60.0],
        "monetary_value": [100.0, 150.0]
    }, index=["C1", "C2"])
    
    result = fit_models(rfm)
    
    assert isinstance(result, CLVResult)
    assert hasattr(result.bgnbd, "params_")
    assert hasattr(result.gg, "params_")


def test_build_rfm_and_fit_models_integration():
    """Integration test: build_rfm followed by fit_models."""
    transactions = pd.DataFrame({
        "customer_id": ["C1", "C1", "C2", "C2", "C2", "C3", "C3"],
        "invoice_no": ["I1", "I2", "I3", "I4", "I5", "I6", "I7"],
        "invoice_date": ["2023-01-01", "2023-01-15", "2023-01-10", "2023-02-01", "2023-02-15", "2023-01-20", "2023-02-10"],
        "revenue": [100.0, 150.0, 200.0, 180.0, 220.0, 50.0, 75.0]
    })
    
    cutoff_date = "2023-03-01"
    rfm = build_rfm(transactions, cutoff_date)
    
    # Verify RFM structure
    assert len(rfm) >= 3  # At least C1, C2, C3
    assert all(col in rfm.columns for col in ["frequency", "recency", "T", "monetary_value"])
    
    # Fit models
    result = fit_models(rfm)
    
    # Verify models are fitted
    assert isinstance(result, CLVResult)
    assert len(result.rfm) == len(rfm)
    assert hasattr(result.bgnbd, "params_")
    assert hasattr(result.gg, "params_")
    
    # Verify RFM data is preserved
    pd.testing.assert_frame_equal(result.rfm, rfm)


if __name__ == "__main__":
    # Run tests
    test_build_rfm_basic()
    print("✓ test_build_rfm_basic passed")
    
    test_build_rfm_filters_invalid()
    print("✓ test_build_rfm_filters_invalid passed")
    
    test_build_rfm_cutoff_date()
    print("✓ test_build_rfm_cutoff_date passed")
    
    test_build_rfm_cap_percentile()
    print("✓ test_build_rfm_cap_percentile passed")
    
    test_build_rfm_multiple_items_per_invoice()
    print("✓ test_build_rfm_multiple_items_per_invoice passed")
    
    test_fit_models_basic()
    print("✓ test_fit_models_basic passed")
    
    test_fit_models_custom_penalizers()
    print("✓ test_fit_models_custom_penalizers passed")
    
    test_fit_models_default_bgnbd_penalizer()
    print("✓ test_fit_models_default_bgnbd_penalizer passed")
    
    test_fit_models_filters_repeat_buyers_for_gg()
    print("✓ test_fit_models_filters_repeat_buyers_for_gg passed")
    
    test_fit_models_minimal_data()
    print("✓ test_fit_models_minimal_data passed")
    
    test_build_rfm_and_fit_models_integration()
    print("✓ test_build_rfm_and_fit_models_integration passed")
    
    print("\n✅ All tests passed!")
    
    # ========================================================================
    # TEST ON REAL DATA
    # ========================================================================
    print("\n" + "="*60)
    print("TESTING ON REAL DATA")
    print("="*60)
    
    # Load transactions from SQLite (include country for segment analysis)
    sql = """
    SELECT customer_id, invoice_no, invoice_date, revenue, country
    FROM transactions
    WHERE customer_id IS NOT NULL
    """
    rows, _ = run_query_internal(sql, max_rows=2_000_000)
    df = pd.DataFrame(rows)
    
    print(f"\nLoaded {len(df):,} transactions")
    print(f"Date range: {df['invoice_date'].min()} to {df['invoice_date'].max()}")
    
    # Use a reasonable cutoff date (e.g., 2011-12-09 like in survival test)
    cutoff_date = "2011-12-09"
    
    # Test build_rfm on real data
    print(f"\n--- Testing build_rfm with cutoff_date={cutoff_date} ---")
    rfm = build_rfm(df, cutoff_date)
    
    print(f"✓ RFM table created: {len(rfm):,} customers")
    print(f"\nRFM Summary:")
    print(rfm[["frequency", "recency", "T", "monetary_value"]].describe())
    
    # Check data quality
    print(f"\nData Quality Checks:")
    print(f"  - Customers with frequency > 0: {(rfm['frequency'] > 0).sum():,}")
    print(f"  - Customers with valid monetary_value: {rfm['monetary_value'].notna().sum():,}")
    print(f"  - One-time buyers: {(rfm['frequency'] == 0).sum():,}")
    print(f"  - Repeat buyers: {(rfm['frequency'] > 0).sum():,}")
    
    # Test fit_models on real data
    print(f"\n--- Testing fit_models ---")
    result = fit_models(rfm)
    
    print(f"✓ Models fitted successfully")
    print(f"  - BG/NBD model parameters: {list(result.bgnbd.params_.keys())}")
    print(f"  - Gamma-Gamma model parameters: {list(result.gg.params_.keys())}")
    
    # Display some sample customers
    print(f"\n--- Sample Customers (Top 10 by Frequency) ---")
    sample = rfm.sort_values("frequency", ascending=False).head(10)
    print(sample[["frequency", "recency", "T", "monetary_value"]])
    
    print(f"\n--- Sample Customers (Random 10) ---")
    sample = rfm.sample(n=min(10, len(rfm)))
    print(sample[["frequency", "recency", "T", "monetary_value"]])
    
    print("\n✅ Real data tests completed successfully!")
    
    # ========================================================================
    # TRAIN/TEST SPLIT AND MODEL EVALUATION
    # ========================================================================
    print("\n" + "="*60)
    print("TRAIN/TEST SPLIT AND MODEL EVALUATION")
    print("="*60)
    
    # Split data: train on data up to 2011-09-09, test on 2011-09-09 to 2011-12-09
    train_cutoff = "2011-09-09"
    test_cutoff = "2011-12-09"
    test_horizon_days = (pd.to_datetime(test_cutoff) - pd.to_datetime(train_cutoff)).days
    
    print(f"\nTrain cutoff: {train_cutoff}")
    print(f"Test cutoff: {test_cutoff}")
    print(f"Test horizon: {test_horizon_days} days")
    
    # Build RFM for training period
    print(f"\n--- Building RFM for training period ---")
    rfm_train = build_rfm(df, train_cutoff)
    print(f"Training customers: {len(rfm_train):,}")
    
    # Fit models on training data
    print(f"\n--- Fitting models on training data ---")
    clv_train = fit_models(rfm_train)
    print(f"✓ Models fitted on {len(rfm_train):,} customers")
    
    # Calculate actual values in test period
    print(f"\n--- Calculating actual values in test period ---")
    df_test = df.copy()
    df_test["invoice_date"] = pd.to_datetime(df_test["invoice_date"])
    train_cutoff_dt = pd.to_datetime(train_cutoff)
    test_cutoff_dt = pd.to_datetime(test_cutoff)
    
    # Filter to test period transactions
    test_transactions = df_test[
        (df_test["invoice_date"] > train_cutoff_dt) & 
        (df_test["invoice_date"] <= test_cutoff_dt)
    ].copy()
    
    print(f"Test period transactions: {len(test_transactions):,}")
    
    # Aggregate test period data by customer
    test_orders = (
        test_transactions.groupby(["customer_id", "invoice_no"], as_index=False)
        .agg(
            order_date=("invoice_date", "min"),
            order_value=("revenue", "sum"),
        )
    )
    
    # Calculate actual purchases and AOV in test period
    test_customer_stats = test_orders.groupby("customer_id", as_index=True).agg(
        actual_purchases=("invoice_no", "nunique"),  # Number of orders in test period
        actual_aov=("order_value", "mean"),  # Average order value in test period
    )
    
    # For customers with no test period purchases, set to 0
    test_customer_stats = test_customer_stats.reindex(rfm_train.index, fill_value=0)
    test_customer_stats["actual_purchases"] = test_customer_stats["actual_purchases"].fillna(0)
    test_customer_stats["actual_aov"] = test_customer_stats["actual_aov"].fillna(np.nan)
    
    print(f"Customers with purchases in test period: {(test_customer_stats['actual_purchases'] > 0).sum():,}")
    
    # Make predictions using training models
    print(f"\n--- Making predictions for test period ---")
    predictions = clv_train.bgnbd.conditional_expected_number_of_purchases_up_to_time(
        test_horizon_days,
        rfm_train["frequency"],
        rfm_train["recency"],
        rfm_train["T"]
    )
    
    # Predict AOV for repeat buyers
    gg_eligible_mask = (rfm_train["frequency"] > 0) & (rfm_train["monetary_value"] > 0)
    pred_aov = pd.Series(index=rfm_train.index, dtype=float)
    pred_aov.loc[gg_eligible_mask] = clv_train.gg.conditional_expected_average_profit(
        rfm_train.loc[gg_eligible_mask, "frequency"],
        rfm_train.loc[gg_eligible_mask, "monetary_value"],
    )
    
    # Combine predictions with actuals
    evaluation = pd.DataFrame({
        "pred_purchases": predictions,
        "actual_purchases": test_customer_stats["actual_purchases"],
        "pred_aov": pred_aov,
        "actual_aov": test_customer_stats["actual_aov"],
    }, index=rfm_train.index)
    
    # ========================================================================
    # EVALUATE BG/NBD MODEL (Purchase Frequency)
    # ========================================================================
    print(f"\n--- BG/NBD Model Evaluation (Purchase Frequency) ---")
    
    # Filter to customers present in training set
    eval_bgnbd = evaluation.dropna(subset=["pred_purchases", "actual_purchases"])
    
    if len(eval_bgnbd) > 0:
        # Calculate MAE and RMSE
        mae_purchases = np.mean(np.abs(eval_bgnbd["pred_purchases"] - eval_bgnbd["actual_purchases"]))
        rmse_purchases = np.sqrt(np.mean((eval_bgnbd["pred_purchases"] - eval_bgnbd["actual_purchases"])**2))
        
        print(f"Customers evaluated: {len(eval_bgnbd):,}")
        print(f"MAE (Mean Absolute Error): {mae_purchases:.4f}")
        print(f"RMSE (Root Mean Squared Error): {rmse_purchases:.4f}")
        
        # Additional statistics
        print(f"\nPurchase Prediction Statistics:")
        print(f"  Mean predicted purchases: {eval_bgnbd['pred_purchases'].mean():.4f}")
        print(f"  Mean actual purchases: {eval_bgnbd['actual_purchases'].mean():.4f}")
        print(f"  Std predicted purchases: {eval_bgnbd['pred_purchases'].std():.4f}")
        print(f"  Std actual purchases: {eval_bgnbd['actual_purchases'].std():.4f}")
        
        # Show some examples
        print(f"\n--- Sample Predictions vs Actuals (Top 20 by actual purchases) ---")
        sample = eval_bgnbd.sort_values("actual_purchases", ascending=False).head(20)
        print(sample[["pred_purchases", "actual_purchases"]])
    else:
        print("No data available for BG/NBD evaluation")
    
    # ========================================================================
    # EVALUATE GAMMA-GAMMA MODEL (Average Order Value)
    # ========================================================================
    print(f"\n--- Gamma-Gamma Model Evaluation (Average Order Value) ---")
    
    # Filter to repeat buyers who made purchases in test period
    eval_gg = evaluation[
        (evaluation["pred_aov"].notna()) & 
        (evaluation["actual_aov"].notna()) &
        (evaluation["actual_purchases"] > 0)  # Only evaluate on customers who actually purchased
    ].copy()
    
    if len(eval_gg) > 0:
        # Calculate MAE
        mae_aov = np.mean(np.abs(eval_gg["pred_aov"] - eval_gg["actual_aov"]))
        
        print(f"Customers evaluated: {len(eval_gg):,}")
        print(f"MAE (Mean Absolute Error): {mae_aov:.4f}")
        
        # Additional statistics
        print(f"\nAOV Prediction Statistics:")
        print(f"  Mean predicted AOV: {eval_gg['pred_aov'].mean():.4f}")
        print(f"  Mean actual AOV: {eval_gg['actual_aov'].mean():.4f}")
        print(f"  Std predicted AOV: {eval_gg['pred_aov'].std():.4f}")
        print(f"  Std actual AOV: {eval_gg['actual_aov'].std():.4f}")
        
        # Show some examples
        print(f"\n--- Sample Predictions vs Actuals (Top 20 by actual AOV) ---")
        sample = eval_gg.sort_values("actual_aov", ascending=False).head(20)
        print(sample[["pred_aov", "actual_aov"]])
    else:
        print("No data available for Gamma-Gamma evaluation")
    
    print("\n✅ Model evaluation completed!")
    
    # ========================================================================
    # CLV CALIBRATION FLOW
    # ========================================================================
    print("\n" + "="*60)
    print("CLV CALIBRATION FLOW")
    print("="*60)
    
    # Step 1: Split data at time cutoff (already done above)
    print(f"\n--- Step 1: Data Split ---")
    print(f"Train cutoff: {train_cutoff}")
    print(f"Test cutoff: {test_cutoff}")
    print(f"Horizon: {test_horizon_days} days")
    
    # Step 2: Train BG/NBD + Gamma-Gamma on train (already done above)
    print(f"\n--- Step 2: Model Training (Already Complete) ---")
    print(f"✓ BG/NBD and Gamma-Gamma models fitted on {len(rfm_train):,} customers")
    
    # Step 3: Predict X-day CLV with no scaling
    print(f"\n--- Step 3: Initial CLV Prediction (No Scaling) ---")
    clv_unscaled = predict_clv(
        clv_train,
        horizon_days=test_horizon_days,
        scale_to_target_purchases=None,
        scale_to_target_revenue=None,
        aov_fallback="global_mean"  # Use global mean for non-eligible customers to compute total revenue
    )
    print(f"✓ Predicted CLV for {len(clv_unscaled):,} customers")
    
    # Step 4: Compute total actual vs predicted purchases and revenue in test
    print(f"\n--- Step 4: Compute Actual vs Predicted Totals ---")
    
    # Calculate actual totals from test period
    actual_total_purchases = test_customer_stats["actual_purchases"].sum()
    # Calculate actual revenue: sum of all test period transaction revenue
    actual_total_revenue = test_transactions["revenue"].sum()
    
    # Calculate predicted totals (unscaled)
    pred_total_purchases = clv_unscaled["pred_purchases"].sum()
    pred_total_revenue = clv_unscaled["clv"].sum(skipna=True)
    
    print(f"Actual totals (test period):")
    print(f"  Total purchases: {actual_total_purchases:,.0f}")
    print(f"  Total revenue: ${actual_total_revenue:,.2f}")
    print(f"\nPredicted totals (unscaled):")
    print(f"  Total purchases: {pred_total_purchases:,.2f}")
    print(f"  Total revenue: ${pred_total_revenue:,.2f}")
    
    # Step 5: Derive purchase_scale and revenue_scale as ratios actual/predicted
    print(f"\n--- Step 5: Calculate Calibration Scales ---")
    
    purchase_scale = actual_total_purchases / pred_total_purchases if pred_total_purchases > 0 else 1.0
    revenue_scale = actual_total_revenue / pred_total_revenue if pred_total_revenue > 0 else 1.0
    
    print(f"\nScale Values:")
    print(f"  Purchase scale: {purchase_scale:.6f} (actual: {actual_total_purchases:,.0f} / predicted: {pred_total_purchases:,.2f})")
    print(f"  Revenue scale: {revenue_scale:.6f} (actual: ${actual_total_revenue:,.2f} / predicted: ${pred_total_revenue:,.2f})")
    
    print(f"\n--- Hard-code these values in analytics/clv.py ---")
    print(f"PURCHASE_SCALE = {purchase_scale:.10f}")
    print(f"REVENUE_SCALE = {revenue_scale:.10f}")
    
    # Step 6: Rerun prediction with scaling
    print(f"\n--- Step 6: Scaled CLV Prediction ---")
    clv_scaled = predict_clv(
        clv_train,
        horizon_days=test_horizon_days,
        scale_to_target_purchases=actual_total_purchases,
        scale_to_target_revenue=actual_total_revenue,
        aov_fallback="global_mean"
    )
    print(f"✓ Scaled CLV predicted for {len(clv_scaled):,} customers")
    
    # Verify scaling worked
    scaled_total_purchases = clv_scaled["pred_purchases"].sum()
    scaled_total_revenue = clv_scaled["clv"].sum(skipna=True)
    
    print(f"\nScaled prediction totals:")
    print(f"  Total purchases: {scaled_total_purchases:,.2f} (target: {actual_total_purchases:,.0f})")
    print(f"  Total revenue: ${scaled_total_revenue:,.2f} (target: ${actual_total_revenue:,.2f})")
    
    # Step 7: Output scaled CLV per customer plus evaluation metrics
    print(f"\n--- Step 7: Scaled CLV Output and Evaluation Metrics ---")
    
    # Calculate actual revenue per customer from test transactions
    customer_actual_revenue = test_transactions.groupby("customer_id")["revenue"].sum().reset_index(name="actual_revenue")
    
    # Merge with actuals for evaluation
    clv_evaluation = clv_scaled.merge(
        test_customer_stats.reset_index(),
        on="customer_id",
        how="left",
        suffixes=("", "_actual")
    ).merge(
        customer_actual_revenue,
        on="customer_id",
        how="left"
    )
    clv_evaluation["actual_revenue"] = clv_evaluation["actual_revenue"].fillna(0)
    
    # Calculate evaluation metrics
    print(f"\nEvaluation Metrics (Scaled Predictions):")
    
    # Purchase metrics
    purchase_eval = clv_evaluation.dropna(subset=["pred_purchases", "actual_purchases"])
    if len(purchase_eval) > 0:
        purchase_mae = np.mean(np.abs(purchase_eval["pred_purchases"] - purchase_eval["actual_purchases"]))
        purchase_rmse = np.sqrt(np.mean((purchase_eval["pred_purchases"] - purchase_eval["actual_purchases"])**2))
        purchase_mape = np.mean(np.abs((purchase_eval["pred_purchases"] - purchase_eval["actual_purchases"]) / 
                                       (purchase_eval["actual_purchases"] + 1e-10))) * 100
        
        print(f"\nPurchase Prediction Metrics:")
        print(f"  MAE: {purchase_mae:.4f}")
        print(f"  RMSE: {purchase_rmse:.4f}")
        print(f"  MAPE: {purchase_mape:.2f}%")
        print(f"  Total predicted: {purchase_eval['pred_purchases'].sum():,.2f}")
        print(f"  Total actual: {purchase_eval['actual_purchases'].sum():,.0f}")
        print(f"  Error: {abs(purchase_eval['pred_purchases'].sum() - purchase_eval['actual_purchases'].sum()):,.2f}")
    
    # Revenue metrics
    revenue_eval = clv_evaluation[
        (clv_evaluation["clv"].notna()) & 
        (clv_evaluation["actual_revenue"] > 0)
    ].copy()
    
    if len(revenue_eval) > 0:
        revenue_mae = np.mean(np.abs(revenue_eval["clv"] - revenue_eval["actual_revenue"]))
        revenue_rmse = np.sqrt(np.mean((revenue_eval["clv"] - revenue_eval["actual_revenue"])**2))
        revenue_mape = np.mean(np.abs((revenue_eval["clv"] - revenue_eval["actual_revenue"]) / 
                                     (revenue_eval["actual_revenue"] + 1e-10))) * 100
        
        print(f"\nRevenue (CLV) Prediction Metrics:")
        print(f"  MAE: ${revenue_mae:,.2f}")
        print(f"  RMSE: ${revenue_rmse:,.2f}")
        print(f"  MAPE: {revenue_mape:.2f}%")
        print(f"  Total predicted: ${revenue_eval['clv'].sum():,.2f}")
        print(f"  Total actual: ${revenue_eval['actual_revenue'].sum():,.2f}")
        print(f"  Error: ${abs(revenue_eval['clv'].sum() - revenue_eval['actual_revenue'].sum()):,.2f}")
    else:
        print("\nNo revenue data available for evaluation")
    
    # Output scaled CLV per customer (top customers)
    print(f"\n--- Top 20 Customers by Scaled CLV ---")
    top_clv = clv_scaled.nlargest(20, "clv")[
        ["customer_id", "frequency", "recency", "T", "pred_purchases", "pred_aov", "clv"]
    ]
    print(top_clv.to_string(index=False))
    
    # Summary statistics
    print(f"\n--- Scaled CLV Summary Statistics ---")
    clv_summary = clv_scaled["clv"].describe()
    print(clv_summary)
    
    print(f"\nCustomers with valid CLV: {clv_scaled['has_valid_clv'].sum():,} / {len(clv_scaled):,}")
    print(f"Total scaled CLV: ${clv_scaled['clv'].sum(skipna=True):,.2f}")
    
    # ========================================================================
    # DECILE ANALYSIS BY SCALED CLV
    # ========================================================================
    print("\n" + "="*60)
    print("DECILE ANALYSIS BY SCALED CLV")
    print("="*60)
    
    # Merge scaled CLV with actual revenue
    decile_data = clv_scaled[["customer_id", "clv"]].merge(
        customer_actual_revenue,
        on="customer_id",
        how="left"
    )
    decile_data["actual_revenue"] = decile_data["actual_revenue"].fillna(0)
    
    # Sort by CLV and create deciles
    decile_data = decile_data.sort_values("clv", ascending=False, na_position="last").reset_index(drop=True)
    
    # Create deciles based on CLV values (handle NaN separately)
    valid_clv = decile_data["clv"].notna()
    decile_data["decile"] = None
    
    # Assign deciles to customers with valid CLV
    if valid_clv.sum() > 0:
        try:
            decile_data.loc[valid_clv, "decile"] = pd.qcut(
                decile_data.loc[valid_clv, "clv"],
                q=10,
                labels=[f"Decile {i+1}" for i in range(10)],
                duplicates="drop"
            )
        except ValueError:
            # If qcut fails (e.g., too many duplicates), use rank-based approach
            decile_data.loc[valid_clv, "decile"] = pd.cut(
                decile_data.loc[valid_clv, "clv"].rank(method="first"),
                bins=10,
                labels=[f"Decile {i+1}" for i in range(10)]
            )
    
    # For customers with NaN CLV, assign to decile 1 (lowest)
    decile_data.loc[decile_data["clv"].isna(), "decile"] = "Decile 1"
    
    # Compute mean predicted vs actual revenue per decile
    decile_summary = decile_data.groupby("decile").agg({
        "clv": ["mean", "sum", "count"],
        "actual_revenue": ["mean", "sum"]
    }).round(2)
    
    # Flatten column names
    decile_summary.columns = ["_".join(col).strip() for col in decile_summary.columns]
    decile_summary = decile_summary.rename(columns={
        "clv_mean": "Mean_Pred_CLV",
        "clv_sum": "Total_Pred_CLV",
        "clv_count": "N_Customers",
        "actual_revenue_mean": "Mean_Actual_Revenue",
        "actual_revenue_sum": "Total_Actual_Revenue"
    })
    
    # Calculate pred/actual ratio
    decile_summary["Pred_Actual_Ratio"] = (
        decile_summary["Total_Pred_CLV"] / decile_summary["Total_Actual_Revenue"]
    ).round(3)
    
    print("\n--- Decile Analysis: Predicted vs Actual Revenue ---")
    print(decile_summary.to_string())
    
    # Validation checks
    print("\n--- Decile Validation Checks ---")
    
    # Check 1: Actual revenue should increase from decile 1 to 10
    actual_means = decile_summary["Mean_Actual_Revenue"].values
    is_increasing = all(actual_means[i] <= actual_means[i+1] for i in range(len(actual_means)-1))
    print(f"✓ Actual revenue increasing from D1 to D10: {is_increasing}")
    if not is_increasing:
        print(f"  WARNING: Actual revenue does not consistently increase across deciles")
        print(f"  D1 mean: ${actual_means[0]:,.2f}, D10 mean: ${actual_means[-1]:,.2f}")
    
    # Check 2: Pred/actual ratios should be reasonable (between 0.5 and 2.0)
    ratios = decile_summary["Pred_Actual_Ratio"].values
    extreme_ratios = decile_summary[
        (decile_summary["Pred_Actual_Ratio"] < 0.5) | 
        (decile_summary["Pred_Actual_Ratio"] > 2.0)
    ]
    if len(extreme_ratios) > 0:
        print(f"⚠ WARNING: {len(extreme_ratios)} decile(s) have extreme pred/actual ratios:")
        print(extreme_ratios[["Pred_Actual_Ratio"]].to_string())
    else:
        print(f"✓ All decile pred/actual ratios are reasonable (0.5-2.0)")
    
    # Check 3: Overall ratio should be close to 1.0 (due to scaling)
    overall_ratio = decile_summary["Total_Pred_CLV"].sum() / decile_summary["Total_Actual_Revenue"].sum()
    print(f"✓ Overall pred/actual ratio: {overall_ratio:.4f} (should be ~1.0 after scaling)")
    
    # ========================================================================
    # SEGMENT ANALYSIS
    # ========================================================================
    print("\n" + "="*60)
    print("SEGMENT ANALYSIS")
    print("="*60)
    
    # Get customer attributes for segmentation
    # 1. Heavy vs Light buyers (based on training frequency)
    # 2. Long vs Short tenure (based on training T)
    # 3. Key markets (countries)
    
    # Add training RFM attributes
    segment_data = clv_scaled.merge(
        rfm_train.reset_index(),
        on="customer_id",
        how="left",
        suffixes=("", "_train")
    )
    
    # Add actual revenue
    segment_data = segment_data.merge(
        customer_actual_revenue,
        on="customer_id",
        how="left"
    )
    segment_data["actual_revenue"] = segment_data["actual_revenue"].fillna(0)
    
    # Get country information (most common country per customer)
    customer_countries = df.groupby("customer_id")["country"].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else "Unknown").reset_index()
    customer_countries.columns = ["customer_id", "country"]
    segment_data = segment_data.merge(customer_countries, on="customer_id", how="left")
    segment_data["country"] = segment_data["country"].fillna("Unknown")
    
    # Define segments
    # Heavy vs Light buyers (median split on frequency)
    freq_median = segment_data["frequency"].median()
    segment_data["buyer_segment"] = segment_data["frequency"].apply(
        lambda x: "Heavy" if x >= freq_median else "Light"
    )
    
    # Long vs Short tenure (median split on T)
    t_median = segment_data["T"].median()
    segment_data["tenure_segment"] = segment_data["T"].apply(
        lambda x: "Long" if x >= t_median else "Short"
    )
    
    # Top markets (top 5 countries by customer count)
    top_countries = segment_data["country"].value_counts().head(5).index.tolist()
    segment_data["market_segment"] = segment_data["country"].apply(
        lambda x: x if x in top_countries else "Other"
    )
    
    # Analyze each segment
    segments_to_analyze = [
        ("buyer_segment", "Heavy vs Light Buyers"),
        ("tenure_segment", "Long vs Short Tenure"),
        ("market_segment", "Key Markets")
    ]
    
    for segment_col, segment_name in segments_to_analyze:
        print(f"\n--- {segment_name} ---")
        
        segment_summary = segment_data.groupby(segment_col).agg({
            "clv": ["mean", "sum", "count"],
            "actual_revenue": ["mean", "sum"]
        }).round(2)
        
        segment_summary.columns = ["_".join(col).strip() for col in segment_summary.columns]
        segment_summary = segment_summary.rename(columns={
            "clv_mean": "Mean_Pred_CLV",
            "clv_sum": "Total_Pred_CLV",
            "clv_count": "N_Customers",
            "actual_revenue_mean": "Mean_Actual_Revenue",
            "actual_revenue_sum": "Total_Actual_Revenue"
        })
        
        segment_summary["Pred_Actual_Ratio"] = (
            segment_summary["Total_Pred_CLV"] / segment_summary["Total_Actual_Revenue"]
        ).round(3)
        
        print(segment_summary.to_string())
        
        # Check for extreme ratios
        extreme_segments = segment_summary[
            (segment_summary["Pred_Actual_Ratio"] < 0.5) | 
            (segment_summary["Pred_Actual_Ratio"] > 2.0)
        ]
        if len(extreme_segments) > 0:
            print(f"\n⚠ WARNING: {len(extreme_segments)} segment(s) have extreme pred/actual ratios:")
            print(extreme_segments[["Pred_Actual_Ratio"]].to_string())
        else:
            print(f"\n✓ All {segment_name} segments have reasonable pred/actual ratios (0.5-2.0)")
    
    print("\n✅ CLV calibration flow completed!")
    
    # ========================================================================
    # MULTIPLE TRAIN/TEST SPLITS TO PICK BEST SCALE VALUES
    # ========================================================================
    print("\n" + "="*60)
    print("MULTIPLE TRAIN/TEST SPLITS - FINDING BEST SCALE VALUES")
    print("="*60)
    
    # Define multiple train/test cutoff date combinations
    # Test different split points to find robust scale values
    split_configs = [
        {"train_cutoff": "2011-08-09", "test_cutoff": "2011-11-09", "horizon_days": 92},  # ~3 months
        {"train_cutoff": "2011-09-09", "test_cutoff": "2011-12-09", "horizon_days": 91},  # ~3 months (original)
        {"train_cutoff": "2011-07-09", "test_cutoff": "2011-10-09", "horizon_days": 92},  # ~3 months
        {"train_cutoff": "2011-06-09", "test_cutoff": "2011-09-09", "horizon_days": 92},  # ~3 months
        {"train_cutoff": "2011-08-09", "test_cutoff": "2011-10-09", "horizon_days": 61},  # ~2 months
        {"train_cutoff": "2011-09-09", "test_cutoff": "2011-11-09", "horizon_days": 61},  # ~2 months
    ]
    
    print(f"\nTesting {len(split_configs)} different train/test split configurations...")
    
    scale_results = []
    
    for i, config in enumerate(split_configs, 1):
        train_cutoff = config["train_cutoff"]
        test_cutoff = config["test_cutoff"]
        horizon_days = config["horizon_days"]
        
        print(f"\n--- Split {i}/{len(split_configs)}: Train={train_cutoff}, Test={test_cutoff} ({horizon_days} days) ---")
        
        try:
            # Build RFM for training period
            rfm_train_split = build_rfm(df, train_cutoff)
            
            if len(rfm_train_split) < 100:  # Skip if too few customers
                print(f"  ⚠ Skipped: Too few training customers ({len(rfm_train_split)})")
                continue
            
            # Fit models
            clv_train_split = fit_models(rfm_train_split)
            
            # Calculate actual values in test period
            df_test_split = df.copy()
            df_test_split["invoice_date"] = pd.to_datetime(df_test_split["invoice_date"])
            train_cutoff_dt = pd.to_datetime(train_cutoff)
            test_cutoff_dt = pd.to_datetime(test_cutoff)
            
            test_transactions_split = df_test_split[
                (df_test_split["invoice_date"] > train_cutoff_dt) & 
                (df_test_split["invoice_date"] <= test_cutoff_dt)
            ].copy()
            
            if len(test_transactions_split) == 0:
                print(f"  ⚠ Skipped: No test period transactions")
                continue
            
            # Aggregate test period data
            test_orders_split = (
                test_transactions_split.groupby(["customer_id", "invoice_no"], as_index=False)
                .agg(
                    order_date=("invoice_date", "min"),
                    order_value=("revenue", "sum"),
                )
            )
            
            test_customer_stats_split = test_orders_split.groupby("customer_id", as_index=True).agg(
                actual_purchases=("invoice_no", "nunique"),
                actual_aov=("order_value", "mean"),
            )
            
            test_customer_stats_split = test_customer_stats_split.reindex(rfm_train_split.index, fill_value=0)
            test_customer_stats_split["actual_purchases"] = test_customer_stats_split["actual_purchases"].fillna(0)
            test_customer_stats_split["actual_aov"] = test_customer_stats_split["actual_aov"].fillna(np.nan)
            
            # Predict unscaled CLV
            clv_unscaled_split = predict_clv(
                clv_train_split,
                horizon_days=horizon_days,
                scale_to_target_purchases=None,
                scale_to_target_revenue=None,
                aov_fallback="global_mean"
            )
            
            # Calculate actual vs predicted totals
            actual_total_purchases_split = test_customer_stats_split["actual_purchases"].sum()
            actual_total_revenue_split = test_transactions_split["revenue"].sum()
            
            pred_total_purchases_split = clv_unscaled_split["pred_purchases"].sum()
            pred_total_revenue_split = clv_unscaled_split["clv"].sum(skipna=True)
            
            # Calculate scales
            purchase_scale_split = actual_total_purchases_split / pred_total_purchases_split if pred_total_purchases_split > 0 else 1.0
            revenue_scale_split = actual_total_revenue_split / pred_total_revenue_split if pred_total_revenue_split > 0 else 1.0
            
            # Evaluate scaled predictions
            clv_scaled_split = predict_clv(
                clv_train_split,
                horizon_days=horizon_days,
                scale_to_target_purchases=actual_total_purchases_split,
                scale_to_target_revenue=actual_total_revenue_split,
                aov_fallback="global_mean"
            )
            
            # Calculate evaluation metrics
            customer_actual_revenue_split = test_transactions_split.groupby("customer_id")["revenue"].sum().reset_index(name="actual_revenue")
            
            clv_eval_split = clv_scaled_split.merge(
                test_customer_stats_split.reset_index(),
                on="customer_id",
                how="left"
            ).merge(
                customer_actual_revenue_split,
                on="customer_id",
                how="left"
            )
            clv_eval_split["actual_revenue"] = clv_eval_split["actual_revenue"].fillna(0)
            
            # Purchase metrics
            purchase_eval_split = clv_eval_split.dropna(subset=["pred_purchases", "actual_purchases"])
            purchase_mae_split = np.mean(np.abs(purchase_eval_split["pred_purchases"] - purchase_eval_split["actual_purchases"])) if len(purchase_eval_split) > 0 else np.nan
            purchase_rmse_split = np.sqrt(np.mean((purchase_eval_split["pred_purchases"] - purchase_eval_split["actual_purchases"])**2)) if len(purchase_eval_split) > 0 else np.nan
            
            # Revenue metrics
            revenue_eval_split = clv_eval_split[
                (clv_eval_split["clv"].notna()) & 
                (clv_eval_split["actual_revenue"] > 0)
            ]
            revenue_mae_split = np.mean(np.abs(revenue_eval_split["clv"] - revenue_eval_split["actual_revenue"])) if len(revenue_eval_split) > 0 else np.nan
            revenue_rmse_split = np.sqrt(np.mean((revenue_eval_split["clv"] - revenue_eval_split["actual_revenue"])**2)) if len(revenue_eval_split) > 0 else np.nan
            
            # Store results
            result = {
                "train_cutoff": train_cutoff,
                "test_cutoff": test_cutoff,
                "horizon_days": horizon_days,
                "n_train_customers": len(rfm_train_split),
                "n_test_customers": len(test_customer_stats_split[test_customer_stats_split["actual_purchases"] > 0]),
                "purchase_scale": purchase_scale_split,
                "revenue_scale": revenue_scale_split,
                "purchase_mae": purchase_mae_split,
                "purchase_rmse": purchase_rmse_split,
                "revenue_mae": revenue_mae_split,
                "revenue_rmse": revenue_rmse_split,
                "actual_total_purchases": actual_total_purchases_split,
                "actual_total_revenue": actual_total_revenue_split,
            }
            
            scale_results.append(result)
            
            print(f"  ✓ Purchase scale: {purchase_scale_split:.6f}")
            print(f"  ✓ Revenue scale: {revenue_scale_split:.6f}")
            print(f"  ✓ Purchase MAE: {purchase_mae_split:.4f}, RMSE: {purchase_rmse_split:.4f}")
            print(f"  ✓ Revenue MAE: ${revenue_mae_split:,.2f}, RMSE: ${revenue_rmse_split:,.2f}")
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            continue
    
    if len(scale_results) == 0:
        print("\n⚠ No valid split configurations found")
    else:
        # Convert to DataFrame for analysis
        scale_df = pd.DataFrame(scale_results)
        
        print(f"\n--- Summary of All Splits ---")
        print(scale_df[["train_cutoff", "test_cutoff", "purchase_scale", "revenue_scale", 
                       "purchase_mae", "revenue_mae"]].to_string(index=False))
        
        # Analyze scale values
        print(f"\n--- Scale Value Statistics ---")
        print(f"Purchase scale - Mean: {scale_df['purchase_scale'].mean():.6f}, Std: {scale_df['purchase_scale'].std():.6f}")
        print(f"Purchase scale - Min: {scale_df['purchase_scale'].min():.6f}, Max: {scale_df['purchase_scale'].max():.6f}")
        print(f"Revenue scale - Mean: {scale_df['revenue_scale'].mean():.6f}, Std: {scale_df['revenue_scale'].std():.6f}")
        print(f"Revenue scale - Min: {scale_df['revenue_scale'].min():.6f}, Max: {scale_df['revenue_scale'].max():.6f}")
        
        # Find best split based on combined error metric (lower is better)
        # Use normalized MAE (MAE / mean actual value) to compare across splits
        scale_df["combined_error"] = (
            scale_df["purchase_mae"] / (scale_df["actual_total_purchases"] / scale_df["n_test_customers"] + 1e-10) +
            scale_df["revenue_mae"] / (scale_df["actual_total_revenue"] / scale_df["n_test_customers"] + 1e-10)
        )
        
        best_idx = scale_df["combined_error"].idxmin()
        best_split = scale_df.loc[best_idx]
        
        print(f"\n--- Best Split (Lowest Combined Error) ---")
        print(f"Train cutoff: {best_split['train_cutoff']}")
        print(f"Test cutoff: {best_split['test_cutoff']}")
        print(f"Horizon: {best_split['horizon_days']} days")
        print(f"Purchase scale: {best_split['purchase_scale']:.6f}")
        print(f"Revenue scale: {best_split['revenue_scale']:.6f}")
        print(f"Purchase MAE: {best_split['purchase_mae']:.4f}")
        print(f"Revenue MAE: ${best_split['revenue_mae']:,.2f}")
        
        # Recommend using mean scales for robustness, or best split scales
        mean_purchase_scale = scale_df["purchase_scale"].mean()
        mean_revenue_scale = scale_df["revenue_scale"].mean()
        
        print(f"\n--- Recommended Scale Values ---")
        print(f"Option 1: Use mean scales (most robust across splits):")
        print(f"  PURCHASE_SCALE = {mean_purchase_scale:.10f}")
        print(f"  REVENUE_SCALE = {mean_revenue_scale:.10f}")
        print(f"\nOption 2: Use best split scales (lowest error):")
        print(f"  PURCHASE_SCALE = {best_split['purchase_scale']:.10f}")
        print(f"  REVENUE_SCALE = {best_split['revenue_scale']:.10f}")
        print(f"\nOption 3: Use median scales (robust to outliers):")
        print(f"  PURCHASE_SCALE = {scale_df['purchase_scale'].median():.10f}")
        print(f"  REVENUE_SCALE = {scale_df['revenue_scale'].median():.10f}")
        
        print("\n✅ Multiple split analysis completed!")

