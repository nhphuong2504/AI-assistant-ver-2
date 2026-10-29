"""
Customer Lifetime Value (CLV) modeling using BG/NBD and Gamma-Gamma models.

This module implements the classic CLV framework:
- BG/NBD: Models purchase frequency (how often customers buy)
- Gamma-Gamma: Models monetary value (how much customers spend per order)
"""
import pandas as pd
import numpy as np
from scipy.integrate import quad
from scipy.special import hyp2f1
from dataclasses import dataclass
from lifetimes import BetaGeoFitter, GammaGammaFitter
from .time_utils import cutoff_timestamp

# Neutral defaults: no calibration on holdout outcomes. Retained for API compatibility.
PURCHASE_SCALE = 1.0
REVENUE_SCALE = 1.0


@dataclass
class CLVResult:
    """Container for fitted CLV models and RFM data.
    
    Attributes:
        rfm: RFM (Recency, Frequency, Monetary) data frame for all customers
        bgnbd: Fitted BG/NBD model for purchase frequency prediction
        gg: Fitted Gamma-Gamma model for average order value prediction
    """
    rfm: pd.DataFrame
    bgnbd: BetaGeoFitter
    gg: GammaGammaFitter


def build_rfm(
    transactions: pd.DataFrame, 
    cutoff_date: str, 
    cap_percentile: float = 100 
) -> pd.DataFrame:
    """Build RFM (Recency, Frequency, Monetary) table from transaction data.
    
    RFM fields:
    - frequency: Number of repeat purchase-days (n_purchase_days - 1)
    - recency: Days between first and last purchase
    - T: Days between first purchase and cutoff date
    - monetary_value: Mean repeat purchase-day spend, excluding first day (NaN for one-time buyers)
    
    Args:
        transactions: Transaction data with columns: customer_id, invoice_no, 
                     invoice_date, revenue
        cutoff_date: Date string to use as analysis cutoff (YYYY-MM-DD)
        cap_percentile: Percentile to cap order values (99.5 = cap at 99.5th percentile)
                       Set to 100 to disable capping
    
    Returns:
        DataFrame indexed by customer_id with columns: frequency, recency, T, monetary_value
    """
    df = transactions.copy()
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    if not 0 < cap_percentile <= 100:
        raise ValueError("cap_percentile must be in (0, 100]")
    cutoff = cutoff_timestamp(cutoff_date)
    
    # ========================================================================
    # Data cleaning: filter to valid transactions before cutoff
    # ========================================================================
    df = df[df["invoice_date"] <= cutoff]
    df = df[df["customer_id"].notna()]
    df = df[df["revenue"] > 0]
    
    # ========================================================================
    # Invoice-level aggregation: sum revenue per invoice
    # ========================================================================
    orders = (
        df.groupby(["customer_id", "invoice_no"], as_index=False)
        .agg(
            order_date=("invoice_date", "min"),
            order_value=("revenue", "sum"),
        )
    )
    # A daily model counts purchase occasions, not multiple invoices on one day.
    # Match lifetimes' daily summary convention and aggregate daily spend.
    orders["order_date"] = orders["order_date"].dt.normalize()
    orders = orders.groupby(["customer_id", "order_date"], as_index=False).agg(
        order_value=("order_value", "sum")
    )
    
    # Cap extreme order values to stabilize Gamma-Gamma model
    if len(orders) > 0 and cap_percentile < 100:
        cap_threshold = orders["order_value"].quantile(cap_percentile / 100)
        orders["order_value"] = orders["order_value"].clip(upper=cap_threshold)
    
    # ========================================================================
    # Customer-level aggregation
    # ========================================================================
    customer_groups = orders.groupby("customer_id", as_index=True)
    
    first_purchase = customer_groups["order_date"].min()
    last_purchase = customer_groups["order_date"].max()
    n_orders = customer_groups.size()
    
    # ========================================================================
    # Calculate RFM metrics (lifetimes-style, in days)
    # ========================================================================
    # T: time from first purchase to cutoff (customer "age")
    T = (cutoff.normalize() - first_purchase).dt.days.astype(float)
    
    # recency: time from first to last purchase
    recency = (last_purchase - first_purchase).dt.days.astype(float)
    
    # frequency: repeat purchases (BG/NBD uses n_orders - 1)
    frequency = (n_orders - 1).astype(float)
    
    # monetary_value: average order value (only for repeat buyers)
    repeat_orders = orders[orders["order_date"] > orders["customer_id"].map(first_purchase)]
    monetary_value = repeat_orders.groupby("customer_id")["order_value"].mean().reindex(n_orders.index)
    monetary_value = monetary_value.where(n_orders >= 2, np.nan)
    
    # ========================================================================
    # Build RFM DataFrame and filter invalid records
    # ========================================================================
    rfm = pd.DataFrame({
        "frequency": frequency,
        "recency": recency,
        "T": T,
        "monetary_value": monetary_value,
    })
    
    # Keep only valid records: T > 0, recency >= 0, frequency >= 0
    rfm = rfm[(rfm["T"] > 0) & (rfm["recency"] >= 0) & (rfm["frequency"] >= 0)]
    
    return rfm


def fit_models(
    rfm: pd.DataFrame,
    penalizer: float = 0.0,
    bgnbd_penalizer: float = None
) -> CLVResult:
    """Fit BG/NBD and Gamma-Gamma models on RFM data.
    
    Args:
        rfm: RFM DataFrame from build_rfm()
        penalizer: Penalizer for Gamma-Gamma model (default: 0; require q > 1)
        bgnbd_penalizer: Penalizer for BG/NBD model (default: 0.05)
                        If None, uses 0.05 to handle extreme parameter values
    
    Returns:
        CLVResult with fitted models and original RFM data
    
    Raises:
        ValueError: If there are no repeat buyers for Gamma-Gamma fitting
    """
    # Default BG/NBD penalizer (higher to handle extreme parameter values)
    if bgnbd_penalizer is None:
        bgnbd_penalizer = 0.05
    if rfm.empty:
        raise ValueError("Cannot fit CLV models on empty RFM data")
    
    # Fit BG/NBD model (all customers)
    bgnbd = BetaGeoFitter(penalizer_coef=bgnbd_penalizer)
    bgnbd.fit(rfm["frequency"], rfm["recency"], rfm["T"])
    
    # Filter to repeat buyers for Gamma-Gamma fitting
    rfm_gg = rfm[(rfm["frequency"] > 0) & (rfm["monetary_value"] > 0)].copy()
    if rfm_gg.empty:
        raise ValueError("Gamma-Gamma requires positive-spend repeat buyers")
    
    # Fit Gamma-Gamma model (repeat buyers only)
    gg = GammaGammaFitter(penalizer_coef=penalizer)
    gg.fit(rfm_gg["frequency"], rfm_gg["monetary_value"], q_constraint=True)
    if gg.params_["q"] <= 1 + 1e-8:
        raise ValueError("Gamma-Gamma requires q > 1 for a finite population mean; reduce monetary-model penalization")
    
    return CLVResult(rfm=rfm, bgnbd=bgnbd, gg=gg)


def predict_clv(
    clv: CLVResult,
    horizon_days: int = 180,
    discount_rate: float = 0.0,
    scale_to_target_purchases: float | None = None,
    scale_to_target_revenue: float | None = None,
    aov_fallback: str = "none",  # "none" | "global_mean" | "zero"
) -> pd.DataFrame:
    """Predict Customer Lifetime Value for all customers.
    
    Output columns:
    - customer_id: Customer identifier
    - frequency, recency, T, monetary_value: Input RFM values
    - pred_purchases: Expected number of purchases in horizon
    - pred_aov: Expected average order value
    - clv: Predicted CLV (pred_purchases * pred_aov)
    - gg_status: "Eligible for GG" or "Not eligible for GG"
    - has_valid_clv: True if CLV is not NaN
    
    Args:
        clv: Fitted CLV models from fit_models()
        horizon_days: Prediction horizon in days (default: 180)
        discount_rate: Discount rate for future cash flows (default: 0.0 = no discounting)
        scale_to_target_purchases: Optional scenario total; NEVER supply evaluation outcomes
        scale_to_target_revenue: Optional scenario total; NEVER supply evaluation outcomes
        aov_fallback: Strategy for non-GG-eligible customers:
                     - "none": pred_aov = NaN (default)
                     - "global_mean": Use mean AOV of repeat buyers
                     - "zero": Use 0.0
    
    Returns:
        DataFrame with predictions for all customers
    """
    if horizon_days <= 0 or discount_rate < 0:
        raise ValueError("horizon_days must be positive and discount_rate nonnegative")
    if aov_fallback not in {"none", "global_mean", "zero"}:
        raise ValueError("Unknown aov_fallback")
    rfm = clv.rfm.copy()
    
    # ========================================================================
    # Step 1: Predict expected purchases (BG/NBD - all customers)
    # ========================================================================
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        rfm["pred_purchases"] = clv.bgnbd.conditional_expected_number_of_purchases_up_to_time(
            horizon_days, rfm["frequency"], rfm["recency"], rfm["T"]
        )
    bad = ~np.isfinite(rfm.pred_purchases) | (rfm.pred_purchases < 0)
    # Equivalent expected-intensity integral avoids cancellation in lifetimes'
    # closed form near a=0 or a=1. This does not impute/drop difficult customers.
    r, alpha, a, b = [float(clv.bgnbd.params_[k]) for k in ["r", "alpha", "a", "b"]]
    for cid, row in rfm.loc[bad].iterrows():
        shape, rate = r + row.frequency, alpha + row["T"]
        alive = float(np.asarray(clv.bgnbd.conditional_probability_alive(row.frequency, row.recency, row["T"])).item())
        integral, _ = quad(lambda u: shape / rate * hyp2f1(
            shape + 1, a, a + b + row.frequency, -u / rate), 0, horizon_days)
        rfm.loc[cid, "pred_purchases"] = alive * integral
    if not np.isfinite(rfm.pred_purchases).all() or (rfm.pred_purchases < 0).any():
        raise ValueError("BG/NBD produced invalid expected purchase counts")
    
    # ========================================================================
    # Step 2: Predict expected AOV (Gamma-Gamma - repeat buyers only)
    # ========================================================================
    gg_eligible_mask = (rfm["frequency"] > 0) & (rfm["monetary_value"] > 0)
    rfm["pred_aov"] = np.nan
    
    rfm.loc[gg_eligible_mask, "pred_aov"] = clv.gg.conditional_expected_average_profit(
        rfm.loc[gg_eligible_mask, "frequency"],
        rfm.loc[gg_eligible_mask, "monetary_value"],
    )
    
    # Apply fallback strategy for non-eligible customers
    if aov_fallback == "global_mean":
        global_aov = rfm.loc[gg_eligible_mask, "monetary_value"].mean()
        rfm.loc[~gg_eligible_mask, "pred_aov"] = global_aov
    elif aov_fallback == "zero":
        rfm.loc[~gg_eligible_mask, "pred_aov"] = 0.0
    
    rfm["gg_status"] = np.where(
        gg_eligible_mask, 
        "Eligible for GG", 
        "Not eligible for GG"
    )
    
    # ========================================================================
    # Step 3: Calculate CLV (purchases * AOV)
    # ========================================================================
    rfm["clv"] = rfm["pred_purchases"] * rfm["pred_aov"]
    
    # ========================================================================
    # Step 4: Optional scaling
    # ========================================================================
    # Scale purchases to match target total
    if scale_to_target_purchases is not None:
        total_pred_purchases = rfm["pred_purchases"].sum()
        if total_pred_purchases > 0:
            purchase_scale = scale_to_target_purchases / total_pred_purchases
            rfm["pred_purchases"] *= purchase_scale
            rfm["clv"] = rfm["pred_purchases"] * rfm["pred_aov"]
    
    # Scale revenue to match target total
    if scale_to_target_revenue is not None:
        total_pred_revenue = rfm["clv"].sum(skipna=True)
        if total_pred_revenue > 0:
            revenue_scale = scale_to_target_revenue / total_pred_revenue
            rfm.loc[rfm["pred_aov"].notna(), "pred_aov"] *= revenue_scale
            rfm["clv"] = rfm["pred_purchases"] * rfm["pred_aov"]
    
    # ========================================================================
    # Step 5: Finalize output
    # ========================================================================
    rfm["has_valid_clv"] = rfm["clv"].notna()
    
    # Apply discounting (simple present value calculation)
    if discount_rate > 0:
        rfm["clv"] = rfm["clv"] / (1 + discount_rate)
    
    # Convert index to column and return
    output = rfm.reset_index().rename(columns={"index": "customer_id"})
    return output
