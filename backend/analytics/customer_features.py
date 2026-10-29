"""Cutoff-safe customer features, shared by training, evaluation and serving."""
import numpy as np
import pandas as pd
from .landmark import FEATURES

WINDOW_FEATURES = [f"log_{kind}_{days}" for days in (30, 60, 90) for kind in ("purchase_days", "spend")]
WINDOW_FEATURES += ["purchase_trend_30", "spend_trend_30"]
CADENCE_FEATURES = ["log_mean_gap", "gap_cv", "recency_to_mean_gap", "has_repeat_day", "prior_inactivity_episodes"]
FULL_FEATURES = FEATURES + WINDOW_FEATURES + CADENCE_FEATURES


def customer_features(transactions, cutoff_date):
    """One row per existing customer with >0 days of observed history.

    All date windows are daily, (cutoff - window, cutoff]. A zero-history
    customer is excluded consistently with the original RFM cohort.
    """
    cutoff = pd.Timestamp(cutoff_date).normalize()
    tx = transactions.copy()
    tx["day"] = pd.to_datetime(tx.invoice_date).dt.normalize()
    tx = tx[(tx.day <= cutoff) & tx.customer_id.notna() & (tx.revenue > 0)]
    orders = tx.groupby(["customer_id", "invoice_no"]).agg(day=("day", "min"), value=("revenue", "sum"))
    out = orders.groupby(level=0).agg(first=("day", "min"), last=("day", "max"),
        n_orders=("value", "size"), monetary_value=("value", "mean"))
    out["tenure_days"] = (cutoff-out["first"]).dt.days.astype(float)
    out["recency_from_cutoff"] = (cutoff-out["last"]).dt.days.astype(float)
    out["product_diversity"] = tx.groupby("customer_id").stock_code.nunique()
    for col in ("n_orders", "monetary_value", "product_diversity"):
        out["log_"+col] = np.log1p(out[col])
    daily = tx.groupby(["customer_id", "day"]).revenue.sum().rename("value").reset_index()
    for days in (30, 60, 90):
        group = daily[daily.day > cutoff-pd.Timedelta(days=days)].groupby("customer_id")
        out[f"purchase_days_{days}"] = group.size().reindex(out.index, fill_value=0)
        out[f"spend_{days}"] = group.value.sum().reindex(out.index, fill_value=0)
        for kind in ("purchase_days", "spend"):
            out[f"log_{kind}_{days}"] = np.log1p(out[f"{kind}_{days}"])
    for kind in ("purchase_days", "spend"):
        prior = out[f"{kind}_60"]-out[f"{kind}_30"]
        out[f"{kind if kind == 'spend' else 'purchase'}_trend_30"] = np.log1p(out[f"{kind}_30"])-np.log1p(prior)
    gaps = daily.sort_values(["customer_id", "day"]).groupby("customer_id").day.diff().dt.days
    daily["gap"] = gaps
    cadence = daily.groupby("customer_id").gap.agg(["mean", "std", "count"])
    # Explicit prior (90 days) for one-time buyers; not estimated from future cohorts.
    mean_gap = cadence["mean"].reindex(out.index).fillna(90.)
    out["log_mean_gap"] = np.log1p(mean_gap)
    out["gap_cv"] = cadence["std"].reindex(out.index).fillna(0.)/mean_gap.clip(lower=1)
    out["recency_to_mean_gap"] = out.recency_from_cutoff/mean_gap.clip(lower=1)
    out["has_repeat_day"] = (cadence["count"].reindex(out.index).fillna(0) > 0).astype(float)
    out["prior_inactivity_episodes"] = daily.assign(inactive=daily.gap >= 90).groupby("customer_id").inactive.sum().reindex(out.index, fill_value=0)
    out = out[out.tenure_days > 0].sort_index()
    if not np.isfinite(out[FULL_FEATURES].to_numpy()).all():
        raise ValueError("Nonfinite customer features")
    return out


def future_spend(transactions, cutoff_date, horizon_days=90, customer_index=None):
    """Outcome-only aggregation; never call this to construct serving features."""
    cutoff = pd.Timestamp(cutoff_date).normalize()
    end = cutoff+pd.Timedelta(days=horizon_days)
    tx = transactions.copy()
    tx["day"] = pd.to_datetime(tx.invoice_date).dt.normalize()
    tx = tx[(tx.day > cutoff) & (tx.day <= end) & (tx.revenue > 0)]
    daily = tx.groupby(["customer_id", "day"]).revenue.sum()
    out = pd.DataFrame({"actual_purchases": daily.groupby(level=0).size(), "actual_revenue": daily.groupby(level=0).sum()})
    return out.reindex(customer_index, fill_value=0) if customer_index is not None else out
