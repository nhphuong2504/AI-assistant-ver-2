"""Prospective inactivity survival: freeze features BEFORE follow-up starts.

An event is the first future crossing of an inactivity threshold, including
customers who later reactivate. Durations are measured from the landmark.
"""
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from .time_utils import cutoff_timestamp

FEATURES = ["log_n_orders", "log_monetary_value", "log_product_diversity",
            "recency_from_cutoff", "tenure_days"]


def snapshot(transactions, cutoff_date, inactivity_days=90):
    if inactivity_days <= 0:
        raise ValueError("inactivity_days must be positive")
    cutoff = cutoff_timestamp(cutoff_date)
    tx = transactions.copy()
    tx["invoice_date"] = pd.to_datetime(tx["invoice_date"]).dt.normalize()
    tx = tx[(tx.invoice_date <= cutoff) & tx.customer_id.notna() & (tx.revenue > 0)]
    orders = tx.groupby(["customer_id", "invoice_no"]).agg(
        date=("invoice_date", "min"), value=("revenue", "sum"))
    g = orders.groupby(level=0)
    out = g.agg(first=("date", "min"), last=("date", "max"),
                n_orders=("value", "size"), monetary_value=("value", "mean"))
    day = cutoff.normalize()
    out["tenure_days"] = (day - out["first"]).dt.days.astype(float)
    out["recency_from_cutoff"] = (day - out["last"]).dt.days.astype(float)
    out["product_diversity"] = tx.groupby("customer_id").stock_code.nunique()
    for raw in ["n_orders", "monetary_value", "product_diversity"]:
        out["log_" + raw] = np.log1p(out[raw])
    return out[(out.recency_from_cutoff < inactivity_days) & (out.tenure_days > 0)].copy()


def outcomes(transactions, cutoff_date, end_date, inactivity_days=90):
    """Keep snapshot features immutable; only labels use future transactions."""
    start = pd.Timestamp(cutoff_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if end <= start:
        raise ValueError("end_date must follow cutoff_date")
    out = snapshot(transactions, cutoff_date, inactivity_days)
    tx = transactions.copy()
    tx["invoice_date"] = pd.to_datetime(tx["invoice_date"]).dt.normalize()
    future = tx[(tx.invoice_date > start) & (tx.invoice_date <= end)
                & (tx.revenue > 0)].groupby("customer_id").invoice_date.unique()
    labels = []
    events = []
    for cid, row in out.iterrows():
        last = row["last"]
        event_date = None
        for purchase in sorted(future.get(cid, [])):
            crossing = last + pd.Timedelta(days=inactivity_days)
            if pd.Timestamp(purchase) >= crossing:
                event_date = crossing
                break
            last = pd.Timestamp(purchase)
        if event_date is None and last + pd.Timedelta(days=inactivity_days) <= end:
            event_date = last + pd.Timedelta(days=inactivity_days)
        labels.append(((event_date or end) - start).days)
        events.append(int(event_date is not None))
    out["duration"] = labels
    # A crossing exactly on the end date is an event, not censoring.
    out["event"] = events
    return out


def fit_landmark_cox(transactions, as_of, inactivity_days=90, followup_days=90):
    """Fit one completed historical landmark; never train on scoring outcomes."""
    end = pd.Timestamp(as_of).normalize()
    start = end - pd.Timedelta(days=followup_days)
    train = outcomes(transactions, start, end, inactivity_days)
    if len(train) < 30 or train.event.sum() < 5:
        raise ValueError("Insufficient completed historical follow-up for landmark Cox")
    model = CoxPHFitter(penalizer=0.1)
    model.fit(train[FEATURES + ["duration", "event"]], "duration", "event")
    model.landmark_followup_days_ = followup_days
    model.landmark_training_end_ = end.isoformat()
    model.inactivity_days_ = inactivity_days
    return model, train


def probabilities(model, features, horizon_days=90):
    if not 0 < horizon_days <= model.landmark_followup_days_:
        raise ValueError("Horizon exceeds completed landmark follow-up (1–90 days supported)")
    sf = model.predict_survival_function(features[FEATURES])
    # Survival curves are step functions, not linear interpolants.
    eligible = sf.index <= horizon_days
    survival = sf.loc[eligible].iloc[-1].to_numpy() if eligible.any() else np.ones(len(features))
    return pd.Series(np.clip(1 - survival, 0, 1), index=features.index)


def first_inactivity_table(transactions, cutoff_date, inactivity_days=90):
    """First observed inactivity episode from first observed purchase, for KM.

    Reactivation does not undo a historical first event. This differs from
    current active/inactive status and is not permanent customer dropout.
    """
    cutoff = cutoff_timestamp(cutoff_date).normalize()
    tx = transactions.copy()
    tx["invoice_date"] = pd.to_datetime(tx.invoice_date).dt.normalize()
    tx = tx[(tx.invoice_date <= cutoff) & tx.customer_id.notna() & (tx.revenue > 0)]
    rows = []
    for cid, group in tx.groupby("customer_id"):
        dates = sorted(group.invoice_date.unique())
        first, last = pd.Timestamp(dates[0]), pd.Timestamp(dates[0])
        event_date = None
        for date in dates[1:]:
            if pd.Timestamp(date) >= last + pd.Timedelta(days=inactivity_days):
                event_date = last + pd.Timedelta(days=inactivity_days)
                break
            last = pd.Timestamp(date)
        if event_date is None and last + pd.Timedelta(days=inactivity_days) <= cutoff:
            event_date = last + pd.Timedelta(days=inactivity_days)
        rows.append({"customer_id": cid, "duration": ((event_date or cutoff)-first).days,
                     "event": int(event_date is not None)})
    return pd.DataFrame(rows)
