"""Serve the development-selected policy; no evaluation labels are returned."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .challengers import (logistic, boosting, TwoPartSpend, PROTOCOL)
from .customer_features import customer_features, future_spend, FULL_FEATURES, WINDOW_FEATURES
from .landmark import FEATURES, outcomes, fit_landmark_cox, probabilities
from .clv import build_rfm, fit_models, predict_clv
from .targeting import policy_scores, rank_order, tie_keys

SELECTION_PATH = Path(__file__).resolve().parents[1]/"reports/model_selection.json"
_cache = {}


def read_selection():
    if not SELECTION_PATH.exists():
        raise RuntimeError("Run python backend/evaluate_decisions.py to create development-only model selection")
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    protocol_sha = hashlib.sha256((Path(__file__).parent/"decision_protocol.json").read_bytes()).hexdigest()
    if selection["protocol_sha256"] != protocol_sha:
        raise RuntimeError("Model selection is stale relative to decision protocol; rerun evaluation")
    if selection["churn_model"] not in PROTOCOL["churn_candidates"] or selection["spend_model"] not in PROTOCOL["spend_candidates"] or selection["policy"] not in PROTOCOL["policies"] or selection["policy"] == "random_expected":
        raise RuntimeError("Invalid selected model/policy")
    return selection


def selected_predictions(transactions, cutoff_date, selection=None):
    """Fit selected families on completed historical labels, then score cutoff.

    A content fingerprint prevents reusing fits on different same-length data.
    Feature construction and fitted models see only history through cutoff.
    """
    selection = read_selection() if selection is None else selection
    cutoff = pd.Timestamp(cutoff_date).normalize()
    observed = pd.to_datetime(transactions.invoice_date)
    if cutoff > observed.max().normalize() or cutoff < observed.min().normalize()+pd.Timedelta(days=90):
        raise ValueError("Cutoff must be observed and have at least 90 days of completed historical follow-up")
    # Trim future rows before fingerprint, features, or fitting.
    tx = transactions.loc[observed.dt.normalize() <= cutoff].copy()
    data_sha = hashlib.sha256(pd.util.hash_pandas_object(tx, index=True).values.tobytes()).hexdigest()
    key = (cutoff.isoformat(), data_sha, json.dumps(selection, sort_keys=True))
    if key in _cache:
        return _cache[key].copy(), selection
    historical = cutoff-pd.Timedelta(days=90)
    train_x = customer_features(tx, historical)
    train_y = outcomes(tx, historical, cutoff)
    all_features = customer_features(tx, cutoff)
    active = all_features[all_features.recency_from_cutoff < 90].copy()
    name = selection["churn_model"]
    with threadpool_limits(limits=1):
        if name == "landmark_cox":
            fitted, _ = fit_landmark_cox(tx, cutoff)
            p = probabilities(fitted, active).to_numpy()
        else:
            columns = ["recency_from_cutoff"] if name == "recency_logistic" else FEATURES
            if name == "logistic_windows": columns = FEATURES+WINDOW_FEATURES
            if name in ("logistic_full", "boosting_full"): columns = FULL_FEATURES
            model = boosting() if name.startswith("boosting") else logistic()
            model.fit(train_x.loc[train_y.index, columns], train_y.event)
            p = model.predict_proba(active[columns])[:, 1]
        spend_name = selection["spend_model"]
        if spend_name == "last90":
            spend = active.spend_90.to_numpy()
        elif spend_name == "bgnbd_gamma_gamma":
            clv = fit_models(build_rfm(tx, cutoff))
            spend = predict_clv(clv, 90, aov_fallback="global_mean").set_index("customer_id").loc[active.index, "clv"].to_numpy()
        else:
            labels = future_spend(tx, historical, customer_index=train_x.index)
            model = TwoPartSpend(spend_name == "two_part_boosting").fit(train_x, labels.actual_revenue)
            spend = model.predict(active)
    scores = policy_scores(active, pd.Series(p, index=active.index), pd.Series(spend, index=active.index))
    result = active[["recency_from_cutoff", "spend_90"]].rename(columns={"spend_90": "historical_spend_90_gbp"})
    result["churn_probability_90"] = p
    result["predicted_gross_spend_90_gbp"] = spend
    result["priority_score"] = scores[selection["policy"]]
    result = result.iloc[rank_order(result.priority_score, tie_keys(result.index))]
    result["priority_rank"] = np.arange(1, len(result)+1)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("Invalid selected-model predictions")
    # Bounded process-local cache: reproducibility does not require persistent pickle files.
    if len(_cache) >= 8:
        _cache.pop(next(iter(_cache)))
    _cache[key] = result.copy()
    return result, selection


def retention_targets(transactions, cutoff_date, contact_fraction=.2):
    if not 0 < contact_fraction <= 1:
        raise ValueError("contact_fraction must be in (0, 1]")
    predictions, selection = selected_predictions(transactions, cutoff_date)
    contacts = math.ceil(len(predictions)*contact_fraction)
    return {"cutoff_date": pd.Timestamp(cutoff_date).date().isoformat(), "horizon_days": 90,
        "inactivity_days": 90, "active_customers": len(predictions), "contacts": contacts,
        "requested_contact_fraction": contact_fraction,
        "actual_contact_fraction": contacts/len(predictions) if len(predictions) else 0.,
        "churn_model": selection["churn_model"], "spend_model": selection["spend_model"], "policy": selection["policy"],
        "score_definition": "Predicted inactivity probability times trailing-90-day gross sales" if selection["policy"] == "churn_x_historical_spend" else selection["policy"],
        "limitations": "Retrospectively evaluated prioritization. Historical gross-sales exposure is not recoverable revenue, profit, causal uplift or guaranteed campaign ROI. Inactivity may be followed by reactivation.",
        "customers": predictions.head(contacts).reset_index().to_dict(orient="records")}
