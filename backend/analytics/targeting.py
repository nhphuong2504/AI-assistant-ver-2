"""Budgeted contact ranking. Exposure measures are not causal campaign returns."""
import hashlib
import math
import numpy as np
import pandas as pd


def tie_keys(customer_index):
    return np.array([int.from_bytes(hashlib.sha256(f"42:{cid}".encode()).digest()[:8], "big") for cid in customer_index], dtype=np.uint64)


def policy_scores(features, churn_probability, predicted_spend):
    return pd.DataFrame({"recency": features.recency_from_cutoff,
        "historical_spend": features.spend_90, "churn": churn_probability,
        "clv": predicted_spend, "churn_x_clv": churn_probability*predicted_spend,
        "churn_x_historical_spend": churn_probability*features.spend_90}, index=features.index)


def rank_order(scores, keys):
    return np.lexsort((keys, -np.asarray(scores, dtype=float)))


def budget_metrics(events, historical_value, scores, budget, keys):
    y, value = np.asarray(events), np.asarray(historical_value)
    k = max(1, math.ceil(budget*len(y)))
    idx = rank_order(scores, keys)[:k]
    total_exposure = (y*value).sum()
    return {"contacts": k, "contact_fraction": k/len(y), "events_identified": int(y[idx].sum()),
        "precision": float(y[idx].mean()), "recall": float(y[idx].sum()/y.sum()) if y.sum() else 0.,
        "historical_value_weighted_recall": float((y[idx]*value[idx]).sum()/total_exposure) if total_exposure else 0.,
        "historical_sales_exposure_identified_gbp": float((y[idx]*value[idx]).sum())}


def evaluate_policies(events, historical_value, scores, budgets=(.05, .1, .2)):
    keys = tie_keys(scores.index)
    n = len(events)
    result = {}
    for name in ["random_expected", *scores.columns]:
        rows = []
        for budget in budgets:
            if name == "random_expected":
                k = max(1, math.ceil(budget*n)); fraction = k/n
                row = {"contacts": k, "contact_fraction": fraction,
                    "events_identified": float(np.sum(events)*fraction), "precision": float(np.mean(events)),
                    "recall": fraction, "historical_value_weighted_recall": fraction,
                    "historical_sales_exposure_identified_gbp": float(np.sum(np.asarray(events)*np.asarray(historical_value))*fraction)}
            else:
                row = budget_metrics(events, historical_value, scores[name], budget, keys)
            rows.append({"budget": budget, **row})
        result[name] = rows
    return result


def minimum_contacts(events, historical_value, scores, target=.5):
    weighted = np.asarray(events)*np.asarray(historical_value)
    if weighted.sum() <= 0:
        return None
    idx = rank_order(scores, tie_keys(scores.index))
    count = int(np.searchsorted(np.cumsum(weighted[idx]), target*weighted.sum())+1)
    return {"target_weighted_recall": target, "contacts": count, "contact_fraction": count/len(events)}


def policy_bootstrap(events, historical_value, candidate, baseline, budgets=(.05, .1, .2), repeats=1000):
    """Resample customers and recompute capacity-limited rankings each time."""
    y, value, a, b = map(np.asarray, (events, historical_value, candidate, baseline))
    keys = tie_keys(candidate.index)
    rng = np.random.default_rng(42)
    samples = {budget: {"recall": [], "historical_value_weighted_recall": []} for budget in budgets}
    for _ in range(repeats):
        idx = rng.integers(0, len(y), len(y))
        for budget in budgets:
            ma = budget_metrics(y[idx], value[idx], a[idx], budget, keys[idx])
            mb = budget_metrics(y[idx], value[idx], b[idx], budget, keys[idx])
            for metric in samples[budget]:
                samples[budget][metric].append(ma[metric]-mb[metric])
    return {str(budget): {metric: {"difference": budget_metrics(y, value, a, budget, keys)[metric]-budget_metrics(y, value, b, budget, keys)[metric],
        "ci95": np.quantile(values, [.025, .975]).tolist(), "resamples": repeats}
        for metric, values in metrics.items()} for budget, metrics in samples.items()}
