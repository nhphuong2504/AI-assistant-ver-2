"""Reproducible offline evaluation. Run: python backend/evaluate.py.

Development: cutoff 2011-06-09, outcomes through 2011-09-07 (90d).
Final test: cutoff 2011-09-09, outcomes through 2011-12-08 (90d).
No outcome-based scaling, no new-customer revenue in existing-cohort targets.
"""
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from scipy.optimize import minimize
from scipy.special import expit
from lifelines.statistics import proportional_hazard_test
from analytics.clv import build_rfm, fit_models, predict_clv
from analytics.landmark import fit_landmark_cox, outcomes, probabilities, FEATURES
from etl.load_online_retail import load_csv, clean_and_flag, RAW_PATH

OUT = Path(__file__).resolve().parent / "reports"
SPLITS = [("development", "2011-06-09"), ("final_test", "2011-09-09")]


def regression_metrics(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    if not np.isfinite(pred).all():
        raise ValueError("Evaluation requires finite predictions for every cohort customer")
    k = max(1, math.ceil(len(y) * .2))
    order = np.argsort(-pred, kind="stable")
    share = y[order[:k]].sum() / y.sum() if y.sum() else None
    return {"n": len(y), "mae": float(np.mean(abs(pred-y))),
            "rmse": float(np.sqrt(np.mean((pred-y)**2))),
            "actual_total": float(y.sum()), "predicted_total": float(pred.sum()),
            "aggregate_bias_pct": float(100*(pred.sum()/y.sum()-1)) if y.sum() else None,
            "top20_capture": float(share) if share is not None else None,
            "top20_lift": float(share/(k/len(y))) if share is not None else None}


def binary_metrics(y, p):
    y, p = np.asarray(y, dtype=int), np.asarray(p)
    n1, n0 = y.sum(), len(y)-y.sum()
    auc = (rankdata(p)[y == 1].sum()-n1*(n1+1)/2)/(n1*n0)
    order = np.argsort(-p, kind="stable")
    ys, ps = y[order], p[order]
    # Average precision at distinct thresholds, including all tied predictions.
    ends = np.r_[np.flatnonzero(np.diff(ps) != 0), len(ps)-1]
    tp = np.cumsum(ys)[ends]
    recall = tp/n1
    ap = np.sum(np.diff(np.r_[0, recall]) * tp/(ends+1))
    k = math.ceil(.2*len(y))
    return {"n": len(y), "events": int(n1), "prevalence": float(y.mean()),
            "roc_auc": float(auc), "average_precision": float(ap),
            "brier": float(np.mean((p-y)**2)), "mean_probability": float(p.mean()),
            "top20_precision": float(ys[:k].mean()),
            "top20_recall": float(ys[:k].sum()/n1)}


def paired_bootstrap(y, candidate, baseline, kind, repeats=1000):
    """Customer-level paired percentile CI; conditional on this temporal fold."""
    rng = np.random.default_rng(42)
    y, candidate, baseline = map(np.asarray, (y, candidate, baseline))
    deltas = []
    for _ in range(repeats):
        idx = rng.integers(0, len(y), len(y))
        if kind == "mae":
            delta = np.mean(abs(candidate[idx]-y[idx])-abs(baseline[idx]-y[idx]))
        else:
            delta = np.mean((candidate[idx]-y[idx])**2-(baseline[idx]-y[idx])**2)
        deltas.append(delta)
    return {"metric": kind, "candidate_minus_baseline": float(np.mean(
        abs(candidate-y)-abs(baseline-y) if kind == "mae" else (candidate-y)**2-(baseline-y)**2)),
        "ci95": np.quantile(deltas, [.025, .975]).tolist(), "resamples": repeats}


def evaluate_clv(tx, cutoff, end):
    rfm = build_rfm(tx, cutoff)
    models = fit_models(rfm)
    predicted = predict_clv(models, horizon_days=90, aov_fallback="global_mean").set_index("customer_id")
    dates = pd.to_datetime(tx.invoice_date).dt.normalize()
    future = tx[(dates > pd.Timestamp(cutoff)) & (dates <= end)]
    future = future[future.customer_id.isin(rfm.index)]
    daily = future.assign(day=pd.to_datetime(future.invoice_date).dt.normalize()).groupby(["customer_id", "day"]).revenue.sum()
    actual_count = daily.groupby(level=0).size().reindex(rfm.index, fill_value=0)
    actual_rev = daily.groupby(level=0).sum().reindex(rfm.index, fill_value=0)
    history = tx[(dates <= pd.Timestamp(cutoff)) & (dates > pd.Timestamp(cutoff)-pd.Timedelta(days=90))]
    hist_daily = history.assign(day=pd.to_datetime(history.invoice_date).dt.normalize()).groupby(["customer_id", "day"]).revenue.sum()
    naive_count = hist_daily.groupby(level=0).size().reindex(rfm.index, fill_value=0)
    naive_rev = hist_daily.groupby(level=0).sum().reindex(rfm.index, fill_value=0)
    result = {"customers": len(rfm), "repeat_customers": int((rfm.frequency > 0).sum()),
        "zero_future_purchase_customers": int((actual_count == 0).sum()),
        "purchase_unit": "customer purchase-day", "currency": "GBP gross positive sales, not profit or net revenue",
        "purchase_models": {}, "revenue_models": {},
        "bgnbd_parameters": models.bgnbd.params_.to_dict(), "gamma_gamma_parameters": models.gg.params_.to_dict(),
        "frequency_monetary_spearman": float(spearmanr(rfm.loc[rfm.frequency > 0, "frequency"], rfm.loc[rfm.frequency > 0, "monetary_value"]).statistic)}
    for label, count, revenue in [("bgnbd_gamma_gamma", predicted.pred_purchases, predicted.clv),
                                  ("last90", naive_count, naive_rev),
                                  ("zero", np.zeros(len(rfm)), np.zeros(len(rfm)))]:
        result["purchase_models"][label] = regression_metrics(actual_count, count)
        result["revenue_models"][label] = regression_metrics(actual_rev, revenue)
    result["paired_purchase_mae_vs_last90"] = paired_bootstrap(actual_count, predicted.pred_purchases, naive_count, "mae")
    result["paired_revenue_mae_vs_last90"] = paired_bootstrap(actual_rev, predicted.clv, naive_rev, "mae")
    rows = pd.DataFrame({"actual_purchases": actual_count, "actual_revenue": actual_rev,
                         "pred_purchases": predicted.pred_purchases, "pred_revenue": predicted.clv,
                         "baseline_purchases": naive_count, "baseline_revenue": naive_rev,
                         "repeat_customer": rfm.frequency > 0})
    result["segments"] = {str(bool(repeat)): {"revenue": regression_metrics(g.actual_revenue, g.pred_revenue),
        "purchases": regression_metrics(g.actual_purchases, g.pred_purchases)} for repeat, g in rows.groupby("repeat_customer")}
    return result, rows


def evaluate_churn(tx, cutoff, end, inactivity_days=90):
    model, train = fit_landmark_cox(tx, cutoff, inactivity_days)
    test = outcomes(tx, cutoff, end, inactivity_days)
    p = probabilities(model, test)
    bins = [-1, 14, 30, 60, inactivity_days]
    bins = sorted(set(b for b in bins if b < inactivity_days) | {inactivity_days})
    train_bins = pd.cut(train.recency_from_cutoff, bins)
    test_bins = pd.cut(test.recency_from_cutoff, bins)
    rates = train.groupby(train_bins, observed=False).event.agg(["sum", "count"])
    # Add-one smoothing estimated only on completed historical outcomes.
    rates["p"] = (rates["sum"]+1)/(rates["count"]+2)
    base = test_bins.map(rates.p).astype(float).fillna(train.event.mean())
    # Continuous recency-only baseline, fitted solely on historical outcomes.
    x_train = np.column_stack([np.ones(len(train)), train.recency_from_cutoff/inactivity_days])
    target = train.event.to_numpy()
    opt = minimize(lambda w: np.mean(np.logaddexp(0, x_train@w)-target*(x_train@w)),
        np.zeros(2), jac=lambda w: x_train.T@(expit(x_train@w)-target)/len(train), method="BFGS")
    if not opt.success:
        raise ValueError(f"Recency logistic fit failed: {opt.message}")
    logistic = expit(np.column_stack([np.ones(len(test)), test.recency_from_cutoff/inactivity_days])@opt.x)
    constant = np.full(len(test), train.event.mean())
    ph = proportional_hazard_test(model, train[FEATURES+["duration", "event"]], time_transform="rank").summary
    return {"training_customers": len(train), "training_start": (pd.Timestamp(cutoff)-pd.Timedelta(days=90)).date().isoformat(),
        "training_end": cutoff, "inactivity_days": inactivity_days,
        "models": {"landmark_cox": binary_metrics(test.event, p), "recency_bins": binary_metrics(test.event, base),
                   "recency_logistic": binary_metrics(test.event, logistic),
                   "historical_prevalence": binary_metrics(test.event, constant)},
        "paired_brier_vs_recency_logistic": paired_bootstrap(test.event, p, logistic, "brier"),
        "calibration_deciles": pd.DataFrame({"p": p, "y": test.event}).groupby(pd.qcut(p.rank(method="first"), 10), observed=True).agg(
            customers=("y", "size"), mean_probability=("p", "mean"), event_rate=("y", "mean")).to_dict(orient="records"),
        "paired_brier_vs_recency_bins": paired_bootstrap(test.event, p, base, "brier"),
        "ph_rank_p_values": ph.p.to_dict(), "coefficients": model.params_.to_dict()}, pd.DataFrame(
            {"event": test.event, "cox_probability": p, "recency_probability": base, "recency_logistic_probability": logistic})


def main():
    OUT.mkdir(exist_ok=True)
    raw = load_csv()
    flagged = clean_and_flag(raw)
    tx = flagged.loc[flagged.include_clean].copy()
    report = {"protocol": {"seed": 42, "horizon_days": 90, "splits": SPLITS,
        "selection": "BG/NBD .05, Cox .1. Gamma-Gamma unpenalized: development-only parameter diagnostic found .01 forces q=1 (infinite population mean). No holdout-based scaling or model selection.",
        "bootstrap_scope": "Paired customer resampling within fold; no refitting, no seasonal or cross-period uncertainty.",
        "evaluation_status": "Retrospective temporal holdout; final window rerun for numerical bug fixes. Not a pristine prospective experiment."},
        "data": {"sha256": hashlib.sha256(RAW_PATH.read_bytes()).hexdigest(), "raw_rows": len(raw),
        "clean_rows": len(tx), "customers": int(tx.customer_id.nunique()), "invoices": int(tx.invoice_no.nunique()),
        "gross_sales_gbp": float(tx.revenue.sum()), "start": str(tx.invoice_date.min()), "end": str(tx.invoice_date.max()),
        "flags": {c: int(flagged[c].sum()) for c in flagged if c.startswith("is_")},
        "duplicate_policy": "Exclude exact source duplicates from analytics; retain audit rows. Identical legitimate lines cannot be distinguished without source event IDs."},
        "environment": {"python": platform.python_version(), **{p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scipy", "lifetimes", "lifelines"]}}, "splits": {}}
    for name, cutoff in SPLITS:
        end = pd.Timestamp(cutoff)+pd.Timedelta(days=90)
        clv, clv_rows = evaluate_clv(tx, cutoff, end)
        churn, churn_rows = evaluate_churn(tx, cutoff, end)
        clv_rows.to_csv(OUT/f"{name}_clv_predictions.csv", index_label="customer_id")
        churn_rows.to_csv(OUT/f"{name}_churn_predictions.csv", index_label="customer_id")
        report["splits"][name] = {"cutoff": cutoff, "end": end.date().isoformat(), "clv": clv, "churn": churn}
        print(name, json.dumps({"clv_revenue": clv["revenue_models"], "churn": churn["models"]}), flush=True)
    report["sensitivity"] = {}
    for threshold in [60, 180]:
        sensitivity, _ = evaluate_churn(tx, "2011-09-09", pd.Timestamp("2011-12-08"), threshold)
        report["sensitivity"][str(threshold)] = sensitivity
    # Persist a dependency snapshot for reproducing the numerical evaluation.
    (OUT/"evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print("Saved", OUT/"evaluation.json", flush=True)


if __name__ == "__main__":
    main()
