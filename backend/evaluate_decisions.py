"""Fixed-protocol challenger evaluation and budgeted customer targeting.

Run from repo root: python backend/evaluate_decisions.py
Selection uses only three development windows. The already-viewed final window
is a retrospective audit, NOT a pristine unseen test. See decision_protocol.json.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from analytics.challengers import (PROTOCOL, fit_churn_candidates, predict_churn_candidates,
    fit_spend_candidates, predict_spend_candidates)
from analytics.customer_features import customer_features, future_spend, FULL_FEATURES
from analytics.landmark import outcomes
from analytics.targeting import policy_scores, evaluate_policies, minimum_contacts, policy_bootstrap
from etl.load_online_retail import load_csv, clean_and_flag, RAW_PATH
from evaluate import binary_metrics, regression_metrics, paired_bootstrap

OUT = Path(__file__).resolve().parent/"reports"


def calibration(y, p):
    frame = pd.DataFrame({"p": np.asarray(p), "y": np.asarray(y)})
    groups = pd.qcut(frame.p.rank(method="first"), min(10, len(frame)), duplicates="drop")
    return frame.groupby(groups, observed=True).agg(customers=("y", "size"), mean_probability=("p", "mean"), event_rate=("y", "mean")).to_dict(orient="records")


def evaluate_fold(tx, cutoff):
    all_features = customer_features(tx, cutoff)
    churn_labels = outcomes(tx, cutoff, pd.Timestamp(cutoff)+pd.Timedelta(days=90))
    active = all_features.loc[churn_labels.index]
    fitted, train_features, train_labels = fit_churn_candidates(tx, cutoff)
    churn = predict_churn_candidates(fitted, active)
    spend_fitted, spend_train_features, spend_train_labels = fit_spend_candidates(tx, cutoff)
    spend, bg = predict_spend_candidates(spend_fitted, tx, cutoff, all_features)
    spend_labels = future_spend(tx, cutoff, customer_index=all_features.index)
    churn_metrics = {name: {**binary_metrics(churn_labels.event, p),
        "log_loss": float(log_loss(churn_labels.event, p)), "calibration_deciles": calibration(churn_labels.event, p)} for name, p in churn.items()}
    spend_metrics = {name: regression_metrics(spend_labels.actual_revenue, p) for name, p in spend.items()}
    # Oracle diagnostic only: exactly decomposes actual minus predicted gross sales.
    actual_count = spend_labels.actual_purchases
    predicted_aov = bg.pred_aov
    purchase_component = float(((actual_count-bg.pred_purchases)*predicted_aov).sum())
    value_component = float((spend_labels.actual_revenue-actual_count*predicted_aov).sum())
    cohort = {"churn_train_customers": len(train_features), "spend_train_customers": len(spend_train_features),
        "churn_evaluation_customers": len(active), "spend_evaluation_customers": len(all_features),
        "training_landmark": (pd.Timestamp(cutoff)-pd.Timedelta(days=90)).date().isoformat(),
        "training_outcomes_complete_through": cutoff, "evaluation_end": (pd.Timestamp(cutoff)+pd.Timedelta(days=90)).date().isoformat()}
    diagnostics = {"bgnbd_underprediction_decomposition": {"purchase_frequency_component_gbp": purchase_component,
        "purchase_day_value_component_gbp": value_component,
        "actual_minus_predicted_gbp": float(spend_labels.actual_revenue.sum()-spend.bgnbd_gamma_gamma.sum()),
        "warning": "Oracle diagnostic uses observed future counts; these are not deployable forecasts or causal attribution."},
        "largest_customer_share_of_actual_revenue": float(spend_labels.actual_revenue.max()/spend_labels.actual_revenue.sum())}
    churn_rows = active[["recency_from_cutoff", "spend_90"]].join(churn_labels[["event", "duration"]]).join(churn.add_prefix("p_"))
    spend_rows = all_features[["n_orders", "tenure_days", "has_repeat_day", "spend_90"]].join(spend_labels).join(spend.add_prefix("pred_"))
    return {"cutoff": cutoff, "cohort": cohort, "churn_models": churn_metrics, "spend_models": spend_metrics,
        "diagnostics": diagnostics}, churn_rows, spend_rows


def choose_models(development):
    churn_mean = {name: float(np.mean([fold["churn_models"][name]["brier"] for fold in development])) for name in PROTOCOL["churn_candidates"]}
    spend_mean = {name: float(np.mean([fold["spend_models"][name]["mae"] for fold in development])) for name in PROTOCOL["spend_candidates"]}
    return {"churn_model": min(churn_mean, key=churn_mean.get), "spend_model": min(spend_mean, key=spend_mean.get),
        "development_mean_brier": churn_mean, "development_mean_spend_mae": spend_mean}


def scores_for(rows, spend_rows, selection):
    value = spend_rows.loc[rows.index, "pred_"+selection["spend_model"]]
    return policy_scores(rows, rows["p_"+selection["churn_model"]], value)


def main():
    OUT.mkdir(exist_ok=True)
    flagged = clean_and_flag(load_csv())
    tx = flagged.loc[flagged.include_clean].copy()
    last_day = pd.to_datetime(tx.invoice_date).max().normalize()
    for cutoff in [*PROTOCOL["development_cutoffs"], PROTOCOL["retrospective_test_cutoff"]]:
        if pd.Timestamp(cutoff)+pd.Timedelta(days=90) >= last_day:
            raise ValueError("Evaluation follow-up must end before the possibly partial last dataset day")
    protocol_path = Path(__file__).parent/"analytics/decision_protocol.json"
    report = {"protocol": PROTOCOL, "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(RAW_PATH.read_bytes()).hexdigest(), "feature_columns": FULL_FEATURES, "development": [],
        "data": {"raw_rows": len(flagged), "clean_rows": len(tx), "customers": int(tx.customer_id.nunique())}}
    development_rows = []
    for cutoff in PROTOCOL["development_cutoffs"]:
        metrics, churn_rows, spend_rows = evaluate_fold(tx, cutoff)
        report["development"].append(metrics)
        development_rows.append((churn_rows, spend_rows))
        churn_rows.to_csv(OUT/f"decision_{cutoff}_churn.csv", index_label="customer_id")
        spend_rows.to_csv(OUT/f"decision_{cutoff}_spend.csv", index_label="customer_id")
        print("development", cutoff, {k: round(v["brier"], 4) for k, v in metrics["churn_models"].items()}, flush=True)
    selection = choose_models(report["development"])
    for metrics, (rows, spend_rows) in zip(report["development"], development_rows):
        scores = scores_for(rows, spend_rows, selection)
        metrics["policies"] = evaluate_policies(rows.event, rows.spend_90, scores, PROTOCOL["budget_fractions"])
    policy_means = {name: float(np.mean([fold["policies"][name][-1]["historical_value_weighted_recall"] for fold in report["development"]]))
        for name in PROTOCOL["policies"] if name != "random_expected"}
    selection.update({"policy": max(policy_means, key=policy_means.get), "development_mean_weighted_recall_at20": policy_means,
        "protocol_sha256": report["protocol_sha256"], "source_sha256": report["source_sha256"],
        "horizon_days": 90, "inactivity_days": 90, "selected_using": "Development windows only; no final-window scores used in selection"})
    # Persist selection BEFORE touching the final evaluation fold.
    (OUT/"model_selection.json").write_text(json.dumps(selection, indent=2, allow_nan=False), encoding="utf-8")
    print("FROZEN SELECTION", selection["churn_model"], selection["spend_model"], selection["policy"], flush=True)
    report["selection"] = selection
    final, churn_rows, spend_rows = evaluate_fold(tx, PROTOCOL["retrospective_test_cutoff"])
    scores = scores_for(churn_rows, spend_rows, selection)
    final["policies"] = evaluate_policies(churn_rows.event, churn_rows.spend_90, scores)
    final["policy_curve"] = evaluate_policies(churn_rows.event, churn_rows.spend_90, scores, budgets=np.arange(.01, 1.01, .01))
    final["contacts_for_50pct_weighted_recall"] = {name: minimum_contacts(churn_rows.event, churn_rows.spend_90, scores[name]) for name in scores}
    final["selected_policy_vs_historical_spend_bootstrap"] = policy_bootstrap(churn_rows.event, churn_rows.spend_90,
        scores[selection["policy"]], scores.historical_spend)
    selected_p = churn_rows["p_"+selection["churn_model"]]
    final["selected_churn_vs_recency_brier_bootstrap"] = paired_bootstrap(churn_rows.event, selected_p, churn_rows.p_recency_logistic, "brier")
    final["selected_churn_vs_cox_brier_bootstrap"] = paired_bootstrap(churn_rows.event, selected_p, churn_rows.p_landmark_cox, "brier")
    # Added after inspecting the primary results. This stronger comparator is
    # explicitly post-hoc and does not participate in selecting models/policies.
    recency_value = churn_rows.p_recency_logistic*churn_rows.spend_90
    final["posthoc_robustness"] = {"status": "Added after primary results were inspected; not used for selection",
        "recency_probability_x_historical_spend": evaluate_policies(churn_rows.event, churn_rows.spend_90,
            pd.DataFrame({"recency_probability_x_historical_spend": recency_value}))["recency_probability_x_historical_spend"],
        "selected_policy_vs_recency_value_bootstrap": policy_bootstrap(churn_rows.event, churn_rows.spend_90, scores[selection["policy"]], recency_value)}
    selected_value = spend_rows["pred_"+selection["spend_model"]]
    final["selected_spend_vs_recent_spend_mae_bootstrap"] = paired_bootstrap(spend_rows.actual_revenue, selected_value, spend_rows.pred_last90, "mae")
    final["selected_spend_vs_bgnbd_mae_bootstrap"] = paired_bootstrap(spend_rows.actual_revenue, selected_value, spend_rows.pred_bgnbd_gamma_gamma, "mae")
    final["spend_segments"] = {str(bool(repeat)): {name: regression_metrics(group.actual_revenue, group["pred_"+name])
        for name in PROTOCOL["spend_candidates"]} for repeat, group in spend_rows.groupby("has_repeat_day")}
    report["retrospective_test"] = final
    churn_rows.join(scores.add_prefix("score_")).to_csv(OUT/"decision_final_churn.csv", index_label="customer_id")
    spend_rows.to_csv(OUT/"decision_final_spend.csv", index_label="customer_id")
    import sklearn
    report["sklearn_version"] = sklearn.__version__
    (OUT/"decision_evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print("FINAL CHURN", {k: {m: round(v[m], 4) for m in ("roc_auc", "brier", "log_loss")} for k, v in final["churn_models"].items()}, flush=True)
    print("FINAL SPEND", {k: {m: round(v[m], 4) for m in ("mae", "rmse", "aggregate_bias_pct", "top20_capture")} for k, v in final["spend_models"].items()}, flush=True)
    print("Saved", OUT/"decision_evaluation.json", flush=True)


if __name__ == "__main__":
    main()
