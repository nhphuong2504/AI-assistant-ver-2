"""Fixed candidate families; selection is performed only on development folds."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, GammaRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from .landmark import FEATURES, outcomes, fit_landmark_cox, probabilities
from .customer_features import customer_features, future_spend, FULL_FEATURES, WINDOW_FEATURES
from .clv import build_rfm, fit_models, predict_clv

PROTOCOL = json.loads((Path(__file__).parent/"decision_protocol.json").read_text())


def logistic():
    return make_pipeline(StandardScaler(), LogisticRegression(C=PROTOCOL["model_settings"]["logistic_C"], max_iter=2000, random_state=PROTOCOL["seed"]))


def boosting(regression=False):
    cls = HistGradientBoostingRegressor if regression else HistGradientBoostingClassifier
    settings = PROTOCOL["model_settings"]
    kwargs = dict(max_iter=settings["boosting_max_iter"], max_leaf_nodes=settings["boosting_max_leaf_nodes"], min_samples_leaf=settings["boosting_min_samples_leaf"],
        learning_rate=settings["boosting_learning_rate"], l2_regularization=settings["boosting_l2"], early_stopping=settings["early_stopping"], random_state=PROTOCOL["seed"])
    return cls(**kwargs, **({"loss": "poisson"} if regression else {}))


def fit_churn_candidates(transactions, as_of):
    historical = pd.Timestamp(as_of).normalize()-pd.Timedelta(days=90)
    labels = outcomes(transactions, historical, as_of)
    features = customer_features(transactions, historical).loc[labels.index]
    fitted = {}
    with threadpool_limits(limits=1):
        for name, columns, model in [
            ("recency_logistic", ["recency_from_cutoff"], logistic()),
            ("logistic_base", FEATURES, logistic()),
            ("logistic_windows", FEATURES+WINDOW_FEATURES, logistic()),
            ("logistic_full", FULL_FEATURES, logistic()),
            ("boosting_base", FEATURES, boosting()),
            ("boosting_full", FULL_FEATURES, boosting())]:
            model.fit(features[columns], labels.event)
            fitted[name] = (model, columns)
        model, _ = fit_landmark_cox(transactions, as_of)
        fitted["landmark_cox"] = (model, FEATURES)
    return fitted, features, labels


def predict_churn_candidates(fitted, features):
    with threadpool_limits(limits=1):
        return pd.DataFrame({name: probabilities(model, features).to_numpy() if name == "landmark_cox"
            else model.predict_proba(features[columns])[:, 1]
            for name, (model, columns) in fitted.items()}, index=features.index)


class TwoPartSpend:
    """P(any purchase) × E(gross spend | any purchase); all targets in GBP."""
    def __init__(self, boosted=False):
        self.probability = boosting() if boosted else logistic()
        self.positive_spend = boosting(True) if boosted else make_pipeline(StandardScaler(), GammaRegressor(alpha=PROTOCOL["model_settings"]["gamma_alpha"], max_iter=2000))

    def fit(self, features, target):
        positive = target > 0
        if positive.sum() < 30 or (~positive).sum() < 5:
            raise ValueError("Insufficient outcomes for two-part spend")
        with threadpool_limits(limits=1):
            self.probability.fit(features[FULL_FEATURES], positive.astype(int))
            self.positive_spend.fit(features.loc[positive, FULL_FEATURES], target[positive])
        return self

    def components(self, features):
        with threadpool_limits(limits=1):
            p = self.probability.predict_proba(features[FULL_FEATURES])[:, 1]
            value = self.positive_spend.predict(features[FULL_FEATURES])
        return p, np.maximum(value, 0.)

    def predict(self, features):
        p, value = self.components(features)
        return p*value


def fit_spend_candidates(transactions, as_of):
    historical = pd.Timestamp(as_of).normalize()-pd.Timedelta(days=90)
    features = customer_features(transactions, historical)
    labels = future_spend(transactions, historical, customer_index=features.index)
    fitted = {"two_part_glm": TwoPartSpend().fit(features, labels.actual_revenue),
              "two_part_boosting": TwoPartSpend(True).fit(features, labels.actual_revenue)}
    return fitted, features, labels


def predict_spend_candidates(fitted, transactions, as_of, features):
    bg = fit_models(build_rfm(transactions, as_of))
    bg_predictions = predict_clv(bg, 90, aov_fallback="global_mean").set_index("customer_id").loc[features.index]
    predictions = pd.DataFrame({"last90": features.spend_90, "bgnbd_gamma_gamma": bg_predictions.clv,
        **{name: model.predict(features) for name, model in fitted.items()}}, index=features.index)
    if not np.isfinite(predictions.to_numpy()).all() or (predictions < 0).any().any():
        raise ValueError("Invalid spend predictions")
    return predictions, bg_predictions
