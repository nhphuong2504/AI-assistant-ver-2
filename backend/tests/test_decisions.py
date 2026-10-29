"""Temporal invariance, feature semantics and capacity-limited policy tests."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analytics.customer_features import customer_features, future_spend, FULL_FEATURES
from analytics.targeting import (budget_metrics, evaluate_policies, minimum_contacts,
    rank_order, tie_keys, policy_bootstrap)
from evaluate_decisions import choose_models
from analytics.challengers import PROTOCOL


def tx(dates, values=None):
    return pd.DataFrame({"customer_id": [1]*len(dates), "invoice_no": list(map(str, range(len(dates)))),
        "invoice_date": pd.to_datetime(dates), "revenue": values or [10.]*len(dates),
        "stock_code": ["A"]*len(dates), "country": ["UK"]*len(dates)})


class FeatureTests(unittest.TestCase):
    def test_daily_windows_trends_and_first_purchase(self):
        data = tx(["2020-01-01", "2020-01-01", "2020-01-11", "2020-03-15", "2020-04-01", "2020-04-30"], [10.,20.,30.,40.,50.,60.])
        row = customer_features(data, "2020-04-30").loc[1]
        self.assertEqual(row.purchase_days_30, 2)
        self.assertEqual(row.purchase_days_60, 3)
        self.assertEqual(row.spend_30, 110.)
        self.assertAlmostEqual(row.spend_trend_30, np.log1p(110.)-np.log1p(40.))
        self.assertEqual(row.n_orders, 6)
        self.assertEqual(len(FULL_FEATURES), 18)

    def test_future_transactions_and_new_customers_do_not_change_features(self):
        data = tx(["2020-01-01", "2020-02-01", "2020-06-01"])
        altered = data.copy(); altered.loc[2, "revenue"] = 1e9; altered.loc[2, "customer_id"] = 999
        pd.testing.assert_frame_equal(customer_features(data, "2020-03-01"), customer_features(altered, "2020-03-01"))

    def test_reactivation_and_one_time_buyer_defaults(self):
        row = customer_features(tx(["2020-01-01", "2020-04-10"]), "2020-04-11").loc[1]
        self.assertEqual(row.prior_inactivity_episodes, 1)
        one = customer_features(tx(["2020-01-01"]), "2020-01-02").loc[1]
        self.assertEqual(one.has_repeat_day, 0)
        self.assertAlmostEqual(one.recency_to_mean_gap, 1/90)
        self.assertTrue(np.isfinite(one[FULL_FEATURES].astype(float)).all())

    def test_outcome_window_excludes_cutoff_and_includes_last_day(self):
        data = tx(["2020-01-01", "2020-01-02", "2020-03-31", "2020-04-01"])
        labels = future_spend(data, "2020-01-01", 90, pd.Index([1, 2], name="customer_id"))
        self.assertEqual(labels.loc[1, "actual_purchases"], 2)
        self.assertEqual(labels.loc[2, "actual_revenue"], 0)


class TargetingTests(unittest.TestCase):
    def test_hand_calculated_metrics_and_random_expectation(self):
        y, value, scores = [1,0,1,1], [10.,100.,20.,0.], [.4,.3,.2,.1]
        metrics = budget_metrics(y, value, scores, .5, tie_keys([1,2,3,4]))
        self.assertEqual(metrics["contacts"], 2)
        self.assertEqual(metrics["precision"], .5)
        self.assertAlmostEqual(metrics["historical_value_weighted_recall"], 1/3)
        random = evaluate_policies(y, value, pd.DataFrame({"p":scores}), budgets=[.5])["random_expected"][0]
        self.assertEqual(random["recall"], .5)
        self.assertEqual(random["historical_value_weighted_recall"], .5)

    def test_tie_breaking_independent_of_row_order(self):
        data = pd.Series(1., index=[1,2,3,4,5])
        before = data.index[rank_order(data, tie_keys(data.index))[:3]].tolist()
        shuffled = data.sample(frac=1, random_state=7)
        after = shuffled.index[rank_order(shuffled, tie_keys(shuffled.index))[:3]].tolist()
        self.assertEqual(before, after)

    def test_identical_policy_bootstrap_difference_zero(self):
        scores = pd.Series([.4,.3,.2,.1])
        intervals = policy_bootstrap([1,0,1,1], [10.,100.,20.,0.], scores, scores, repeats=50)
        for budget in intervals.values():
            for result in budget.values(): self.assertEqual(result["ci95"], [0.,0.])

    def test_contact_efficiency_uses_cumulative_weight(self):
        result = minimum_contacts([1,0,1,1], [10.,100.,20.,0.], pd.Series([.4,.3,.2,.1]), target=.5)
        self.assertEqual(result["contacts"], 3)


class SelectionTests(unittest.TestCase):
    def test_selection_uses_declared_development_metrics(self):
        fold = {"churn_models": {name: {"brier": .2} for name in PROTOCOL["churn_candidates"]},
            "spend_models": {name: {"mae": 100.} for name in PROTOCOL["spend_candidates"]}}
        fold["churn_models"]["logistic_base"]["brier"] = .1
        fold["spend_models"]["last90"]["mae"] = 90.
        chosen = choose_models([fold])
        self.assertEqual(chosen["churn_model"], "logistic_base")
        self.assertEqual(chosen["spend_model"], "last90")


if __name__ == "__main__":
    unittest.main()
