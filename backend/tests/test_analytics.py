"""Offline regression tests for scientific correctness; no OpenAI calls."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import pandas as pd
from lifetimes import BetaGeoFitter
from lifetimes.utils import summary_data_from_transaction_data

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analytics.clv import build_rfm, predict_clv, fit_models, CLVResult
from analytics.landmark import snapshot, outcomes, first_inactivity_table, probabilities
from analytics.monte_carlo import compute_erl_days
from etl.load_online_retail import clean_and_flag


def transactions(dates, customers=None, revenue=None):
    n = len(dates)
    return pd.DataFrame({"customer_id": customers or [1]*n,
        "invoice_no": [str(i) for i in range(n)], "invoice_date": pd.to_datetime(dates),
        "revenue": revenue or [10.]*n, "stock_code": ["A"]*n, "country": ["United Kingdom"]*n})


class RFMTests(unittest.TestCase):
    def test_daily_summary_matches_reference_including_cutoff_day(self):
        tx = transactions(["2020-01-01 10:00", "2020-01-01 11:00", "2020-01-02 09:00",
            "2020-01-03 23:59", "2020-01-04 01:00"], revenue=[10., 20., 40., 60., 900.])
        actual = build_rfm(tx, "2020-01-03")
        reference = summary_data_from_transaction_data(tx, "customer_id", "invoice_date",
            monetary_value_col="revenue", observation_period_end="2020-01-03")
        pd.testing.assert_frame_equal(actual, reference[actual.columns], check_dtype=False)
        self.assertEqual(actual.loc[1, "monetary_value"], 50.)
        self.assertEqual(actual.loc[1, "frequency"], 2.)

    def test_future_changes_do_not_change_features(self):
        tx = transactions(["2020-01-01", "2020-01-10", "2020-03-01"])
        altered = tx.copy(); altered.loc[2, "revenue"] = 1e9
        pd.testing.assert_frame_equal(build_rfm(tx, "2020-01-31"), build_rfm(altered, "2020-01-31"))
        pd.testing.assert_frame_equal(snapshot(tx, "2020-01-31"), snapshot(altered, "2020-01-31"))

    def test_empty_and_no_repeat_fail_explicitly(self):
        with self.assertRaises(ValueError):
            fit_models(pd.DataFrame(columns=["frequency", "recency", "T", "monetary_value"]))

    def test_stable_purchase_integral_matches_closed_form(self):
        bg = BetaGeoFitter(); bg.params_ = pd.Series({"r": .7, "alpha": 40., "a": 1.8, "b": 3.})
        rfm = pd.DataFrame({"frequency": [0., 3.], "recency": [0., 20.], "T": [30., 50.], "monetary_value": [np.nan, 50.]}, index=pd.Index([1, 2], name="customer_id"))
        gg = SimpleNamespace(conditional_expected_average_profit=lambda f, m: np.full(len(f), 50.))
        expected = bg.conditional_expected_number_of_purchases_up_to_time(90, rfm.frequency, rfm.recency, rfm["T"])
        with patch.object(bg, "conditional_expected_number_of_purchases_up_to_time", return_value=np.array([np.nan, np.nan])):
            actual = predict_clv(CLVResult(rfm, bg, gg), 90, aov_fallback="global_mean")
        np.testing.assert_allclose(actual.pred_purchases, expected, rtol=1e-8)

    def test_boundary_parameters_have_finite_predictions(self):
        bg = BetaGeoFitter(); bg.params_ = pd.Series({"r": .57, "alpha": 47., "a": 4.5e-23, "b": 3e-13})
        rfm = pd.DataFrame({"frequency": [0.], "recency": [0.], "T": [1.], "monetary_value": [np.nan]}, index=pd.Index([1], name="customer_id"))
        gg = SimpleNamespace(conditional_expected_average_profit=lambda f, m: np.full(len(f), 50.))
        pred = predict_clv(CLVResult(rfm, bg, gg), 90, aov_fallback="zero")
        self.assertTrue(np.isfinite(pred.pred_purchases).all())


class LandmarkTests(unittest.TestCase):
    def test_first_gap_is_event_even_after_reactivation(self):
        tx = transactions(["2020-01-01", "2020-02-01", "2020-06-01"])
        result = outcomes(tx, "2020-02-02", "2020-06-02", 30)
        self.assertEqual(result.loc[1, "event"], 1)
        self.assertEqual(result.loc[1, "duration"], 29)
        km = first_inactivity_table(tx, "2020-06-02", 30)
        self.assertEqual(km.iloc[0].duration, 30)

    def test_purchase_on_crossing_date_and_censor_boundary(self):
        tx = transactions(["2020-01-01", "2020-01-31"])
        event = outcomes(tx, "2020-01-02", "2020-01-31", 30)
        censored = outcomes(tx, "2020-01-02", "2020-01-30", 30)
        self.assertEqual(event.loc[1, "event"], 1)
        self.assertEqual(censored.loc[1, "event"], 0)

    def test_new_purchase_before_crossing_resets_clock(self):
        tx = transactions(["2020-01-01", "2020-01-20"])
        result = outcomes(tx, "2020-01-02", "2020-02-10", 30)
        self.assertEqual(result.loc[1, "event"], 0)

    def test_survival_step_and_horizon_guard(self):
        model = SimpleNamespace(landmark_followup_days_=90,
            predict_survival_function=lambda x: pd.DataFrame({1: [.9, .7]}, index=[10, 20]))
        data = pd.DataFrame(0., index=[1], columns=["log_n_orders", "log_monetary_value", "log_product_diversity", "recency_from_cutoff", "tenure_days"])
        self.assertAlmostEqual(probabilities(model, data, 15).iloc[0], .1)
        with self.assertRaises(ValueError): probabilities(model, data, 91)


class SimulationTests(unittest.TestCase):
    def test_cap_all_paths_and_stable_customer_streams(self):
        bg = BetaGeoFitter(); bg.params_ = pd.Series({"r": 1., "alpha": 30., "a": 2., "b": 3.})
        summary = pd.DataFrame({"customer_id": [1, 2], "frequency": [2., 3.], "recency": [5., 10.], "T": [20., 20.], "last_purchase_age_days": [15., 10.]})
        a = compute_erl_days(bg, summary, "2020-01-01", 90, N=100, max_days=30)
        b = compute_erl_days(bg, summary.iloc[::-1], "2020-01-01", 90, N=100, max_days=30)
        self.assertTrue(a.ERL_days.between(0, 30).all())
        pd.testing.assert_frame_equal(a.sort_values("customer_id").reset_index(drop=True), b.sort_values("customer_id").reset_index(drop=True))
        self.assertTrue(a.ERL_mc_se.ge(0).all())

    def test_inactive_schema_and_invalid_arguments(self):
        bg = BetaGeoFitter(); bg.params_ = pd.Series({"r": 1., "alpha": 30., "a": 2., "b": 3.})
        summary = pd.DataFrame({"customer_id": [1], "frequency": [1.], "recency": [10.], "T": [120.], "last_purchase_age_days": [110.]})
        out = compute_erl_days(bg, summary, "2020-01-01", 90)
        self.assertEqual(out.ERL_days.iloc[0], 0)
        self.assertIn("ERL_mc_se", out)
        with self.assertRaises(ValueError): compute_erl_days(bg, summary, "2020-01-01", 0)


class ETLTests(unittest.TestCase):
    def test_duplicates_and_nonfinite_values_retained_but_excluded(self):
        raw = pd.DataFrame({"InvoiceNo": ["1"]*3, "StockCode": ["A"]*3, "Description": ["A"]*3,
            "Quantity": [1., 1., np.nan], "InvoiceDate": pd.to_datetime(["2020-01-01"]*3),
            "UnitPrice": [2.]*3, "CustomerID": [1]*3, "Country": ["UK"]*3})
        flagged = clean_and_flag(raw)
        self.assertEqual(len(flagged), 3)
        self.assertEqual(flagged.include_clean.sum(), 1)
        self.assertEqual(flagged.is_duplicate.sum(), 1)


if __name__ == "__main__":
    unittest.main()
