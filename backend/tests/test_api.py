"""Integration with the locally regenerated source-data SQLite database.

No LLM invocations. Dummy API key only permits importing agent configuration.
"""
import os
import sys
import unittest
import json
from contextlib import closing
from pathlib import Path

os.environ.setdefault("OPENAI_API_KEY", "offline-test-no-network")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from app.main import app
from app.db import get_conn


class APITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_clv_serializes_single_purchase_customers_without_nan(self):
        response = self.client.post("/clv", json={"cutoff_date": "2011-09-09", "horizon_days": 90})
        self.assertEqual(response.status_code, 200, response.text[:300])
        body = response.json()
        self.assertEqual(body["summary"]["customers_total"], 3362)
        self.assertTrue(any(c["monetary_value"] is None for c in body["top_customers"]))
        self.assertNotIn("NaN", response.text)

    def test_served_churn_matches_offline_evaluation(self):
        response = self.client.post("/survival/churn-probability?cutoff_date=2011-09-09&X_days=90")
        self.assertEqual(response.status_code, 200, response.text[:300])
        expected = json.loads((Path(__file__).resolve().parents[1]/"reports/evaluation.json").read_text())["splits"]["final_test"]["churn"]["models"]["landmark_cox"]
        self.assertEqual(response.json()["n_customers"], expected["n"])
        self.assertAlmostEqual(response.json()["summary"]["churn_probability_mean"], expected["mean_probability"], places=10)

    def test_invalid_horizons_and_readonly_queries(self):
        self.assertEqual(self.client.post("/survival/churn-probability?X_days=91").status_code, 422)
        self.assertEqual(self.client.post("/survival/score?inactivity_days=0").status_code, 422)
        response = self.client.post("/query", json={"sql": "SELECT COUNT(*) AS n FROM transactions;"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["rows"][0]["n"], 392692)
        with closing(get_conn()) as conn:
            with self.assertRaises(Exception): conn.execute("DELETE FROM transactions WHERE 1=0")

    def test_restricted_lifetime_respects_requested_horizon(self):
        response = self.client.post("/survival/expected-lifetime?H_days=30")
        self.assertEqual(response.status_code, 200, response.text[:300])
        self.assertLessEqual(response.json()["summary"]["expected_lifetime_max"], 30.)

    def test_segmentation_uses_requested_threshold_and_restriction(self):
        response = self.client.post("/survival/segmentation?inactivity_days=60&H_days=30")
        self.assertEqual(response.status_code, 200, response.text[:300])
        self.assertTrue(response.json()["segments"])
        self.assertTrue(all(row["erl_days"] <= 30 for row in response.json()["segments"]))
        self.assertTrue(all(row["life_bucket"] == "At-Risk" for row in response.json()["segments"]))

    def test_km_survival_curve_is_monotone(self):
        response = self.client.post("/survival/km")
        self.assertEqual(response.status_code, 200, response.text[:300])
        survival = [row["survival"] for row in response.json()["survival_curve"]]
        self.assertTrue(all(a >= b for a, b in zip(survival, survival[1:])))

    def test_budgeted_targets_match_frozen_offline_policy(self):
        import pandas as pd
        response = self.client.post("/retention/targets", json={"cutoff_date":"2011-09-09", "contact_fraction":.2})
        self.assertEqual(response.status_code, 200, response.text[:300])
        body = response.json()
        report_path = Path(__file__).resolve().parents[1]/"reports"
        offline = pd.read_csv(report_path/"decision_final_churn.csv").set_index("customer_id")
        from analytics.targeting import rank_order, tie_keys
        self.assertEqual(body["contacts"], 380)
        self.assertEqual(body["active_customers"], 1899)
        self.assertEqual(body["policy"], "churn_x_historical_spend")
        expected_ids = offline.index[rank_order(offline.score_churn_x_historical_spend, tie_keys(offline.index))[:380]].tolist()
        self.assertEqual([row["customer_id"] for row in body["customers"]], expected_ids)
        for row in body["customers"]:
            reference = offline.loc[row["customer_id"]]
            self.assertAlmostEqual(row["priority_score"], reference.score_churn_x_historical_spend, places=8)
            self.assertAlmostEqual(row["churn_probability_90"], reference.p_logistic_full, places=10)
            self.assertNotIn("event", row)
        self.assertEqual(self.client.post("/retention/targets", json={"contact_fraction":0}).status_code, 422)
        self.assertEqual(self.client.post("/retention/targets", json={"cutoff_date":"2012-01-01"}).status_code, 422)

    def test_selected_predictions_ignore_modified_future(self):
        from app.data import get_transactions_df
        from analytics.decision_service import selected_predictions, _cache
        import pandas as pd
        data = get_transactions_df()
        before, _ = selected_predictions(data, "2011-09-09")
        altered = data.copy()
        future = pd.to_datetime(altered.invoice_date).dt.normalize() > pd.Timestamp("2011-09-09")
        altered.loc[future, "revenue"] *= 1000
        _cache.clear()  # Refit, rather than allowing cached outputs to mask lookahead.
        after, _ = selected_predictions(altered, "2011-09-09")
        pd.testing.assert_frame_equal(before, after)

    def test_retention_tool_forwards_budget_and_reports_display_count(self):
        from unittest.mock import patch
        from app.llm_langchain import prioritize_retention_targets_tool
        stub = {"contacts": 20, "active_customers":100, "customers":[
            {"customer_id": i, "priority_rank":i+1, "priority_score":10-i} for i in range(5)]}
        with patch("app.llm_langchain.retention_targets", return_value=stub) as mocked:
            result = json.loads(prioritize_retention_targets_tool.invoke({"contact_fraction":.2,"limit_customers":3}))
        self.assertEqual(result["contacts_in_budget"], 20)
        self.assertEqual(result["returned_customers"], 3)
        self.assertEqual(mocked.call_args.args[2], .2)

    def test_default_retention_cutoff_runs_without_llm(self):
        import math
        response = self.client.post("/retention/targets", json={"contact_fraction":.05})
        self.assertEqual(response.status_code, 200, response.text[:300])
        body = response.json()
        self.assertEqual(body["cutoff_date"], "2011-12-09")
        self.assertEqual(body["contacts"], math.ceil(.05*body["active_customers"]))
        self.assertEqual(len(body["customers"]), body["contacts"])


if __name__ == "__main__":
    unittest.main()
