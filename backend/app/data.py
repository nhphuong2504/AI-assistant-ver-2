"""
Shared data loading and model caches for the backend.
Used by main.py and llm_langchain.py to avoid redundant DB reads and model fits.
"""
import pandas as pd
from typing import Dict, Any, Optional

from app.db import run_query_internal
from analytics.clv import build_rfm, fit_models, CLVResult

# ---------------------------------------------------------------------------
# Transactions cache
# ---------------------------------------------------------------------------
_transactions_cache: Optional[pd.DataFrame] = None

TRANSACTIONS_SQL = """
SELECT customer_id, invoice_no, invoice_date, revenue, stock_code, country
FROM transactions
WHERE customer_id IS NOT NULL
"""


def get_transactions_df() -> pd.DataFrame:
    """Load transactions data - cached for performance. Safe to call from any module."""
    global _transactions_cache
    if _transactions_cache is None:
        rows, _ = run_query_internal(TRANSACTIONS_SQL, max_rows=2_000_000)
        _transactions_cache = pd.DataFrame(rows)
    return _transactions_cache.copy()


# ---------------------------------------------------------------------------
# CLV models cache (keyed by cutoff_date)
# ---------------------------------------------------------------------------
_clv_models_cache: Dict[str, CLVResult] = {}


def get_clv_models(cutoff_date: str) -> CLVResult:
    """Build RFM and fit BG/NBD + Gamma-Gamma models - cached by cutoff_date."""
    global _clv_models_cache
    if cutoff_date not in _clv_models_cache:
        df = get_transactions_df()
        rfm = build_rfm(df, cutoff_date=cutoff_date)
        _clv_models_cache[cutoff_date] = fit_models(rfm)
    return _clv_models_cache[cutoff_date]
