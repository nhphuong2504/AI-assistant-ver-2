import os
import json
import re
import time
import hashlib
import uuid
import warnings
from typing import Dict, Any, Optional, Tuple, Union, List
from functools import lru_cache
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from openai import RateLimitError, APIError, APIConnectionError, APITimeoutError
import pandas as pd
from analytics.clv import build_rfm, fit_models, predict_clv, PURCHASE_SCALE, REVENUE_SCALE
from analytics.survival import (
    build_covariate_table,
    fit_cox_baseline,
    score_customers,
    predict_churn_probability,
    expected_remaining_lifetime,
    build_segmentation_table,
    CUTOFF_DATE,
    INACTIVITY_DAYS,
)
from app.db import run_query, run_query_internal, get_schema, ensure_select_only

load_dotenv()

# Fixed constants
FIXED_CUTOFF_DATE = "2011-12-09"
FIXED_INACTIVITY_DAYS = 90

# Global transactions cache
_transactions_cache = None

# Model fit cache (for expensive Cox model fits)
_model_cache: Dict[str, Tuple[Any, float]] = {}
CACHE_TTL = 3600  # 1 hour cache TTL


def get_transactions_df() -> pd.DataFrame:
    """Load transactions data - cached for performance"""
    global _transactions_cache
    if _transactions_cache is None:
        sql = """
        SELECT customer_id, invoice_no, invoice_date, revenue, stock_code, country
        FROM transactions
        WHERE customer_id IS NOT NULL
        """
        rows, _ = run_query_internal(sql, max_rows=2_000_000)
        _transactions_cache = pd.DataFrame(rows)
    return _transactions_cache.copy()


def get_cached_cox_model(cache_key: str) -> Optional[Any]:
    """Get cached Cox model if available and not expired"""
    if cache_key in _model_cache:
        model, timestamp = _model_cache[cache_key]
        if time.time() - timestamp < CACHE_TTL:
            return model
        else:
            # Cache expired
            del _model_cache[cache_key]
    return None


def cache_cox_model(cache_key: str, model: Any):
    """Cache a Cox model with timestamp"""
    _model_cache[cache_key] = (model, time.time())


def get_or_fit_cox_model(transactions_df: pd.DataFrame, cutoff_date: str, inactivity_days: int) -> Dict[str, Any]:
    """Get cached Cox model or fit a new one. Returns full cox_result dict."""
    # Create cache key based on parameters
    cache_key = hashlib.md5(
        f"{cutoff_date}_{inactivity_days}_{len(transactions_df)}".encode()
    ).hexdigest()
    
    # Try to get cached model
    cached_model = get_cached_cox_model(cache_key)
    if cached_model is not None:
        # Rebuild full result dict (we only cache the model, need to rebuild other parts)
        # For now, we'll still need to rebuild covariate table, but model is cached
        cov = build_covariate_table(
            transactions=transactions_df,
            cutoff_date=cutoff_date,
            inactivity_days=inactivity_days,
        ).df
        
        return {
            'model': cached_model,
            'summary': None,  # Will be regenerated if needed
            'interpretation': None,
            'flags': None,
            'n_train': len(cov),
            'n_validation': 0,
            'n_dropped': 0,
            'train_df': cov,
            'validation_df': pd.DataFrame(),
        }
    
    # Fit new model
    cov = build_covariate_table(
        transactions=transactions_df,
        cutoff_date=cutoff_date,
        inactivity_days=inactivity_days,
    ).df
    
    cox_result = fit_cox_baseline(
        covariates=cov,
        covariate_cols=['n_orders', 'log_monetary_value', 'product_diversity'],
        train_frac=0.8,
        random_state=42,
        penalizer=0.1,
    )
    
    # Cache the model
    cache_cox_model(cache_key, cox_result['model'])
    
    return cox_result


def validate_sql(sql: str) -> None:
    """
    Validate SQL query to block malicious/invalid SQL before execution.
    Raises ValueError with descriptive message if validation fails.
    """
    if not sql or not sql.strip():
        raise ValueError("SQL query cannot be empty.")
    
    sql_upper = sql.upper().strip()
    
    # Must start with SELECT or WITH (CTE)
    if not (sql_upper.startswith("SELECT") or sql_upper.startswith("WITH")):
        raise ValueError("Only SELECT queries (including WITH/CTE) are allowed.")
    
    # Use the existing ensure_select_only validation from db module
    # This checks for forbidden keywords (INSERT, UPDATE, DELETE, etc.) and multiple statements
    try:
        ensure_select_only(sql)
    except ValueError as e:
        raise ValueError(f"SQL validation failed: {str(e)}")
    
    # Additional checks for suspicious/malicious patterns
    suspicious_patterns = [
        # SQL injection attempts
        (r"'\s*;\s*--", "SQL injection pattern detected (comment injection)"),
        (r"'\s*OR\s*'\s*=\s*'", "SQL injection pattern detected (OR 1=1 style)"),
        (r"'\s*UNION\s+ALL\s+SELECT", "SQL injection pattern detected (UNION injection)"),
        # Dangerous SQLite functions
        (r'\bLOAD_EXTENSION\s*\(', "LOAD_EXTENSION is not allowed"),
        (r'\.read\s*\(', "File read operations are not allowed"),
        (r'\.import\s+', "Import operations are not allowed"),
        # System table access attempts
        (r'sqlite_master\s*WHERE\s*type\s*=\s*["\']table["\']', "Direct sqlite_master access is restricted"),
        # Function calls that could be dangerous
        (r'\bEXEC\s*\(', "EXEC() calls are not allowed"),
        (r'\bEXECUTE\s*\(', "EXECUTE() calls are not allowed"),
    ]
    
    for pattern, message in suspicious_patterns:
        if re.search(pattern, sql, re.IGNORECASE | re.DOTALL):
            raise ValueError(f"Malicious SQL pattern detected: {message}")
    
    # Check for excessive nesting or complexity (basic check to prevent DoS)
    if sql.count('(') > 100 or sql.count(')') > 100:
        raise ValueError("SQL query is too complex (excessive nesting).")
    
    # Check for suspicious string concatenation that might be used for injection
    # This pattern looks for string concatenation that could be used to bypass validation
    if re.search(r"'\s*\+\s*[^']", sql, re.IGNORECASE) or re.search(r"[^']\s*\+\s*'", sql, re.IGNORECASE):
        # Allow simple concatenation but flag suspicious patterns
        if re.search(r"'\s*\+\s*SELECT", sql, re.IGNORECASE):
            raise ValueError("Suspicious string concatenation with SELECT detected.")


def filter_by_customer_ids(df: pd.DataFrame, customer_id: Optional[Union[int, str]] = None, customer_ids: Optional[List[Union[int, str]]] = None) -> Tuple[pd.DataFrame, List[int], List[int]]:
    """
    Filter dataframe by customer_id(s). Handles type conversion and validation.
    
    Args:
        df: DataFrame with customer_id column
        customer_id: Single customer ID (int or str)
        customer_ids: List of customer IDs (int or str)
    
    Returns:
        Tuple of (filtered_df, list_of_customer_ids_found, list_of_customer_ids_not_found)
    """
    # Validate that both aren't provided
    if customer_id is not None and customer_ids is not None:
        raise ValueError("Cannot provide both customer_id and customer_ids. Use one or the other.")
    
    # If neither provided, return original dataframe
    if customer_id is None and customer_ids is None:
        return df, [], []
    
    # Combine single and list parameters into one list
    target_ids = []
    if customer_id is not None:
        target_ids = [customer_id]
    elif customer_ids is not None:
        target_ids = customer_ids
    
    # Convert all to int (handle str to int conversion)
    target_ids_int = []
    for cid in target_ids:
        try:
            target_ids_int.append(int(cid))
        except (ValueError, TypeError):
            raise ValueError(f"Invalid customer_id format: {cid}. Must be int or convertible to int.")
    
    # Ensure customer_id column exists
    if 'customer_id' not in df.columns:
        raise ValueError("DataFrame does not contain 'customer_id' column")
    
    # Convert customer_id column to int for comparison (handle mixed types)
    df_customer_ids = pd.to_numeric(df['customer_id'], errors='coerce')
    
    # Filter dataframe
    mask = df_customer_ids.isin(target_ids_int)
    filtered_df = df[mask].copy()
    
    # Find which IDs were found and which were not
    found_ids = filtered_df['customer_id'].unique().tolist()
    found_ids_int = [int(cid) for cid in found_ids]
    not_found_ids = [cid for cid in target_ids_int if cid not in found_ids_int]
    
    return filtered_df, found_ids_int, not_found_ids


@tool
def predict_clv_tool(horizon_days: int = 90, limit_customers: int = 10, customer_id: Optional[Union[int, str]] = None, customer_ids: Optional[List[Union[int, str]]] = None) -> str:
    """Predict Customer Lifetime Value (CLV) using BG/NBD and Gamma-Gamma models. Use this for questions about: customer lifetime value, CLV, future customer value, predicted revenue per customer, customer worth, or which customers are most valuable. Calibration cutoff date is fixed at 2011-12-09. Parameters: horizon_days (default: 90), limit_customers (default: 10), customer_id (optional, single customer ID), customer_ids (optional, list of customer IDs). If customer_id or customer_ids provided, returns predictions only for those customers. If not provided, returns top N customers (default behavior)."""
    try:
        transactions_df = get_transactions_df()
        cutoff_date = FIXED_CUTOFF_DATE
        
        # Suppress RuntimeWarning about invalid value encountered in log
        # This is a known issue with the lifetimes library when processing edge cases
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in log')
            
            rfm = build_rfm(transactions_df, cutoff_date=cutoff_date)
            models = fit_models(rfm)
            
            pred_unscaled = predict_clv(models, horizon_days=horizon_days, aov_fallback="global_mean")
            pred_total_purchases = pred_unscaled["pred_purchases"].sum()
            pred_total_revenue = pred_unscaled["clv"].sum(skipna=True)
            
            target_purchases = pred_total_purchases * PURCHASE_SCALE if PURCHASE_SCALE != 1.0 else None
            target_revenue = pred_total_revenue * REVENUE_SCALE if REVENUE_SCALE != 1.0 else None
            
            pred = predict_clv(
                models,
                horizon_days=horizon_days,
                scale_to_target_purchases=target_purchases,
                scale_to_target_revenue=target_revenue,
                aov_fallback="global_mean"
            )
        
        # Filter by customer_id(s) if provided
        if customer_id is not None or customer_ids is not None:
            pred, found_ids, not_found_ids = filter_by_customer_ids(pred, customer_id, customer_ids)
            
            if len(pred) == 0:
                return json.dumps({
                    "status": "error",
                    "error": f"Customer(s) not found: {not_found_ids}",
                    "error_type": "NotFoundError"
                })
            
            if len(not_found_ids) > 0:
                # Some customers not found, but return results for found ones
                warning = f"Warning: Some customers not found: {not_found_ids}. Returning results for found customers: {found_ids}"
            else:
                warning = None
        else:
            # Default behavior: top N customers
            pred = pred.sort_values("clv", ascending=False).head(limit_customers)
            found_ids = []
            not_found_ids = []
            warning = None
        
        result = {
            "status": "success",
            "total_customers": len(pred),
            "horizon_days": horizon_days,
            "summary": {
                "top_clv": float(pred['clv'].max()),
                "mean_clv": float(pred['clv'].mean()),
                "min_clv": float(pred['clv'].min()),
            },
            "customers": pred[["customer_id", "clv", "pred_purchases", "monetary_value"]].to_dict(orient="records")
        }
        
        if warning:
            result["warning"] = warning
        if found_ids:
            result["customer_ids_found"] = found_ids
        if not_found_ids:
            result["customer_ids_not_found"] = not_found_ids
        
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e), "error_type": type(e).__name__})


@tool
def score_churn_risk_tool(question: str = "", customer_id: Optional[Union[int, str]] = None, customer_ids: Optional[List[Union[int, str]]] = None) -> str:
    """Rank and score customers by relative churn risk (risk_score, risk_rank, risk_bucket: High/Medium/Low). Returns RISK SCORES for ranking/prioritization, NOT probabilities. Use this for: ranking customers by risk, identifying high-risk customers, risk-based prioritization, or which customers need attention first. Returns risk_score (higher = higher risk), risk_rank, risk_percentile, and risk_bucket. Parameters: customer_id (optional, single customer ID), customer_ids (optional, list of customer IDs). If customer_id or customer_ids provided, returns scores only for those customers. If not provided, returns top 25 high-risk customers (default behavior). Do NOT use for probability questions."""
    try:
        transactions_df = get_transactions_df()
        cutoff_date = FIXED_CUTOFF_DATE
        inactivity_days = FIXED_INACTIVITY_DAYS
        
        cox_result = get_or_fit_cox_model(transactions_df, cutoff_date, inactivity_days)
        
        scored = score_customers(
            model=cox_result['model'],
            transactions=transactions_df,
            cutoff_date=cutoff_date,
        )
        
        # Filter by customer_id(s) if provided
        if customer_id is not None or customer_ids is not None:
            scored, found_ids, not_found_ids = filter_by_customer_ids(scored, customer_id, customer_ids)
            
            if len(scored) == 0:
                return json.dumps({
                    "status": "error",
                    "error": f"Customer(s) not found: {not_found_ids}",
                    "error_type": "NotFoundError"
                })
            
            if len(not_found_ids) > 0:
                warning = f"Warning: Some customers not found: {not_found_ids}. Returning results for found customers: {found_ids}"
            else:
                warning = None
        else:
            # Default behavior: top 25 high-risk
            scored = scored.head(25)
            found_ids = []
            not_found_ids = []
            warning = None
        
        high_risk = (scored["risk_bucket"] == "High").sum()
        medium_risk = (scored["risk_bucket"] == "Medium").sum()
        low_risk = (scored["risk_bucket"] == "Low").sum()
        
        result = {
            "status": "success",
            "total_customers": len(scored),
            "risk_distribution": {
                "high_risk_count": int(high_risk),
                "medium_risk_count": int(medium_risk),
                "low_risk_count": int(low_risk),
            },
            "customers": scored[["customer_id", "risk_score", "risk_rank", "risk_bucket", "risk_percentile"]].to_dict(orient="records")
        }
        
        if warning:
            result["warning"] = warning
        if found_ids:
            result["customer_ids_found"] = found_ids
        if not_found_ids:
            result["customer_ids_not_found"] = not_found_ids
        
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e), "error_type": type(e).__name__})


@tool
def predict_churn_probability_tool(X_days: int = 90, customer_id: Optional[Union[int, str]] = None, customer_ids: Optional[List[Union[int, str]]] = None) -> str:
    """Predict the PROBABILITY (0-1) that active customers will churn in the next X days. Returns actual churn probabilities, not risk scores. Use this for: 'what is the probability customer X will churn in 90 days?', 'likelihood of churn', 'churn probability', or 'probability of leaving in X days'. Returns churn_probability (0.0 to 1.0), survival_at_t0, and survival_at_t0_plus_X. Parameters: X_days (default: 90), customer_id (optional, single customer ID), customer_ids (optional, list of customer IDs). If customer_id or customer_ids provided, returns predictions only for those customers. If not provided, returns top 25 highest probability customers (default behavior). Do NOT use for risk ranking or lifetime questions."""
    try:
        transactions_df = get_transactions_df()
        cutoff_date = FIXED_CUTOFF_DATE
        inactivity_days = FIXED_INACTIVITY_DAYS
        
        cox_result = get_or_fit_cox_model(transactions_df, cutoff_date, inactivity_days)
        
        predictions = predict_churn_probability(
            model=cox_result['model'],
            transactions=transactions_df,
            cutoff_date=cutoff_date,
            X_days=X_days,
            inactivity_days=inactivity_days,
        )
        
        # Filter by customer_id(s) if provided
        if customer_id is not None or customer_ids is not None:
            predictions, found_ids, not_found_ids = filter_by_customer_ids(predictions, customer_id, customer_ids)
            
            if len(predictions) == 0:
                return json.dumps({
                    "status": "error",
                    "error": f"Customer(s) not found: {not_found_ids}",
                    "error_type": "NotFoundError"
                })
            
            if len(not_found_ids) > 0:
                warning = f"Warning: Some customers not found: {not_found_ids}. Returning results for found customers: {found_ids}"
            else:
                warning = None
        else:
            # Default behavior: top 25 highest probability
            predictions = predictions.head(25)
            found_ids = []
            not_found_ids = []
            warning = None
        
        result = {
            "status": "success",
            "total_customers": len(predictions),
            "X_days": X_days,
            "summary": {
                "mean_churn_probability": float(predictions["churn_probability"].mean()),
                "max_churn_probability": float(predictions["churn_probability"].max()),
                "min_churn_probability": float(predictions["churn_probability"].min()),
                "median_churn_probability": float(predictions["churn_probability"].median()),
            },
            "customers": predictions[["customer_id", "churn_probability", "t0", "X_days"]].to_dict(orient="records")
        }
        
        if warning:
            result["warning"] = warning
        if found_ids:
            result["customer_ids_found"] = found_ids
        if not_found_ids:
            result["customer_ids_not_found"] = not_found_ids
        
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e), "error_type": type(e).__name__})


@tool
def expected_remaining_lifetime_tool(H_days: int = 365, customer_id: Optional[Union[int, str]] = None, customer_ids: Optional[List[Union[int, str]]] = None) -> str:
    """Compute EXPECTED REMAINING LIFETIME in days for active customers (how long they will stay). Returns expected_remaining_life_days (number of days), NOT probabilities or risk scores. Use this for: 'how long will customer X stay?', 'expected remaining lifetime', 'customer lifetime expectancy', 'how many days until churn', or 'remaining customer duration'. Returns expected_remaining_life_days (numeric days), t0 (current tenure), and H_days (horizon). Parameters: H_days (default: 365), customer_id (optional, single customer ID), customer_ids (optional, list of customer IDs). If customer_id or customer_ids provided, returns predictions only for those customers. If not provided, returns top 25 longest lifetime customers (default behavior). Do NOT use for probability or risk ranking questions."""
    try:
        transactions_df = get_transactions_df()
        cutoff_date = FIXED_CUTOFF_DATE
        inactivity_days = FIXED_INACTIVITY_DAYS
        
        cox_result = get_or_fit_cox_model(transactions_df, cutoff_date, inactivity_days)
        
        expected_lifetimes = expected_remaining_lifetime(
            model=cox_result['model'],
            covariates_df=cox_result['train_df'],
            H_days=H_days,
            inactivity_days=inactivity_days,
        )
        
        # Filter by customer_id(s) if provided
        if customer_id is not None or customer_ids is not None:
            expected_lifetimes, found_ids, not_found_ids = filter_by_customer_ids(expected_lifetimes, customer_id, customer_ids)
            
            if len(expected_lifetimes) == 0:
                return json.dumps({
                    "status": "error",
                    "error": f"Customer(s) not found: {not_found_ids}",
                    "error_type": "NotFoundError"
                })
            
            if len(not_found_ids) > 0:
                warning = f"Warning: Some customers not found: {not_found_ids}. Returning results for found customers: {found_ids}"
            else:
                warning = None
        else:
            # Default behavior: top 25 longest lifetime
            expected_lifetimes = expected_lifetimes.head(25)
            found_ids = []
            not_found_ids = []
            warning = None
        
        result = {
            "status": "success",
            "total_customers": len(expected_lifetimes),
            "H_days": H_days,
            "summary": {
                "mean_expected_lifetime_days": float(expected_lifetimes["expected_remaining_life_days"].mean()),
                "max_expected_lifetime_days": float(expected_lifetimes["expected_remaining_life_days"].max()),
                "min_expected_lifetime_days": float(expected_lifetimes["expected_remaining_life_days"].min()),
                "median_expected_lifetime_days": float(expected_lifetimes["expected_remaining_life_days"].median()),
            },
            "customers": expected_lifetimes[["customer_id", "expected_remaining_life_days", "t0"]].to_dict(orient="records")
        }
        
        if warning:
            result["warning"] = warning
        if found_ids:
            result["customer_ids_found"] = found_ids
        if not_found_ids:
            result["customer_ids_not_found"] = not_found_ids
        
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e), "error_type": type(e).__name__})


@tool
def customer_segmentation_tool(H_days: int = 365, customer_id: Optional[Union[int, str]] = None, customer_ids: Optional[List[Union[int, str]]] = None) -> str:
    """Build comprehensive customer segmentation combining RISK LABELS (High/Medium/Low from churn risk) and EXPECTED REMAINING LIFETIME buckets (Short/Medium/Long). Returns 9 segments (e.g., 'High-Long', 'Medium-Medium', 'Low-Short') with action tags and recommended actions. Use this for: customer segmentation, segment analysis, action recommendations, customer groups by risk and lifetime, strategic customer management, or which actions to take for different customer types. Parameters: H_days (default: 365), customer_id (optional, single customer ID), customer_ids (optional, list of customer IDs). If customer_id or customer_ids provided, returns segmentation only for those customers. If not provided, returns all customers with samples per segment (default behavior). This is a COMPREHENSIVE segmentation that combines both risk and lifetime - do NOT use for individual risk scores, probabilities, or lifetime values alone."""
    try:
        transactions_df = get_transactions_df()
        cutoff_date = FIXED_CUTOFF_DATE
        inactivity_days = FIXED_INACTIVITY_DAYS
        
        cox_result = get_or_fit_cox_model(transactions_df, cutoff_date, inactivity_days)
        
        segmentation_df, cutoffs = build_segmentation_table(
            model=cox_result['model'],
            transactions=transactions_df,
            covariates_df=cox_result['train_df'],
            cutoff_date=cutoff_date,
            H_days=H_days,
        )
        
        # Filter by customer_id(s) if provided
        if customer_id is not None or customer_ids is not None:
            segmentation_df, found_ids, not_found_ids = filter_by_customer_ids(segmentation_df, customer_id, customer_ids)
            
            if len(segmentation_df) == 0:
                return json.dumps({
                    "status": "error",
                    "error": f"Customer(s) not found: {not_found_ids}",
                    "error_type": "NotFoundError"
                })
            
            if len(not_found_ids) > 0:
                warning = f"Warning: Some customers not found: {not_found_ids}. Returning results for found customers: {found_ids}"
            else:
                warning = None
        else:
            found_ids = []
            not_found_ids = []
            warning = None
        
        segment_counts = segmentation_df['segment'].value_counts().to_dict()
        segment_summary = {}
        for segment in segment_counts.keys():
            segment_data = segmentation_df[segmentation_df['segment'] == segment]
            # If filtered, show all customers; if not filtered, show sample
            if customer_id is not None or customer_ids is not None:
                customers = segment_data[["customer_id", "risk_label", "life_bucket", "action_tag", "recommended_action"]].to_dict(orient="records")
            else:
                customers = segment_data.head(5)[["customer_id", "risk_label", "life_bucket", "action_tag", "recommended_action"]].to_dict(orient="records")
            segment_summary[segment] = {
                "count": int(segment_counts[segment]),
                "customers": customers
            }
        
        result = {
            "status": "success",
            "total_customers": len(segmentation_df),
            "total_segments": len(segment_counts),
            "segment_distribution": {k: int(v) for k, v in segment_counts.items()},
            "segment_details": segment_summary,
            "cutoffs": cutoffs
        }
        
        if warning:
            result["warning"] = warning
        if found_ids:
            result["customer_ids_found"] = found_ids
        if not_found_ids:
            result["customer_ids_not_found"] = not_found_ids
        
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e), "error_type": type(e).__name__})


@tool
def execute_sql_query_tool(sql: str, explanation: str = "") -> str:
    """Execute a SQL SELECT query to answer questions about historical data, aggregations, filtering, reporting, or data exploration. Use this for descriptive questions that don't require predictive modeling, such as: revenue by country, top customers, sales trends, product analysis, or any data aggregation/filtering questions. Parameters: sql (required, SQL SELECT query), explanation (optional, description of query)."""
    try:
        validate_sql(sql)
        rows, cols = run_query(sql, limit=1000)
        
        result = {
            "status": "success",
            "total_rows": len(rows),
            "columns": cols,
        }
        
        if len(rows) == 0:
            result["message"] = "Query executed successfully but returned no results"
            return json.dumps(result, indent=2, default=str)
        
        if len(rows) > 100:
            result["note"] = f"Showing first 25 rows out of {len(rows)} total rows"
            result["sample_rows"] = rows[:25]
        else:
            result["rows"] = rows
        
        return json.dumps(result, indent=2, default=str)
    except ValueError as e:
        return json.dumps({"status": "error", "error": str(e), "error_type": "ValidationError"})
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e), "error_type": type(e).__name__})


# Define LangChain Tools
tools = [
    predict_clv_tool,
    score_churn_risk_tool,
    predict_churn_probability_tool,
    expected_remaining_lifetime_tool,
    customer_segmentation_tool,
    execute_sql_query_tool,
]

# Initialize LLM
llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    temperature=0,
)

# Load database schema
try:
    SCHEMA = get_schema()
    SCHEMA_JSON = json.dumps(SCHEMA, indent=2)
except Exception as e:
    # Fallback schema if loading fails
    SCHEMA_JSON = json.dumps({
        "transactions": [
            {"name": "customer_id", "type": "INTEGER"},
            {"name": "invoice_no", "type": "TEXT"},
            {"name": "invoice_date", "type": "TEXT"},
            {"name": "revenue", "type": "REAL"},
            {"name": "stock_code", "type": "TEXT"},
            {"name": "country", "type": "TEXT"},
            {"name": "quantity", "type": "INTEGER"},
            {"name": "unit_price", "type": "REAL"},
            {"name": "description", "type": "TEXT"}
        ]
    }, indent=2)

# Enhanced system prompt with clearer tool selection guidance and schema
SYSTEM_PROMPT = f"""You are a data assistant for an Online Retail SQLite database.
You can answer questions using either SQL queries or specialized analytics functions.

DATABASE SCHEMA:
{SCHEMA_JSON}

TOOL SELECTION GUIDE:

1. execute_sql_query - Use for HISTORICAL/DESCRIPTIVE questions:
   ✓ Revenue by country/month/product
   ✓ Top customers/products by sales
   ✓ Sales trends, aggregations, counts, sums, averages
   ✓ Filtering, grouping, sorting historical data
   ✓ "What happened" questions
   ✓ "Show me" questions about past data
   ✗ Do NOT use for predictions, probabilities, or future values

2. score_churn_risk - Use for RISK RANKING questions:
   ✓ "Which customers are at high risk?"
   ✓ "Rank customers by churn risk"
   ✓ "Show me high-risk customers"
   ✓ Returns: risk_score, risk_rank, risk_bucket (High/Medium/Low)
   ✗ Do NOT use for probabilities or "how long" questions

3. predict_churn_probability - Use for PROBABILITY questions:
   ✓ "What is the probability customer X will churn?"
   ✓ "Likelihood of churn"
   ✓ "Churn probability"
   ✓ Returns: churn_probability (0.0 to 1.0)
   ✗ Do NOT use for risk ranking or lifetime questions

4. expected_remaining_lifetime - Use for LIFETIME/DURATION questions:
   ✓ "How long will customer X stay?"
   ✓ "Expected remaining lifetime"
   ✓ "How many days until churn?"
   ✓ Returns: expected_remaining_life_days (number of days)
   ✗ Do NOT use for probabilities or risk ranking

5. customer_segmentation - Use for SEGMENTATION/ACTION questions:
   ✓ "Show me customer segments"
   ✓ "What actions should I take?"
   ✓ "Customer groups by risk and lifetime"
   ✓ Returns: segments (High-Long, Medium-Medium, etc.) with actions
   ✗ Do NOT use for individual metrics (use specific tools instead)

6. predict_customer_lifetime_value - Use for CLV/VALUE questions:
   ✓ "Customer lifetime value"
   ✓ "CLV"
   ✓ "Future customer value"
   ✓ "Which customers are most valuable?"

DECISION RULES:
- If question asks about "risk" or "ranking" → use score_churn_risk
- If question asks about "probability" or "likelihood" → use predict_churn_probability
- If question asks about "how long" or "lifetime" → use expected_remaining_lifetime
- If question asks about "segments" or "actions" → use customer_segmentation
- If question asks about "value" or "CLV" → use predict_customer_lifetime_value
- If question asks about historical data, aggregations, or descriptive statistics → use execute_sql_query
- For multi-step questions, chain tools in logical order (e.g., get high-risk customers first, then their probabilities)

Always choose the most appropriate tool(s) and chain them if needed for complex questions."""

# Initialize MemorySaver for proper conversation memory
memory = MemorySaver()

# Create agent using LangChain's create_agent (replaces deprecated create_react_agent)
agent_executor = create_agent(
    model=llm,
    tools=tools,
    checkpointer=memory,
    system_prompt=SYSTEM_PROMPT,
)



def ask_question(question: str, use_memory: bool = True, thread_id: str = "default") -> str:
    """
    Ask a question and get response from LangGraph agent.
    
    Args:
        question: Natural language question
        use_memory: Whether to use conversation memory (default: True)
        thread_id: Thread ID for conversation memory (default: "default")
    
    Returns:
        Natural language answer from the agent
    """
    from langchain_core.messages import HumanMessage
    
    try:
        # Configure memory based on use_memory flag
        if use_memory:
            config = {"configurable": {"thread_id": thread_id}}
        else:
            # Use a temporary thread_id that won't persist
            config = {"configurable": {"thread_id": f"temp_{uuid.uuid4().hex[:8]}"}}
        
        # Invoke agent with proper configuration
        response = agent_executor.invoke(
            {"messages": [HumanMessage(content=question)]},
            config=config
        )
        
        # Extract the final answer from messages
        if isinstance(response, dict):
            messages = response.get("messages", [])
            if messages:
                # Find the last AIMessage with content
                for msg in reversed(messages):
                    if isinstance(msg, AIMessage) and msg.content:
                        answer = str(msg.content)
                        break
                else:
                    # Fallback: use last message
                    last_msg = messages[-1]
                    answer = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
            else:
                answer = response.get("output", str(response))
        else:
            answer = str(response)
        
        return answer
        
    except RateLimitError as e:
        return f"Rate limit exceeded. Please wait a minute and try again. Error: {str(e)}"
    except APITimeoutError as e:
        return f"Request timed out. The query may be too complex. Please try again or simplify your question. Error: {str(e)}"
    except APIConnectionError as e:
        return f"Connection error. Please check your internet connection and try again. Error: {str(e)}"
    except APIError as e:
        return f"OpenAI API error: {str(e)}. Please try again later."
    except ValueError as e:
        return f"Validation error: {str(e)}. Please check your question and try again."
    except Exception as e:
        return f"Error processing question: {str(e)}. Please try rephrasing your question."


def clear_memory(thread_id: str = "default"):
    """Clear the conversation memory for a specific thread"""
    # MemorySaver stores state per thread_id
    # To clear, we can use a new thread_id or the memory will naturally expire
    # For now, users should use a new thread_id to start fresh
    # In production, you might implement thread deletion if LangGraph supports it
    pass
