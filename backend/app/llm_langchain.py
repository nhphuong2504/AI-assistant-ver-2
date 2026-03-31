import os
import json
import re
import time
import hashlib
import uuid
import warnings
import sqlite3
from typing import Dict, Any, Optional, Tuple, Union, List
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import create_agent
from langchain_community.callbacks import get_openai_callback
from langgraph.checkpoint.memory import MemorySaver
from openai import RateLimitError, APIError, APIConnectionError, APITimeoutError
import pandas as pd
from analytics.clv import predict_clv, PURCHASE_SCALE, REVENUE_SCALE
from analytics.survival import (
    build_covariate_table,
    fit_cox_baseline,
    score_customers,
    predict_churn_probability,
    prioritize_retention_targets,
    build_segmentation_table,
    CUTOFF_DATE,
    INACTIVITY_DAYS,
)
from analytics.monte_carlo import compute_erl_days
from app.db import run_query, run_query_internal, get_schema, ensure_select_only
from app.data import get_transactions_df, get_clv_models

load_dotenv()

# Model fit cache (for expensive Cox model fits)
_model_cache: Dict[str, Tuple[Any, float]] = {}
CACHE_TTL = 3600  # 1 hour cache TTL


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
        train_frac=1.0,
        random_state=42,
        penalizer=0.1,
    )
    
    # Cache the model
    cache_cox_model(cache_key, cox_result['model'])
    
    # Use full covariate table as train_df so downstream (build_segmentation_table, etc.)
    # get customer_id, tenure_days, monetary_value, etc.
    cox_result['train_df'] = cov
    
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
def predict_clv_tool(horizon_days: int = 90, limit_customers: int = 10, order: str = "descending", customer_id: Optional[Union[int, str]] = None, customer_ids: Optional[List[Union[int, str]]] = None) -> str:
    """Predict forward-looking Customer Lifetime Value (CLV) using BG/NBD and Gamma-Gamma models. Use for future value, predicted revenue, and most valuable customers. Returns top N by CLV and supports customer_id/customer_ids filtering. Do NOT use for historical sales totals."""
    try:
        transactions_df = get_transactions_df()
        cutoff_date = CUTOFF_DATE
        
        # Suppress RuntimeWarning about invalid value encountered in log
        # This is a known issue with the lifetimes library when processing edge cases
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in log')
            
            models = get_clv_models(cutoff_date)
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
            # Default behavior: top N customers (order: descending=highest first, ascending=lowest first)
            pred = pred.sort_values("clv", ascending=(order == "ascending"), na_position="last").head(limit_customers)
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
def score_churn_risk_tool(limit_customers: int = 10, order: str = "descending", customer_id: Optional[Union[int, str]] = None, customer_ids: Optional[List[Union[int, str]]] = None) -> str:
    """Rank ACTIVE customers by relative churn risk using risk_score, risk_rank, and risk_bucket. Returns ranking-oriented risk scores (not probabilities) and flags already-churned customers separately. Use for high-risk lists and risk-based prioritization. Supports customer_id/customer_ids filtering."""
    try:
        transactions_df = get_transactions_df()
        cutoff_date = CUTOFF_DATE
        inactivity_days = INACTIVITY_DAYS
        
        cox_result = get_or_fit_cox_model(transactions_df, cutoff_date, inactivity_days)
        train_df = cox_result["train_df"]
        churned_ids = set(train_df.loc[train_df["event"] == 1, "customer_id"].astype(int).tolist())

        # Score active customers only (already churned are skipped and labeled)
        scored = score_customers(
            model=cox_result["model"],
            transactions=transactions_df,
            cutoff_date=cutoff_date,
            covariate_df=train_df,
        )

        # Filter by customer_id(s) if provided
        result_note_extra = ""
        if customer_id is not None or customer_ids is not None:
            if customer_id is not None:
                target_ids = [customer_id]
            else:
                target_ids = list(customer_ids)
            target_ids_int = []
            for cid in target_ids:
                try:
                    target_ids_int.append(int(cid))
                except (TypeError, ValueError):
                    pass
            churned_requested = [c for c in target_ids_int if c in churned_ids]
            active_requested = [c for c in target_ids_int if c not in churned_ids]
            scored, found_ids, not_found_ids = filter_by_customer_ids(scored, customer_ids=active_requested)

            if len(scored) == 0 and len(churned_requested) == 0:
                return json.dumps({
                    "status": "error",
                    "error": f"Customer(s) not found: {not_found_ids}",
                    "error_type": "NotFoundError"
                })

            if len(not_found_ids) > 0:
                warning = f"Warning: Some customers not found: {not_found_ids}. Returning results for found customers: {found_ids}"
            else:
                warning = None
            already_churned_in_response = churned_requested
            already_churned_count_val = len(already_churned_in_response)
        else:
            # Default behavior: top N by risk (order: descending=highest risk first, ascending=lowest/safest first)
            scored = scored.sort_values("risk_score", ascending=(order == "ascending")).head(limit_customers)
            found_ids = []
            not_found_ids = []
            warning = None
            # When no filter, cap listed IDs to avoid huge payloads
            _churned_list = list(churned_ids)
            already_churned_in_response = _churned_list[:500]
            already_churned_count_val = len(_churned_list)
            if len(_churned_list) > 500:
                result_note_extra = f" (showing first 500 of {len(_churned_list)} already-churned customers)"

        high_risk = (scored["risk_bucket"] == "High").sum() if len(scored) > 0 else 0
        medium_risk = (scored["risk_bucket"] == "Medium").sum() if len(scored) > 0 else 0
        low_risk = (scored["risk_bucket"] == "Low").sum() if len(scored) > 0 else 0

        result = {
            "status": "success",
            "total_customers": len(scored),
            "already_churned_count": already_churned_count_val,
            "already_churned_customer_ids": already_churned_in_response,
            "note": "Churn risk is computed for active customers only. Already churned customers (no purchases for 90+ days as of cutoff) are listed above and not scored." + result_note_extra,
            "risk_distribution": {
                "high_risk_count": int(high_risk),
                "medium_risk_count": int(medium_risk),
                "low_risk_count": int(low_risk),
            },
            "customers": scored[["customer_id", "risk_score", "risk_rank", "risk_bucket", "risk_percentile"]].to_dict(orient="records") if len(scored) > 0 else []
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
def predict_churn_probability_tool(X_days: int = 90, limit_customers: int = 10, order: str = "descending", customer_id: Optional[Union[int, str]] = None, customer_ids: Optional[List[Union[int, str]]] = None) -> str:
    """Predict churn probability (0-1) for ACTIVE customers over the next X_days. Use for likelihood/chance questions and probability-based churn analysis. Returns churn_probability with survival context and supports customer_id/customer_ids filtering. Do NOT use for risk-score ranking or lifetime-duration questions."""
    try:
        transactions_df = get_transactions_df()
        cutoff_date = CUTOFF_DATE
        inactivity_days = INACTIVITY_DAYS
        
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
            # Default behavior: top N by churn probability (order: descending=highest first, ascending=lowest first)
            predictions = predictions.sort_values("churn_probability", ascending=(order == "ascending")).head(limit_customers)
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
def prioritize_retention_targets_tool(prediction_horizon: int = 90, limit_customers: int = 10, order: str = "descending") -> str:
    """Prioritize ACTIVE customers for retention by combining CLV and churn probability into prioritize_score (clv * churn_prob). Use for action-oriented questions like who to contact/save first. Returns highest-impact retention targets ranked by business value at risk."""
    try:
        transactions_df = get_transactions_df()
        cutoff_date = CUTOFF_DATE
        inactivity_days = INACTIVITY_DAYS

        cox_result = get_or_fit_cox_model(transactions_df, cutoff_date, inactivity_days)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in log")
            models = get_clv_models(cutoff_date)
            pred_unscaled = predict_clv(models, horizon_days=prediction_horizon, aov_fallback="global_mean")
            pred_total_purchases = pred_unscaled["pred_purchases"].sum()
            pred_total_revenue = pred_unscaled["clv"].sum(skipna=True)
            target_purchases = pred_total_purchases * PURCHASE_SCALE if PURCHASE_SCALE != 1.0 else None
            target_revenue = pred_total_revenue * REVENUE_SCALE if REVENUE_SCALE != 1.0 else None
            clv_df = predict_clv(
                models,
                horizon_days=prediction_horizon,
                scale_to_target_purchases=target_purchases,
                scale_to_target_revenue=target_revenue,
                aov_fallback="global_mean",
            )

        prioritized = prioritize_retention_targets(
            model=cox_result["model"],
            transactions=transactions_df,
            clv_df=clv_df,
            prediction_horizon=prediction_horizon,
            cutoff_date=cutoff_date,
            inactivity_days=inactivity_days,
        )

        prioritized = prioritized.sort_values("prioritize_score", ascending=(order == "ascending")).head(limit_customers)

        result = {
            "status": "success",
            "total_customers": len(prioritized),
            "prediction_horizon_days": prediction_horizon,
            "summary": {
                "mean_prioritize_score": float(prioritized["prioritize_score"].mean()),
                "max_prioritize_score": float(prioritized["prioritize_score"].max()),
                "min_prioritize_score": float(prioritized["prioritize_score"].min()),
            },
            "customers": prioritized[["customer_id", "clv", "churn_prob", "prioritize_score"]].to_dict(orient="records"),
        }
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e), "error_type": type(e).__name__})


@tool
def compute_erl_days_tool(limit_customers: int = 10, order: str = "descending", customer_id: Optional[Union[int, str]] = None, customer_ids: Optional[List[Union[int, str]]] = None) -> str:
    """Compute expected remaining lifetime (ERL) in days for customers. Use for 'how long will they stay' or 'days until churn' questions. Returns expected_remaining_life_days and tenure context, and supports customer_id/customer_ids filtering. Use order='ascending' for soonest expected churn. Do NOT use for probability or risk-score requests."""
    try:
        transactions_df = get_transactions_df()
        cutoff_date = CUTOFF_DATE
        inactivity_days = INACTIVITY_DAYS
        
        clv_result = get_clv_models(cutoff_date)
        rfm = clv_result.rfm
        transactions_df = transactions_df.copy()
        transactions_df["invoice_date"] = pd.to_datetime(transactions_df["invoice_date"])
        cutoff_dt = pd.to_datetime(cutoff_date)
        last_purchases = (
            transactions_df[transactions_df["invoice_date"] <= cutoff_dt]
            .groupby("customer_id")["invoice_date"]
            .max()
            .reset_index()
        )
        last_purchases.columns = ["customer_id", "last_purchase_date"]
        last_purchases["last_purchase_date"] = last_purchases["last_purchase_date"].dt.strftime("%Y-%m-%d")
        customer_summary = rfm.reset_index().merge(last_purchases, on="customer_id", how="left")
        erl_result = compute_erl_days(
            bgf=clv_result.bgnbd,
            customer_summary_df=customer_summary,
            cutoff_date=cutoff_date,
            INACTIVITY_DAYS=inactivity_days,
        )
        expected_lifetimes = erl_result.merge(
            customer_summary[["customer_id", "T"]], on="customer_id", how="left"
        ).rename(columns={"ERL_days": "expected_remaining_life_days", "T": "t0"})
        
        # Add flag to indicate if customer is already churned (ERL = 0 means already churned by business rule)
        expected_lifetimes["is_already_churned"] = (
            expected_lifetimes["expected_remaining_life_days"] == 0.0
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
            # Default behavior: Filter out already churned customers, then sort and return top N
            # This ensures "most likely to churn soonest" returns active customers only
            active_only = expected_lifetimes[~expected_lifetimes["is_already_churned"]].copy()
            
            if len(active_only) == 0:
                return json.dumps({
                    "status": "error",
                    "error": "No active customers found. All customers are already churned.",
                    "error_type": "NoActiveCustomersError"
                })
            
            expected_lifetimes = active_only.sort_values("expected_remaining_life_days", ascending=(order == "ascending")).head(limit_customers)
            found_ids = []
            not_found_ids = []
            warning = None

        # Calculate summary statistics (excluding already churned customers for mean/median)
        active_customers = expected_lifetimes[~expected_lifetimes["is_already_churned"]]
        already_churned_count = int(expected_lifetimes["is_already_churned"].sum())
        
        result = {
            "status": "success",
            "total_customers": len(expected_lifetimes),
            "summary": {
                "mean_expected_lifetime_days": float(expected_lifetimes["expected_remaining_life_days"].mean()),
                "max_expected_lifetime_days": float(expected_lifetimes["expected_remaining_life_days"].max()),
                "min_expected_lifetime_days": float(expected_lifetimes["expected_remaining_life_days"].min()),
                "median_expected_lifetime_days": float(expected_lifetimes["expected_remaining_life_days"].median()),
                "already_churned_count": already_churned_count,
                "active_customers_count": len(active_customers),
            },
            "customers": expected_lifetimes[["customer_id", "expected_remaining_life_days", "t0", "is_already_churned"]].to_dict(orient="records")
        }
        
        # Add note if there are already churned customers
        if already_churned_count > 0:
            result["note"] = f"{already_churned_count} customer(s) have 0 days expected remaining lifetime, meaning they are already churned (no purchases for {inactivity_days}+ days as of cutoff date {cutoff_date})."
        
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
    """Build comprehensive customer segmentation by combining churn risk labels and ERL lifetime buckets into 12 actionable segments. Use for strategy/action guidance, segment analysis, and customer-level segment assignment with recommended actions. Supports customer_id/customer_ids filtering. Do NOT use for standalone probability, risk-score, or single-metric lifetime requests."""
    try:
        transactions_df = get_transactions_df()
        cutoff_date = CUTOFF_DATE
        inactivity_days = INACTIVITY_DAYS
        
        cox_result = get_or_fit_cox_model(transactions_df, cutoff_date, inactivity_days)
        
        segmentation_df, cutoffs = build_segmentation_table(
            model=cox_result['model'],
            transactions=transactions_df,
            covariates_df=cox_result['train_df'],
            cutoff_date=cutoff_date,
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
                customers = segment_data.head(10)[["customer_id", "risk_label", "life_bucket", "action_tag", "recommended_action"]].to_dict(orient="records")
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
def get_customer_counts_tool() -> str:
    """Return total customers, churned customers, and active (not churned) customers as of the cutoff date. Use this for total/churned/active customer counts. Churn means no purchases for 90+ days as of 2011-12-09. Returns total_customers, churned_customers, and active_customers."""
    try:
        transactions_df = get_transactions_df()
        cov_result = build_covariate_table(
            transactions=transactions_df,
            cutoff_date=CUTOFF_DATE,
            inactivity_days=INACTIVITY_DAYS,
        )
        cov = cov_result.df
        total = len(cov)
        churned = int((cov["event"] == 1).sum())
        active = int((cov["event"] == 0).sum())
        result = {
            "status": "success",
            "total_customers": total,
            "churned_customers": churned,
            "active_customers": active,
            "cutoff_date": CUTOFF_DATE,
            "inactivity_days": INACTIVITY_DAYS,
            "note": "Churned = no purchases for 90+ days as of cutoff. Active = not churned. total_customers = churned_customers + active_customers.",
        }
        return json.dumps(result, indent=2, default=str)
    except FileNotFoundError as e:
        return json.dumps({
            "status": "error",
            "error": str(e),
            "error_type": "FileNotFoundError",
        })
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e), "error_type": type(e).__name__})


@tool
def execute_sql_query_tool(sql: str, explanation: str = "") -> str:
    """Execute SQL SELECT queries for historical reporting, aggregations, and descriptive analytics (orders, invoices, revenue, products, trends). Parameters: sql (required) and explanation (optional). Do NOT use for prediction tasks, and do NOT use for total/churned/active customer counts; use get_customer_counts_tool for those cohort counts."""
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
    except FileNotFoundError as e:
        error_msg = (
            f"Database file not found. {str(e)}\n"
            f"Please ensure the database has been generated by running the ETL script."
        )
        return json.dumps({"status": "error", "error": error_msg, "error_type": "FileNotFoundError"})
    except ValueError as e:
        return json.dumps({"status": "error", "error": str(e), "error_type": "ValidationError"})
    except sqlite3.Error as e:
        return json.dumps({"status": "error", "error": f"SQLite error: {str(e)}", "error_type": "DatabaseError"})
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e), "error_type": type(e).__name__})


# Define LangChain Tools
tools = [
    predict_clv_tool,
    score_churn_risk_tool,
    predict_churn_probability_tool,
    prioritize_retention_targets_tool,
    compute_erl_days_tool,
    customer_segmentation_tool,
    get_customer_counts_tool,
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
SYSTEM_PROMPT = f"""
You are an expert Data Science Assistant for an Online Retail business.
You have access to a SQLite database and a suite of advanced predictive analytics tools.

Your goal is to answer user questions by selecting the BEST tool.
Follow this hierarchy: **Strategy/Action** > **Specific Prediction** > **Historical Fact**.

========================
RESPONSE SYNTHESIS
========================
1. **Direct Answer:** Start immediately with the answer or key insight. The audience is a business owner or manager who is not technical. 
2. **Context:** Briefly contextualize the metric relative to the analysis cutoff date (2011-12-09). Clarify if the insight is retrospective (past performance) or prospective (forecasted risk/value). Explain the logic of the analysis in a way that is easy to understand for a business owner or manager who is not technical.
3. **Formatting:** Lists: Use concise bullet points for rankings or "Top N" requests. Avoid raw JSON or unformatted code dumps; present data in clean text or markdown tables.
4. **Limited result sets:** When a tool returns a capped list (e.g. top 10 by default), and the user asked for a share of customers (e.g. "20%", "a subset", "who should we prioritize") without a specific number, describe the list explicitly as a subset: e.g. "Here are the **first 10** of the prioritized retention targets" or "Below are the **top 10** (you can ask for more if needed)." Do not say "here are the 10 customers to focus on" in a way that implies the full set is exactly 10.
5. **Uncertainty framing:** For predictions (CLV, churn probability, ERL), describe outputs as model estimates with uncertainty. Prefer phrasing like "the model estimates X% probability" instead of deterministic claims.

========================
DATA CONTEXT (CRITICAL)
========================
- **Cutoff Date:** 2011-12-09 (All predictions are calculated as of this date).
- **Churn Definition:** A customer is 'churned' if they have made NO purchases for 90 consecutive days as of the cutoff date.
- **Future Dates:** You do NOT have future transactions. When asked "when will they churn", estimate using `compute_erl_days_tool` + the cutoff date.
- Money values are in £.

========================
TOOL SELECTION HIERARCHY
========================

### LEVEL 1: STRATEGY & ACTION (Highest Value)

1. **prioritize_retention_targets_tool**
   - **Trigger:** "Who should I contact?", "Who to save?", "High value at risk", "Retention priority".
   - **Why:** Correct when BOTH value and churn likelihood matter.

2. **customer_segmentation_tool**
   - **Trigger:** "Segment my customers", "Analyze the customer base", "What actions should I take?"; or questions about a **specific customer or group of customers**: "What segment is customer X in?", "Which segment are these customers in?", "What action for customer 12346?", "Risk and lifetime bucket for this customer.", "What should we do for customers A and B?".
   - **Why:** Combines Risk and Lifetime into actionable strategic buckets. Use for questions about one or more specific customers (segment, risk label, ERL bucket, recommended action).

### LEVEL 2: SPECIFIC PREDICTIONS

3. **predict_clv_tool**
   - **Trigger:** "CLV", "future value", "predicted revenue", "most valuable customers".
   - **Note:** Forward-looking. Do NOT use for historical sales.

4. **score_churn_risk_tool**
   - **Trigger:** "Rank by risk", "Who is high risk?", "Risk scores".
   - **Note:** Relative ranking, NOT probability.

5. **predict_churn_probability_tool**
   - **Trigger:** "Probability", "likelihood", "chance", "% of churn".
   - **Note:** Returns 0–1 probability.

6. **compute_erl_days_tool**
   - **Trigger:** "How long will they stay?", "Days until churn", "When will they leave?".
   - **Critical:** "Soonest churn" → `order="ascending"`.

### LEVEL 3: HISTORICAL & DESCRIPTIVE

7. **get_customer_counts_tool**
   - **Trigger:** "How many customers do we have?", "How many churned?", "How many active / not churned?", "Total/churned/active customer counts".
   - **Why:** Returns total_customers, churned_customers, active_customers (total = churned + active) from the same cohort. Use this for consistent total/churned/active numbers.

8. **execute_sql_query_tool**
   - **Trigger:** Historical reporting, aggregation, and other COUNTING (e.g. orders, invoices, revenue by country).
   - **Limit:** NEVER use for predictions. For total/churned/active customer counts use **get_customer_counts_tool** instead.

========================
AMBIGUITY RESOLUTION
========================
- User: "Who are our best customers?"
  - Tool: **predict_clv_tool** (forward-looking value, not historical sales).
- User: "Which customers should we save first?"
  - Tool: **prioritize_retention_targets_tool** (value + churn likelihood for action).
- User: "Who will churn soonest?"
  - Tool: **compute_erl_days_tool** with `order="ascending"` (time-to-churn view).
- User: "What's the chance customer 16444 leaves?"
  - Tool: **predict_churn_probability_tool** (probability estimate 0-1).

========================
MULTI-STEP REASONING
========================
For complex requests (budgets, percentages, or multi-constraint decisions):
1. Break the request into sub-goals.
2. Identify the right tool(s) and parameter mapping for each sub-goal.
3. Execute tools in the needed sequence.
4. Synthesize one coherent final answer tied to the user's business objective.

========================
ERROR HANDLING
========================
- If a tool returns `status: error`, explain the failure in plain language and suggest a concrete retry (e.g., refine dates, provide a valid customer ID, or ask for a narrower query).
- If customers are not found, explicitly list the missing IDs and ask the user to verify them.
- If the request is outside available data or tool capabilities, state the limitation clearly and avoid hallucinated values.

========================
PARAMETER MAPPING RULES
========================
- "Next 30/60/90 days" → `X_days` / `horizon_days`
- "6 months" → ~180 days; "1 year" → 365 days
- Default `order="descending"` (Highest Value, Highest Risk, Longest Life)
- "Safest" / "Least Likely" / "Soonest Churn" → `order="ascending"`

========================
COMMON PARAMETERS
========================
- `limit_customers`: Default is 10 when not provided.
- `order`: Use `descending` by default; use `ascending` for lowest/soonest/safest style requests.
- `customer_id`: Single-customer lookup.
- `customer_ids`: Multi-customer lookup.

========================
DATABASE SCHEMA
========================
Use this schema ONLY when constructing SQL for `execute_sql_query_tool`. Follow these exact table and column names.
{SCHEMA_JSON}
"""




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
        
        # Invoke agent with proper configuration and track token usage
        with get_openai_callback() as cb:
            response = agent_executor.invoke(
                {"messages": [HumanMessage(content=question)]},
                config=config
            )

        # Print token usage for this request
        print(
            f"Token usage - prompt: {cb.prompt_tokens}, completion: {cb.completion_tokens}, "
            f"total: {cb.total_tokens}"
            + (f", cost: ${cb.total_cost:.4f}" if getattr(cb, "total_cost", None) is not None else "")
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
