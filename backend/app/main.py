from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from app.db import get_schema, run_query, run_query_internal
from app.llm_langchain import ask_question, clear_memory
import pandas as pd
from analytics.clv import build_rfm, fit_models, predict_clv, PURCHASE_SCALE, REVENUE_SCALE
from analytics.survival import (
    build_covariate_table,
    fit_km_all,
    fit_cox_baseline,
    score_customers,
    predict_churn_probability,
    expected_remaining_lifetime,
    build_segmentation_table,
    CUTOFF_DATE,
    INACTIVITY_DAYS,
)

app = FastAPI(title="Retail Data Assistant API", version="0.1")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#uvicorn app.main:app --reload --host 127.0.0.1 --port 8000


class QueryRequest(BaseModel):
    sql: str = Field(..., description="Read-only SQL query (SELECT / WITH ... SELECT)")
    limit: int = Field(500, ge=1, le=5000, description="Default LIMIT applied if SQL has none")


class QueryResponse(BaseModel):
    columns: List[str]
    rows: List[Dict[str, Any]]
    row_count: int

class AskLangChainRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural language question")
    use_memory: bool = Field(True, description="Whether to use conversation memory")
    thread_id: str = Field("default", description="Thread ID for conversation memory (allows multiple concurrent conversations)")

class AskLangChainResponse(BaseModel):
    question: str
    answer: str

class CLVRequest(BaseModel):
    cutoff_date: str = Field("2011-09-30", description="Calibration cutoff date (YYYY-MM-DD)")
    horizon_days: int = Field(180, ge=1, le=3650)
    limit_customers: int = Field(5000, ge=10, le=200000)


class CLVResponse(BaseModel):
    cutoff_date: str
    horizon_days: int
    top_customers: List[Dict[str, Any]]
    summary: Dict[str, Any]


class KMResponse(BaseModel):
    cutoff_date: str
    inactivity_days: int
    n_customers: int
    churn_rate: float
    survival_curve: List[Dict[str, float]]


class ScoreCustomersResponse(BaseModel):
    cutoff_date: str
    inactivity_days: int
    n_customers: int
    scored_customers: List[Dict[str, Any]]
    summary: Dict[str, Any]


class ChurnProbabilityResponse(BaseModel):
    cutoff_date: str
    inactivity_days: int
    X_days: int
    n_customers: int
    predictions: List[Dict[str, Any]]
    summary: Dict[str, Any]


class ExpectedLifetimeResponse(BaseModel):
    cutoff_date: str
    inactivity_days: int
    H_days: int
    n_customers: int
    expected_lifetimes: List[Dict[str, Any]]
    summary: Dict[str, Any]


class SegmentationResponse(BaseModel):
    cutoff_date: str
    inactivity_days: int
    H_days: int
    n_customers: int
    segments: List[Dict[str, Any]]
    cutoffs: Dict[str, float]
    summary: Dict[str, Any]




@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/schema")
def schema() -> Dict[str, Any]:
    try:
        return get_schema()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    try:
        rows, cols = run_query(req.sql, limit=req.limit)
        return QueryResponse(columns=cols, rows=rows, row_count=len(rows))
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def execute_analytics_function(
    function_name: str,
    parameters: Dict[str, Any],
    transactions_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Execute an analytics function and return results in a dictionary format.
    Fixed values: cutoff_date="2011-12-09", inactivity_days=90 for most functions.
    
    Args:
        function_name: Name of the analytics function to execute
        parameters: Dictionary of function parameters
        transactions_df: DataFrame with transaction data
        
    Returns:
        Dictionary with: columns, rows, row_count, answer
    """
    # Fixed constants
    FIXED_CUTOFF_DATE = "2011-12-09"
    FIXED_INACTIVITY_DAYS = 90
    
    if function_name == "predict_customer_lifetime_value":
        # Fixed cutoff_date at 2011-12-09
        cutoff_date = FIXED_CUTOFF_DATE
        horizon_days = parameters.get("horizon_days", 90)  # Required, but provide default
        limit_customers = parameters.get("limit_customers", 10)  # Updated default to 10
        
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
        
        pred = pred.sort_values("clv", ascending=False).head(limit_customers)
        
        columns = ["customer_id", "frequency", "recency", "T", "monetary_value", 
                  "pred_purchases", "pred_aov", "clv"]
        rows = pred[columns].to_dict(orient="records")
        
        answer = f"Predicted Customer Lifetime Value for {len(pred)} customers over {horizon_days} days (cutoff: {cutoff_date}). "
        answer += f"Top customer CLV: {pred['clv'].max():.2f}, Mean CLV: {pred['clv'].mean():.2f}."
        
        return {
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
            "answer": answer
        }
    
    elif function_name == "score_churn_risk":
        # Fixed cutoff_date and inactivity_days
        cutoff_date = FIXED_CUTOFF_DATE
        inactivity_days = FIXED_INACTIVITY_DAYS
        
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
        
        scored = score_customers(
            model=cox_result['model'],
            transactions=transactions_df,
            cutoff_date=cutoff_date,
        )
        
        columns = ["customer_id", "n_orders", "log_monetary_value", "product_diversity",
                  "risk_score", "risk_rank", "risk_percentile", "risk_bucket"]
        rows = scored[columns].to_dict(orient="records")
        
        high_risk = (scored["risk_bucket"] == "High").sum()
        answer = f"Scored {len(scored)} customers for churn risk (cutoff: {cutoff_date}, inactivity: {inactivity_days} days). "
        answer += f"{high_risk} customers in High risk category."
        
        return {
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
            "answer": answer
        }
    
    elif function_name == "predict_churn_probability":
        # Fixed cutoff_date and inactivity_days, X_days is required (default 90)
        cutoff_date = FIXED_CUTOFF_DATE
        inactivity_days = FIXED_INACTIVITY_DAYS
        X_days = parameters.get("X_days", 90)  # Required, but provide default
        
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
        
        predictions = predict_churn_probability(
            model=cox_result['model'],
            transactions=transactions_df,
            cutoff_date=cutoff_date,
            X_days=X_days,
            inactivity_days=inactivity_days,
        )
        
        columns = ["customer_id", "t0", "X_days", "churn_probability",
                  "survival_at_t0", "survival_at_t0_plus_X"]
        rows = predictions.to_dict(orient="records")
        
        mean_prob = predictions["churn_probability"].mean()
        answer = f"Predicted churn probability for {len(predictions)} active customers. "
        answer += f"Mean probability of churn in next {X_days} days: {mean_prob:.2%} "
        answer += f"(cutoff: {cutoff_date}, inactivity: {inactivity_days} days)."
        
        return {
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
            "answer": answer
        }
    
    elif function_name == "expected_remaining_lifetime":
        # Fixed cutoff_date and inactivity_days, H_days is optional (default 365)
        cutoff_date = FIXED_CUTOFF_DATE
        inactivity_days = FIXED_INACTIVITY_DAYS
        H_days = parameters.get("H_days", 365)
        
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
        
        expected_lifetimes = expected_remaining_lifetime(
            model=cox_result['model'],
            covariates_df=cov,
            H_days=H_days,
            inactivity_days=inactivity_days,
        )
        
        columns = ["customer_id", "t0", "H_days", "expected_remaining_life_days"]
        rows = expected_lifetimes[columns].to_dict(orient="records")
        
        mean_lifetime = expected_lifetimes["expected_remaining_life_days"].mean()
        answer = f"Computed expected remaining lifetime for {len(expected_lifetimes)} active customers. "
        answer += f"Mean expected remaining lifetime: {mean_lifetime:.2f} days "
        answer += f"(cutoff: {cutoff_date}, inactivity: {inactivity_days} days, H: {H_days} days)."
        
        return {
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
            "answer": answer
        }
    
    elif function_name == "customer_segmentation":
        # Fixed cutoff_date and inactivity_days, H_days is optional (default 365)
        cutoff_date = FIXED_CUTOFF_DATE
        inactivity_days = FIXED_INACTIVITY_DAYS
        H_days = parameters.get("H_days", 365)
        
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
        
        segmentation_df, cutoffs = build_segmentation_table(
            model=cox_result['model'],
            transactions=transactions_df,
            covariates_df=cov,
            cutoff_date=cutoff_date,
            H_days=H_days,
        )
        
        columns = ["customer_id", "risk_label", "t0", "erl_365_days", 
                  "life_bucket", "segment", "action_tag", "recommended_action"]
        rows = segmentation_df[columns].to_dict(orient="records")
        
        segment_counts = segmentation_df['segment'].value_counts().to_dict()
        answer = f"Segmented {len(segmentation_df)} customers into {len(segment_counts)} segments "
        answer += f"(cutoff: {cutoff_date}, inactivity: {inactivity_days} days, H: {H_days} days). "
        answer += f"Top segments: {', '.join(list(segment_counts.keys())[:3])}."
        
        return {
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
            "answer": answer
        }
    
    else:
        raise ValueError(f"Unknown analytics function: {function_name}")


@app.post("/ask-langchain", response_model=AskLangChainResponse)
def ask_langchain(req: AskLangChainRequest) -> AskLangChainResponse:
    """
    Ask a question using LangChain agent - supports multi-step reasoning and conversation memory.
    This endpoint uses LangChain for more natural language responses and multi-tool orchestration.
    """
    try:
        answer = ask_question(req.question, use_memory=req.use_memory, thread_id=req.thread_id)
        return AskLangChainResponse(
            question=req.question,
            answer=answer
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LangChain ask failed: {str(e)}")

@app.post("/ask-langchain/clear-memory")
def clear_langchain_memory() -> Dict[str, str]:
    """Clear the conversation memory for LangChain agent"""
    try:
        clear_memory()
        return {"status": "success", "message": "Memory cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear memory: {str(e)}")

@app.post("/clv", response_model=CLVResponse)
def clv(req: CLVRequest) -> CLVResponse:
    # Pull only what we need
    sql = """
    SELECT customer_id, invoice_no, invoice_date, revenue
    FROM transactions
    WHERE customer_id IS NOT NULL
    """
    rows, _ = run_query_internal(sql, max_rows=2_000_000)  # internal call, not user SQL
    df = pd.DataFrame(rows)

    rfm = build_rfm(df, cutoff_date=req.cutoff_date)
    models = fit_models(rfm)
    
    # First get unscaled predictions to calculate target totals
    pred_unscaled = predict_clv(models, horizon_days=req.horizon_days, aov_fallback="global_mean")
    
    # Calculate target totals using hard-coded scales
    pred_total_purchases = pred_unscaled["pred_purchases"].sum()
    pred_total_revenue = pred_unscaled["clv"].sum(skipna=True)
    
    target_purchases = pred_total_purchases * PURCHASE_SCALE if PURCHASE_SCALE != 1.0 else None
    target_revenue = pred_total_revenue * REVENUE_SCALE if REVENUE_SCALE != 1.0 else None
    
    # Get scaled predictions
    pred = predict_clv(
        models, 
        horizon_days=req.horizon_days,
        scale_to_target_purchases=target_purchases,
        scale_to_target_revenue=target_revenue,
        aov_fallback="global_mean"
    )

    pred = pred.sort_values("clv", ascending=False)

    # Return all customers (or up to limit_customers if less than total)
    customer_limit = min(req.limit_customers, len(pred))
    all_customers = pred.head(customer_limit)[["customer_id", "frequency", "recency", "T", "monetary_value", "pred_purchases", "pred_aov", "clv"]]
    summary = {
        "customers_total": int(len(pred)),
        "customers_with_repeat": int((pred["frequency"] > 0).sum()),
        "clv_mean": float(pred["clv"].mean(skipna=True)) if len(pred) > 0 else 0.0,  # Mean of all customers
        "clv_max": float(all_customers["clv"].max()) if len(all_customers) > 0 else 0.0,
    }

    return CLVResponse(
        cutoff_date=req.cutoff_date,
        horizon_days=req.horizon_days,
        top_customers=all_customers.to_dict(orient="records"),
        summary=summary,
    )


@app.post("/survival/km", response_model=KMResponse)
def km_all(inactivity_days: int = Query(INACTIVITY_DAYS, description="Inactivity days threshold for churn definition")) -> KMResponse:
    """
    Fit Kaplan-Meier survival model on all customers.
    """
    sql = """
    SELECT customer_id, invoice_no, invoice_date, revenue, stock_code, country
    FROM transactions
    WHERE customer_id IS NOT NULL
    """
    rows, _ = run_query_internal(sql, max_rows=2_000_000)
    df = pd.DataFrame(rows)

    cov = build_covariate_table(
        transactions=df,
        cutoff_date=CUTOFF_DATE,
        inactivity_days=inactivity_days,
    ).df

    kmf = fit_km_all(cov)

    curve = [
        {"time": float(t), "survival": float(s)}
        for t, s in zip(kmf.timeline, kmf.survival_function_.iloc[:, 0])
    ]

    return KMResponse(
        cutoff_date=CUTOFF_DATE,
        inactivity_days=inactivity_days,
        n_customers=len(cov),
        churn_rate=float(cov["event"].mean()),
        survival_curve=curve,
    )


@app.post("/survival/score", response_model=ScoreCustomersResponse)
def score_customers_endpoint(
    inactivity_days: int = Query(INACTIVITY_DAYS, description="Inactivity days threshold for churn definition"),
    cutoff_date: str = Query(CUTOFF_DATE, description="Cutoff date for scoring (YYYY-MM-DD)"),
) -> ScoreCustomersResponse:
    """
    Score customers using a fitted Cox model to predict churn risk.
    """
    sql = """
    SELECT customer_id, invoice_no, invoice_date, revenue, stock_code, country
    FROM transactions
    WHERE customer_id IS NOT NULL
    """
    rows, _ = run_query_internal(sql, max_rows=2_000_000)
    df = pd.DataFrame(rows)

    # Build covariate table for model fitting
    cov = build_covariate_table(
        transactions=df,
        cutoff_date=cutoff_date,
        inactivity_days=inactivity_days,
    ).df

    # Fit Cox model with standard covariates
    cox_result = fit_cox_baseline(
        covariates=cov,
        covariate_cols=['n_orders', 'log_monetary_value', 'product_diversity'],
        train_frac=1.0,
        random_state=42,
        penalizer=0.1,
    )
    cox_model = cox_result['model']

    # Score customers
    scored = score_customers(
        model=cox_model,
        transactions=df,
        cutoff_date=cutoff_date,
    )

    # Create summary
    summary = {
        "n_customers": int(len(scored)),
        "risk_score_mean": float(scored["risk_score"].mean()),
        "risk_score_max": float(scored["risk_score"].max()),
        "risk_bucket_counts": scored["risk_bucket"].value_counts().to_dict(),
    }

    return ScoreCustomersResponse(
        cutoff_date=cutoff_date,
        inactivity_days=inactivity_days,
        n_customers=len(scored),
        scored_customers=scored.to_dict(orient="records"),
        summary=summary,
    )


@app.post("/survival/churn-probability", response_model=ChurnProbabilityResponse)
def churn_probability_endpoint(
    inactivity_days: int = Query(INACTIVITY_DAYS, description="Inactivity days threshold for churn definition"),
    cutoff_date: str = Query(CUTOFF_DATE, description="Cutoff date for prediction (YYYY-MM-DD)"),
    X_days: int = Query(90, ge=1, le=3650, description="Prediction horizon in days"),
) -> ChurnProbabilityResponse:
    """
    Predict conditional churn probability for active customers.
    Computes P(churn in next X days | survived to cutoff).
    """
    sql = """
    SELECT customer_id, invoice_no, invoice_date, revenue, stock_code, country
    FROM transactions
    WHERE customer_id IS NOT NULL
    """
    rows, _ = run_query_internal(sql, max_rows=2_000_000)
    df = pd.DataFrame(rows)

    # Build covariate table for model fitting
    cov = build_covariate_table(
        transactions=df,
        cutoff_date=cutoff_date,
        inactivity_days=inactivity_days,
    ).df

    # Fit Cox model with standard covariates
    cox_result = fit_cox_baseline(
        covariates=cov,
        covariate_cols=['n_orders', 'log_monetary_value', 'product_diversity'],
        train_frac=1.0,
        random_state=42,
        penalizer=0.1,
    )
    cox_model = cox_result['model']

    # Predict churn probabilities
    predictions = predict_churn_probability(
        model=cox_model,
        transactions=df,
        cutoff_date=cutoff_date,
        X_days=X_days,
        inactivity_days=inactivity_days,
    )

    # Create summary
    summary = {
        "n_customers": int(len(predictions)),
        "churn_probability_mean": float(predictions["churn_probability"].mean()),
        "churn_probability_median": float(predictions["churn_probability"].median()),
        "churn_probability_max": float(predictions["churn_probability"].max()),
        "churn_probability_min": float(predictions["churn_probability"].min()),
        "survival_at_t0_mean": float(predictions["survival_at_t0"].mean()),
        "survival_at_t0_plus_X_mean": float(predictions["survival_at_t0_plus_X"].mean()),
    }

    return ChurnProbabilityResponse(
        cutoff_date=cutoff_date,
        inactivity_days=inactivity_days,
        X_days=X_days,
        n_customers=len(predictions),
        predictions=predictions.to_dict(orient="records"),
        summary=summary,
    )


@app.post("/survival/expected-lifetime", response_model=ExpectedLifetimeResponse)
def expected_lifetime_endpoint(
    inactivity_days: int = Query(INACTIVITY_DAYS, description="Inactivity days threshold for churn definition"),
    cutoff_date: str = Query(CUTOFF_DATE, description="Cutoff date for computation (YYYY-MM-DD)"),
    H_days: int = Query(365, ge=1, le=3650, description="Horizon in days for restricted expectation"),
) -> ExpectedLifetimeResponse:
    """
    Compute restricted expected remaining lifetime for active customers.
    """
    sql = """
    SELECT customer_id, invoice_no, invoice_date, revenue, stock_code, country
    FROM transactions
    WHERE customer_id IS NOT NULL
    """
    rows, _ = run_query_internal(sql, max_rows=2_000_000)
    df = pd.DataFrame(rows)

    # Build covariate table for model fitting
    cov = build_covariate_table(
        transactions=df,
        cutoff_date=cutoff_date,
        inactivity_days=inactivity_days,
    ).df

    # Fit Cox model with standard covariates
    cox_result = fit_cox_baseline(
        covariates=cov,
        covariate_cols=['n_orders', 'log_monetary_value', 'product_diversity'],
        train_frac=1.0,
        random_state=42,
        penalizer=0.1,
    )
    cox_model = cox_result['model']

    # Compute expected remaining lifetime
    expected_lifetimes = expected_remaining_lifetime(
        model=cox_model,
        covariates_df=cov,
        H_days=H_days,
        inactivity_days=inactivity_days,
    )

    # Create summary
    summary = {
        "n_customers": int(len(expected_lifetimes)),
        "expected_lifetime_mean": float(expected_lifetimes["expected_remaining_life_days"].mean()),
        "expected_lifetime_median": float(expected_lifetimes["expected_remaining_life_days"].median()),
        "expected_lifetime_max": float(expected_lifetimes["expected_remaining_life_days"].max()),
        "expected_lifetime_min": float(expected_lifetimes["expected_remaining_life_days"].min()),
        "t0_mean": float(expected_lifetimes["t0"].mean()),
    }

    return ExpectedLifetimeResponse(
        cutoff_date=cutoff_date,
        inactivity_days=inactivity_days,
        H_days=H_days,
        n_customers=len(expected_lifetimes),
        expected_lifetimes=expected_lifetimes.to_dict(orient="records"),
        summary=summary,
    )


@app.post("/survival/segmentation", response_model=SegmentationResponse)
def segmentation_endpoint(
    inactivity_days: int = Query(INACTIVITY_DAYS, description="Inactivity days threshold for churn definition"),
    cutoff_date: str = Query(CUTOFF_DATE, description="Cutoff date for segmentation (YYYY-MM-DD)"),
    H_days: int = Query(365, ge=1, le=3650, description="Horizon in days for expected remaining lifetime"),
) -> SegmentationResponse:
    """
    Build segmentation table combining risk labels and expected remaining lifetime.
    """
    sql = """
    SELECT customer_id, invoice_no, invoice_date, revenue, stock_code, country
    FROM transactions
    WHERE customer_id IS NOT NULL
    """
    rows, _ = run_query_internal(sql, max_rows=2_000_000)
    df = pd.DataFrame(rows)

    # Build covariate table
    cov = build_covariate_table(
        transactions=df,
        cutoff_date=cutoff_date,
        inactivity_days=inactivity_days,
    ).df

    # Fit Cox model with standard covariates
    cox_result = fit_cox_baseline(
        covariates=cov,
        covariate_cols=['n_orders', 'log_monetary_value', 'product_diversity'],
        train_frac=1.0,
        random_state=42,
        penalizer=0.1,
    )
    cox_model = cox_result['model']

    # Build segmentation table
    segmentation_df, cutoffs = build_segmentation_table(
        model=cox_model,
        transactions=df,
        covariates_df=cov,
        cutoff_date=cutoff_date,
        H_days=H_days,
    )

    # Create summary
    summary = {
        "n_customers": int(len(segmentation_df)),
        "segment_counts": segmentation_df['segment'].value_counts().to_dict(),
        "risk_label_counts": segmentation_df['risk_label'].value_counts().to_dict(),
        "life_bucket_counts": segmentation_df['life_bucket'].value_counts().to_dict(),
        "action_tag_counts": segmentation_df['action_tag'].value_counts().to_dict(),
        "erl_mean": float(segmentation_df['erl_365_days'].mean()),
        "erl_median": float(segmentation_df['erl_365_days'].median()),
    }

    return SegmentationResponse(
        cutoff_date=cutoff_date,
        inactivity_days=inactivity_days,
        H_days=H_days,
        n_customers=len(segmentation_df),
        segments=segmentation_df.to_dict(orient="records"),
        cutoffs=cutoffs,
        summary=summary,
    )

