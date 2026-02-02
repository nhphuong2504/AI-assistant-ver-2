from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from app.db import get_schema, run_query
from app.llm_langchain import ask_question, clear_memory, get_or_fit_cox_model
from app.data import get_transactions_df, get_clv_models
import pandas as pd
from analytics.clv import predict_clv, PURCHASE_SCALE, REVENUE_SCALE
from analytics.survival import (
    build_covariate_table,
    fit_km_all,
    score_customers,
    predict_churn_probability,
    build_segmentation_table,
    CUTOFF_DATE,
    INACTIVITY_DAYS,
)
from analytics.monte_carlo import compute_erl_days


def _build_expected_lifetimes_df(
    df: pd.DataFrame,
    cutoff_date: str,
    inactivity_days: int,
    H_days: int = 365,
) -> pd.DataFrame:
    """Build expected lifetimes DataFrame (Monte Carlo ERL) with columns customer_id, t0, H_days, expected_remaining_life_days."""
    clv_result = get_clv_models(cutoff_date)
    rfm = clv_result.rfm
    df = df.copy()
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    cutoff_dt = pd.to_datetime(cutoff_date)
    last_purchases = (
        df[df["invoice_date"] <= cutoff_dt]
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
    expected_lifetimes["H_days"] = H_days
    return expected_lifetimes


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
    already_churned_count: int
    already_churned_customer_ids: List[int]
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
    models = get_clv_models(req.cutoff_date)
    
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
    df = get_transactions_df()
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
    Score active customers using a fitted Cox model (already-churned customers are excluded and listed).
    """
    df = get_transactions_df()
    cox_result = get_or_fit_cox_model(df, cutoff_date, inactivity_days)
    train_df = cox_result["train_df"]
    churned_ids = train_df.loc[train_df["event"] == 1, "customer_id"].astype(int).tolist()

    # Score active customers only (covariate_df = train_df filters to event==0)
    scored = score_customers(
        model=cox_result["model"],
        transactions=df,
        cutoff_date=cutoff_date,
        covariate_df=train_df,
    )

    summary = {
        "n_customers": int(len(scored)),
        "risk_score_mean": float(scored["risk_score"].mean()) if len(scored) > 0 else 0.0,
        "risk_score_max": float(scored["risk_score"].max()) if len(scored) > 0 else 0.0,
        "risk_bucket_counts": scored["risk_bucket"].value_counts().to_dict() if len(scored) > 0 else {},
    }

    return ScoreCustomersResponse(
        cutoff_date=cutoff_date,
        inactivity_days=inactivity_days,
        n_customers=len(scored),
        already_churned_count=len(churned_ids),
        already_churned_customer_ids=churned_ids[:500],
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
    df = get_transactions_df()
    cox_result = get_or_fit_cox_model(df, cutoff_date, inactivity_days)

    # Predict churn probabilities
    predictions = predict_churn_probability(
        model=cox_result["model"],
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
    H_days: int = Query(365, ge=1, le=3650, description="Horizon in days (nominal; reported in response)"),
) -> ExpectedLifetimeResponse:
    """
    Compute expected remaining lifetime in days (Monte Carlo, BG/NBD).
    """
    df = get_transactions_df()
    expected_lifetimes = _build_expected_lifetimes_df(df, cutoff_date, inactivity_days, H_days)

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
    df = get_transactions_df()
    cox_result = get_or_fit_cox_model(df, cutoff_date, inactivity_days)
    cov = cox_result["train_df"]

    # Build segmentation table
    segmentation_df, cutoffs = build_segmentation_table(
        model=cox_result["model"],
        transactions=df,
        covariates_df=cov,
        cutoff_date=cutoff_date,
    )

    # Create summary
    summary = {
        "n_customers": int(len(segmentation_df)),
        "segment_counts": segmentation_df['segment'].value_counts().to_dict(),
        "risk_label_counts": segmentation_df['risk_label'].value_counts().to_dict(),
        "life_bucket_counts": segmentation_df['life_bucket'].value_counts().to_dict(),
        "action_tag_counts": segmentation_df['action_tag'].value_counts().to_dict(),
        "erl_mean": float(segmentation_df['erl_days'].mean()),
        "erl_median": float(segmentation_df['erl_days'].median()),
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

