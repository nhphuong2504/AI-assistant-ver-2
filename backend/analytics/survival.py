import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.utils import concordance_index

from .clv import build_rfm, fit_models
from .monte_carlo import compute_erl_days as compute_erl_days_mc

# --------------------
# GLOBAL MODEL ASSUMPTIONS
# --------------------
CUTOFF_DATE = "2011-12-09"
INACTIVITY_DAYS = 90


@dataclass
class CovariateTable:
    df: pd.DataFrame
    cutoff_date: pd.Timestamp


def build_covariate_table(
    transactions: pd.DataFrame,
    cutoff_date: str = CUTOFF_DATE,
    inactivity_days: int = INACTIVITY_DAYS,
) -> CovariateTable:
    """
    Builds a customer-level table at a fixed cutoff date (inclusive),
    using invoice-level orders.

    Output columns:
      1. customer_id: Unique identifier of the customer; one row per customer.
      2. duration: Number of days from first order to churn time (if event=1) or cutoff (if event=0).
      3. event: Churn indicator (1=churned, 0=censored).
      4. tenure_days: Number of days between first order date and cutoff date.
      5. recency_from_cutoff: Number of days between last order date and cutoff date.
      6. n_orders: Total number of distinct orders (invoices) up to cutoff.
      7. frequency_rate: Purchase frequency as orders per month (n_orders / (tenure_days/30)).
      8. monetary_value: Mean order value (total_revenue / n_orders).
      9. product_diversity: Count of distinct product codes (stock_code) purchased.
      10. is_uk: Binary indicator (1 if any order from UK, 0 otherwise).
    """
    df = transactions.copy()
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    cutoff = pd.to_datetime(cutoff_date)

    # Observation window (inclusive)
    df = df[df["invoice_date"] <= cutoff]
    df = df[df["customer_id"].notna()]
    df = df[df["revenue"] > 0]

    # Invoice-level orders
    orders = (
        df.groupby(["customer_id", "invoice_no"], as_index=False)
          .agg(
              order_date=("invoice_date", "min"),
              order_value=("revenue", "sum"),
          )
    )

    g = orders.groupby("customer_id")
    first = g["order_date"].min()
    last = g["order_date"].max()
    n_orders = g["invoice_no"].nunique()
    total_revenue = g["order_value"].sum()

    # Time quantities (days)
    tenure_days = (cutoff - first).dt.days
    recency_from_cutoff = (cutoff - last).dt.days

    # Survival event: churn if inactive for >= inactivity_days at cutoff
    event = (recency_from_cutoff >= inactivity_days).astype(int)

    # Duration: if churned, event time = last + inactivity_days; else censored at cutoff
    duration = pd.Series(index=first.index, dtype=float)
    duration[event == 1] = (
        last[event == 1]
        + pd.to_timedelta(inactivity_days, unit="D")
        - first[event == 1]
    ).dt.days
    duration[event == 0] = tenure_days[event == 0]

    # Behavioral covariates
    n_orders_float = n_orders.astype(float)

    # Monetary value: mean order value (total_revenue / n_orders)
    monetary_value = total_revenue / n_orders
    monetary_value = monetary_value.astype(float)

    # Frequency rate: orders per month
    frequency_rate = n_orders / (tenure_days / 30.0)
    frequency_rate = frequency_rate.replace([np.inf, -np.inf], np.nan).astype(float)

    product_diversity = (
        df.groupby("customer_id")["stock_code"]
          .nunique()
          .reindex(first.index)
          .fillna(0)
          .astype(float)
    )

    is_uk = (
        df.groupby("customer_id")["country"]
          .apply(lambda x: int((x == "United Kingdom").any()))
          .reindex(first.index)
          .fillna(0)
          .astype(int)
    )

    cov = pd.DataFrame({
        "customer_id": first.index,
        "duration": duration,
        "event": event.astype(int),
        "tenure_days": tenure_days.astype(float),
        "recency_from_cutoff": recency_from_cutoff.astype(float),
        "n_orders": n_orders_float,
        "frequency_rate": frequency_rate,
        "monetary_value": monetary_value,
        "product_diversity": product_diversity,
        "is_uk": is_uk,
    }).reset_index(drop=True)

    # Safety filters
    cov = cov[
        (cov["duration"] > 0) &
        (cov["tenure_days"] > 0) &
        (cov["recency_from_cutoff"] >= 0)
    ].copy()

    return CovariateTable(df=cov, cutoff_date=cutoff)


def fit_km_all(covariates: pd.DataFrame) -> KaplanMeierFitter:
    """
    Fits a Kaplan-Meier survival model on all customers.
    
    Args:
        covariates: DataFrame with 'duration' and 'event' columns from build_covariate_table
        
    Returns:
        Fitted KaplanMeierFitter model
    """
    kmf = KaplanMeierFitter()
    kmf.fit(
        durations=covariates["duration"],
        event_observed=covariates["event"],
        label="All customers",
    )
    return kmf


def prepare_cox_data(
    covariates: pd.DataFrame,
    covariate_cols: List[str] = None,
) -> pd.DataFrame:
    """
    Prepares data for Cox model by selecting covariates and dropping missing values.
    Applies log transformation to product_diversity if requested.
    
    Args:
        covariates: DataFrame from build_covariate_table
        covariate_cols: List of covariate column names to include. 
                       Default: ['n_orders', 'log_product_diversity']
    
    Returns:
        DataFrame with duration, event, and selected covariates (no missing values)
    """
    if covariate_cols is None:
        covariate_cols = ['n_orders', 'log_product_diversity']
    
    df = covariates.copy()
    
    # Create log transformation of monetary_value if log_monetary_value is requested
    if 'log_monetary_value' in covariate_cols and 'monetary_value' in df.columns:
        df['log_monetary_value'] = np.log1p(df['monetary_value'])
    
    # Create log transformation of product_diversity if log_product_diversity is requested
    if 'log_product_diversity' in covariate_cols and 'product_diversity' in df.columns:
        df['log_product_diversity'] = np.log1p(df['product_diversity'])
    
    # Select required columns: duration, event, and requested covariates
    # For log_monetary_value, we need the transformed column, not monetary_value
    required_cols = ['duration', 'event']
    for col in covariate_cols:
        if col in df.columns:
            required_cols.append(col)
    
    df = df[required_cols].copy()
    
    # Drop rows with missing values in the final covariate columns
    final_covariate_cols = [col for col in covariate_cols if col in df.columns]
    df = df.dropna(subset=final_covariate_cols)
    
    return df


def split_train_validation(
    df: pd.DataFrame,
    train_frac: float = 0.8,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits data into train and validation sets by customer (random split).
    
    Args:
        df: DataFrame with customer data
        train_frac: Fraction of data for training (default: 0.8)
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (train_df, validation_df)
    """
    np.random.seed(random_state)
    n = len(df)
    indices = np.arange(n)
    np.random.shuffle(indices)
    
    n_train = int(n * train_frac)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_df = df.iloc[train_indices].copy()
    val_df = df.iloc[val_indices].copy()
    
    return train_df, val_df


def fit_cox_baseline(
    covariates: pd.DataFrame,
    covariate_cols: List[str] = None,
    train_frac: float = 1.0,
    random_state: int = 42,
    penalizer: float = 0.1,
) -> Dict[str, Any]:
    """
    Fits a baseline Cox proportional hazards model to study customer churn timing.
    
    Steps:
    1. Drop rows with missing values in selected covariates
    2. If train_frac < 1, split into train/validation; otherwise use all data for training
    3. Fit Cox model on training data
    4. Generate summary with interpretation
    
    Args:
        covariates: DataFrame from build_covariate_table
        covariate_cols: List of covariate column names. 
                       Default: ['n_orders', 'log_product_diversity']
        train_frac: Fraction for training (default: 1.0 = use all data). Use < 1 (e.g. 0.8)
                    only when a held-out validation set is needed (e.g. for validate_cox_model).
        random_state: Random seed for split (used only when train_frac < 1)
        penalizer: L2 penalizer for Cox model (default: 0.1)
    
    Returns:
        Dictionary with:
        - model: Fitted CoxPHFitter
        - summary: DataFrame with coefficients, hazard ratios, p-values
        - interpretation: Dictionary with coefficient interpretations
        - flags: Dictionary with warnings (unexpected signs, large SE, non-significant)
        - n_train: Number of training samples
        - n_validation: Number of validation samples
        - n_dropped: Number of rows dropped due to missing values
        - train_df: Training dataset
        - validation_df: Validation dataset
    """
    if covariate_cols is None:
        covariate_cols = ['n_orders', 'log_product_diversity']
    
    # Step 1: Prepare data (drop missing)
    n_before = len(covariates)
    df_cox = prepare_cox_data(covariates, covariate_cols)
    n_after = len(df_cox)
    n_dropped = n_before - n_after
    
    # Step 2: Split train/validation
    train_df, val_df = split_train_validation(df_cox, train_frac, random_state)
    
    # Step 3: Fit Cox model on training data
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(train_df, duration_col='duration', event_col='event')
    
    # Step 4: Generate summary
    summary = cph.summary.copy()
    summary['hazard_ratio'] = np.exp(summary['coef'])
    summary = summary.reset_index().rename(columns={'index': 'covariate'})
    
    # Interpretation and flags
    interpretation = {}
    flags = {
        'unexpected_signs': [],
        'large_se': [],
        'non_significant': [],
    }
    
    # Expected signs based on business logic:
    # - n_orders: negative (more orders → lower churn risk)
    # - monetary_value: negative (higher value → lower churn risk)
    # - log_monetary_value: negative (higher value → lower churn risk)
    # - product_diversity: negative (more diversity → lower churn risk)
    # - log_product_diversity: negative (more diversity → lower churn risk)
    
    expected_signs = {
        'n_orders': 'negative',
        'monetary_value': 'negative',
        'log_monetary_value': 'negative',
        'product_diversity': 'negative',
        'log_product_diversity': 'negative',
    }
    
    for _, row in summary.iterrows():
        cov = row['covariate']
        coef = row['coef']
        se = row['se(coef)']
        p = row['p']
        hr = row['hazard_ratio']
        
        # Interpretation
        if coef > 0:
            interpretation[cov] = {
                'sign': 'positive',
                'meaning': 'Higher churn risk (shorter lifetime)',
                'hazard_ratio': hr,
                'effect': f'Each unit increase → {hr:.3f}x higher hazard'
            }
        else:
            interpretation[cov] = {
                'sign': 'negative',
                'meaning': 'Lower churn risk (longer lifetime)',
                'hazard_ratio': hr,
                'effect': f'Each unit increase → {hr:.3f}x lower hazard'
            }
        
        # Flags
        expected = expected_signs.get(cov)
        if expected is not None:
            actual_sign = 'positive' if coef > 0 else 'negative'
            if actual_sign != expected:
                flags['unexpected_signs'].append({
                    'covariate': cov,
                    'expected': expected,
                    'actual': actual_sign,
                    'coef': coef,
                })
        
        # Large standard error (relative to coefficient)
        if abs(se) > abs(coef) * 2:
            flags['large_se'].append({
                'covariate': cov,
                'coef': coef,
                'se': se,
                'ratio': abs(se / coef) if coef != 0 else float('inf'),
            })
        
        # Non-significant (p > 0.05)
        if p > 0.05:
            flags['non_significant'].append({
                'covariate': cov,
                'p': p,
                'coef': coef,
            })
    
    return {
        'model': cph,
        'summary': summary,
        'interpretation': interpretation,
        'flags': flags,
        'n_train': len(train_df),
        'n_validation': len(val_df),
        'n_dropped': n_dropped,
        'train_df': train_df,
        'validation_df': val_df,
    }


def validate_cox_model(
    model: CoxPHFitter,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Validates a fitted Cox proportional hazards model.
    
    Performs:
    1. Proportional hazards assumption testing using Schoenfeld residuals
    2. Predictive performance evaluation using C-index on validation set
    
    Args:
        model: Fitted CoxPHFitter model
        train_df: Training dataset with duration, event, and covariates
        validation_df: Validation dataset with duration, event, and covariates
    
    Returns:
        Dictionary with:
        - ph_tests: DataFrame with PH test results (global and per-covariate)
        - ph_violations: List of covariates violating PH assumption (p < 0.05)
        - c_index: Concordance index on validation set
        - interpretation: Dictionary with validation results interpretation
    """
    from scipy import stats
    from scipy.stats import combine_pvalues
    
    # Expected covariates (should match model)
    expected_covariates = list(model.params_.index)
    
    # Step 1: Check proportional hazards assumption on training data
    ph_test_results = []
    ph_violations = []
    ph_test_failed = False
    ph_test_error_msg = None
    
    try:
        # Use check_assumptions method for reliable PH testing
        # This method performs Schoenfeld residual tests
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # check_assumptions may print to stdout, but we'll capture the results
            model.check_assumptions(train_df, p_value_threshold=0.05, show_plots=False)
        
        # Get Schoenfeld residuals
        residuals = model.compute_residuals(train_df, kind='schoenfeld')
        
        # Verify we have residuals for all expected covariates
        missing_covariates = [c for c in expected_covariates if c not in residuals.columns]
        if missing_covariates:
            ph_test_failed = True
            ph_test_error_msg = f"Missing residuals for covariates: {missing_covariates}"
        else:
            # For each covariate, test if residuals are correlated with time
            for covariate in expected_covariates:
                # Get valid residuals (non-null)
                valid_mask = residuals[covariate].notna()
                n_valid = valid_mask.sum()
                
                if n_valid < 3:
                    # Not enough data for test
                    ph_test_results.append({
                        'covariate': covariate,
                        'test_statistic': np.nan,
                        'p_value': np.nan,
                        'violates_ph': False,
                        'note': f'Insufficient data (n={n_valid})'
                    })
                    ph_test_failed = True
                    continue
                
                res_values = residuals.loc[valid_mask, covariate].values
                time_values = train_df.loc[valid_mask, 'duration'].values
                
                # Test correlation between residuals and time using Kendall's tau
                # (rank correlation, more robust than Pearson)
                tau, p_value = stats.kendalltau(time_values, res_values)
                
                # Check for NaN results
                if np.isnan(p_value) or np.isnan(tau):
                    ph_test_results.append({
                        'covariate': covariate,
                        'test_statistic': np.nan,
                        'p_value': np.nan,
                        'violates_ph': False,
                        'note': 'Test computation produced NaN'
                    })
                    ph_test_failed = True
                else:
                    ph_test_results.append({
                        'covariate': covariate,
                        'test_statistic': tau,
                        'p_value': p_value,
                        'violates_ph': p_value < 0.05,
                    })
                    
                    if p_value < 0.05:
                        ph_violations.append(covariate)
            
            # Global test (overall PH assumption)
            # Use Fisher's combined probability test across all covariates
            valid_p_values = [r['p_value'] for r in ph_test_results 
                            if not np.isnan(r['p_value'])]
            
            if len(valid_p_values) > 0:
                global_stat, global_p = combine_pvalues(valid_p_values, method='fisher')
                
                # Check if global test is valid
                if np.isnan(global_p):
                    ph_test_failed = True
                    global_entry = {
                        'covariate': 'global',
                        'test_statistic': np.nan,
                        'p_value': np.nan,
                        'violates_ph': False,
                        'note': 'Global test computation produced NaN'
                    }
                else:
                    global_entry = {
                        'covariate': 'global',
                        'test_statistic': global_stat,
                        'p_value': global_p,
                        'violates_ph': global_p < 0.05,
                    }
            else:
                ph_test_failed = True
                global_entry = {
                    'covariate': 'global',
                    'test_statistic': np.nan,
                    'p_value': np.nan,
                    'violates_ph': False,
                    'note': 'No valid p-values for global test'
                }
            
            # Create DataFrame with global test first, then per-covariate tests
            ph_tests_df = pd.concat([
                pd.DataFrame([global_entry]),
                pd.DataFrame(ph_test_results)
            ], ignore_index=True)
            
    except Exception as e:
        # If PH test computation fails, report as failure
        ph_test_failed = True
        ph_test_error_msg = str(e)
        ph_tests_df = pd.DataFrame([{
            'covariate': 'computation_error',
            'test_statistic': np.nan,
            'p_value': np.nan,
            'violates_ph': False,
            'note': f'PH test failed: {ph_test_error_msg}'
        }])
        ph_violations = []
    
    # Verify all expected covariates have valid test results
    if not ph_test_failed:
        for cov in expected_covariates:
            cov_results = ph_tests_df[ph_tests_df['covariate'] == cov]
            if len(cov_results) == 0 or cov_results['p_value'].isna().any():
                ph_test_failed = True
                break
    
    # Step 2: Evaluate predictive performance on validation set
    # Compute risk scores (partial hazards) for validation data
    c_index = np.nan
    c_index_failed = False
    
    try:
        # Get covariates (exclude duration and event)
        covariate_cols = [col for col in validation_df.columns 
                         if col not in ['duration', 'event']]
        
        # Predict partial hazard (risk scores)
        # Higher partial hazard = higher churn risk = shorter survival
        risk_scores = model.predict_partial_hazard(validation_df[covariate_cols])
        
        # INVERT risk scores: higher score should correspond to longer survival
        # For C-index, we want: higher predicted score = longer survival time
        # So we use negative of risk scores (or 1/risk_scores, but negative is more stable)
        survival_scores = -risk_scores.values
        
        # Compute C-index
        # C-index measures: P(predicted longer survival > predicted shorter survival | actual longer survival > actual shorter survival)
        c_index = concordance_index(
            event_times=validation_df['duration'].values,
            predicted_scores=survival_scores,
            event_observed=validation_df['event'].values,
        )
        
        # Validate C-index is in valid range [0, 1]
        if np.isnan(c_index) or c_index < 0 or c_index > 1:
            c_index_failed = True
            if np.isnan(c_index):
                c_index = np.nan
            else:
                c_index = np.clip(c_index, 0, 1)
                
    except Exception as e:
        c_index = np.nan
        c_index_failed = True
    
    # Interpretation
    # PH assumption holds only if:
    # 1. Tests were computed successfully (no failures)
    # 2. All covariates have valid (non-NaN) test results
    # 3. No covariate has p < 0.05
    ph_assumption_holds = (
        not ph_test_failed and
        len(ph_violations) == 0 and
        not ph_tests_df.empty and
        ph_tests_df['covariate'].iloc[0] != 'computation_error' and
        not ph_tests_df['p_value'].isna().any()
    )
    
    ph_evaluable = not ph_test_failed and not ph_tests_df.empty and \
                   ph_tests_df['covariate'].iloc[0] != 'computation_error'
    
    # C-index interpretation
    if np.isnan(c_index) or c_index_failed:
        c_index_interpretation = "C-index computation failed or produced invalid result"
        acceptable_performance = False
    elif c_index < 0.5:
        c_index_interpretation = f"Poor discriminative performance (C-index={c_index:.4f} < 0.5, worse than random)"
        acceptable_performance = False
    elif c_index < 0.6:
        c_index_interpretation = f"Weak discriminative performance (C-index={c_index:.4f} < 0.6)"
        acceptable_performance = False
    elif c_index >= 0.6 and c_index < 0.7:
        c_index_interpretation = f"Moderate discriminative performance (C-index={c_index:.4f})"
        acceptable_performance = True
    elif c_index >= 0.7:
        c_index_interpretation = f"Good discriminative performance (C-index={c_index:.4f} >= 0.7)"
        acceptable_performance = True
    else:
        c_index_interpretation = f"Discriminative performance (C-index={c_index:.4f})"
        acceptable_performance = c_index >= 0.6
    
    interpretation = {
        'ph_assumption_holds': ph_assumption_holds,
        'ph_evaluable': ph_evaluable,
        'ph_violations': ph_violations,
        'ph_test_failed': ph_test_failed,
        'ph_test_error': ph_test_error_msg,
        'acceptable_performance': acceptable_performance,
        'c_index_interpretation': c_index_interpretation,
        'c_index_failed': c_index_failed,
    }
    
    return {
        'ph_tests': ph_tests_df,
        'ph_violations': ph_violations,
        'c_index': c_index,
        'interpretation': interpretation,
    }


def score_customers(
    model: CoxPHFitter,
    transactions: pd.DataFrame,
    cutoff_date: str = CUTOFF_DATE,
) -> pd.DataFrame:
    """
    Leakage-free scoring and ranking pipeline using a fitted Cox model.
    
    Computes risk scores for customers at a cutoff date using only historical data.
    Risk scores represent relative churn risk for prioritization, not probabilities.
    
    Args:
        model: Fitted CoxPHFitter model (must NOT be refit)
        transactions: DataFrame with customer_id, invoice_no, invoice_date, revenue, stock_code
        cutoff_date: Cutoff date (inclusive) for feature computation (YYYY-MM-DD)
    
    Returns:
        DataFrame with columns:
        - customer_id: Customer identifier
        - n_orders: Total number of distinct orders per customer
        - log_monetary_value: Log-transformed mean order value per customer
        - product_diversity: Number of unique products purchased per customer
        - risk_score: Partial hazard (higher = higher churn risk)
        - risk_rank: Rank by risk_score (1 = highest risk)
        - risk_percentile: Percentile rank 
        - risk_bucket: Risk category (High/Medium/Low based on percentiles)
    
    Note:
        Risk scores represent relative churn risk and are intended for prioritization,
        not probability estimation.
    """
    # Step 1: Feature construction at cutoff (leakage-free)
    df = transactions.copy()
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    cutoff = pd.to_datetime(cutoff_date)
    
    # Filter to data up to cutoff (inclusive)
    df = df[df["invoice_date"] <= cutoff]
    df = df[df["customer_id"].notna()]
    df = df[df["revenue"] > 0]
    
    # Invoice-level orders
    orders = (
        df.groupby(["customer_id", "invoice_no"], as_index=False)
          .agg(
              order_date=("invoice_date", "min"),
              order_value=("revenue", "sum"),
          )
    )
    
    g = orders.groupby("customer_id")
    n_orders = g["invoice_no"].nunique()
    total_revenue = g["order_value"].sum()
    
    # Compute features (NO duration, event, recency_from_cutoff, or tenure_days)
    # n_orders: total number of distinct orders per customer
    n_orders_series = n_orders.astype(float)
    
    # monetary_value: mean order value
    monetary_value = total_revenue / n_orders
    monetary_value = monetary_value.astype(float)
    
    # log_monetary_value: log-transformed mean order value
    log_monetary_value = np.log1p(monetary_value)
    
    # product_diversity: number of unique products purchased per customer
    product_diversity = (
        df.groupby("customer_id")["stock_code"]
          .nunique()
          .reindex(n_orders.index)
          .fillna(0)
          .astype(float)
    )
    
    # Create feature dataframe
    feature_df = pd.DataFrame({
        "customer_id": n_orders.index,
        "n_orders": n_orders_series,
        "log_monetary_value": log_monetary_value,
        "product_diversity": product_diversity,
    }).reset_index(drop=True)
    
    # Drop rows with missing values in required features
    feature_df = feature_df.dropna(subset=["n_orders", "log_monetary_value", "product_diversity"])
    
    # Step 2: Risk score computation using fitted model
    # Get covariate columns (must match model's expected covariates)
    covariate_cols = list(model.params_.index)
    
    # Verify we have all required covariates
    missing_cols = [col for col in covariate_cols if col not in feature_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required covariates in feature_df: {missing_cols}")
    
    # Compute risk scores (partial hazards)
    # Higher score = higher churn risk
    risk_scores = model.predict_partial_hazard(feature_df[covariate_cols])
    
    # Step 3: Customer ranking
    result_df = feature_df.copy()
    result_df["risk_score"] = risk_scores.values
    
    # Rank by risk_score (descending: highest risk = rank 1)
    result_df["risk_rank"] = result_df["risk_score"].rank(method="min", ascending=False).astype(int)
    
    # Compute percentile (0-100, higher = higher risk)
    result_df["risk_percentile"] = (
        result_df["risk_score"].rank(method="min", pct=True, ascending=True) * 100
    ).round(2)
    
    # Assign risk buckets based on percentiles
    # Top 10% (90-100%) → High risk
    # Next 20% (70-90%) → Medium risk
    # Remaining (0-70%) → Low risk
    def assign_risk_bucket(percentile):
        if percentile >= 90:
            return "High"
        elif percentile >= 70:
            return "Medium"
        else:
            return "Low"
    
    result_df["risk_bucket"] = result_df["risk_percentile"].apply(assign_risk_bucket)
    
    # Sort by risk_score descending (highest risk first)
    result_df = result_df.sort_values("risk_score", ascending=False).reset_index(drop=True)
    
    # Select and order output columns
    output_cols = [
        "customer_id",
        "n_orders",
        "log_monetary_value",
        "product_diversity",
        "risk_score",
        "risk_rank",
        "risk_percentile",
        "risk_bucket",
    ]
    
    return result_df[output_cols]


def predict_churn_probability(
    model: CoxPHFitter,
    transactions: pd.DataFrame,
    cutoff_date: str = CUTOFF_DATE,
    X_days: int = 90,
    inactivity_days: int = INACTIVITY_DAYS,
) -> pd.DataFrame:
    """
    Computes conditional churn probability for active customers.
    
    For each active customer (event == 0), computes:
    P(churn in next X days | survived to t0) = 1 - S(t0 + X | x) / S(t0 | x)
    
    Where:
    - t0 = customer's tenure_days at cutoff
    - X = prediction horizon (X_days)
    - S(t | x) = individual survival function from Cox model
    
    Args:
        model: Fitted CoxPHFitter model (must NOT be refit)
        transactions: DataFrame with customer_id, invoice_no, invoice_date, revenue, stock_code
        cutoff_date: Cutoff date (inclusive) for feature computation (YYYY-MM-DD)
        X_days: Prediction horizon in days (default: 90)
        inactivity_days: Inactivity days threshold for churn definition (default: 90)
    
    Returns:
        DataFrame with columns:
        - customer_id: Customer identifier
        - t0: Current duration (tenure_days) at cutoff
        - X_days: Prediction horizon
        - churn_probability: Conditional probability of churn in next X days
        - survival_at_t0: Survival probability at t0
        - survival_at_t0_plus_X: Survival probability at t0 + X
    """
    # Step 1: Build covariate table to get tenure_days and event
    cov_table = build_covariate_table(
        transactions=transactions,
        cutoff_date=cutoff_date,
        inactivity_days=inactivity_days,
    )
    cov = cov_table.df
    
    # Step 2: Filter to only active customers (event == 0)
    active_customers = cov[cov["event"] == 0].copy()
    
    if len(active_customers) == 0:
        return pd.DataFrame(columns=[
            "customer_id", "t0", "X_days", "churn_probability",
            "survival_at_t0", "survival_at_t0_plus_X"
        ])
    
    # Step 3: Get covariate columns needed for model
    covariate_cols = list(model.params_.index)
    
    # Create log transformations if needed
    if 'log_monetary_value' in covariate_cols and 'monetary_value' in active_customers.columns:
        active_customers['log_monetary_value'] = np.log1p(active_customers['monetary_value'])
    
    if 'log_product_diversity' in covariate_cols and 'product_diversity' in active_customers.columns:
        active_customers['log_product_diversity'] = np.log1p(active_customers['product_diversity'])
    
    # Verify we have all required covariates
    missing_cols = [col for col in covariate_cols if col not in active_customers.columns]
    if missing_cols:
        raise ValueError(f"Missing required covariates: {missing_cols}")
    
    # Drop rows with missing values in covariates
    active_customers = active_customers.dropna(subset=covariate_cols)
    
    # Helper function to interpolate survival at time t
    def survival_at(sf, t):
        """Interpolate survival function at time t."""
        if t <= sf.index.min():
            return float(sf.iloc[0])
        elif t >= sf.index.max():
            return float(sf.iloc[-1])
        else:
            return float(np.interp(t, sf.index.values, sf.values))
    
    # Step 4: Compute conditional churn probability for each active customer
    results = []
    
    for idx, row in active_customers.iterrows():
        customer_id = row["customer_id"]
        t0 = row["tenure_days"]  # Current duration at cutoff
        
        # Get covariates for this customer (as DataFrame with single row)
        X_df = pd.DataFrame([row[covariate_cols]], columns=covariate_cols)
        
        # Get individual survival curve
        # Returns DataFrame with time as index, one column per customer
        sf = model.predict_survival_function(X_df)
        
        # Extract survival curve (first column, as Series with time index)
        sf_series = sf.iloc[:, 0]  # Series with time index
        
        # Read survival at t0 and t0 + X_days
        s_t0 = survival_at(sf_series, t0)
        s_t1 = survival_at(sf_series, t0 + X_days)
        
        # Step 5: Compute conditional churn probability
        # P(churn in next X | survived to t0) = 1 - S(t0 + X) / S(t0)
        if s_t0 > 0:
            p_churn = 1.0 - (s_t1 / s_t0)
            # Clamp to [0, 1] for numerical stability
            p_churn = max(0.0, min(1.0, p_churn))
        else:
            # If survival at t0 is 0, customer has already churned (shouldn't happen for event=0)
            p_churn = 1.0
        
        results.append({
            "customer_id": customer_id,
            "t0": float(t0),
            "X_days": X_days,
            "churn_probability": p_churn,
            "survival_at_t0": s_t0,
            "survival_at_t0_plus_X": s_t1,
        })
    
    result_df = pd.DataFrame(results)
    
    # Sort by churn probability descending (highest risk first)
    result_df = result_df.sort_values("churn_probability", ascending=False).reset_index(drop=True)
    
    return result_df


def prioritize_retention_targets(
    model: CoxPHFitter,
    transactions: pd.DataFrame,
    clv_df: pd.DataFrame,
    prediction_horizon: int = 90,
    cutoff_date: str = CUTOFF_DATE,
    inactivity_days: int = INACTIVITY_DAYS,
) -> pd.DataFrame:
    """
    Scores active customers by multiplying CLV and churn probability (prioritize_score = clv * churn_prob).
    Use for retention targeting: high-value at-risk customers first.

    Only customers present in BOTH clv_df and churn probability (active) are scored. Customers
    with CLV but no churn probability, or with churn probability but no (valid) CLV, are excluded.

    CLV should come from predict_customer_lifetime_value (e.g. predict_clv from analytics.clv)
    with the same horizon. Churn probability comes from predict_churn_probability over the
    same prediction horizon.

    Args:
        model: Fitted CoxPHFitter model (must NOT be refit)
        transactions: DataFrame with customer_id, invoice_no, invoice_date, revenue, stock_code
        clv_df: DataFrame with columns customer_id and clv (from predict_clv / predict_customer_lifetime_value)
        prediction_horizon: Prediction horizon in days for churn probability and CLV alignment (default: 90)
        cutoff_date: Cutoff date (inclusive) for feature computation (YYYY-MM-DD)
        inactivity_days: Inactivity days threshold for churn definition (default: 90)

    Returns:
        DataFrame with columns:
        - customer_id: Customer identifier
        - clv: Customer lifetime value (from clv_df)
        - churn_prob: Conditional probability of churn in next prediction_horizon days
        - prioritize_score: clv * churn_prob, sorted from highest to lowest
    """
    churn_df = predict_churn_probability(
        model=model,
        transactions=transactions,
        cutoff_date=cutoff_date,
        X_days=prediction_horizon,
        inactivity_days=inactivity_days,
    )
    if "customer_id" not in clv_df.columns or "clv" not in clv_df.columns:
        raise ValueError("clv_df must contain columns 'customer_id' and 'clv'")
    merged = churn_df[["customer_id", "churn_probability"]].merge(
        clv_df[["customer_id", "clv"]].drop_duplicates(subset="customer_id"),
        on="customer_id",
        how="inner",
    )
    merged = merged.dropna(subset=["clv"])
    merged["prioritize_score"] = merged["clv"] * merged["churn_probability"]
    result = merged[["customer_id", "clv", "churn_probability", "prioritize_score"]].rename(
        columns={"churn_probability": "churn_prob"}
    )
    return result.sort_values("prioritize_score", ascending=False).reset_index(drop=True)


def build_segmentation_table(
    model: CoxPHFitter,
    transactions: pd.DataFrame,
    covariates_df: pd.DataFrame,
    cutoff_date: str = CUTOFF_DATE,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Builds final segmentation table combining risk labels and expected remaining lifetime.
    
    Segments active customers based on:
    - Risk label (High/Medium/Low) from score_customers
    - ERL bucket by fixed thresholds: At-Risk (0-90 days), Stable (91-270), Valued (271-720), VIP (>720)
    - Provides action tags and recommended actions for each segment
    
    Args:
        model: Fitted CoxPHFitter model (must NOT be refit)
        transactions: DataFrame with customer_id, invoice_no, invoice_date, revenue, stock_code
        covariates_df: DataFrame with customer_id, event, duration (or tenure_days), and Cox covariates
        cutoff_date: Cutoff date for scoring (YYYY-MM-DD)
    
    Returns:
        Tuple of:
        - final_df: DataFrame with columns:
            customer_id, risk_label, t0, erl_days, life_bucket, segment, action_tag, recommended_action
        - cutoffs: Dictionary with ERL thresholds (erl_at_risk_max, erl_stable_max, erl_valued_max)
    """
    # Step 1: Compute risk labels using score_customers
    df_scores = score_customers(
        model=model,
        transactions=transactions,
        cutoff_date=cutoff_date,
    )
    
    # Extract risk_label from risk_bucket
    df_scores = df_scores[['customer_id', 'risk_bucket']].copy()
    df_scores.rename(columns={'risk_bucket': 'risk_label'}, inplace=True)
    
    # Step 2: Compute ERL using Monte Carlo (BG/NBD)
    transactions_df = transactions.copy()
    transactions_df["invoice_date"] = pd.to_datetime(transactions_df["invoice_date"])
    cutoff_dt = pd.to_datetime(cutoff_date)
    rfm = build_rfm(transactions, cutoff_date)
    last_purchases = (
        transactions_df[transactions_df["invoice_date"] <= cutoff_dt]
        .groupby("customer_id")["invoice_date"]
        .max()
        .reset_index()
    )
    last_purchases.columns = ["customer_id", "last_purchase_date"]
    last_purchases["last_purchase_date"] = last_purchases["last_purchase_date"].dt.strftime("%Y-%m-%d")
    customer_summary = rfm.reset_index().merge(last_purchases, on="customer_id", how="left")
    clv_result = fit_models(rfm)
    df_erl = compute_erl_days_mc(
        bgf=clv_result.bgnbd,
        customer_summary_df=customer_summary,
        cutoff_date=cutoff_date,
        INACTIVITY_DAYS=INACTIVITY_DAYS,
    )
    df_erl = df_erl.rename(columns={"ERL_days": "erl_days"})
    # Keep only active customers (Monte Carlo: erl_days > 0; ERL=0 means already churned)
    df_erl = df_erl[df_erl["erl_days"] > 0].copy()
    df_erl = df_erl.merge(
        customer_summary[["customer_id", "T"]], on="customer_id", how="left"
    ).rename(columns={"T": "t0"})

    # Step 3: Join results
    merged = df_erl.merge(
        df_scores[['customer_id', 'risk_label']],
        on='customer_id',
        how='inner'
    )
    
    # Step 4: Create ERL life_bucket by fixed thresholds (days)
    # At-Risk: 0-90, Stable: 91-270, Valued: 271-720, VIP: >720
    def assign_life_bucket(erl):
        if erl <= 90:
            return "At-Risk"
        elif erl <= 270:
            return "Stable"
        elif erl <= 720:
            return "Valued"
        else:
            return "VIP"
    
    merged['life_bucket'] = merged['erl_days'].apply(assign_life_bucket)
    
    # Step 5: Build segment (risk_label/life_bucket) and map to action_tag + recommended_action
    # segment: e.g. High/At-Risk; action_tag: e.g. Urgent_Winback; recommended_action: e.g. 50% discount + phone survey
    merged['segment'] = merged['risk_label'] + "/" + merged['life_bucket']
    
    # segment -> (action_tag, recommended_action)
    action_map = {
        'High/At-Risk': ('Urgent_Winback', '50% discount + phone survey'),
        'High/Stable': ('Intensive_Nurture', 'Weekly personalized emails + loyalty points'),
        'High/Valued': ('Monitor_Upsell', 'Monthly bundles + churn alerts'),
        'High/VIP': ('VIP_Protect', 'Dedicated concierge + exclusive previews'),
        'Medium/At-Risk': ('Fast_Nurture', 'Reminder emails + 20% off next buy'),
        'Medium/Stable': ('Standard_Cadence', 'Biweekly newsletters + recommendations'),
        'Medium/Valued': ('Targeted_Upsell', 'Product bundles + cross-sell'),
        'Medium/VIP': ('Deep_Engage', 'Beta tests + early access'),
        'Low/At-Risk': ('Light_Observe', 'Monthly check-in emails'),
        'Low/Stable': ('Regular_Engage', 'Quarterly surveys + rewards'),
        'Low/Valued': ('Premium_Upsell', 'VIP bundles + referrals'),
        'Low/VIP': ('Maximize_LTV', 'Custom exclusives + advocacy program'),
    }
    
    def get_action(segment):
        action_tag, recommended_action = action_map.get(segment, ('Unknown', 'No action defined.'))
        return action_tag, recommended_action
    
    merged[['action_tag', 'recommended_action']] = merged['segment'].apply(
        lambda s: pd.Series(get_action(s))
    )
    
    # Step 6: Select and order output columns
    final_df = merged[[
        'customer_id',
        'risk_label',
        't0',
        'erl_days',
        'life_bucket',
        'segment',
        'action_tag',
        'recommended_action'
    ]].copy()
    
    # Sort by risk_label (High first) then by erl_days (descending)
    risk_order = {'High': 0, 'Medium': 1, 'Low': 2}
    final_df['_risk_order'] = final_df['risk_label'].map(risk_order)
    final_df = final_df.sort_values(['_risk_order', 'erl_days'], ascending=[True, False])
    final_df = final_df.drop(columns=['_risk_order']).reset_index(drop=True)
    
    cutoffs = {
        'erl_at_risk_max': 90,
        'erl_stable_max': 270,
        'erl_valued_max': 720,
    }
    return final_df, cutoffs
