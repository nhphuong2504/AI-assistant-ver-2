"""
Test all combinations of Cox model covariates to find the best model.
Tests: n_orders + (monetary_value OR log_monetary_value OR product_diversity OR log_product_diversity)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path to import analytics module
sys.path.insert(0, str(Path(__file__).parent.parent))

from analytics.survival import build_covariate_table, fit_cox_baseline, validate_cox_model
from app.db import run_query_internal

# Load transactions from SQLite
print("Loading data...")
sql = """
SELECT customer_id, invoice_no, invoice_date, revenue, stock_code, country
FROM transactions
WHERE customer_id IS NOT NULL
"""
rows, _ = run_query_internal(sql, max_rows=2_000_000)
df = pd.DataFrame(rows)

# Build covariate table
print("Building covariate table...")
cov_table = build_covariate_table(
    transactions=df,
    cutoff_date="2011-12-09",
    inactivity_days=90,
)
cov = cov_table.df

print(f"Total customers: {len(cov)}")
print(f"Churn rate: {cov['event'].mean():.3f}\n")

# Define all combinations to test
# 2-covariate combinations (n_orders + 1 other)
combinations_2 = [
    ['n_orders', 'monetary_value'],
    ['n_orders', 'log_monetary_value'],
    ['n_orders', 'product_diversity'],
    ['n_orders', 'log_product_diversity'],
]

# 3-covariate combinations (n_orders + 2 others)
combinations_3 = [
    ['n_orders', 'monetary_value', 'product_diversity'],
    ['n_orders', 'monetary_value', 'log_product_diversity'],
    ['n_orders', 'log_monetary_value', 'product_diversity'],
    ['n_orders', 'log_monetary_value', 'log_product_diversity'],
]

# Combine all combinations
combinations = combinations_2 + combinations_3

results = []

print("="*100)
print("TESTING ALL COVARIATE COMBINATIONS")
print("="*100)
print(f"\n2-covariate combinations: {len(combinations_2)}")
print(f"3-covariate combinations: {len(combinations_3)}")
print(f"Total combinations: {len(combinations)}")

for i, covariate_cols in enumerate(combinations, 1):
    print(f"\n{'='*100}")
    print(f"COMBINATION {i}/{len(combinations)}: {covariate_cols}")
    print(f"{'='*100}")
    
    try:
        # Fit Cox model
        print(f"\nFitting Cox model...")
        result = fit_cox_baseline(
            covariates=cov,
            covariate_cols=covariate_cols,
            train_frac=0.8,
            random_state=42,
            penalizer=0.1,
        )
        
        print(f"Training set: {result['n_train']} customers")
        print(f"Validation set: {result['n_validation']} customers")
        print(f"Dropped due to missing: {result['n_dropped']} customers")
        
        # Validate model
        print(f"\nValidating model...")
        validation_result = validate_cox_model(
            model=result['model'],
            train_df=result['train_df'],
            validation_df=result['validation_df'],
        )
        
        # Extract key metrics
        c_index = validation_result['c_index']
        ph_assumption_holds = validation_result['interpretation']['ph_assumption_holds']
        ph_evaluable = validation_result['interpretation']['ph_evaluable']
        ph_violations = validation_result['ph_violations']
        acceptable_performance = validation_result['interpretation']['acceptable_performance']
        
        # Get summary statistics
        summary = result['summary']
        n_significant = (summary['p'] <= 0.05).sum()
        n_covariates = len(summary)
        
        # Calculate average absolute coefficient (measure of effect size)
        avg_abs_coef = summary['coef'].abs().mean()
        
        # Store results
        results.append({
            'combination': ' + '.join(covariate_cols),
            'covariates': covariate_cols,
            'n_train': result['n_train'],
            'n_validation': result['n_validation'],
            'n_dropped': result['n_dropped'],
            'c_index': c_index,
            'ph_assumption_holds': ph_assumption_holds,
            'ph_evaluable': ph_evaluable,
            'ph_violations': len(ph_violations),
            'ph_violation_list': ph_violations,
            'acceptable_performance': acceptable_performance,
            'n_significant': n_significant,
            'n_covariates': n_covariates,
            'avg_abs_coef': avg_abs_coef,
            'summary': summary,
            'validation_interpretation': validation_result['interpretation'],
        })
        
        # Print summary
        print(f"\n--- RESULTS ---")
        print(f"C-index: {c_index:.4f}")
        print(f"PH assumption holds: {ph_assumption_holds}")
        print(f"PH evaluable: {ph_evaluable}")
        if ph_violations:
            print(f"PH violations: {ph_violations}")
        print(f"Acceptable performance: {acceptable_performance}")
        print(f"Significant covariates: {n_significant}/{n_covariates}")
        print(f"\nModel Summary:")
        print(summary[['covariate', 'coef', 'hazard_ratio', 'p']].to_string(index=False))
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        results.append({
            'combination': ' + '.join(covariate_cols),
            'covariates': covariate_cols,
            'error': str(e),
            'c_index': np.nan,
            'ph_assumption_holds': False,
            'ph_evaluable': False,
        })

# Compare all results
print("\n" + "="*100)
print("COMPARISON OF ALL COMBINATIONS")
print("="*100)

results_df = pd.DataFrame([
    {
        'Combination': r['combination'],
        'C-index': r.get('c_index', np.nan),
        'PH Holds': r.get('ph_assumption_holds', False),
        'PH Evaluable': r.get('ph_evaluable', False),
        'PH Violations': r.get('ph_violations', 0),
        'Acceptable': r.get('acceptable_performance', False),
        'Significant': f"{r.get('n_significant', 0)}/{r.get('n_covariates', 0)}",
        'Avg |Coef|': r.get('avg_abs_coef', np.nan),
    }
    for r in results
])

print("\n" + results_df.to_string(index=False))

# Find best combination
print("\n" + "="*100)
print("BEST COMBINATION ANALYSIS")
print("="*100)

# Filter out failed models
valid_results = [r for r in results if 'error' not in r and not np.isnan(r.get('c_index', np.nan))]

if valid_results:
    # Score each model (higher is better)
    # Criteria: C-index (weight: 0.5), PH holds (0.3), acceptable performance (0.2)
    for r in valid_results:
        score = 0.0
        
        # C-index component (0-0.5 points, normalized to 0-1 scale)
        c_idx = r.get('c_index', 0)
        if not np.isnan(c_idx):
            # Normalize: 0.5 -> 0 points, 1.0 -> 0.5 points
            score += (c_idx - 0.5) * 0.5  # 0.5 * (c_index - 0.5) / 0.5
        
        # PH assumption holds (0.3 points)
        if r.get('ph_assumption_holds', False):
            score += 0.3
        
        # Acceptable performance (0.2 points)
        if r.get('acceptable_performance', False):
            score += 0.2
        
        r['score'] = score
    
    # Sort by score
    valid_results.sort(key=lambda x: x.get('score', -1), reverse=True)
    
    print("\nRanked by composite score (C-index + PH assumption + acceptable performance):")
    print("-" * 100)
    for i, r in enumerate(valid_results, 1):
        print(f"\n{i}. {r['combination']}")
        print(f"   Score: {r.get('score', 0):.4f}")
        print(f"   C-index: {r.get('c_index', np.nan):.4f}")
        print(f"   PH assumption holds: {r.get('ph_assumption_holds', False)}")
        print(f"   Acceptable performance: {r.get('acceptable_performance', False)}")
        print(f"   PH violations: {r.get('ph_violations', 0)}")
    
    best = valid_results[0]
    print("\n" + "="*100)
    print(f"🏆 BEST COMBINATION: {best['combination']}")
    print("="*100)
    print(f"C-index: {best.get('c_index', np.nan):.4f}")
    print(f"PH assumption holds: {best.get('ph_assumption_holds', False)}")
    print(f"PH evaluable: {best.get('ph_evaluable', False)}")
    print(f"PH violations: {best.get('ph_violations', 0)}")
    if best.get('ph_violations', 0) > 0:
        print(f"PH violation list: {best.get('ph_violation_list', [])}")
    print(f"Acceptable performance: {best.get('acceptable_performance', False)}")
    print(f"Significant covariates: {best.get('n_significant', 0)}/{best.get('n_covariates', 0)}")
    print(f"\nModel Summary:")
    print(best['summary'][['covariate', 'coef', 'hazard_ratio', 'p']].to_string(index=False))
    print(f"\nInterpretation: {best['validation_interpretation']['c_index_interpretation']}")
else:
    print("\n❌ No valid models to compare!")

print("\n" + "="*100)

