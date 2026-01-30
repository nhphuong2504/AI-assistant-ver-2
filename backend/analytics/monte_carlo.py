import pandas as pd
import numpy as np
from datetime import datetime
from lifetimes import BetaGeoFitter
from typing import Union


def compute_erl_days(
    bgf: BetaGeoFitter,
    customer_summary_df: pd.DataFrame,
    cutoff_date: Union[str, datetime],
    INACTIVITY_DAYS: int,
    N: int = 1000,
    seed: int = 42,
    max_days: int = 1825,
) -> pd.DataFrame:
    """
    Compute Expected Remaining Lifetime (ERL_days) using Monte Carlo simulation.

    ERL_days = expected days until BUSINESS-DEFINED churn:
      churn occurs when NO purchases for INACTIVITY_DAYS consecutive days.

    Key corrections vs common pitfalls:
    - Uses P(Alive) to decide if customer is already latent-dead at cutoff *per simulation path*.
      If already dead at cutoff, observed churn occurs at (INACTIVITY_DAYS - current_age).
    - Still simulates future latent dropout after purchases (dropout coin flip).
      If latent dropout occurs at time t (after a purchase), observed churn is t + INACTIVITY_DAYS.
    """
    rng = np.random.default_rng(seed)

    cutoff_date = pd.to_datetime(cutoff_date)
    df = customer_summary_df.copy()

    # 1) last_purchase_age_days
    if "last_purchase_date" in df.columns:
        df["last_purchase_date"] = pd.to_datetime(df["last_purchase_date"])
        df["last_purchase_age_days"] = (cutoff_date - df["last_purchase_date"]).dt.days
    elif "last_purchase_age_days" not in df.columns:
        raise ValueError("Must provide 'last_purchase_date' or 'last_purchase_age_days'")

    # 2) Already churned by business rule => ERL=0
    already_inactive = df["last_purchase_age_days"] >= INACTIVITY_DAYS
    df["ERL_days"] = 0.0

    # 3) P(alive at cutoff) from BG/NBD
    # (lifetimes accepts array-like)
    df["prob_alive"] = bgf.conditional_probability_alive(
        df["frequency"].astype(float),
        df["recency"].astype(float),
        df["T"].astype(float),
    )

    active_mask = ~already_inactive
    active_df = df.loc[active_mask].copy()
    if active_df.empty:
        return df[["customer_id", "ERL_days", "prob_alive", "last_purchase_age_days"]]

    # BG/NBD hyperparameters
    r = float(bgf.params_["r"])
    alpha = float(bgf.params_["alpha"])
    a = float(bgf.params_["a"])
    b = float(bgf.params_["b"])

    erl_list = []
    erl_if_alive_list = []  # optional diagnostic

    # Iterate customers; vectorize across N simulations per customer
    for _, row in active_df.iterrows():
        freq = float(row["frequency"])
        T = float(row["T"])
        current_age = float(row["last_purchase_age_days"])
        p_alive = float(row["prob_alive"])

        # ---- A) Decide "already latent-dead at cutoff" per simulation path ----
        alive_flags = rng.random(N) < p_alive  # True => alive at cutoff in that world

        # If already dead at cutoff, observed churn happens when inactivity window completes
        dead_churn_time = max(0.0, INACTIVITY_DAYS - current_age)

        # Initialize arrays
        times = np.full(N, dead_churn_time, dtype=float)     # churn time for dead-at-cutoff paths
        ages = np.full(N, current_age, dtype=float)
        active_sims = alive_flags.copy()  # only alive-at-cutoff sims continue simulating

        # If nobody is alive in any sim, ERL is just dead_churn_time
        if not np.any(active_sims):
            erl_list.append(times.mean())
            erl_if_alive_list.append(0.0)
            continue

        # ---- B) Sample per-path parameters for the alive sims ----
        # Purchase rate (heuristic posterior draw)
        lambdas = np.empty(N, dtype=float)
        lambdas[active_sims] = rng.gamma(shape=r + freq, scale=1.0 / (alpha + T), size=active_sims.sum())

        # Dropout probability per path
        # NOTE: Beta(a, b+freq) is a *heuristic* that shrinks dropout lower for frequent buyers.
        # If you want strict BG/NBD without this heuristic, use rng.beta(a, b, size=...) instead.
        ps = np.empty(N, dtype=float)
        ps[active_sims] = rng.beta(a, b + freq, size=active_sims.sum())

        # Initialize times for alive sims at 0 (time starts at cutoff)
        times[active_sims] = 0.0

        # ---- C) Simulate forward for alive sims ----
        while np.any(active_sims):
            idx = np.where(active_sims)[0]

            # draw next interpurchase times: delta ~ Exp(rate=lambda)
            deltas = rng.exponential(scale=1.0 / lambdas[idx])
            times[idx] += deltas
            ages[idx] += deltas

            # Case 1: inactivity triggers before next purchase
            churn_inact = ages[idx] >= INACTIVITY_DAYS
            if np.any(churn_inact):
                churn_idx = idx[churn_inact]
                # exact crossing time when age hits INACTIVITY_DAYS
                excess = ages[churn_idx] - INACTIVITY_DAYS
                times[churn_idx] -= excess
                active_sims[churn_idx] = False

            # survivors had a purchase at time t
            surv_idx = idx[~churn_inact]
            if surv_idx.size == 0:
                continue

            ages[surv_idx] = 0.0

            # Case 2: latent dropout after purchase
            drop = rng.binomial(1, ps[surv_idx])
            died_idx = surv_idx[drop == 1]
            if died_idx.size > 0:
                # observed churn is INACTIVITY_DAYS after the final purchase
                times[died_idx] += INACTIVITY_DAYS
                active_sims[died_idx] = False

            # Safety cap
            cap_idx = surv_idx[times[surv_idx] > max_days]
            if cap_idx.size > 0:
                times[cap_idx] = max_days
                active_sims[cap_idx] = False

        # ---- D) Aggregate ----
        erl = float(times.mean())
        erl_list.append(erl)

        # Optional: conditional ERL if alive at cutoff (for debugging / reporting)
        if alive_flags.any():
            erl_if_alive_list.append(float(times[alive_flags].mean()))
        else:
            erl_if_alive_list.append(0.0)

    df.loc[active_mask, "ERL_days"] = erl_list
    df.loc[active_mask, "ERL_simulated_if_alive"] = erl_if_alive_list  # optional column

    return df[["customer_id", "ERL_days", "prob_alive", "last_purchase_age_days", "ERL_simulated_if_alive"]]
