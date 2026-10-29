"""Shared date-only cutoff semantics: include the entire named calendar day."""
import pandas as pd


def cutoff_timestamp(value):
    return pd.Timestamp(value).normalize() + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
