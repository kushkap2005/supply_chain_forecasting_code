"""
metrics.py
Forecast evaluation: MAPE, RMSE, MAE, Bias.
"""

import numpy as np
import pandas as pd


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual, predicted = np.array(actual), np.array(predicted)
    mask = actual > 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.array(actual) - np.array(predicted)) ** 2)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(np.array(actual) - np.array(predicted))))


def bias(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean bias (positive = over-forecast, negative = under-forecast)."""
    return float(np.mean(np.array(predicted) - np.array(actual)))


def evaluate_all(actual: np.ndarray, predicted: np.ndarray, model_name: str) -> dict:
    return {
        "model": model_name,
        "MAPE": round(mape(actual, predicted), 3),
        "RMSE": round(rmse(actual, predicted), 3),
        "MAE":  round(mae(actual, predicted), 3),
        "Bias": round(bias(actual, predicted), 3),
    }


def summarize_results(results: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    return df.sort_values("MAPE")
