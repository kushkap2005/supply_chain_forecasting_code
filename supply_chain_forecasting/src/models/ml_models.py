"""
ml_models.py
XGBoost regressor and ARIMA wrapper for SKU-level demand forecasting.
"""

import numpy as np
import pandas as pd
import pickle
import warnings
from pathlib import Path

from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")



# XGBoost


class XGBDemandForecaster:
    def __init__(self, feature_cols: list):
        self.feature_cols = feature_cols
        self.model = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )

    def fit(self, df_train: pd.DataFrame):
        X = df_train[self.feature_cols]
        y = df_train["demand"]
        self.model.fit(X, y)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.clip(self.model.predict(df[self.feature_cols]), 0, None)

    def feature_importance(self) -> pd.Series:
        return pd.Series(
            self.model.feature_importances_,
            index=self.feature_cols
        ).sort_values(ascending=False)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "XGBDemandForecaster":
        with open(path, "rb") as f:
            return pickle.load(f)


# ARIMA (baseline)

class ARIMAForecaster:
    """
    Per-SKU ARIMA(2,1,2) wrapper.
    Fits on a univariate demand series and produces rolling forecasts.
    """

    def __init__(self, order: tuple = (2, 1, 2)):
        self.order = order
        self._fitted = {}   # sku_id → fitted ARIMA result

    def fit_sku(self, sku_id: str, series: pd.Series):
        try:
            result = ARIMA(series.values, order=self.order).fit()
            self._fitted[sku_id] = result
        except Exception:
            self._fitted[sku_id] = None
        return self

    def predict_sku(self, sku_id: str, steps: int = 28) -> np.ndarray:
        result = self._fitted.get(sku_id)
        if result is None:
            return np.zeros(steps)
        fc = result.forecast(steps=steps)
        return np.clip(fc, 0, None)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "ARIMAForecaster":
        with open(path, "rb") as f:
            return pickle.load(f)
