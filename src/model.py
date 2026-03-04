"""Model training and evaluation for the equity signal pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.pipeline import FoldRange, WalkForwardSplitter


LOGGER = logging.getLogger(__name__)


@dataclass
class ModelOutputs:
    """Container for model outputs and aggregate statistics."""

    predictions: pd.DataFrame
    metrics: pd.DataFrame
    monthly_ic: pd.Series
    feature_importance: pd.Series
    fold_ranges: list[FoldRange]


class SignalModel:
    """Wrapper for model training under walk-forward cross validation."""

    def __init__(
        self,
        model_type: str,
        random_seed: int,
        xgb_params: dict[str, Any],
        rf_params: dict[str, Any],
        logistic_params: dict[str, Any],
    ):
        """Build a model wrapper with configuration-provided hyperparameters."""
        self.model_type = model_type
        self.random_seed = random_seed
        self.xgb_params = xgb_params
        self.rf_params = rf_params
        self.logistic_params = logistic_params

    def _build_estimator(self):
        """Instantiate the configured estimator."""
        if self.model_type == "xgboost":
            try:
                from xgboost import XGBClassifier  # pylint: disable=import-outside-toplevel
            except Exception as exc:  # pylint: disable=broad-except
                raise ImportError("xgboost is required for MODEL_TYPE='xgboost'.") from exc
            return XGBClassifier(random_state=self.random_seed, **self.xgb_params)

        if self.model_type == "random_forest":
            return RandomForestClassifier(random_state=self.random_seed, **self.rf_params)

        if self.model_type == "logistic":
            return LogisticRegression(random_state=self.random_seed, **self.logistic_params)

        raise ValueError(f"Unsupported model type: {self.model_type}")

    @staticmethod
    def _safe_auc(y_true: pd.Series, y_prob: pd.Series) -> float:
        """Compute AUC safely when a sample has only one class."""
        if y_true.nunique() < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_prob))

    @staticmethod
    def _compute_ic(y_prob: pd.Series, forward_return: pd.Series) -> float:
        """Compute Information Coefficient (Spearman rank correlation)."""
        valid = pd.concat([y_prob, forward_return], axis=1).dropna()
        if valid.empty:
            return float("nan")
        corr, _ = spearmanr(valid.iloc[:, 0], valid.iloc[:, 1])
        return float(corr)

    @staticmethod
    def _extract_feature_importance(estimator, feature_cols: list[str]) -> pd.Series:
        """Extract feature importance for models that expose it."""
        if hasattr(estimator, "feature_importances_"):
            values = pd.Series(estimator.feature_importances_, index=feature_cols)
            return values
        if hasattr(estimator, "coef_"):
            values = pd.Series(np.abs(estimator.coef_[0]), index=feature_cols)
            return values
        return pd.Series(dtype=float)

    def run_walk_forward(
        self,
        data: pd.DataFrame,
        feature_cols: list[str],
        splitter: WalkForwardSplitter,
        classification_threshold: float,
    ) -> ModelOutputs:
        """Train and evaluate the model over walk-forward folds using date-based splits."""
        unique_dates = pd.DatetimeIndex(sorted(data["date"].unique()))
        fold_ranges = splitter.fold_ranges(unique_dates)

        oof_frames: list[pd.DataFrame] = []
        fold_metric_rows: list[dict[str, Any]] = []
        feat_importances: list[pd.Series] = []

        for fold_idx, ((train_idx, test_idx), fold_range) in enumerate(
            zip(splitter.split(unique_dates), fold_ranges),
            start=1,
        ):
            train_dates = unique_dates[train_idx]
            test_dates = unique_dates[test_idx]

            train_mask = data["date"].isin(train_dates)
            test_mask = data["date"].isin(test_dates)

            train_df = data.loc[train_mask].copy()
            test_df = data.loc[test_mask].copy()

            x_train = train_df[feature_cols].to_numpy()
            y_train = train_df["label"].to_numpy()
            x_test = test_df[feature_cols].to_numpy()
            y_test = test_df["label"].to_numpy()

            scaler = StandardScaler()
            x_train_scaled = scaler.fit_transform(x_train)
            x_test_scaled = scaler.transform(x_test)

            estimator = self._build_estimator()
            estimator.fit(x_train_scaled, y_train)
            y_prob = estimator.predict_proba(x_test_scaled)[:, 1]
            y_pred = (y_prob >= classification_threshold).astype(int)

            fold_preds = test_df[["date", "ticker", "forward_return", "label", "Open", "Close"]].copy()
            fold_preds["pred_prob"] = y_prob
            fold_preds["pred_label"] = y_pred
            fold_preds["fold"] = fold_idx
            oof_frames.append(fold_preds)

            fold_auc = self._safe_auc(test_df["label"], pd.Series(y_prob, index=test_df.index))
            fold_ic = self._compute_ic(pd.Series(y_prob, index=test_df.index), test_df["forward_return"])

            fold_metric_rows.extend(
                [
                    {"scope": "fold", "fold": fold_idx, "metric": "accuracy", "value": float(accuracy_score(y_test, y_pred))},
                    {"scope": "fold", "fold": fold_idx, "metric": "auc_roc", "value": fold_auc},
                    {
                        "scope": "fold",
                        "fold": fold_idx,
                        "metric": "brier_score",
                        "value": float(brier_score_loss(y_test, y_prob)),
                    },
                    {"scope": "fold", "fold": fold_idx, "metric": "ic", "value": fold_ic},
                ]
            )

            feat_importances.append(self._extract_feature_importance(estimator, feature_cols))

            LOGGER.info(
                "Fold %d/%d complete: Train %s -> %s | Test %s -> %s",
                fold_idx,
                splitter.n_splits,
                fold_range.train_start.date(),
                fold_range.train_end.date(),
                fold_range.test_start.date(),
                fold_range.test_end.date(),
            )

        predictions = pd.concat(oof_frames, axis=0, ignore_index=True).sort_values(["date", "ticker"])

        metrics_rows = list(fold_metric_rows)
        agg_auc = self._safe_auc(predictions["label"], predictions["pred_prob"])
        agg_ic = self._compute_ic(predictions["pred_prob"], predictions["forward_return"])
        metrics_rows.extend(
            [
                {
                    "scope": "aggregate",
                    "fold": np.nan,
                    "metric": "accuracy",
                    "value": float(accuracy_score(predictions["label"], predictions["pred_label"])),
                },
                {"scope": "aggregate", "fold": np.nan, "metric": "auc_roc", "value": agg_auc},
                {
                    "scope": "aggregate",
                    "fold": np.nan,
                    "metric": "brier_score",
                    "value": float(brier_score_loss(predictions["label"], predictions["pred_prob"])),
                },
                {"scope": "aggregate", "fold": np.nan, "metric": "ic", "value": agg_ic},
            ]
        )

        monthly_ic = (
            predictions.set_index("date")
            .groupby(pd.Grouper(freq="M"))
            .apply(lambda x: self._compute_ic(x["pred_prob"], x["forward_return"]))
            .rename("monthly_ic")
        )

        if feat_importances and not all(series.empty for series in feat_importances):
            feature_importance = pd.concat(feat_importances, axis=1).mean(axis=1).sort_values(ascending=False)
        else:
            feature_importance = pd.Series(dtype=float)

        metrics = pd.DataFrame(metrics_rows)

        return ModelOutputs(
            predictions=predictions,
            metrics=metrics,
            monthly_ic=monthly_ic,
            feature_importance=feature_importance,
            fold_ranges=fold_ranges,
        )
