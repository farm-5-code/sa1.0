"""sport_analyzer.models.ml_predictor

Machine-learning correction layer over Poisson/ELO.

Target classes:
  0 = away win
  1 = draw
  2 = home win
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_MODEL_PATH = Path("models/match_predictor.pkl")


def build_features(
    lambda_h: float,
    lambda_a: float,
    elo_home: float,
    elo_away: float,
    form_home: float,
    form_away: float,
    xg_home: Dict,
    xg_away: Dict,
    weather: Dict,
    h2h: Dict,
    fatigue_home: float = 0.0,
    fatigue_away: float = 0.0,
    match_importance: float = 1.0,
) -> np.ndarray:
    elo_diff = elo_home - elo_away
    form_diff = form_home - form_away

    features = [
        # Poisson
        lambda_h,
        lambda_a,
        lambda_h - lambda_a,
        lambda_h / max(lambda_a, 0.1),
        # ELO
        elo_home / 1500.0,
        elo_away / 1500.0,
        elo_diff / 400.0,
        # Form
        form_home / 100.0,
        form_away / 100.0,
        form_diff / 100.0,
        # xG
        float(xg_home.get("xg_for", 1.35)),
        float(xg_home.get("xg_against", 1.35)),
        float(xg_home.get("xg_diff", 0.0)),
        float(xg_away.get("xg_for", 1.35)),
        float(xg_away.get("xg_against", 1.35)),
        float(xg_away.get("xg_diff", 0.0)),
        float(xg_home.get("xg_for", 1.35)) - float(xg_away.get("xg_for", 1.35)),
        # Weather
        float(weather.get("impact_score", 0.0)) / 100.0,
        # H2H
        float(h2h.get("home_win_pct", 33.3)) / 100.0,
        float(h2h.get("draw_pct", 33.3)) / 100.0,
        float(h2h.get("matches", 0.0)) / 50.0,
        # Fatigue
        float(fatigue_home),
        float(fatigue_away),
        float(fatigue_home) - float(fatigue_away),
        # Importance
        float(match_importance),
    ]

    return np.array(features, dtype=np.float32)


FEATURE_NAMES = [
    "lambda_h",
    "lambda_a",
    "lambda_diff",
    "lambda_ratio",
    "elo_home_norm",
    "elo_away_norm",
    "elo_diff_norm",
    "form_home",
    "form_away",
    "form_diff",
    "xgf_home",
    "xga_home",
    "xgd_home",
    "xgf_away",
    "xga_away",
    "xgd_away",
    "xgf_advantage",
    "weather_impact",
    "h2h_home_rate",
    "h2h_draw_rate",
    "h2h_matches_norm",
    "fatigue_home",
    "fatigue_away",
    "fatigue_diff",
    "match_importance",
]


class TrainingDataCollector:
    """Build a training dataset from elo_updates table."""

    def __init__(self, db_path: str, sports_collector):
        self.db_path = db_path
        self.sports = sports_collector

    def collect(self, seasons_back: int = 3) -> pd.DataFrame:
        rows: List[Dict] = []
        matches = self._load_finished_matches(seasons_back)
        logger.info(f"Collecting features for {len(matches)} matches...")

        for m in matches:
            row = self._match_to_row(m)
            if row:
                rows.append(row)
        return pd.DataFrame(rows)

    def _load_finished_matches(self, seasons_back: int) -> List[Dict]:
        with sqlite3.connect(self.db_path, timeout=10) as c:
            rows = c.execute(
                """
                SELECT match_id, home_team, away_team,
                       home_goals, away_goals,
                       elo_before_home, elo_before_away
                FROM elo_updates
                ORDER BY updated_at DESC
                """
            ).fetchall()
        return [
            {
                "id": r[0],
                "home_team": r[1],
                "away_team": r[2],
                "home_goals": r[3],
                "away_goals": r[4],
                "elo_home": r[5],
                "elo_away": r[6],
            }
            for r in rows
        ]

    def _match_to_row(self, m: Dict) -> Optional[Dict]:
        try:
            hg = int(m.get("home_goals") or 0)
            ag = int(m.get("away_goals") or 0)
        except Exception:
            return None

        outcome = 2 if hg > ag else (1 if hg == ag else 0)
        elo_h = float(m.get("elo_home") or 1500.0)
        elo_a = float(m.get("elo_away") or 1500.0)

        exp_h = 1.0 / (1.0 + 10 ** ((elo_a - elo_h - 100.0) / 400.0))
        lambda_h = 1.5 * (1.0 + (exp_h - 0.5) * 0.3)
        lambda_a = 1.15 * (1.0 - (exp_h - 0.5) * 0.3)

        row = {"outcome": outcome}
        row.update(
            {
                "lambda_h": lambda_h,
                "lambda_a": lambda_a,
                "lambda_diff": lambda_h - lambda_a,
                "lambda_ratio": lambda_h / max(lambda_a, 0.1),
                "elo_home_norm": elo_h / 1500.0,
                "elo_away_norm": elo_a / 1500.0,
                "elo_diff_norm": (elo_h - elo_a) / 400.0,
                # Defaults for unavailable features
                "form_home": 0.5,
                "form_away": 0.5,
                "form_diff": 0.0,
                "xgf_home": 1.35,
                "xga_home": 1.35,
                "xgd_home": 0.0,
                "xgf_away": 1.35,
                "xga_away": 1.35,
                "xgd_away": 0.0,
                "xgf_advantage": 0.0,
                "weather_impact": 0.0,
                "h2h_home_rate": 0.333,
                "h2h_draw_rate": 0.333,
                "h2h_matches_norm": 0.0,
                "fatigue_home": 0.0,
                "fatigue_away": 0.0,
                "fatigue_diff": 0.0,
                "match_importance": 1.0,
            }
        )
        return row


class MatchPredictor:
    """Train/predict outcome probabilities."""

    def __init__(self):
        self.bundle: Optional[Dict] = None
        self.is_trained: bool = False
        self._load_if_exists()

    def _load_if_exists(self):
        if not _MODEL_PATH.exists():
            return
        try:
            import pickle

            with open(_MODEL_PATH, "rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, dict) and "model" in obj and "scaler" in obj:
                self.bundle = obj
            else:
                self.bundle = {"model": obj, "scaler": None}
            self.is_trained = True
            logger.info("ML model loaded")
        except Exception as e:
            logger.warning(f"Failed to load ML model: {e}")

    def train(self, df: pd.DataFrame) -> Dict:
        if df is None or df.empty:
            return {"status": "empty"}
        if len(df) < 200:
            return {"status": "insufficient_data", "n_samples": int(len(df))}

        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.preprocessing import StandardScaler

        X = df[FEATURE_NAMES].values.astype(np.float32)
        y = df["outcome"].values.astype(int)

        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)

        model = self._build_model()
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_sc, y, cv=cv, scoring="accuracy")
        model.fit(X_sc, y)

        bundle = {"model": model, "scaler": scaler}
        _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        import pickle

        with open(_MODEL_PATH, "wb") as f:
            pickle.dump(bundle, f)

        self.bundle = bundle
        self.is_trained = True

        return {
            "status": "trained",
            "n_samples": int(len(df)),
            "cv_accuracy": round(float(scores.mean()), 4),
            "cv_std": round(float(scores.std()), 4),
            "model_type": type(model).__name__,
        }

    def predict(self, features: np.ndarray) -> Optional[Dict]:
        if not self.is_trained or not self.bundle:
            return None

        model = self.bundle.get("model")
        scaler = self.bundle.get("scaler")

        try:
            X = features.reshape(1, -1)
            if scaler is not None:
                X = scaler.transform(X)
            probs = model.predict_proba(X)[0]
            classes = list(getattr(model, "classes_", [0, 1, 2]))
            return {
                "away_win": float(probs[classes.index(0)]) if 0 in classes else 0.33,
                "draw": float(probs[classes.index(1)]) if 1 in classes else 0.33,
                "home_win": float(probs[classes.index(2)]) if 2 in classes else 0.33,
                "source": "ml",
            }
        except Exception as e:
            logger.error(f"ML predict error: {e}")
            return None

    @staticmethod
    def _build_model():
        try:
            import lightgbm as lgb  # type: ignore

            return lgb.LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
            )
        except Exception:
            pass

        try:
            import xgboost as xgb  # type: ignore

            return xgb.XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="mlogloss",
                verbosity=0,
            )
        except Exception:
            pass

        from sklearn.ensemble import GradientBoostingClassifier

        logger.info("Using GradientBoostingClassifier (install lightgbm for best results)")
        return GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
        )

    def feature_importance(self) -> Optional[pd.DataFrame]:
        if not self.is_trained or not self.bundle:
            return None
        model = self.bundle.get("model")
        if not hasattr(model, "feature_importances_"):
            return None
        return (
            pd.DataFrame({"feature": FEATURE_NAMES, "importance": model.feature_importances_})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
