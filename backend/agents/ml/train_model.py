"""
Training script scaffold for item #8 ("replace the weighted formula with a
trained LightGBM/XGBoost/CatBoost/Random Forest model").

This is intentionally NOT wired to run automatically: training a return
predictor needs a labelled historical dataset (features at time T, actual
forward return at T+horizon) that only you can assemble from your own
data licence (5 years of NSE OHLCV + fundamentals + news, per the review
doc's "Feature Store" item #9). Feeding it fabricated or scraped-without-a-
licence data would make the "prediction" meaningless or legally risky, so
this script defines the CONTRACT the rest of the system expects and leaves
the actual data loading to you.

Usage once you have a labelled dataset as a DataFrame with columns:
    rsi, macd_hist, ema20, ema50, pe, pb, roe, debt_to_equity,
    sentiment_score, volume_zscore, ...  (your feature set)
    forward_return_pct   <- the label

    python train_model.py --data path/to/dataset.csv --out model.joblib

Then set PREDICTION_MODEL_PATH=model.joblib in .env and
agents/prediction_agent.py will load and use it automatically, falling
back to the weighted formula if loading fails for any reason.
"""
import argparse
import sys


class TrainedReturnModel:
    """Thin wrapper so prediction_agent.py has one stable call signature
    regardless of which library the underlying model was trained with."""

    def __init__(self, sk_model, feature_names: list[str]):
        self.sk_model = sk_model
        self.feature_names = feature_names

    def predict_feature_vector(self, feature_vector: dict) -> dict:
        import numpy as np

        x = np.array([[feature_vector.get(f, 0.0) for f in self.feature_names]])
        expected_return_pct = float(self.sk_model.predict(x)[0])
        # A single point estimate isn't a probability distribution - derive
        # a rough confidence range and profit probability the same way the
        # weighted-formula fallback does, from residual std if available.
        std = getattr(self.sk_model, "residual_std_", 1.5)
        return {
            "expected_return_pct": round(expected_return_pct, 3),
            "range_low_pct": round(expected_return_pct - std, 3),
            "range_high_pct": round(expected_return_pct + std, 3),
            "profit_probability_pct": round(min(max(50 + expected_return_pct * 10, 5), 95), 1),
            "loss_probability_pct": round(100 - min(max(50 + expected_return_pct * 10, 5), 95), 1),
        }


def main():
    parser = argparse.ArgumentParser(description="Train a return-prediction model for the Prediction Agent.")
    parser.add_argument("--data", required=True, help="CSV with feature columns + a forward_return_pct label")
    parser.add_argument("--out", default="model.joblib")
    parser.add_argument("--model", choices=["lightgbm", "xgboost", "random_forest"], default="random_forest")
    args = parser.parse_args()

    try:
        import joblib
        import pandas as pd
    except ImportError:
        print("Install scikit-learn/pandas/joblib (and lightgbm/xgboost if selected) first: "
              "pip install scikit-learn pandas joblib lightgbm xgboost", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.data)
    if "forward_return_pct" not in df.columns:
        print("Dataset must contain a 'forward_return_pct' label column.", file=sys.stderr)
        sys.exit(1)

    feature_names = [c for c in df.columns if c != "forward_return_pct"]
    X, y = df[feature_names], df["forward_return_pct"]

    if args.model == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42)
    elif args.model == "lightgbm":
        import lightgbm as lgb
        model = lgb.LGBMRegressor(n_estimators=300, max_depth=8, random_state=42)
    else:
        import xgboost as xgb
        model = xgb.XGBRegressor(n_estimators=300, max_depth=6, random_state=42)

    model.fit(X, y)
    residuals = y - model.predict(X)
    model.residual_std_ = float(residuals.std())

    wrapped = TrainedReturnModel(model, feature_names)
    joblib.dump(wrapped, args.out)
    print(f"Saved trained model to {args.out} (features: {feature_names})")


if __name__ == "__main__":
    main()
