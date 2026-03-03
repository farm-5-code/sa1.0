"""Train ML model.

Run:
  python scripts/train_model.py
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
)

from sport_analyzer.config.settings import Config  # noqa: E402
from sport_analyzer.collectors.sports_collector import SportsCollector  # noqa: E402
from sport_analyzer.models.ml_predictor import TrainingDataCollector, MatchPredictor  # noqa: E402


def main():
    cfg = Config()
    sports = SportsCollector(cfg)

    print("\n🤖 ML training")
    print("=" * 50)

    print("\n1) Collect training data...")
    collector = TrainingDataCollector(cfg.DB_PATH, sports)
    df = collector.collect(seasons_back=3)

    if df.empty:
        print("❌ No data found. Run scripts/update_elo.py first.")
        return

    print(f"   Samples: {len(df)}")
    print(
        f"   Outcomes: away={(df['outcome'] == 0).sum()} | "
        f"draw={(df['outcome'] == 1).sum()} | home={(df['outcome'] == 2).sum()}"
    )

    print("\n2) Train model...")
    predictor = MatchPredictor()
    res = predictor.train(df)

    if res.get("status") == "insufficient_data":
        print(
            f"❌ Not enough samples: {res.get('n_samples')} (need >= 200, recommend 1000+)"
        )
        return
    if res.get("status") != "trained":
        print(f"❌ Training failed: {res}")
        return

    print("\n✅ Trained!")
    print(f"   Model: {res['model_type']}")
    print(f"   CV accuracy: {res['cv_accuracy']:.1%} ± {res['cv_std']:.1%}")
    print("   Saved to: models/match_predictor.pkl")


if __name__ == "__main__":
    main()
