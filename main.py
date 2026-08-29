"""
Entry point.

Usage:
    python main.py
    python main.py --config config/config.yaml
"""

import argparse
import sys

from src.exceptions import FraudDetectionError
from src.logger import get_logger
from src.pipeline.training_pipeline import TrainingPipeline

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fraud Detection - Training Pipeline")
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Path to config.yaml"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        pipeline = TrainingPipeline(config_path=args.config)
        pipeline.run()
    except FraudDetectionError as e:
        logger.error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
