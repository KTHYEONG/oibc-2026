"""데이터 파이프라인 CLI."""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Sequence
from pathlib import Path

from src.oibc.config import CompetitionSpec, DataPaths, FeatureConfig, QualityConfig
from src.oibc.data import derived, ingest
from src.oibc.features import panel as features_panel
from src.oibc.submit.validate import validate_submission

logger = logging.getLogger(__name__)


def main(argv: Sequence[str] | None = None) -> int:
    """하위 명령에 해당하는 구축 단계를 실행한다."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(prog="pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest")
    sub.add_parser("derive")
    sub.add_parser("features")
    validate_cmd = sub.add_parser("validate-submission")
    validate_cmd.add_argument("--composition", type=Path, required=True)
    validate_cmd.add_argument("--forecast", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    start = time.perf_counter()
    logger.info("[SYS] pipeline %s start", args.command)
    exit_code = 0
    if args.command == "ingest":
        ingest.build_canonical(DataPaths(), CompetitionSpec(), QualityConfig())
    elif args.command == "derive":
        derived.build_derived(DataPaths(), CompetitionSpec())
    elif args.command == "features":
        features_panel.build_features(DataPaths(), CompetitionSpec(), FeatureConfig())
    elif args.command == "validate-submission":
        report = validate_submission(args.composition, args.forecast)
        for warn in report.warnings:
            logger.info("[SUBMIT] warning: %s", warn)
        exit_code = 0 if report.ok else 1
    elapsed = time.perf_counter() - start
    logger.info("[SYS] pipeline %s done in %.1fs", args.command, elapsed)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
