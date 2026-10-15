"""데이터 파이프라인 CLI."""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Sequence

from src.oibc.config import CompetitionSpec, DataPaths, QualityConfig
from src.oibc.data import derived, ingest

logger = logging.getLogger(__name__)


def main(argv: Sequence[str] | None = None) -> int:
    """하위 명령에 해당하는 구축 단계를 실행한다."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(prog="pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest")
    sub.add_parser("derive")
    args = parser.parse_args(list(argv) if argv is not None else None)
    start = time.perf_counter()
    logger.info("[SYS] pipeline %s start", args.command)
    if args.command == "ingest":
        ingest.build_canonical(DataPaths(), CompetitionSpec(), QualityConfig())
    else:
        derived.build_derived(DataPaths(), CompetitionSpec())
    elapsed = time.perf_counter() - start
    logger.info("[SYS] pipeline %s done in %.1fs", args.command, elapsed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
