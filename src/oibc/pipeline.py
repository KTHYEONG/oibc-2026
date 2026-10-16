"""데이터 파이프라인 CLI."""

from __future__ import annotations

import argparse
import hashlib
import logging
import math
import time
from collections.abc import Sequence
from pathlib import Path

from src.oibc.config import (
    REFERENCE_COMP_ID,
    CompetitionSpec,
    ComposeConfig,
    DataPaths,
    FeatureConfig,
    ModelConfig,
    QualityConfig,
    RunConfig,
)
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
    val = sub.add_parser("validate-run")
    val.add_argument("--run-id", required=True)
    val.add_argument("--hypothesis", required=True)
    val.add_argument("--baseline", default=None)
    val.add_argument("--composition", default=REFERENCE_COMP_ID)
    val.add_argument("--drop-group", action="append", choices=sorted(FeatureConfig().groups), default=[])
    val.add_argument("--weight", choices=["smp_k_cap", "k_cap", "none"], default="smp_k_cap")
    val.add_argument("--allow-commit-drift", action="store_true")
    val.add_argument("--save-oof", action="store_true")
    comp_cmd = sub.add_parser("compose")
    comp_cmd.add_argument("--comp-id", required=True)
    rn = sub.add_parser("report-netting")
    rn.add_argument("--run-id", required=True)
    rn.add_argument("--composition", action="append", required=True)
    rn.add_argument("--n-random", type=int, default=200)
    rn.add_argument("--seed", type=int, default=0)
    sub_cmd = sub.add_parser("submit")
    sub_cmd.add_argument("--run-id", required=True)
    sub_cmd.add_argument("--sub-id", required=True)
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
    elif args.command == "validate-run":
        from src.oibc.train.runner import run_validation

        groups = set(FeatureConfig().groups) - set(args.drop_group)
        run_validation(
            RunConfig(
                run_id=args.run_id,
                hypothesis=args.hypothesis,
                baseline_run_id=args.baseline,
                composition_id=args.composition,
                features=FeatureConfig(groups=frozenset(groups)),
                model=ModelConfig(weight=args.weight),
                allow_commit_drift=args.allow_commit_drift,
            ),
            save_oof=args.save_oof,
        )
    elif args.command == "compose":
        import pandas as pd

        from src.oibc.compose.anticluster import build_composition
        from src.oibc.compose.constraints import validate_assignment
        from src.oibc.compose.store import write_composition

        compose_cfg = ComposeConfig()
        sites = pd.read_parquet(DataPaths().canonical_dir / "sites.parquet")
        assignment = build_composition(sites, compose_cfg)
        out_path = write_composition(assignment, args.comp_id, compose_cfg, sites)
        caps = validate_assignment(assignment, sites)
        logger.info("[VPP] compose id=%s path=%s caps=%s", args.comp_id, out_path, caps)
    elif args.command == "submit":
        import pandas as pd

        from src.oibc.evaluate.registry import read_records
        from src.oibc.submit.build import write_submission
        from src.oibc.train.runner import RunnerError, forecast_eval, resolve_composition

        records = read_records()
        match = next((r for r in records if r.run_id == args.run_id), None)
        if match is None:
            raise RunnerError(f"submit: unknown run_id {args.run_id!r}")
        comparison = match.comparison or {}
        bag_map = comparison.get("bag", {})
        if not isinstance(bag_map, dict) or not bag_map:
            raise RunnerError("submit: missing bag metrics")
        if set(bag_map) != set(match.folds):
            raise RunnerError("submit: incomplete bag folds")
        try:
            local_metrics = {
                name: sum(float(bag_map[fold][name]) for fold in match.folds)
                for name in ("W_all", "W_warm")
            }
            config = match.config
            if not {"features", "model", "panel_path", "panel_sha256", "competition"} <= set(config):
                raise ValueError("missing reproducibility config")
            cfg = RunConfig.model_validate(config)
            spec_data = dict(config["competition"])
            for key in ("train_end", "eval_start", "eval_end"):
                spec_data[key] = pd.Timestamp(spec_data[key]).tz_convert("Asia/Seoul")
            spec = CompetitionSpec.model_validate(spec_data)
            panel_path = Path(config["panel_path"])
            with panel_path.open("rb") as fh:
                panel_hash = hashlib.file_digest(fh, "sha256").hexdigest()
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise RunnerError(f"submit: invalid recorded config or metrics ({exc})") from exc
        if (
            cfg.run_id != match.run_id or cfg.model.seeds != match.seeds
            or panel_hash != config["panel_sha256"]
            or not all(math.isfinite(v) and v >= 0 for v in local_metrics.values())
        ):
            raise RunnerError("submit: changed panel, seeds or invalid metrics")
        sites = pd.read_parquet(DataPaths().canonical_dir / "sites.parquet")
        assignment, comp_sha = resolve_composition(cfg.composition_id, sites, spec=spec)
        if "composition_sha256" in config:
            if config["composition_sha256"] != comp_sha:
                raise RunnerError("submit: composition hash drift")
        elif cfg.composition_id != REFERENCE_COMP_ID:
            raise RunnerError("submit: missing composition_sha256")
        forecast = forecast_eval(cfg, panel_path=panel_path, spec=spec)
        _, final_sha = resolve_composition(
            cfg.composition_id, pd.read_parquet(DataPaths().canonical_dir / "sites.parquet"), spec=spec,
        )
        if final_sha != comp_sha:
            raise RunnerError("submit: composition changed during inference")
        write_submission(
            assignment,
            forecast,
            args.sub_id,
            run_id=args.run_id,
            composition_id=cfg.composition_id,
            local_metrics=local_metrics,
            spec=spec,
        )
    elif args.command == "report-netting":
        import json

        from src.oibc.config import ArtifactPaths as _ArtifactPaths
        from src.oibc.train.runner import netting_report

        net_report = netting_report(
            args.run_id,
            list(args.composition),
            n_random=args.n_random,
            seed=args.seed,
        )
        run_dir = _ArtifactPaths().runs / args.run_id
        out_path = run_dir / f"netting_n{args.n_random}_seed{args.seed}.json"
        if out_path.exists() or out_path.is_symlink():
            raise FileExistsError(f"report-netting: already exists {out_path}")
        run_dir.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(net_report, ensure_ascii=False, indent=2, allow_nan=False)
        with out_path.open("x", encoding="utf-8") as fh:
            fh.write(payload)
            fh.write("\n")
        logger.info("[VPP] netting report run=%s path=%s", args.run_id, out_path)
    elapsed = time.perf_counter() - start
    logger.info("[SYS] pipeline %s done in %.1fs", args.command, elapsed)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
