"""데이터 파이프라인 CLI."""

from __future__ import annotations

import argparse
import hashlib
import logging
import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from src.oibc.config import (
    REFERENCE_COMP_ID,
    CompetitionSpec,
    ComposeConfig,
    DataPaths,
    DriftConfig,
    EcmwfConfig,
    FamilyConfig,
    FeatureConfig,
    ModelConfig,
    NoaaGfsConfig,
    QualityConfig,
    RegionalConfig,
    RunConfig,
)
from src.oibc.data import derived, ingest
from src.oibc.features import panel as features_panel
from src.oibc.submit.validate import validate_submission

if TYPE_CHECKING:
    import pandas as pd

    from src.oibc.submit.build import SubmissionManifest

logger = logging.getLogger(__name__)


def main(argv: Sequence[str] | None = None) -> int:
    """하위 명령에 해당하는 구축 단계를 실행한다."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(prog="pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest")
    sub.add_parser("derive")
    sub.add_parser("features")
    gfs_fetch = sub.add_parser("gfs-fetch")
    gfs_fetch.add_argument("--workers", type=int, default=16)
    gfs_fetch.add_argument("--source", choices=("dynamical", "noaa"), default="dynamical")
    gfs_cmp = sub.add_parser("gfs-compare")
    gfs_cmp.add_argument("--inits", nargs="*", default=None)
    gfs_cmp.add_argument("--sample", type=int, default=None)
    sub.add_parser("gfs-audit")
    sub.add_parser("gfs-hourly")
    ec_fetch = sub.add_parser("ecmwf-fetch")
    ec_fetch.add_argument("--runs", nargs="+", default=None)
    ec_fetch.add_argument("--workers", type=int, default=8)
    ec_audit = sub.add_parser("ecmwf-audit")
    ec_audit.add_argument("--workers", type=int, default=8)
    ec_equiv = sub.add_parser("ecmwf-equiv")
    ec_equiv.add_argument("--workers", type=int, default=6)
    ec_corr = sub.add_parser("ecmwf-correct")
    ec_corr.add_argument("--base-sub", required=True)
    ec_corr.add_argument("--sub-id", required=True)
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
    val.add_argument("--regional-weight", type=float, default=0.0)
    val.add_argument("--cat-weight", type=float, default=0.0)
    val.add_argument("--mlp-weight", type=float, default=0.0)
    raw_vars = sorted((*FeatureConfig().wa_vars, *FeatureConfig().wb_vars))
    val.add_argument("--drop-var", action="append", choices=raw_vars, default=[])
    val.add_argument("--mlp-clip", nargs=2, type=float, metavar=("LO", "HI"), default=None)
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
    sub_cmd.add_argument("--calibration", action="store_true")
    sub_cmd.add_argument("--allow-break", action="append", choices=raw_vars, default=[])
    splice = sub.add_parser("splice-submission")
    splice.add_argument("--base-sub", required=True)
    splice.add_argument("--fallback-sub", required=True)
    splice.add_argument("--sub-id", required=True)
    scale_cmd = sub.add_parser("scale-submission")
    scale_cmd.add_argument("--base-sub", required=True)
    scale_cmd.add_argument("--scale", type=float, required=True)
    scale_cmd.add_argument("--sub-id", required=True)
    scale_cmd.add_argument("--months", nargs="+", type=int, choices=range(1, 13), default=None)
    rc = sub.add_parser("report-calibration")
    rc.add_argument("--run-id", required=True)
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
    elif args.command == "gfs-fetch":  # pragma: no cover - 네트워크 수집
        from src.oibc.data.external_gfs import fetch_gfs

        fetch_gfs(DataPaths(), FeatureConfig(), workers=args.workers, source=args.source, noaa=NoaaGfsConfig())
    elif args.command == "gfs-compare":
        from src.oibc.data.external_gfs import compare_gfs_sources, parse_init_id

        inits = [parse_init_id(x) for x in args.inits] if args.inits else None
        noaa_cfg = NoaaGfsConfig() if args.sample is None else NoaaGfsConfig(compare_sample=args.sample)
        compare_gfs_sources(DataPaths(), FeatureConfig(), noaa_cfg, inits)
    elif args.command == "gfs-audit":  # pragma: no cover - 네트워크 수집
        from src.oibc.data.external_gfs import audit_publication

        audit_publication(DataPaths(), FeatureConfig())
    elif args.command == "gfs-hourly":
        from src.oibc.data.external_gfs import build_gfs_hourly

        build_gfs_hourly(DataPaths(), CompetitionSpec(), FeatureConfig())
    elif args.command == "ecmwf-fetch":  # pragma: no cover - 네트워크 수집
        from src.oibc.data.external_ecmwf import fetch_ecmwf

        exit_code = 1 if fetch_ecmwf(DataPaths(), EcmwfConfig(), runs=args.runs, workers=args.workers) else 0
    elif args.command == "ecmwf-audit":  # pragma: no cover - 네트워크 수집
        from src.oibc.data.external_ecmwf import audit_ecmwf

        audit_ecmwf(DataPaths(), EcmwfConfig(), workers=args.workers)
    elif args.command == "ecmwf-equiv":  # pragma: no cover - 네트워크 목록 대조
        from src.oibc.data.external_ecmwf import audit_mirror

        audit_mirror(DataPaths(), EcmwfConfig(), workers=args.workers)
    elif args.command == "ecmwf-correct":
        ecmwf_correct_submission(args.base_sub, args.sub_id)
    elif args.command == "validate-submission":
        report = validate_submission(args.composition, args.forecast)
        for warn in report.warnings:
            logger.info("[SUBMIT] warning: %s", warn)
        exit_code = 0 if report.ok else 1
    elif args.command == "validate-run":
        from src.oibc.train.runner import run_validation

        groups = set(FeatureConfig().groups) - set(args.drop_group)
        drop = set(args.drop_var)
        defaults = FeatureConfig()
        features = FeatureConfig(
            groups=frozenset(groups),
            wa_vars=tuple(v for v in defaults.wa_vars if v not in drop),
            wb_vars=tuple(v for v in defaults.wb_vars if v not in drop),
        )
        clip = tuple(args.mlp_clip) if args.mlp_clip else None
        run_validation(
            RunConfig(
                run_id=args.run_id,
                hypothesis=args.hypothesis,
                baseline_run_id=args.baseline,
                composition_id=args.composition,
                features=features,
                model=ModelConfig(
                    weight=args.weight,
                    regional=RegionalConfig(weight=args.regional_weight),
                    families=FamilyConfig(
                        cat_weight=args.cat_weight,
                        mlp_weight=args.mlp_weight,
                        mlp_clip_quantiles=clip,
                    ),
                ),
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
        cal = None
        cal_cfg = None
        cal_record = None
        note = ""
        if args.calibration:
            import json as _json

            from src.oibc.config import ArtifactPaths as _ArtifactPaths
            from src.oibc.config import CalibrationConfig as _CalibrationConfig
            from src.oibc.models.calibrate import CloudCalibration as _CloudCalibration

            cal_path = _ArtifactPaths().runs / args.run_id / "calibration.json"
            try:
                cal_report = _json.loads(cal_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise RunnerError(f"submit: unreadable calibration report ({exc})") from exc
            if not isinstance(cal_report, dict):
                raise RunnerError("submit: invalid calibration report")
            if cal_report.get("run_id") != args.run_id:
                raise RunnerError("submit: calibration run_id mismatch")
            if cal_report.get("panel_sha256") != panel_hash:
                raise RunnerError("submit: calibration panel hash mismatch")
            if cal_report.get("decision") != "adopt":
                raise RunnerError("submit: calibration not adopted")
            prod = cal_report.get("production")
            if not isinstance(prod, dict):
                raise RunnerError("submit: invalid calibration production")
            try:
                cal_a = float(prod["a"])
                cal_b = float(prod["b"])
                cal_n = int(prod["n_fit_hours"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RunnerError(f"submit: invalid calibration production ({exc})") from exc
            if not (math.isfinite(cal_a) and math.isfinite(cal_b)):
                raise RunnerError("submit: non-finite calibration")
            if prod.get("at_boundary"):
                raise RunnerError("submit: calibration at boundary")
            try:
                cal_cfg = _CalibrationConfig.model_validate(cal_report["config"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RunnerError(f"submit: invalid calibration config ({exc})") from exc
            cal = _CloudCalibration(a=cal_a, b=cal_b, at_boundary=False, n_fit_hours=cal_n)
            cal_record = {"a": cal_a, "b": cal_b}
            note = "calibration: production a,b from calibration.json"
        breaks = _check_input_breaks(cfg, spec, set(args.allow_break))
        if breaks:
            note = "; ".join(part for part in (note, *breaks) if part)
        forecast = forecast_eval(cfg, panel_path=panel_path, spec=spec, calibration=cal, calibration_cfg=cal_cfg)
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
            note=note,
            calibration=cal_record,
            spec=spec,
        )
    elif args.command == "splice-submission":
        splice_submission(args.base_sub, args.fallback_sub, args.sub_id)
    elif args.command == "scale-submission":
        months = tuple(args.months) if args.months else None
        scale_submission(args.base_sub, args.scale, args.sub_id, months=months)
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
    elif args.command == "report-calibration":
        import json

        from src.oibc.config import ArtifactPaths as _ArtifactPaths
        from src.oibc.train.runner import calibration_report

        cal_report = calibration_report(args.run_id)
        run_dir = _ArtifactPaths().runs / args.run_id
        out_path = run_dir / "calibration.json"
        run_dir.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(cal_report, ensure_ascii=False, indent=2, allow_nan=False)
        with out_path.open("x", encoding="utf-8") as fh:
            fh.write(payload)
            fh.write("\n")
        logger.info(
            "[EVAL] calibration report run=%s decision=%s path=%s",
            args.run_id, cal_report.get("decision"), out_path,
        )
    elapsed = time.perf_counter() - start
    logger.info("[SYS] pipeline %s done in %.1fs", args.command, elapsed)
    return exit_code


_PROVIDERS: tuple[Literal["wa", "wb"], ...] = ("wa", "wb")


def _check_input_breaks(cfg: RunConfig, spec: CompetitionSpec, allowed: set[str]) -> list[str]:
    """사용 원천 변수의 단절을 탐지해 허용되지 않은 단절은 RunnerError, 허용된 단절은 note 항목으로 반환한다."""
    import pandas as pd

    from src.oibc.data.drift import detect_breaks, used_raw_variables
    from src.oibc.train.runner import RunnerError

    notes: list[str] = []
    used = used_raw_variables(cfg.features)
    for provider in _PROVIDERS:
        variables = used[provider]
        fc = pd.read_parquet(
            DataPaths().canonical_dir / f"forecast_{provider}.parquet",
            columns=[f"{provider}_station", "round", "ts", *variables],
        )
        found = detect_breaks(fc, provider, variables, spec.train_end, DriftConfig())
        for var, first in found.items():
            if first is None:
                continue
            day = first.strftime("%Y-%m-%d")
            logger.info("[DATA] drift var=%s first_break=%s", var, day)
            if var not in allowed:
                raise RunnerError(f"submit: input break {var} from {day}")
            notes.append(f"break:{var}:{day}")
    return notes


def _read_submission(name: str, label: str) -> tuple[SubmissionManifest, pd.DataFrame, pd.DataFrame]:
    """제출 폴더의 manifest·배정·예측(ts는 KST)을 읽는다. 읽기·검증 실패 시 RunnerError."""
    import pandas as pd

    from src.oibc.config import ArtifactPaths
    from src.oibc.submit.build import SubmissionManifest
    from src.oibc.train.runner import RunnerError

    sub_dir = ArtifactPaths().submissions / name
    try:
        manifest = SubmissionManifest.model_validate_json((sub_dir / "manifest.json").read_text(encoding="utf-8"))
        comp = pd.read_csv(sub_dir / "composition.csv", dtype=str)
        fc = pd.read_csv(sub_dir / "forecast.csv", dtype={"group": str})
        fc["ts"] = pd.to_datetime(fc["ts"], format="ISO8601").dt.tz_convert("Asia/Seoul")
    except (OSError, KeyError, ValueError, TypeError) as exc:
        raise RunnerError(f"{label}: unreadable submission {name} ({exc})") from exc
    return manifest, comp, fc


def scale_submission(base_sub: str, scale: float, sub_id: str, *, months: tuple[int, ...] | None = None) -> Path:
    """base 제출 예측에 scale을 곱한 수준 보정 제출을 만든다. months가 있으면 운영일(right-closed) 월이 그 안인 행만 곱한다.

    scale이 (0, 2] 밖이거나 months가 비었거나 1~12 밖·중복이면, 또는 계약 위반이면 RunnerError.
    """
    import numpy as np

    from src.oibc.submit.build import write_submission
    from src.oibc.timeaxis import op_date
    from src.oibc.train.runner import RunnerError

    if not (math.isfinite(scale) and 0.0 < scale <= 2.0):
        raise RunnerError("scale: scale must be finite and in (0, 2]")
    if months is not None and (not months or len(set(months)) != len(months) or any(not 1 <= m <= 12 for m in months)):
        raise RunnerError("scale: months must be non-empty, distinct, and in 1..12")
    base_m, base_comp, base_fc = _read_submission(base_sub, "scale")
    values = base_fc["forecast_kwh"].to_numpy(dtype="float64")
    scaled = base_fc[["group", "ts"]].copy()
    if months is None:
        scaled["forecast_kwh"] = values * scale
        note = f"level-scale base={base_sub} scale={scale!r}"
    else:
        hit = op_date(base_fc["ts"]).dt.month.isin(months).to_numpy()
        scaled["forecast_kwh"] = np.where(hit, values * scale, values)
        note = f"level-scale base={base_sub} scale={scale!r} months={','.join(map(str, months))}"
    if base_m.note:
        note = f"{note}; base_note={base_m.note}"
    logger.info("[SUBMIT] scale base=%s scale=%r rows=%d", base_sub, scale, len(scaled))
    return write_submission(
        base_comp,
        scaled,
        sub_id,
        run_id=base_m.run_id,
        composition_id=base_m.composition_id,
        local_metrics=base_m.local_metrics,
        note=note,
        calibration=base_m.calibration,
    )


def splice_submission(base_sub: str, fallback_sub: str, sub_id: str) -> Path:
    """단절 전은 base, 단절 후는 fallback 예측을 쓴 제출을 만든다. 계약 위반 시 RunnerError."""
    import pandas as pd

    from src.oibc.data.drift import detect_breaks, splice_forecasts, used_raw_variables
    from src.oibc.evaluate.registry import read_records
    from src.oibc.submit.build import write_submission
    from src.oibc.timeaxis import op_date
    from src.oibc.train.runner import RunnerError

    base_m, base_comp, base_fc = _read_submission(base_sub, "splice")
    fb_m, _, fb_fc = _read_submission(fallback_sub, "splice")
    if base_m.composition_sha256 != fb_m.composition_sha256:
        raise RunnerError("splice: composition hash mismatch")
    if base_m.calibration is not None or fb_m.calibration is not None:
        raise RunnerError("splice: calibrated submissions are not spliced")
    records = read_records()
    configs: dict[str, RunConfig] = {}
    specs: dict[str, CompetitionSpec] = {}
    for label, manifest in (("base", base_m), ("fallback", fb_m)):
        match = next((r for r in records if r.run_id == manifest.run_id), None)
        if match is None:
            raise RunnerError(f"splice: unknown run_id for {label}")
        try:
            configs[label] = RunConfig.model_validate(match.config)
            spec_data = dict(match.config["competition"])
            for key in ("train_end", "eval_start", "eval_end"):
                spec_data[key] = pd.Timestamp(spec_data[key]).tz_convert("Asia/Seoul")
            specs[label] = CompetitionSpec.model_validate(spec_data)
        except (KeyError, TypeError, ValueError) as exc:
            raise RunnerError(f"splice: invalid recorded config for {label} ({exc})") from exc
    if specs["base"].train_end != specs["fallback"].train_end:
        raise RunnerError("splice: train_end mismatch")
    spec = specs["fallback"]

    base_used = used_raw_variables(configs["base"].features)
    fb_used = used_raw_variables(configs["fallback"].features)
    diff = [(p, v) for p in base_used for v in base_used[p] if v not in fb_used[p]]
    if not diff:
        raise RunnerError("splice: base and fallback use the same raw variables")

    switches: list[pd.Timestamp] = []
    for provider in _PROVIDERS:
        needed = tuple(dict.fromkeys((*base_used[provider], *fb_used[provider])))
        canon = pd.read_parquet(
            DataPaths().canonical_dir / f"forecast_{provider}.parquet",
            columns=[f"{provider}_station", "round", "ts", *needed],
        )
        found = detect_breaks(canon, provider, needed, spec.train_end, DriftConfig())
        for var in fb_used[provider]:
            first = found.get(var)
            if first is not None:
                raise RunnerError(f"splice: fallback input break {var} from {first.strftime('%Y-%m-%d')}")
        switches.extend(found[v] for p, v in diff if p == provider and found.get(v) is not None)
    if not switches:
        raise RunnerError("splice: no break in base-only variables")
    switch = min(switches)
    spliced = splice_forecasts(base_fc, fb_fc, switch)
    from_fallback = int((op_date(spliced["ts"]) >= switch).sum())
    switch_day = switch.strftime("%Y-%m-%d")
    diff_names = ",".join(v for _, v in diff)
    note = f"splice base={base_sub} fallback={fallback_sub} switch={switch_day} vars={diff_names}"
    logger.info(
        "[SUBMIT] splice base=%s fallback=%s switch=%s rows_from_fallback=%d",
        base_sub, fallback_sub, switch_day, from_fallback,
    )
    return write_submission(
        base_comp,
        spliced,
        sub_id,
        run_id=fb_m.run_id,
        composition_id=fb_m.composition_id,
        local_metrics=fb_m.local_metrics,
        note=note,
        spec=spec,
    )


def ecmwf_correct_submission(
    base_sub: str,
    sub_id: str,
    *,
    cfg: EcmwfConfig | None = None,
    paths: DataPaths | None = None,
    spec: CompetitionSpec | None = None,
) -> Path:
    """base 제출에 ECMWF 2단 보정 1+βx를 곱한 제출을 만든다.

    β는 base 제출 run의 학습 기간 OOF(그룹 합, SMP·k 가중 L1)로만 맞춘다. 계약 위반 시 RunnerError.
    """
    import pandas as pd

    from src.oibc.data.external_ecmwf import (
        correction_multiplier,
        ec_blocks,
        fit_beta,
        fit_frame,
        gfs_blocks,
        x_table,
    )
    from src.oibc.data.external_gfs import ExternalError, hourly_path
    from src.oibc.submit.build import write_submission
    from src.oibc.train.runner import RunnerError, load_oof

    ec_cfg = cfg if cfg is not None else EcmwfConfig()
    data_paths = paths if paths is not None else DataPaths()
    comp_spec = spec if spec is not None else CompetitionSpec()
    base_m, base_comp, base_fc = _read_submission(base_sub, "ecmwf")
    if base_m.run_id is None:
        raise RunnerError("ecmwf: base submission has no run_id")
    group_of = base_comp.set_index("site_id")["group"]
    parts = []
    for fold in ec_cfg.fit_folds:
        oof = load_oof(base_m.run_id, fold)
        grp = oof["site_id"].astype(str).map(group_of)
        if grp.isna().any():
            raise RunnerError(f"ecmwf: OOF site missing from composition (fold={fold})")
        parts.append(oof.assign(group=grp).groupby(["group", "ts"])[["actual_kwh", "pred_bag"]].sum().reset_index())
    group_oof = pd.concat(parts, ignore_index=True)
    if (group_oof["ts"] > comp_spec.train_end).any():
        raise RunnerError("ecmwf: OOF beyond train_end")
    smp = pd.read_parquet(data_paths.canonical_dir / "smp.parquet", columns=["ts", "smp", "k"])
    weights = pd.Series((smp["smp"].astype("float64") * smp["k"].astype("float64")).to_numpy(), index=smp["ts"])
    try:
        xt = x_table(
            ec_blocks(data_paths, ec_cfg, comp_spec),
            gfs_blocks(pd.read_parquet(hourly_path(data_paths)), ec_cfg),
            ec_cfg,
        )
        h = fit_frame(group_oof, weights, xt, ec_cfg)
        beta = fit_beta(
            h["pred_bag"].to_numpy(), h["actual_kwh"].to_numpy(), h["w"].to_numpy(), h["x"].to_numpy(), ec_cfg
        )
        mult = correction_multiplier(base_fc["ts"], xt, beta, ec_cfg)
    except ExternalError as exc:
        raise RunnerError(f"ecmwf: {exc}") from exc
    out = base_fc[["group", "ts"]].copy()
    out["forecast_kwh"] = base_fc["forecast_kwh"].to_numpy(dtype="float64") * mult
    src_counts = xt["src"].value_counts().to_dict()
    note = f"ecmwf-correct base={base_sub} beta={beta!r} fit_rows={len(h)} x_src={src_counts}"
    if base_m.note:
        note = f"{note}; base_note={base_m.note}"
    logger.info(
        "[SUBMIT] ecmwf-correct base=%s beta=%r fit_rows=%d corrected=%d",
        base_sub,
        beta,
        len(h),
        int((mult != 1.0).sum()),
    )
    return write_submission(
        base_comp,
        out,
        sub_id,
        run_id=base_m.run_id,
        composition_id=base_m.composition_id,
        local_metrics=base_m.local_metrics,
        note=note,
        calibration=base_m.calibration,
    )


if __name__ == "__main__":
    raise SystemExit(main())
