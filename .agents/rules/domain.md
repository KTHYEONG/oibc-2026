# OIBC Competition Domain Directives

> **Maximize out-of-sample settlement revenue `S` (sales revenue − forecast-error penalty `W`) on unseen evaluation data under strict competition rules. Correctness, causality, and reproducibility precede model sophistication.**

## 1. Objective & Settlement Fidelity
- **Settlement Is the Metric:** Evaluate every forecasting and composition decision by `S` and `W`. MAE/RMSE are diagnostics only.
- **Verified Formula:** `S = Σ_t SMP(t)·A(t) − 0.10·Σ_t SMP(t)·k(t)·|F(t) − A(t)|`, where `A` is group actual generation (kWh), `F` the group forecast, `k(t)=3` if the `ts` hour ∈ {11..15} else 1 (as labeled in `smp.csv`). Revenue is a constant under fixed assignment given `A`, so forecasting reduces to minimizing the `SMP·k`-weighted absolute error of the *group sum*; the optimal point forecast is the median of the group-sum distribution (symmetric L1), not the mean. Tie-breaks and the canonical submission contract remain `UNVERIFIED` until the problem definition is available.
- **Group-Level Netting:** The penalty is computed per group then summed. Over/under-forecast errors cancel only among plants *within* a group, never across groups; this is the sole lever of composition optimization.
- **Night & Zero Hours:** Zero-generation hours are still settled; any nonzero forecast there is penalized. Force exact zeros when generation is physically zero and keep forecasts ≥ 0.
- **Weights Matter:** Errors at SMP=0 hours cost nothing; errors in 11–15h weigh 3×. Evaluate and train with the same weights.
- **No Magic Numbers:** Competition constants (20,000 kW group minimum, 17:00 cutoff, 253 plants, 0.10 penalty rate, k hours/values) live in typed `Spec`/`Config` contracts with a source reference, never as literals in business logic.

## 2. Temporal Causality & Information Set
- **Cutoff Rule:** A prediction for target date `D` may use only data available by 17:00 KST on `D-1`, including the 10:00 and 17:00 forecast issuance rounds of `D-1`.
- **Right-Closed Hours:** Timestamp `HH:00` denotes the interval ending at `HH:00`. Misaligning by one hour between generation, SMP, and weather is a fatal defect; verify alignment explicitly.
- **Forbidden Inputs:** Observed weather and actual generation from the evaluation period, and any information issued after the cutoff.
- **Perturbation Invariance:** Corrupting data after the cutoff must leave predictions for date `D` unchanged.

## 3. Validation & Leakage Control
- **Chronological Splits:** Folds simulate the real cutoff and are time-ordered over multiple expanding-window folds (`ValidationConfig.folds`). The last 12 training months were already used for model comparison and are no longer an untouched holdout; the evaluation-period leaderboard is the only untouched test. Never tune on leaderboard feedback.
- **Seed Replication:** compare configurations by the mean over `ValidationConfig.seeds`; training noise is the pooled seed-to-seed relative SD across folds and both sides, and a gain must exceed `noise_z`× its standard error with the combined (bootstrap + training) CI excluding zero. Never use the seed range as the noise threshold, and never compare runs from different code versions (commit, dirty tree, panel hash).
- **Paired Uncertainty:** report per-fold relative ΔW with a day-block bootstrap CI and per-quarter signs; report warm-site and cold-start subsets separately.
- **Leakage Surfaces:** Lag/rolling features, target encodings, scaler/imputer fits, plant-level statistics, and residuals used for VPP assignment must be fit on training data only.
- **Observation vs. Forecast Weather:** Weather observations exist only for training (raw, with sentinel outliers such as `wa_wind_speed`=9999, `wb_cloud`>100). Models consumed at inference must be trained on forecast-weather inputs (or demonstrably robust to the shift); never train on observed weather and infer on forecasts without validating that gap.
- **Statistical Humility:** Account for trial multiplicity across experiments and leaderboard submissions. Prefer performance plateaus over isolated parameter spikes, and report results across multiple periods and conditions.

## 4. VPP Composition Integrity
- **Hard Constraints:** Exactly two groups; all 253 plants assigned exactly once; each group ≥ 20,000 kW installed capacity; one fixed assignment for the entire evaluation period. Validate before any submission and fail closed on violation.
- **Joint Objective:** Evaluate both VPPs together on `S`; do not optimize one VPP in isolation.
- **Out-of-Fold Residuals:** Assignment search (local search, annealing, etc.) may use only training or out-of-fold residuals. Residuals fit on data the model has memorized yield phantom error cancellation.
- **Baselines Required:** Compare any optimized assignment against random and capacity-balanced assignments under identical validation.

## 5. Forecasting Discipline
- **Baselines First:** Seasonal-naive / capacity-scaled baselines precede GBMs; complexity must beat the baseline on `S`.
- **Aggregation Choice Is an Experiment:** Plant-level forecast-then-aggregate versus direct VPP-level forecasting must be compared under the same folds.
- **Missing Forecasts:** 264 provider-A test rows are absent; left-joins yield NaN features. Impute causally with a documented method (or use a NaN-tolerant model) and keep every plant-hour. Train/test forecast availability patterns can differ (provider-B precipitation columns have 1,298 NaN rows each in train, none in test).
- **Non-Gaussian Errors:** Solar error is heteroskedastic and heavy-tailed (clouds, sunrise/sunset edges, night zeros). Check penalty behavior in tails, not just average error.

## 6. Reproducibility & Fail-Closed
- **Deterministic Pipelines:** Fixed seeds, versioned configs, and scripted preprocessing → training → optimization → inference. Raw data is never modified.
- **Fail-Closed:** On schema drift, missing columns, timestamp gaps, or submission-format violations, abort with a clear error rather than substituting defaults.
- **Conservation:** Aggregated VPP generation must equal the sum of member plants within numerical tolerance; assignments must partition the plant set.
- **Anonymity & Data Use:** Never attempt coordinate recovery or re-identification; competition data is used only for this competition.

## 7. Data Facts (Verified from Provided Files)
- **Layout:** raw archive extracted to `data/raw/01_participant_dataset/` (gitignored, read-only). Train: 2021-01-01 → 2025-06-01 00:00 KST; test/evaluation: 2025-06-01 01:00 → 2026-06-01 00:00 KST (8,760 hours).
- **Timestamps:** all `ts` are tz-aware `Asia/Seoul`, right-closed; generation is `datetime64[ms]`, others `[us]`. Normalize before joins.
- **Generation:** long format, no NaN; absent `(site_id, ts)` rows are missing observations (per-plant history length differs; starts 2021-01-19). Contains 40 negative `energy` values and values up to ~1.03× capacity; no duplicate keys.
- **Forecasts:** keyed `(station, round, ts)` with `round ∈ {10, 17}`, both before the cutoff. Providers differ in units (wind: A m/s, B km/h). Several plants share a station.
- **Reference Groups:** `current_group` is a random initial assignment, not an optimized one; do not treat it as a performance reference for composition.
- **Plant Scale:** Capacities range 14–1,388 kW (median ~99 kW; the briefing's "100–900 kW" is approximate); each group needs many plants to reach 20,000 kW.
- **Baseline:** `baseline.py` (organizer-provided) trains LightGBM on per-plant capacity factor from forecasts + solar geometry, averages rounds 10/17, and uses the reference assignment. Treat it as the score floor to beat.
