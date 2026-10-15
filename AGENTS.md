# Operating Manual: OIBC-2026

> **Project Identity:** 2026 POSTECH OIBC competition — (1) partition 253 solar plants into exactly two VPPs, (2) forecast each VPP's hourly day-ahead generation. Goal: first place by maximizing settlement revenue `S` = sales revenue − forecast-error penalty `W`.

## 1. Competition Ground Truth (Non-Negotiable)
Every rule below is a verified competition requirement. If a task conflicts with one, stop and report instead of working around it. Anything not listed here is an **assumption** and must be labeled as such; never invent undocumented rules or formulas.

- **Assignment:** all 253 plants in `sites.csv` assigned, each to exactly one of two groups `G1`/`G2`; none omitted or duplicated. One assignment is held fixed for the whole evaluation period. **Each group's installed capacity must be ≥ 20,000 kW** (total 42,655.4 kW, so slack is only ~2,655 kW). Both groups are optimized jointly. `current_group` (A/B) in `sites.csv` is a reference assignment only.
- **Prediction cutoff:** 17:00 KST on the day before the target date. Use nothing unavailable at that cutoff. Day-ahead weather forecasts from both the 10:00 and 17:00 issuance rounds are permitted.
- **Time convention:** all timestamps KST, right-closed hourly intervals (`01:00` = `00:00–01:00`).
- **Data:** competition-provided datasets only. No external data, weather APIs, supplementary geography, or externally pretrained models. Evaluation-period observations and actual generation are unavailable and must never be used.
- **Data egress ban (disqualification risk):** competition data must never leave the local machine/own server: no uploads to external generative-AI services or APIs, external storage, public repos, Artifact/Docs/MCP publishing, or web tools (WebFetch/WebSearch queries). AI assistance on code and analysis is allowed, and agents may read local data through CLI tools as needed for development. Keep exposure proportionate: prefer schemas and aggregates, use small samples only when debugging requires them, and avoid bulk dumps of raw rows or full series into the conversation. Data on own servers must be deleted within 30 days after the competition ends. Real generation for the evaluation period must not be obtained externally. The user owns the interpretation of the AI-usage rules; do not refuse or stall data work over it.
- **Anonymity:** plant coordinates are displaced up to 2 km. Never attempt coordinate recovery or plant re-identification.
- **Missing forecasts:** 264 provider-A test forecast rows are *absent* (not NaN; provider B has none). Joins must not drop the affected plant-hours; every scoring hour needs a forecast. Handle them with explicit, documented imputation.
- **Use restriction:** competition data is for this competition only.
- **Submissions:** 3 per day, 20 total, Nov 2–13 2026. Two files: `composition.csv` (`site_id,group`; 254 lines incl. header) and `forecast.csv` (`group,ts,forecast_kwh`; 17,521 lines incl. header; `ts` timezone-aware KST; `forecast_kwh` ≥ 0; both groups × all 8,760 hours of `smp.csv`). The leaderboard keeps each team's best score. Forecasts must be finite non-negative reals (no NaN/Inf) with exact timestamps and timezones. Validate locally before every submission and record config, file, and leaderboard score.
- **Finalists (top 7):** source code, presentation, and a reproducible preprocessing → training → inference pipeline by Nov 23 2026. The final award is decided by the online presentation, not leaderboard rank alone, so keep a log of experiments, rejected ideas, and strategy rationale from day one.

## 2. Evaluation & Validation Invariants
- **Objective first:** decisions are judged by `S` (and `W`), not MAE/RMSE. Official formula (confirmed by the participant README and the online briefing): `S = Σ SMP(t)·actual(t) − 0.10·Σ SMP(t)·k(t)·|forecast(t) − actual(t)|`, summed over both groups and all 8,760 hours; `k(t)=3` when the `ts` hour is 11–15, else 1. Tie-breaking, the canonical column/row-count contract, and full invalid-submission conditions are defined only in the problem definition (`OIBC2026-문제정의서.md`, not in the repo) and `scoring.py` (not provided): treat them as `UNVERIFIED` and build the validator from the README and briefing checks meanwhile.
- **Cutoff-faithful validation:** chronological validation that reproduces the 17:00 D-1 information set, over multiple expanding-window folds (`ValidationConfig.folds`). The last 12 training months were already used for model comparison and are no longer an untouched holdout; the evaluation-period leaderboard is the only untouched test. Decisions follow the adoption rule (all folds improve, pooled improvement beyond seed noise, quarter consistency); never tune on leaderboard feedback.
- **Zero leakage:** features, joins, normalization, imputation, residual-based VPP assignment, and folds must use only data observable at the cutoff. VPP assignment may use only training/out-of-fold residuals.
- **Multiple testing:** leaderboard and validation feedback are finite. Do not optimize for the public leaderboard alone, and do not present repeated trial-and-error gains as independent discoveries.
- **Fail-closed:** on schema, timestamp-alignment, or assignment violations, abort with an error. Never silently substitute defaults.
- **No fabricated results:** never claim a score or improvement without a reproducible experiment backing it.

## 3. Autonomy & Execution Contract
- **Bias toward action:** for data queries, scratch diagnostics, and root-cause isolation, execute immediately. Run experiments under `scratch/`. Never modify production code or commit in response to open-ended diagnostic questions.
- **Inspect before building:** before significant changes, read the relevant code, actual data schemas/units/timestamps/missing values, and existing experiments.
- **Smallest measurable step:** simple baselines before complex models; every experiment has a hypothesis and a measurable outcome under identical validation conditions. Preserve working baselines.
- **Skills are independent tools** (`probe`, `spec`, `implement`, `check`, `refactor`, `commit`), not a pipeline. When a slash command is invoked, execute only that skill and halt for review. Specs live under `docs/specs/` (gitignored).
- **Workspace hygiene:** temporary files and exploration stay in `scratch/`. Never commit raw data, model checkpoints, caches, or large generated files; never overwrite original data.

## 4. Toolchain & Verification
- **Quality gate:** `uv run python tools/verify.py`
- **Tests:** `uv run pytest`
- **Commits:** run the project's `commit` skill.
- Required tests cover: settlement formula, timestamp alignment / right-closed convention, cutoff leakage prevention, VPP assignment validity (253 plants, exactly-once, capacity), and submission schema.
- Record config, random seeds, validation periods, model parameters, and results for every meaningful experiment.

## 5. Communication & Language
- **Korean for conversation:** explain rationales and report findings in natural Korean (한국어). In structured output cards, keep English keys/badges with Korean descriptions.
- **Korean for documents and comments:** docs (`docs/architecture/`, `docs/specs/`), docstrings, and in-line comments are written in Korean; identifiers, code, paths, and column names stay in English.
- **English for agent instructions:** `AGENTS.md`, `.agents/rules/`, and `.agents/skills/` are written in technical English.
- **Report honestly:** distinguish verified requirements, assumptions, and proposals; state test results and unresolved limitations accurately.

## 6. Rule Routing
- **Domain invariants:** [domain.md](.agents/rules/domain.md) — *Settlement objective, temporal causality, VPP composition, forecasting validation, and reproducibility.*
- **Testing & quality:** [testing.md](.agents/rules/testing.md) — *Invariant-driven testing, boundary conditions, failure isolation, and diff-coverage.*
- **Architecture & standards:** [code-style.md](.agents/rules/code-style.md) — *Module boundaries, typed contracts, and toolchain alignment.*
- **Documentation & comments:** [documentation.md](.agents/rules/documentation.md) — *Docstrings, specs, and non-obvious rationale.*
- **Performance:** [performance.md](.agents/rules/performance.md) — *Vectorized panel builds, optimizer hot loops, and resource budgets.*
- **Logging:** [logging.md](.agents/rules/logging.md) — *Structured logging, category taxonomy, and data-use hygiene.*
