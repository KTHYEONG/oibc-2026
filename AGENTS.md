# Operating Manual: OIBC-2026

> **Domain Identity:** POSTECH OIBC big data competition & quantitative optimization platform.

## 1. Domain Ground Truth & Hard Boundaries
- **Strict Temporal Causality:** Zero look-ahead bias. Decision-making state, feature engineering, and execution timing must be strictly causal. Never evaluate models or optimization decisions on unobservable future data.
- **Data Leakage & Embargo Integrity:** Model training, validation, and feature pipelines must enforce strict temporal separation and embargo periods to prevent label and feature leakage.
- **Evaluation Metric & Constraint Realism:** All optimization and scoring metrics must strictly respect competition constraints (e.g. physical resource boundaries, grid capacity, penalty schedules).
- **Deterministic Fail-Closed & Conservation:** Numerical results, accounting, and resource allocations must reconcile consistently. On missing or corrupted inputs, safely fail-closed rather than guessing phantom values.
- **Workspace Hygiene:** Keep exploratory experiments and temporary files strictly isolated under `scratch/`. Never commit raw cache or scratch files.

## 2. Autonomy & Execution Contract
- **Bias Toward Action & Diagnostic Autonomy:** For data queries, exploratory scratch diagnostics, and empirical root-cause isolation, execute immediately without asking for permission. When asked open-ended questions about bugs or data anomalies, proactively run scratch experiments under `scratch/` to discover truth. Never modify production code or commit in response to open-ended diagnostic queries.
- **Skills as On-Demand Tools:** Skills (`probe`, `spec`, `implement`, `check`, `refactor`, `commit`) are modular, independent utilities—NOT a mandatory sequential pipeline. When explicitly invoked via slash commands (`/probe`, `/spec`, etc.), execute only that targeted skill and halt for user review. Specs live under `docs/specs/` (gitignored for model/tool handoffs without repo bloat).

## 3. Project Toolchain & Verification
Verify code changes against the project's native toolchains before concluding tasks:
- **Quality Gate:** `uv run python tools/verify.py`
- **Test Runner:** `uv run pytest`
- **Git Commits:** Run the project's `commit` skill.

## 4. Communication & Language
- **Natural Korean:** Converse, explain rationales, and report findings in Korean (한국어). Inside structured output cards, retain English keys/badges while writing descriptions in Korean.
- **Technical English:** System instructions, rules, specifications (`docs/specs/`), code, and docstrings are written in English.

## 5. Domain Rule Routing
- **Domain Invariants:** [domain.md](.agents/rules/domain.md) — *Financial invariants, temporal causality, market frictions, and conservation laws.*
- **Testing & Quality:** [testing.md](.agents/rules/testing.md) — *Invariant-driven testing, boundary conditions, failure isolation, and diff-coverage.*
- **Architecture & Standards:** [code-style.md](.agents/rules/code-style.md) — *Module boundaries, strong static typing contracts, and toolchain alignment.*
- **Documentation & Comments:** [documentation.md](.agents/rules/documentation.md) — *Production docstrings, architecture specs, and non-obvious rationale.*
- **Performance & Optimization:** [performance.md](.agents/rules/performance.md) — *Vectorized panel builds, hot-loop profiling, and resource budgets.*
- **Logging & Diagnostics:** [logging.md](.agents/rules/logging.md) — *Operational logging, 6 fixed category taxonomy, and credential redaction.*
