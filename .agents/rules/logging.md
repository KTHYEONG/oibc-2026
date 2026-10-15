# Unified Logging & Diagnostic Directives

> **Logs must be concise, structured, and machine-parsable without sacrificing diagnostic correctness. Never truncate required numerical precision or suppress exception tracebacks to save tokens. Never log secrets or credentials, and never delete diagnostic evidence unilaterally.**

## 1. Information Integrity & Diagnostic Value
- **Correctness Over Token Economy:** Token thrift must never compromise the ability to diagnose bugs or reconstruct events. Preserve full exception tracebacks, root causes, and error payloads for `ERROR` and `CRITICAL` levels.
- **Precision Preservation:** Retain sufficient precision for monetary and numerical quantities; do not round small non-zero quantities (such as penalty rates, settlement amounts, optimizer tolerances, or model weights) into meaningless zeros.
- **Correlation Identifiers:** Include stable correlation identifiers (such as experiment, fold, target date, VPP, plant, or stage identifiers) when concurrent or multi-stage pipelines make event attribution ambiguous.
- **Strict Credential Redaction:** NEVER log API keys, tokens, secrets, or credentials. Do not write raw competition data rows or per-plant series into logs; log shapes, counts, and aggregate statistics. Do not log information that could aid plant re-identification (e.g. attempted coordinate matching).

## 2. Standard Category Taxonomy & Structured Output
- **Fixed Domain Category Taxonomy:** Classify all operational log events using one of six fixed category tags:
  - `[SYS]`: Environment, configuration, seeds, and host runtime resources.
  - `[DATA]`: Dataset loading, schema/timestamp validation, imputation, and panel construction.
  - `[FCST]`: Feature engineering, model training, inference, and forecast-error diagnostics.
  - `[VPP]`: VPP composition search, assignment constraints, and residual-correlation analysis.
  - `[EVAL]`: Settlement `S`/`W` computation, validation folds, holdout results, and experiment comparison.
  - `[SUBMIT]`: Submission file generation, local validation, and submission records.
- **Consistent Log Format:** Operational pipeline transitions and lifecycle log messages must include the category prefix: `logger.info("[<CATEGORY>] <event description>")`. Existing legacy log statements without category tags do not need to be refactored unless their immediate scope is being modified.
- **Structured Fields:** Prefer structured key-value pairs or machine-readable records (such as JSONL or standardized delimited fields) over long conversational sentences.
- **Large Collection Summaries:** Never dump massive arrays, DataFrames, or raw plant-level panels directly into logs. Summarize collections using concise descriptors: count, shape, min/max range, null count, or representative head/tail samples.

## 3. Separation of Concerns & High-Frequency Output
- **Operational vs. High-Frequency Logs:** Keep high-volume diagnostics (such as per-plant per-hour predictions, optimizer trial traces, or per-fold feature importances) isolated in dedicated diagnostic destinations rather than spamming primary system logs.
- **INFO vs. DEBUG/TRACE:**
  - **`INFO`**: High-level, human-readable phase transitions and milestone summaries (typically 1 line per phase).
  - **`DEBUG` / `TRACE`**: Fine-grained, structured diagnostic events targeted for troubleshooting and programmatic inspection.

## 4. Lifecycle & Path Hygiene
- **Configured Storage Paths:** Persistent service logs must reside in the designated project log path. Temporary diagnostic logs should use distinct scratch locations.
- **Non-Destructive Cleanup:** Temporary diagnostic logs may be cleared via project-designated maintenance scripts or explicit lifecycle commands. The AI must never unilaterally delete diagnostic logs or historical evidence needed for unresolved investigations.


