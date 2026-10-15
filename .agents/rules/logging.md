# Unified Logging & Diagnostic Directives

> **Logs must be concise, structured, and machine-parsable without sacrificing diagnostic correctness. Never truncate required numerical precision or suppress exception tracebacks to save tokens. Never log secrets or credentials, and never delete diagnostic evidence unilaterally.**

## 1. Information Integrity & Diagnostic Value
- **Correctness Over Token Economy:** Token thrift must never compromise the ability to diagnose bugs or reconstruct events. Preserve full exception tracebacks, root causes, and error payloads for `ERROR` and `CRITICAL` levels.
- **Precision Preservation:** Retain sufficient precision for financial and numerical quantities; do not round small non-zero quantities (such as fee ratios, funding rates, optimizer tolerances, or portfolio weights) into meaningless zeros.
- **Correlation Identifiers:** Include stable correlation identifiers (such as session, execution, order, symbol, or stage identifiers) when concurrent or multi-stage pipelines make event attribution ambiguous.
- **Strict Credential Redaction:** NEVER log broker/exchange API credentials, private secret keys, app keys, passphrase tokens, certificates, webhook secrets, or sensitive account/wallet identifiers.

## 2. Standard Category Taxonomy & Structured Output
- **Fixed Domain Category Taxonomy:** Classify all operational log events using one of six fixed category tags:
  - `[SYS]`: Infrastructure lifecycle, configuration, external service connections, and host runtime resources.
  - `[DATA]`: Market data ingestion, collection plans, normalization, validation, and panel construction.
  - `[ALGO]`: Alpha signals, mathematical models, feature computation, and scoring pipelines.
  - `[PORTFOLIO]`: Asset allocation, rebalancing logic, NAV reconciliation, and target weights.
  - `[RISK]`: Exposure boundaries, leverage limits, volatility stops, and order throttling/kill-switches.
  - `[EXEC]`: Broker API communication, order intents, order submission, fills, and slippage tracking.
- **Consistent Log Format:** Operational pipeline transitions and lifecycle log messages must include the category prefix: `logger.info("[<CATEGORY>] <event description>")` (as established in `src/data/operations.py`). Existing legacy log statements without category tags do not need to be refactored unless their immediate scope is being modified.
- **Structured Fields:** Prefer structured key-value pairs or machine-readable records (such as JSONL or standardized delimited fields) over long conversational sentences.
- **Large Collection Summaries:** Never dump massive arrays, DataFrames, or raw market/order book depth directly into logs. Summarize collections using concise descriptors: count, shape, min/max range, null count, or representative head/tail samples.

## 3. Separation of Concerns & High-Frequency Output
- **Operational vs. High-Frequency Logs:** Keep high-frequency streaming data (such as WebSocket tick feeds, order book depth, execution stream dumps, factor evaluations, or optimizer trial traces) isolated in dedicated diagnostic destinations rather than spamming primary system logs.
- **INFO vs. DEBUG/TRACE:**
  - **`INFO`**: High-level, human-readable phase transitions and milestone summaries (typically 1 line per phase).
  - **`DEBUG` / `TRACE`**: Fine-grained, structured diagnostic events targeted for troubleshooting and programmatic inspection.

## 4. Lifecycle & Path Hygiene
- **Configured Storage Paths:** Persistent service logs must reside in the designated project log path. Temporary diagnostic logs should use distinct scratch locations.
- **Non-Destructive Cleanup:** Temporary diagnostic logs may be cleared via project-designated maintenance scripts or explicit lifecycle commands. The AI must never unilaterally delete diagnostic logs or historical evidence needed for unresolved investigations.


