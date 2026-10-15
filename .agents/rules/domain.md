# Quantitative Engineering Core Directives

> **The primary directive is engineering rigorous, reproducible quantitative systems that guarantee financial correctness, temporal causality, and execution realism without phantom alpha. For alpha strategy and portfolio optimization layers, maximize net risk-adjusted compounding growth ($g = \mathbb{E}[\ln(1 + r_{\text{net}})]$) within five non-negotiable constitutional pillars.**

## 1. Temporal Causality (The Arrow of Time)
- **Point-in-Time Availability:** Every signal, feature, universe selection, and portfolio decision at time $T$ must consume strictly data observable prior to or at time $T$.
- **Zero Lookahead:** Executing on bar close $T$ using signals derived from the same bar's close, or referencing unreleased future data, is a fatal causality violation.
- **Perturbation Invariance:** Verification must prove that corrupting or randomizing future data ($t > T$) alters historical decisions at or before time $T$ by zero.

## 2. Universe & Selection Integrity (Point-in-Time Reality)
- **Historical Universe Reconstruction:** Never evaluate strategies on historical windows using present-day survival criteria. The investable universe at time $T$ must reflect real-world constituents at time $T$, including subsequently delisted, bankrupt, or suspended assets.
- **Selection Bias Elimination:** Screening filters and dynamic universe logic must only ingest information certified as known at rebalancing time $T$.

## 3. Execution Friction & Capacity Realism
- **Net Friction Integrity:** Never evaluate strategies under zero-friction illusions. All performance metrics must account for realistic execution drag: commissions, statutory taxes, bid-ask spreads, slippage, and financing/borrow costs.
- **Market Impact & Capacity Constraints:** Never assume infinite liquidity or instant fills. Position sizing must respect market depth and Average Daily Volume (ADV) participation limits to prevent illusory alpha that collapses under scale.
- **Zero Magic Numbers:** Never hardcode strategy thresholds, fee schedules, or filter cutoffs as numeric literals in business logic. Parameters must be declared as typed, configurable contracts (`Spec`/`Config`).

## 4. Multiple Testing & Overfitting Resistance
- **Statistical Humility:** Never present backtest metrics from repeated trial-and-error as independent discoveries without explicit statistical haircuts or adjustments for trial multiplicity.
- **Parameter Surface Robustness:** Never rely on isolated parameter spikes ("knife-edge alpha"). Parameter sensitivity must demonstrate stable performance plateaus across neighboring regimes.
- **Strict Out-of-Sample Isolation:** Maintain strict temporal separation between discovery/training and evaluation. Never leak validation statistics back into model formulation.

## 5. Tail-Risk & Ruin Prevention (Deterministic Fail-Closed)
- **Conservation Law:** Accounting ledgers (cash, asset units, fees, taxes) must reconcile exactly with zero leakage when using exact discrete representations (e.g. integer base units such as KRW/cents/satoshis, or `Decimal`). For continuous valuations, mark-to-market calculations, and floating-point metrics, verify conservation within strict numerical tolerances (`rtol`/`atol`, `pytest.approx`) rather than artificial exact-equality tests.
- **Deterministic Fail-Closed:** On unrecoverable data anomalies or feed disruptions, never substitute arbitrary normal defaults. Safely abort execution (`NO_TRADE`), preserve capital, and protect against catastrophic ruin.
- **Non-Gaussian Survival:** Never evaluate risk assuming pure Gaussian returns. Systems must survive fat-tailed drawdowns, liquidity freezes, and regime shifts.
