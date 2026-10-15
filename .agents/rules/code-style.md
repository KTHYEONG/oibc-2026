# Code Style & Engineering Standards

> **Respect existing architecture and conventions. Maintain strong typed contracts at public boundaries without breaking scope. Reconcile code with specifications objectively, and keep every pipeline stage reproducible.**

## 1. Architectural Integrity & Modularity
- **Layout & Scope Hygiene:** Respect the repository's existing source/test layout and architectural layer boundaries. Maintain high cohesion and avoid out-of-scope sweeping refactorings.
- **Contract & Spec Reconciliation:** Treat validated executable code and contracts as authoritative for current implementation, but actively investigate discrepancies with specifications or documentation rather than assuming either side is automatically correct.

## 2. Typing, Contracts & Configuration
- **Strong Typing at Boundaries:** Enforce strong static typing for public interfaces, domain models, and core production logic. Never weaken existing type guarantees without explicit justification.
- **Explicit Configuration:** Keep runtime settings, competition constants, random seeds, and experiment parameters in validated typed configs; never hardcode environment-dependent paths or secrets.
- **Stage Data Contracts:** Each pipeline stage (preprocess, features, train, assign, evaluate, submit) exchanges data with explicit schemas (column names, dtypes, KST timestamps, units) and validates them at the boundary.
- **Toolchain Alignment:** Use the repository's declared environment and quality tooling (linters, type-checkers, and test runners); do not introduce competing toolchains unnecessarily.

## 3. Language & Extension Boundaries
- **Python Primary:** Python is the primary language. Any additional language or subsystem must meet the same baselines (static checks, linting, automated tests) and define explicit data contracts at the boundary.
- **Reproducible Entry Points:** Preprocessing, training, optimization, and inference run from scripted entry points with fixed seeds and typed configs, so finalist reproduction needs no manual steps.
