# Documentation & Code Commenting Directives

> **Explain the non-obvious "Why" behind system contracts and architectural constraints, never the mechanical "What" visible in code. Keep code self-documenting, comments minimal, and eliminate narrative step-by-step bloat.**

## 1. Commenting Principles: The "Why" Standard
- **Self-Documenting Code First:** Code structure, expressive identifier names, and strong types must communicate what the code does. The default expectation is near-zero inline comments.
- **Rationale-Only Standard:** Write comments exclusively for non-obvious design trade-offs and external constraints that cannot be expressed in code. Omit all mechanical paraphrasing, procedural steps, and session metadata.

## 2. Interface Documentation Standards
- **Caller Contracts, Not Spec Echoes:** Docstrings are compact (1–2 lines) caller contracts (inputs, returns, raised errors). Never paste specification narratives, task metadata, or implementation steps into docstrings or comments.
- **Contract-Driven Documentation:** Document public interfaces when behavior, side effects, units, timing assumptions, or failure conditions are not self-evident from type signatures and names.
- **Consistent Semantic Convention:** Follow established semantic conventions documenting input contracts, return guarantees, side effects, and thrown errors. Avoid conversational prose or historical journaling.
- **Private Helpers:** Omit interface documentation for private helper functions unless the algorithmic logic is complex or non-trivial.

## 3. Architecture & Domain Documentation (`docs/architecture/`)
- **Stable Boundaries & Semantics:** Keep architecture documents focused on stable system boundaries, data flows, formal models, and domain invariants. Avoid cluttering with transient implementation details.
- **Coherent In-Place Updates:** Update existing architecture documents coherently rather than accumulating disconnected append-only notes.
- **Contract & Code Reconciliation:** Treat discrepancies between code contracts and architecture documentation as issues to investigate and resolve; never silently assume either side is automatically correct.

## 4. Repository Language Conventions
- **Korean for Humans:** Documentation (`docs/architecture/`, `docs/specs/`, READMEs), docstrings, and in-line comments are written in concise natural Korean (한국어), because the owner reads and reviews them in Korean.
- **English for Machines:** Identifiers, code, file paths, column names, CLI flags, and commit-independent technical terms stay in English; quote them verbatim inside Korean prose.
- **Agent Instruction Files:** `AGENTS.md`, `.agents/rules/`, and `.agents/skills/` remain in English for deterministic agent behavior.
