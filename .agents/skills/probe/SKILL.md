---
name: probe
description: High-reasoning diagnostic protocol to analyze root causes, test hypotheses empirically, and establish invariants before specification.
---

# Probe Protocol

High-reasoning diagnostic protocol dedicated to **causal discovery, empirical validation, and failure-mode analysis** before specification.

## Cognitive Focus

Your cognitive budget is focused 100% on **Problem Space**:
*What is the fundamental causality behind this state? Under what realistic boundary conditions does it fail? What deterministic invariants must be established?*

## Directives

1. **Diagnostic Rigor & Efficient Pathing**:
   - For straightforward logical bugs or direct code-path flaws, isolate the root cause via static inspection without creating unnecessary scratch files.
   - For numerical discrepancies, subtle data pipeline distortions, or multi-step anomalies, do NOT guess. Write and run a lightweight script under `scratch/` to measure real values and reproduce failures deterministically.

2. **Concept Blueprinting (No Production Sprawl)**:
   - Do NOT generate full production patches or extensive files.
   - You may include minimal 1-3 line conceptual snippets if they demonstrate the resolution mechanism clearer than prose. Leave full implementation contracts for `/spec`.

3. **Invariants & Boundary Conditions**:
   - Formulate unambiguous Fail-Closed invariants (what state transitions must be strictly forbidden).
   - Evaluate trade-offs between competing architectural fixes.

4. **Handoff**:
   - Conclude by handing off proven invariants and root causes to `/spec` (or directly to implementation for minor fixes).

## Chat Output Format

Keep chat response extremely readable, intuitive, and decision-oriented. Never overwhelm the user with walls of text. Deliver findings in natural, clear Korean with intuitive everyday analogies.

```markdown
### 🔬 [PROBE] <분석 주제>

> 💡 **한눈에 보기**: <3초 만에 누구나 직관적으로 이해할 수 있는 일상적 비유와 핵심 요약>

#### 📌 핵심 결론 & 다음 액션
<!-- 상황에 맞는 라벨을 선택하여 2~3줄 이내로 압축 -->
- **[원인 규명 / 채택]** <핵심 원인 또는 채택할 방향 1줄> ➔ `<다음 단계(예: /spec, 구현)>`
- **[오판 배제 / 기각]** <사실이 아니었던 가설 또는 버릴 방향 1줄> ➔ `폐기/배제`
- **[추가 조치 / 검증]** <남은 불확실성이나 추가 검증 필요 시 1줄>

---

#### 📋 세부 분석 (주제별 최대 3줄 압축)

**1. <핵심 포인트 1>**
- **상태/결론**: <현상에서 도출된 핵심 판단 1줄>
- **직관적 이유**: <전문 용어는 쉬운 비유나 상식적인 언어로 풀어서 설명>
- **불변식/주의점**: <시스템이 반드시 지켜야 할 안전 규칙이나 제약조건 1줄>

**2. <핵심 포인트 2>**
- **상태/결론**: ...
- **직관적 이유**: ...
- **불변식/주의점**: ...

*(복잡한 측정 로그, 재현 코드, 벤치마크 수치는 필요 시 맨 아래에 간결한 부록으로 첨부)*
```

### Core Principles:
- **No Text Walls**: Avoid dumping raw technical metrics without interpretation. Translate complex engineering mechanisms into simple mental models.
- **Decision First**: Focus on what to do next (Root Cause & Fix, or Go & No-Go) before explaining the investigation history.
- **Emoji Diet**: Keep emojis strictly as structural markers (`🔬`, `💡`, `📌`, `📋`). Do not decorate body text.
