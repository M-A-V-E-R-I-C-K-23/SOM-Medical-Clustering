# AGENTS.md — Workspace Guidelines & Ponytail Ruleset

## Ponytail: The Lazy Senior Developer

Follow the Ponytail decision ladder before writing, modifying, or refactoring code:

1. **Does this need to exist at all?** (YAGNI — delete dead code, unused imports, and unneeded abstractions).
2. **Already in this codebase?** (Reuse existing helpers, modules, and utilities).
3. **Does the standard library do it?** (Use built-in language features first).
4. **Does an already-installed dependency solve it?** (Use NumPy, Pandas, Scikit-learn, FastAPI directly).
5. **Can this be one line or minimal code?** (Keep solutions concise, idiomatic, and clean).
6. **Only then: write the minimum code necessary.**

## Core Invariants for Medical SOM Project
- **Strict Data Isolation**: Ground truth diagnosis labels ($y$) are held out strictly for post-hoc validation. They must NEVER enter feature scaling, SOM training, or K-Means clustering.
- **Two-Level Clustering**: K-Means is fitted on SOM codebook vectors, NOT raw patient samples.
- **Preserve Pipeline Results**: 9×9 SOM grid, $K=3$, seed=42, 10,000 iterations.
- **Cleanliness**: Zero unused imports, zero dead variables, zero broken routes.
