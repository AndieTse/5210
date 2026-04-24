# MFE5210 Assignment SOP

## 1. Clarify Scope

- Deliverables: code, references, README, correlation matrix, average Sharpe.
- Constraint: maximal factor correlation must be `<= 0.5`.
- Data is self-prepared.

## 2. Define Data Universe

- Choose one primary universe first (stocks or CTA futures).
- Fix one frequency first (daily recommended for baseline).
- Keep schema consistent: `date, asset, close, volume`.

## 3. Build Candidate Factors

- Start with 10-20 factors from mixed categories:
  - momentum/reversal
  - volatility
  - volume-related
- For each factor, maintain:
  - formula
  - intuition
  - expected direction

## 4. Evaluate Single Factors

- Compute forward return horizon (default 1 day).
- Evaluate Long-Short, Long-Only, or Short-Only depending on design.
- Report annualized Sharpe (without cost).

## 5. Correlation Control

- Build factor correlation matrix.
- Greedy select by Sharpe:
  - rank factors by Sharpe descending
  - keep factor only if `abs(corr)` to all selected factors is `<= 0.5`

## 6. Produce Deliverables

- Save all report artifacts under `reports/`.
- Fill README with method and reproducibility commands.
- Fill references in `references.md`.

## 7. Final Check Before Submission

- Pipeline runs end-to-end on clean environment.
- `max_abs_corr <= 0.5` for selected set.
- Average Sharpe is explicitly reported.
- GitHub repo has complete files and clear run instructions.
