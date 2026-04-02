# Critique Fixes Design Spec

**Date**: 2026-04-01
**Scope**: Address all findings from the design critique of `index.html`, prioritized by user preference (validation first).

## Context

The Monte Carlo Stock Simulator serves finance students running simulations for coursework. A design critique scored the interface 24/40 on Nielsen's heuristics and identified 5 priority issues. This spec addresses all of them in 4 discrete sections.

## Section 1: Ticker Input Hardening (P2)

**Problem**: Free-text ticker input has no client-side validation. Errors appear in a global banner far from the input.

**Changes**:
- Add client-side validation on blur and submit: ticker must be 1-5 uppercase alpha characters
- Add inline error element directly below ticker input, styled as a small red hint (matching `--red-soft` pattern)
- When API returns a ticker-specific error, display it inline below the ticker input
- Add a hint line below the label: "US stock symbols only"
- Disable Run Simulation and Run Backtest buttons while ticker is invalid

## Section 2: Stats Grid Restructuring (P2)

**Problem**: 8 metrics at equal visual weight overwhelms working memory (violates 4-item rule).

**Changes**:
- Split into two tiers:
  - **Primary** (3-column row): Current Price, Median Forecast, Prob. of Gain — larger cards, `--surface` background, semantic left borders
  - **Secondary** (5-column row, collapsing on mobile): Bear Case, Bull Case, Annual Volatility, 1-Day VaR, Calibration — smaller text, `--bg` background, subtle borders only
- Two separate `.stats-grid` containers with different column counts and card sizes

## Section 3: AI Pattern Cleanup & Typography (P2/P3)

**Problem**: Decorative gradients signal AI-generated aesthetics. Micro-labels at 10px are too small.

**Changes**:
- Header `::before` shimmer gradient -> solid 3px `var(--accent)` border, no animation
- `#results-card` and `#backtest-card` gradient `border-image` -> solid `border-top: 3px solid var(--accent)`
- Remove `pulse-soft` animation from empty state icon
- Bump all micro-labels (`.stat-label`, `.coverage-bar-label`, `.manual-calib-title`, `.calib-detail-title`) from `0.625rem` to `0.75rem`
- Remove unused `--ease-spring` CSS variable

## Section 4: Results Summary Line (P3)

**Problem**: Raw numbers require interpretation. Students need a "so what" before diving into metrics.

**Changes**:
- Add a dynamic summary `<p>` between the chart and stats grid
- Content: "{TICKER} median forecast is ${price} ({+/-pct}%) over {horizon}, with a {prob}% probability of finishing above today's price."
- Styled: `--text-secondary`, `0.9375rem`, font-weight 500
- Generated in `renderStats()` using existing data fields

## Out of Scope

- Keyboard shortcuts / power user features
- Onboarding flow for first-time users
- Code viewer theme alignment
- Responsive tablet breakpoint
- Metric explanations / glossary
