from http.server import BaseHTTPRequestHandler
import json
import numpy as np
import pandas as pd
import yfinance as yf

yf.set_tz_cache_location("/tmp")

try:
    from arch import arch_model as _arch_model
    _ARCH = True
except Exception:
    _ARCH = False


_TARGET_90 = 0.90
_TARGET_50 = 0.50
_TOL_90 = 0.08
_TOL_50 = 0.10
_K_MIN = 0.5
_K_MAX = 3.0


# ─────────────────────────────────────────────────────────────────────────────
# Simulation engines
# ─────────────────────────────────────────────────────────────────────────────

def _garch_simulate(log_returns, current_price, n_sims, n_days):
    """GARCH(1,1) + Student-t innovations with drift shrinkage."""
    r   = log_returns * 100
    mdl = _arch_model(r, mean='Constant', vol='GARCH', p=1, q=1, dist='t')
    res = mdl.fit(disp='off', show_warning=False)

    mu_pct = float(res.params['mu'])
    omega  = float(res.params['omega'])
    alpha  = float(res.params['alpha[1]'])
    beta   = float(res.params['beta[1]'])
    nu     = float(res.params['nu'])

    # Drift shrinkage: mu is very noisy over short lookbacks.
    # Shrink toward zero proportionally if t-statistic < 2.
    try:
        mu_se  = float(res.std_err['mu'])
        t_stat = abs(mu_pct) / mu_se if mu_se > 0 else 0.0
        mu_pct = mu_pct * min(1.0, t_stat / 2.0)
    except Exception:
        pass

    sig2    = float(res.conditional_volatility[-1]) ** 2
    sig2_v  = np.full(n_sims, sig2)
    price_v = np.full(n_sims, float(current_price))
    paths   = np.empty((n_sims, n_days + 1))
    paths[:, 0] = current_price

    std_scale = np.sqrt(nu / (nu - 2)) if nu > 2 else 1.0
    for d in range(n_days):
        z       = np.random.standard_t(nu, size=n_sims) / std_scale
        shock   = np.sqrt(sig2_v) * z
        ret     = (mu_pct + shock) / 100
        price_v = price_v * np.exp(ret)
        paths[:, d + 1] = price_v
        sig2_v  = omega + alpha * shock ** 2 + beta * sig2_v

    info = {
        'model':        'GARCH(1,1)-t',
        'annual_drift': round(mu_pct / 100 * 252, 4),
        'annual_vol':   round(np.sqrt(sig2) / 100 * np.sqrt(252), 4),
        'garch_alpha':  round(alpha, 4),
        'garch_beta':   round(beta, 4),
        't_df':         round(nu, 2),
    }
    return paths, info


def _gbm_simulate(log_returns, current_price, n_sims, n_days):
    """Plain GBM fallback with drift shrinkage."""
    mu    = float(log_returns.mean())
    sigma = float(log_returns.std())

    # Drift shrinkage
    n      = len(log_returns)
    mu_se  = sigma / np.sqrt(n) if n > 0 else 1.0
    t_stat = abs(mu) / mu_se if mu_se > 0 else 0.0
    mu     = mu * min(1.0, t_stat / 2.0)

    drift = mu - 0.5 * sigma ** 2
    Z     = np.random.normal(0, 1, (n_sims, n_days))
    facts = np.exp(drift + sigma * Z)
    paths = np.empty((n_sims, n_days + 1))
    paths[:, 0] = current_price
    for t in range(1, n_days + 1):
        paths[:, t] = paths[:, t - 1] * facts[:, t - 1]

    info = {
        'model':        'GBM',
        'annual_drift': round(mu * 252, 4),
        'annual_vol':   round(sigma * np.sqrt(252), 4),
    }
    return paths, info


def _coverage_stats(paths, actual):
    p5  = np.percentile(paths[:, 1:],  5, axis=0)
    p25 = np.percentile(paths[:, 1:], 25, axis=0)
    p75 = np.percentile(paths[:, 1:], 75, axis=0)
    p95 = np.percentile(paths[:, 1:], 95, axis=0)
    cov_90 = float(np.mean((actual >= p5)  & (actual <= p95)))
    cov_50 = float(np.mean((actual >= p25) & (actual <= p75)))
    return cov_90, cov_50


def _coverage_score(cov_90, cov_50, k):
    penalty_90 = max(0.0, abs(cov_90 - _TARGET_90) - _TOL_90)
    penalty_50 = max(0.0, abs(cov_50 - _TARGET_50) - _TOL_50)
    inside_bonus = 0.0 if (penalty_90 > 0.0 or penalty_50 > 0.0) else abs(k - 1.0) * 0.01
    return (
        penalty_90 * 100.0
        + penalty_50 * 100.0
        + abs(cov_90 - _TARGET_90)
        + abs(cov_50 - _TARGET_50)
        + inside_bonus
    )


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward sigma calibration
# ─────────────────────────────────────────────────────────────────────────────

def _compute_calib_factor(close, n_sims=200):
    """
    Search for a volatility multiplier k that brings calibration-window
    coverage as close as possible to the acceptable 90% and 50% band targets.
    """
    cal_len = min(63, max(30, len(close) // 6))
    if len(close) < cal_len + 100:
        return 1.0

    inner = close.iloc[:-cal_len]
    cal   = close.iloc[-(cal_len + 1):]   # includes last inner price as day 0

    lr_inner    = np.log(inner / inner.shift(1)).dropna().values
    start_price = float(inner.iloc[-1])

    if len(lr_inner) < 60:
        return 1.0

    # Use a fixed seed so calibration factor is deterministic for the same data
    saved = np.random.get_state()
    np.random.seed(0)
    try:
        if _ARCH and len(lr_inner) >= 100:
            try:
                paths, _ = _garch_simulate(lr_inner, start_price, n_sims, cal_len)
            except Exception:
                paths, _ = _gbm_simulate(lr_inner, start_price, n_sims, cal_len)
        else:
            paths, _ = _gbm_simulate(lr_inner, start_price, n_sims, cal_len)
    except Exception:
        np.random.set_state(saved)
        return 1.0
    np.random.set_state(saved)

    actual = cal.values[1:]
    best_k = 1.0
    best_score = float("inf")

    candidates = np.linspace(_K_MIN, _K_MAX, 51)
    for _ in range(3):
      for k in candidates:
        scaled = _apply_calib(paths, float(k))
        cov_90, cov_50 = _coverage_stats(scaled, actual)
        score = _coverage_score(cov_90, cov_50, float(k))
        if score < best_score:
          best_score = score
          best_k = float(k)

      step = float(candidates[1] - candidates[0]) if len(candidates) > 1 else 0.1
      lo = max(_K_MIN, best_k - 2.0 * step)
      hi = min(_K_MAX, best_k + 2.0 * step)
      candidates = np.linspace(lo, hi, 17)

    return float(np.clip(best_k, _K_MIN, _K_MAX))


def _apply_calib(paths, k):
    """Scale only the volatility component of each path by factor k,
    preserving the original drift.  The cross-simulation mean at each
    timestep estimates drift; deviations from that mean are pure vol."""
    if abs(k - 1.0) < 0.02:
        return paths
    start   = paths[:, 0:1]
    log_ret = np.log(paths / start)                       # (n_sims, n_days+1)
    drift   = log_ret.mean(axis=0, keepdims=True)         # (1, n_days+1)
    scaled  = drift + (log_ret - drift) * k               # scale vol, keep drift
    return start * np.exp(scaled)


# ─────────────────────────────────────────────────────────────────────────────
# Risk metrics (derived from simulated paths)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_risk_metrics(paths, current_price, forecast_days):
    """Derive portfolio/stock risk metrics from the calibrated simulation paths."""
    final_prices  = paths[:, -1]
    final_returns = final_prices / current_price - 1.0

    # Horizon VaR (95%): loss exceeded only 5% of the time
    var_5 = -float(np.percentile(final_returns, 5))

    # CVaR / Expected Shortfall: mean loss in the worst 5%
    cutoff = np.percentile(final_returns, 5)
    tail   = final_returns[final_returns <= cutoff]
    cvar_5 = -float(np.mean(tail)) if len(tail) > 0 else var_5

    # Max drawdown per path (vectorized)
    running_max = np.maximum.accumulate(paths, axis=1)
    drawdowns   = (running_max - paths) / running_max
    max_dd      = drawdowns.max(axis=1)

    # Sharpe ratio (annualized, risk-free = 0)
    log_total  = np.log(final_prices / current_price)
    ann_factor = 252.0 / forecast_days
    mean_ann   = float(np.mean(log_total)) * ann_factor
    vol_ann    = float(np.std(log_total)) * np.sqrt(ann_factor)
    sharpe     = round(mean_ann / vol_ann, 2) if vol_ann > 1e-8 else 0.0

    return {
        'horizon_var_5':       round(var_5, 4),
        'horizon_cvar_5':      round(cvar_5, 4),
        'max_drawdown_median': round(float(np.median(max_dd)), 4),
        'max_drawdown_p95':    round(float(np.percentile(max_dd, 95)), 4),
        'sharpe_ratio':        sharpe,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_portfolio(portfolio, lookback):
    """
    Fetch data for every ticker in the portfolio, align dates, and return
    a synthetic close-price series (indexed to 100) and weighted log returns.

    portfolio: list of {"ticker": "AAPL", "weight": 0.4}
    Returns: (close_series, log_returns_array, label, last_index_entry)
    """
    tickers = [p["ticker"].upper().strip() for p in portfolio]
    weights = np.array([float(p["weight"]) for p in portfolio])

    # Normalise weights to sum to 1
    w_sum = weights.sum()
    if w_sum <= 0:
        raise ValueError("Portfolio weights must be positive and sum to > 0.")
    weights = weights / w_sum

    # Download all tickers at once
    raw = yf.download(tickers, period=lookback, auto_adjust=True,
                      progress=False, threads=True)

    if raw.empty:
        raise ValueError("No data returned for portfolio tickers.")

    # yf.download returns MultiIndex columns (Price, Ticker) for multiple tickers
    if isinstance(raw.columns, pd.MultiIndex):
        close_df = raw["Close"]
    else:
        # Single ticker edge case
        close_df = raw[["Close"]].copy()
        close_df.columns = tickers

    # Drop any ticker columns that are entirely NaN
    close_df = close_df.dropna(axis=1, how="all")
    missing = set(tickers) - set(close_df.columns)
    if missing:
        raise ValueError(f"Tickers not found or no data: {', '.join(sorted(missing))}")

    # Use only rows where ALL tickers have data
    close_df = close_df.dropna()
    if len(close_df) < 30:
        raise ValueError("Insufficient overlapping history for portfolio (need >= 30 days).")

    # Reorder columns to match input order and rebuild aligned weight vector
    close_df = close_df[tickers]

    # Compute daily log returns per ticker
    log_ret_df = np.log(close_df / close_df.shift(1)).dropna()

    # Weighted portfolio log returns (approximate for small daily returns)
    port_log_ret = (log_ret_df.values * weights).sum(axis=1)

    # Build synthetic portfolio close price indexed to 100
    cum_ret = np.concatenate([[0.0], np.cumsum(port_log_ret)])
    port_close = 100.0 * np.exp(cum_ret)

    port_series = pd.Series(port_close, index=close_df.index, name="Close")

    # Build label like "AAPL 40% / MSFT 30% / GOOGL 30%"
    parts = [f"{t} {w*100:.0f}%" for t, w in zip(tickers, weights)]
    label = " / ".join(parts)

    return port_series, port_log_ret, label, close_df.index[-1]


# ─────────────────────────────────────────────────────────────────────────────
# Handler
# ─────────────────────────────────────────────────────────────────────────────

class handler(BaseHTTPRequestHandler):

    def send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_cors_headers()
        self.end_headers()

    def do_POST(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            params = json.loads(body)

            portfolio     = params.get("portfolio")  # list of {ticker, weight}
            ticker        = params.get("ticker", "AAPL").upper().strip()
            lookback      = params.get("lookback", "2y")
            n_simulations = min(int(params.get("simulations", 500)), 10000)
            forecast_days = min(int(params.get("forecast_days", 252)), 504)

            # ── 1. Fetch historical data ──────────────────────────────────────
            is_portfolio = portfolio and isinstance(portfolio, list) and len(portfolio) > 0

            if is_portfolio:
                try:
                    close, port_log_ret, label, last_idx = _fetch_portfolio(portfolio, lookback)
                except ValueError as ve:
                    self._send_error(400, str(ve))
                    return
                ticker        = label
                current_price = float(close.iloc[-1])  # 100-indexed
                log_returns   = port_log_ret
                last_date     = last_idx.to_pydatetime()
            else:
                hist = yf.download(ticker, period=lookback, auto_adjust=True,
                                   progress=False, threads=False)
                if hist.empty or len(hist) < 30:
                    self._send_error(400, f"Ticker '{ticker}' not found or has insufficient history.")
                    return
                if isinstance(hist.columns, pd.MultiIndex):
                    hist.columns = hist.columns.get_level_values(0)

                close         = hist["Close"].dropna()
                current_price = float(close.iloc[-1])
                log_returns   = np.log(close / close.shift(1)).dropna().values
                last_date     = hist.index[-1].to_pydatetime()

            # ── 2. Simulate paths ─────────────────────────────────────────────
            np.random.seed(None)
            if _ARCH and len(log_returns) >= 100:
                try:
                    paths, model_info = _garch_simulate(
                        log_returns, current_price, n_simulations, forecast_days)
                except Exception:
                    paths, model_info = _gbm_simulate(
                        log_returns, current_price, n_simulations, forecast_days)
            else:
                paths, model_info = _gbm_simulate(
                    log_returns, current_price, n_simulations, forecast_days)

            calib_k = _compute_calib_factor(close)
            paths = _apply_calib(paths, calib_k)
            model_info['calib_k'] = round(calib_k, 3)

            # ── 3. Risk metrics ──────────────────────────────────────────────
            risk = _compute_risk_metrics(paths, current_price, forecast_days)

            # ── 4. Date index ─────────────────────────────────────────────────
            future_dates = pd.bdate_range(start=last_date, periods=forecast_days + 1)
            dates        = [d.strftime("%Y-%m-%d") for d in future_dates]

            # ── 6. Percentile bands ───────────────────────────────────────────
            p5  = np.percentile(paths, 5,  axis=0).tolist()
            p50 = np.percentile(paths, 50, axis=0).tolist()
            p95 = np.percentile(paths, 95, axis=0).tolist()

            # ── 7. Summary statistics ─────────────────────────────────────────
            final_prices = paths[:, -1]
            var_95_1day  = float(np.percentile(log_returns, 5))

            stats = {
                'median_final': round(float(np.median(final_prices)), 2),
                'p5_final':     round(float(np.percentile(final_prices, 5)), 2),
                'p95_final':    round(float(np.percentile(final_prices, 95)), 2),
                'prob_gain':    round(float(np.mean(final_prices > current_price)), 4),
                'var_95_1day':  round(var_95_1day, 4),
                **model_info,
            }

            # ── 8. Sample paths for chart ─────────────────────────────────────
            n_display     = min(100, n_simulations)
            idx           = np.random.choice(n_simulations, n_display, replace=False)
            display_paths = paths[idx].tolist()

            # ── 9. CSV ────────────────────────────────────────────────────────
            header_row = "date," + ",".join(f"sim_{i+1}" for i in range(n_simulations))
            rows = [header_row]
            for t, date in enumerate(dates):
                vals = ",".join(f"{paths[s, t]:.4f}" for s in range(n_simulations))
                rows.append(f"{date},{vals}")
            csv_data = "\n".join(rows)

            response = {
                "ticker":        ticker,
                "current_price": round(current_price, 4),
                "forecast_days": forecast_days,
                "is_portfolio":  is_portfolio,
                "dates":         dates,
                "paths":         display_paths,
                "percentile_5":  p5,
                "percentile_50": p50,
                "percentile_95": p95,
                "stats":         stats,
                "risk":          risk,
                "csv_data":      csv_data,
            }

            body_bytes = json.dumps(response).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body_bytes)))
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(body_bytes)

        except Exception as e:
            self._send_error(500, str(e))

    def _send_error(self, code, message):
        body = json.dumps({"error": message}).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass
