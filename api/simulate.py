from http.server import BaseHTTPRequestHandler
import json
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm as _norm

yf.set_tz_cache_location("/tmp")

try:
    from arch import arch_model as _arch_model
    _ARCH = True
except Exception:
    _ARCH = False


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


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward sigma calibration
# ─────────────────────────────────────────────────────────────────────────────

def _compute_calib_factor(close, n_sims=200):
    """
    Find sigma multiplier k such that the 90% band achieves ~90% empirical coverage.

    Method: hold out the last cal_len days of training data as a calibration window.
    Simulate forward from the start of that window, measure coverage, then:
        k = Φ⁻¹(0.95) / Φ⁻¹((1 + observed_coverage) / 2)

    If bands are too narrow (coverage < 90%), k > 1 (widen them).
    If bands are too wide (coverage > 90%), k < 1 (narrow them).
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
    p5     = np.percentile(paths[:, 1:],  5, axis=0)
    p95    = np.percentile(paths[:, 1:], 95, axis=0)
    cov    = float(np.mean((actual >= p5) & (actual <= p95)))

    if cov <= 0.02 or cov >= 0.98:
        return 1.0

    k = float(_norm.ppf(0.95) / _norm.ppf((1.0 + cov) / 2.0))
    return float(np.clip(k, 0.5, 3.0))


def _apply_calib(paths, k):
    """Scale the log-returns of every simulation path by factor k.
    Equivalent to multiplying sigma by k without rerunning the simulation."""
    if abs(k - 1.0) < 0.02:
        return paths
    log_ret = np.log(paths / paths[:, 0:1])   # (n_sims, n_days+1)
    return paths[:, 0:1] * np.exp(log_ret * k)


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

            ticker        = params.get("ticker", "AAPL").upper().strip()
            lookback      = params.get("lookback", "2y")
            n_simulations = min(int(params.get("simulations", 500)), 1000)
            forecast_days = min(int(params.get("forecast_days", 252)), 504)

            # ── 1. Fetch historical data ──────────────────────────────────────
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

            # ── 2. Walk-forward calibration (uses inner holdout) ──────────────
            calib_k = _compute_calib_factor(close)

            # ── 3. Simulate paths ─────────────────────────────────────────────
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

            # ── 4. Apply calibration ──────────────────────────────────────────
            paths = _apply_calib(paths, calib_k)
            model_info['calib_k'] = round(calib_k, 3)

            # ── 5. Date index ─────────────────────────────────────────────────
            last_date    = hist.index[-1].to_pydatetime()
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
                "dates":         dates,
                "paths":         display_paths,
                "percentile_5":  p5,
                "percentile_50": p50,
                "percentile_95": p95,
                "stats":         stats,
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
