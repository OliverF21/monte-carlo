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


def _garch_simulate(log_returns, current_price, n_sims, n_days):
    """GARCH(1,1) with Student-t innovations.

    Captures two well-documented features of equity returns that plain GBM misses:
      - Volatility clustering: volatile days tend to cluster together
      - Fat tails: extreme moves happen more often than a normal distribution predicts

    The model at each step:
        sigma2_t = omega + alpha * shock_{t-1}^2 + beta * sigma2_{t-1}
        shock_t  = sqrt(sigma2_t) * z,  z ~ t(nu) standardised to unit variance
        r_t      = mu + shock_t
        S_t      = S_{t-1} * exp(r_t)
    """
    r = log_returns * 100  # arch library works in percentage returns
    mdl = _arch_model(r, mean='Constant', vol='GARCH', p=1, q=1, dist='t')
    res = mdl.fit(disp='off', show_warning=False)

    mu_pct = float(res.params['mu'])
    omega  = float(res.params['omega'])
    alpha  = float(res.params['alpha[1]'])
    beta   = float(res.params['beta[1]'])
    nu     = float(res.params['nu'])

    # Seed the simulation from the last observed conditional variance
    sig2 = float(res.conditional_volatility[-1]) ** 2

    # Vectorise across simulations; iterate over time (GARCH is path-dependent)
    sig2_v  = np.full(n_sims, sig2)
    price_v = np.full(n_sims, float(current_price))
    paths   = np.empty((n_sims, n_days + 1))
    paths[:, 0] = current_price

    # Scale factor to give t(nu) unit variance: sqrt(nu/(nu-2))
    std_scale = np.sqrt(nu / (nu - 2)) if nu > 2 else 1.0

    for d in range(n_days):
        z       = np.random.standard_t(nu, size=n_sims) / std_scale
        shock   = np.sqrt(sig2_v) * z          # in pct units
        ret     = (mu_pct + shock) / 100       # back to decimal log return
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
    """Fallback: plain GBM with constant volatility and normal innovations."""
    mu    = float(log_returns.mean())
    sigma = float(log_returns.std())
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

            # ── 3. Date index ─────────────────────────────────────────────────
            last_date    = hist.index[-1].to_pydatetime()
            future_dates = pd.bdate_range(start=last_date, periods=forecast_days + 1)
            dates        = [d.strftime("%Y-%m-%d") for d in future_dates]

            # ── 4. Percentile bands ───────────────────────────────────────────
            p5  = np.percentile(paths, 5,  axis=0).tolist()
            p50 = np.percentile(paths, 50, axis=0).tolist()
            p95 = np.percentile(paths, 95, axis=0).tolist()

            # ── 5. Summary statistics ─────────────────────────────────────────
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

            # ── 6. Sample paths for chart (cap at 100) ────────────────────────
            n_display     = min(100, n_simulations)
            idx           = np.random.choice(n_simulations, n_display, replace=False)
            display_paths = paths[idx].tolist()

            # ── 7. CSV ────────────────────────────────────────────────────────
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
