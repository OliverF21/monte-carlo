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


def _garch_simulate(log_returns, current_price, n_sims, n_days):
    r   = log_returns * 100
    mdl = _arch_model(r, mean='Constant', vol='GARCH', p=1, q=1, dist='t')
    res = mdl.fit(disp='off', show_warning=False)
    mu_pct = float(res.params['mu'])
    omega  = float(res.params['omega'])
    alpha  = float(res.params['alpha[1]'])
    beta   = float(res.params['beta[1]'])
    nu     = float(res.params['nu'])
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
    return paths, 'GARCH(1,1)-t'


def _gbm_simulate(log_returns, current_price, n_sims, n_days):
    mu    = float(log_returns.mean())
    sigma = float(log_returns.std())
    n     = len(log_returns)
    mu_se = sigma / np.sqrt(n) if n > 0 else 1.0
    t_stat = abs(mu) / mu_se if mu_se > 0 else 0.0
    mu    = mu * min(1.0, t_stat / 2.0)
    drift = mu - 0.5 * sigma ** 2
    Z     = np.random.normal(0, 1, (n_sims, n_days))
    facts = np.exp(drift + sigma * Z)
    paths = np.empty((n_sims, n_days + 1))
    paths[:, 0] = current_price
    for t in range(1, n_days + 1):
        paths[:, t] = paths[:, t - 1] * facts[:, t - 1]
    return paths, 'GBM'


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


def _compute_calib_factor(close, n_sims=200):
    cal_len = min(63, max(30, len(close) // 6))
    if len(close) < cal_len + 100:
        return 1.0
    inner = close.iloc[:-cal_len]
    cal   = close.iloc[-(cal_len + 1):]
    lr    = np.log(inner / inner.shift(1)).dropna().values
    start = float(inner.iloc[-1])
    if len(lr) < 60:
        return 1.0
    saved = np.random.get_state()
    np.random.seed(0)
    try:
        if _ARCH and len(lr) >= 100:
            try:
                paths, _ = _garch_simulate(lr, start, n_sims, cal_len)
            except Exception:
                paths, _ = _gbm_simulate(lr, start, n_sims, cal_len)
        else:
            paths, _ = _gbm_simulate(lr, start, n_sims, cal_len)
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
    if abs(k - 1.0) < 0.02:
        return paths
    log_ret = np.log(paths / paths[:, 0:1])
    return paths[:, 0:1] * np.exp(log_ret * k)


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
            n_simulations = min(int(params.get("simulations", 500)), 1000)
            test_days     = min(int(params.get("test_days", 252)), 504)

            # Fetch enough history: test period + at least 1 year of training data
            # test_days/252 years of test + 2 years training, rounded up to next period
            total_years = test_days / 252 + 2
            if total_years <= 3:
                fetch_period = "3y"
            elif total_years <= 5:
                fetch_period = "5y"
            else:
                fetch_period = "10y"

            hist = yf.download(ticker, period=fetch_period, auto_adjust=True,
                               progress=False, threads=False)
            if hist.empty or len(hist) < test_days + 100:
                self._send_error(400, f"Insufficient data for '{ticker}'.")
                return
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)

            close = hist["Close"].dropna()
            train = close.iloc[:-test_days]
            test  = close.iloc[-(test_days + 1):]

            current_price = float(train.iloc[-1])
            log_returns   = np.log(train / train.shift(1)).dropna().values

            # Calibration factor derived from training data only (no test data leakage)
            calib_k = _compute_calib_factor(train)

            np.random.seed(None)
            if _ARCH and len(log_returns) >= 100:
                try:
                    paths, model = _garch_simulate(
                        log_returns, current_price, n_simulations, test_days)
                except Exception:
                    paths, model = _gbm_simulate(
                        log_returns, current_price, n_simulations, test_days)
            else:
                paths, model = _gbm_simulate(
                    log_returns, current_price, n_simulations, test_days)

            paths = _apply_calib(paths, calib_k)

            p5  = np.percentile(paths, 5,  axis=0).tolist()
            p25 = np.percentile(paths, 25, axis=0).tolist()
            p50 = np.percentile(paths, 50, axis=0).tolist()
            p75 = np.percentile(paths, 75, axis=0).tolist()
            p95 = np.percentile(paths, 95, axis=0).tolist()

            actual = test.values.tolist()
            dates  = [d.strftime("%Y-%m-%d") for d in test.index]

            act    = test.values[1:]
            cov_90 = float(np.mean((act >= np.array(p5[1:]))  & (act <= np.array(p95[1:]))))
            cov_50 = float(np.mean((act >= np.array(p25[1:])) & (act <= np.array(p75[1:]))))

            response = {
                "ticker":        ticker,
                "model":         model,
                "calib_k":       round(calib_k, 3),
                "test_days":     test_days,
                "dates":         dates,
                "actual":        actual,
                "percentile_5":  p5,
                "percentile_25": p25,
                "percentile_50": p50,
                "percentile_75": p75,
                "percentile_95": p95,
                "coverage": {
                    "band_90": round(cov_90, 4),
                    "band_50": round(cov_50, 4),
                },
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
