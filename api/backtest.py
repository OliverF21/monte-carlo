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
    r = log_returns * 100
    mdl = _arch_model(r, mean='Constant', vol='GARCH', p=1, q=1, dist='t')
    res = mdl.fit(disp='off', show_warning=False)
    mu_pct = float(res.params['mu'])
    omega  = float(res.params['omega'])
    alpha  = float(res.params['alpha[1]'])
    beta   = float(res.params['beta[1]'])
    nu     = float(res.params['nu'])
    sig2   = float(res.conditional_volatility[-1]) ** 2
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
    drift = mu - 0.5 * sigma ** 2
    Z     = np.random.normal(0, 1, (n_sims, n_days))
    facts = np.exp(drift + sigma * Z)
    paths = np.empty((n_sims, n_days + 1))
    paths[:, 0] = current_price
    for t in range(1, n_days + 1):
        paths[:, t] = paths[:, t - 1] * facts[:, t - 1]
    return paths, 'GBM'


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
            test_days     = min(int(params.get("test_days", 252)), 252)

            # Fetch 3 years: ~2 years training, last test_days = held-out test set
            hist = yf.download(ticker, period="3y", auto_adjust=True,
                               progress=False, threads=False)
            if hist.empty or len(hist) < test_days + 100:
                self._send_error(400, f"Insufficient data for '{ticker}'.")
                return
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)

            close = hist["Close"].dropna()

            # Split: training = everything before last test_days
            #        test     = last test_days+1 rows (index 0 = last training price)
            train = close.iloc[:-test_days]
            test  = close.iloc[-(test_days + 1):]

            current_price = float(train.iloc[-1])
            log_returns   = np.log(train / train.shift(1)).dropna().values

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

            p5  = np.percentile(paths, 5,  axis=0).tolist()
            p25 = np.percentile(paths, 25, axis=0).tolist()
            p50 = np.percentile(paths, 50, axis=0).tolist()
            p75 = np.percentile(paths, 75, axis=0).tolist()
            p95 = np.percentile(paths, 95, axis=0).tolist()

            actual = test.values.tolist()
            dates  = [d.strftime("%Y-%m-%d") for d in test.index]

            # Coverage: what % of actual prices fell inside each band?
            # Skip day 0 — it's the shared starting price (always 100% inside)
            act    = test.values[1:]
            cov_90 = float(np.mean((act >= np.array(p5[1:]))  & (act <= np.array(p95[1:]))))
            cov_50 = float(np.mean((act >= np.array(p25[1:])) & (act <= np.array(p75[1:]))))

            response = {
                "ticker":        ticker,
                "model":         model,
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
