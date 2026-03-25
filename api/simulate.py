from http.server import BaseHTTPRequestHandler
import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Vercel's filesystem is read-only except /tmp — point yfinance cache there
yf.set_tz_cache_location("/tmp")


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

            ticker = params.get("ticker", "AAPL").upper().strip()
            lookback = params.get("lookback", "2y")
            n_simulations = min(int(params.get("simulations", 500)), 1000)
            forecast_days = min(int(params.get("forecast_days", 252)), 504)

            # ── 1. Fetch historical data ──────────────────────────────────────
            hist = yf.download(
                ticker,
                period=lookback,
                auto_adjust=True,
                progress=False,
                threads=False,
            )

            if hist.empty or len(hist) < 30:
                self._send_error(400, f"Ticker '{ticker}' not found or has insufficient history. Check the symbol and try again.")
                return

            # Flatten MultiIndex columns (yf.download returns them for single tickers in 0.2.x)
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)

            close = hist["Close"].dropna()
            current_price = float(close.iloc[-1])

            # ── 2. Estimate GBM parameters from log returns ───────────────────
            log_returns = np.log(close / close.shift(1)).dropna().values
            mu = float(log_returns.mean())      # mean daily log return (drift)
            sigma = float(log_returns.std())    # daily volatility

            # ── 3. Simulate N price paths (Geometric Brownian Motion) ─────────
            #   S(t) = S(t-1) * exp( (mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z )
            #   where Z ~ N(0,1), dt = 1 trading day
            np.random.seed(None)  # fresh randomness each run
            drift = mu - 0.5 * sigma ** 2
            Z = np.random.normal(0, 1, (n_simulations, forecast_days))
            daily_factors = np.exp(drift + sigma * Z)  # shape: (N, forecast_days)

            # Build price matrix: shape (N, forecast_days+1), col 0 = current price
            paths = np.empty((n_simulations, forecast_days + 1))
            paths[:, 0] = current_price
            for t in range(1, forecast_days + 1):
                paths[:, t] = paths[:, t - 1] * daily_factors[:, t - 1]

            # ── 4. Build forecast date index (business days) ──────────────────
            last_date = hist.index[-1].to_pydatetime()
            future_dates = pd.bdate_range(start=last_date, periods=forecast_days + 1)
            dates = [d.strftime("%Y-%m-%d") for d in future_dates]

            # ── 5. Percentile bands ───────────────────────────────────────────
            p5  = np.percentile(paths, 5,  axis=0).tolist()
            p50 = np.percentile(paths, 50, axis=0).tolist()
            p95 = np.percentile(paths, 95, axis=0).tolist()

            # ── 6. Summary statistics ─────────────────────────────────────────
            final_prices  = paths[:, -1]
            median_final  = float(np.median(final_prices))
            p5_final      = float(np.percentile(final_prices, 5))
            p95_final     = float(np.percentile(final_prices, 95))
            prob_gain     = float(np.mean(final_prices > current_price))
            var_95_1day   = float(np.percentile(log_returns, 5))   # 1-day 95% VaR (log return)
            annual_vol    = float(sigma * np.sqrt(252))
            annual_drift  = float(mu * 252)

            # ── 7. Sample paths for chart (max 100 for performance) ───────────
            n_display = min(100, n_simulations)
            idx = np.random.choice(n_simulations, n_display, replace=False)
            display_paths = paths[idx].tolist()

            # ── 8. Build CSV (all simulations) ────────────────────────────────
            header_row = "date," + ",".join(f"sim_{i+1}" for i in range(n_simulations))
            rows = [header_row]
            for t, date in enumerate(dates):
                vals = ",".join(f"{paths[s, t]:.4f}" for s in range(n_simulations))
                rows.append(f"{date},{vals}")
            csv_data = "\n".join(rows)

            # ── 9. Build and send response ────────────────────────────────────
            response = {
                "ticker": ticker,
                "current_price": round(current_price, 4),
                "forecast_days": forecast_days,
                "dates": dates,
                "paths": display_paths,
                "percentile_5": p5,
                "percentile_50": p50,
                "percentile_95": p95,
                "stats": {
                    "median_final": round(median_final, 2),
                    "p5_final": round(p5_final, 2),
                    "p95_final": round(p95_final, 2),
                    "prob_gain": round(prob_gain, 4),
                    "var_95_1day": round(var_95_1day, 4),
                    "annual_vol": round(annual_vol, 4),
                    "annual_drift": round(annual_drift, 4),
                },
                "csv_data": csv_data,
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
        pass  # suppress default access log noise
