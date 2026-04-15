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
_K_MIN = 0.3
_K_MAX = 4.0
_N_CALIB_WINDOWS = 5


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


def _simulate(log_returns, current_price, n_sims, n_days):
    """Run GARCH if available, else GBM."""
    if _ARCH and len(log_returns) >= 100:
        try:
            return _garch_simulate(log_returns, current_price, n_sims, n_days)
        except Exception:
            pass
    return _gbm_simulate(log_returns, current_price, n_sims, n_days)


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


def _is_converged(cov_90, cov_50):
    return abs(cov_90 - _TARGET_90) <= _TOL_90 and abs(cov_50 - _TARGET_50) <= _TOL_50


def _build_calib_windows(close, n_windows, cal_len):
    """Create multiple walk-forward calibration windows across the training data.

    Each window is a (train_slice, actual_values) pair.  Windows are spaced
    evenly across the available data so the calibration factor generalises
    across different market regimes rather than overfitting to one period.
    """
    windows = []
    min_train = max(100, cal_len)          # need enough data to fit a model
    usable    = len(close) - cal_len       # last possible train-end index
    if usable < min_train:
        return windows

    # Space window end-points evenly from the earliest feasible point
    # to the latest (end of the series minus cal_len).
    starts = np.linspace(min_train, usable, n_windows, dtype=int)
    for s in starts:
        train_slice = close.iloc[:s]
        actual_slice = close.iloc[s:s + cal_len].values
        if len(actual_slice) == cal_len and len(train_slice) >= 60:
            windows.append((train_slice, actual_slice))
    return windows


def _presimulate_windows(windows, n_sims):
    """Simulate paths once per window and cache them.

    Returns list of (paths, actual) tuples.  The heavy simulation work is
    done here so the k-search loop only needs to rescale, not re-simulate.
    """
    cached = []
    for train_slice, actual in windows:
        lr    = np.log(train_slice / train_slice.shift(1)).dropna().values
        start = float(train_slice.iloc[-1])
        paths, _ = _simulate(lr, start, n_sims, len(actual))
        cached.append((paths, actual))
    return cached


def _score_k_on_cached(cached, k):
    """Evaluate a candidate k on pre-simulated paths.  Fast — no simulation."""
    total_score = 0.0
    total_90 = 0.0
    total_50 = 0.0
    for paths, actual in cached:
        scaled = _apply_calib(paths, k)
        cov_90, cov_50 = _coverage_stats(scaled, actual)
        total_score += _coverage_score(cov_90, cov_50, k)
        total_90 += cov_90
        total_50 += cov_50
    n = len(cached)
    return total_score / n, total_90 / n, total_50 / n


def _find_best_k(cached):
    """Grid search with iterative refinement for the best k on cached paths."""
    best_k = 1.0
    best_score = float("inf")
    candidates = np.linspace(_K_MIN, _K_MAX, 51)

    for _ in range(3):
        for k in candidates:
            score, _, _ = _score_k_on_cached(cached, float(k))
            if score < best_score:
                best_score = score
                best_k = float(k)
        step = float(candidates[1] - candidates[0]) if len(candidates) > 1 else 0.1
        lo = max(_K_MIN, best_k - 2.0 * step)
        hi = min(_K_MAX, best_k + 2.0 * step)
        candidates = np.linspace(lo, hi, 17)

    return float(np.clip(best_k, _K_MIN, _K_MAX)), best_score


def _compute_calib_factor(close, n_sims=100):
    """Iterative multi-window walk-forward calibration.

    1. Try multiple lookback lengths (full data, 75%, 50%) to find the
       training regime that best fits the asset.
    2. For each lookback, build several walk-forward calibration windows
       spread across the training data.
    3. Grid-search for the k that minimises mean coverage error across
       all windows — this prevents overfitting to one period.
    4. Return the best (lookback, k) combination and iteration metadata.
    """
    cal_len = min(63, max(30, len(close) // 6))
    if len(close) < cal_len + 100:
        return 1.0, {"iterations": 0, "converged": False, "windows": 0,
                      "lookback_pct": 100, "avg_cov_90": None, "avg_cov_50": None}

    saved = np.random.get_state()
    np.random.seed(0)

    # Try multiple lookback fractions — shorter captures recent regime,
    # longer captures more market cycles.
    lookback_fractions = [1.0, 0.75, 0.50]
    overall_best_k = 1.0
    overall_best_score = float("inf")
    overall_best_pct = 100
    overall_best_cov90 = None
    overall_best_cov50 = None
    total_iterations = 0
    best_n_windows = 0
    converged = False

    best_cached = None
    for frac in lookback_fractions:
        n_points = max(cal_len + 100, int(len(close) * frac))
        subset = close.iloc[-n_points:] if n_points < len(close) else close

        n_win = min(_N_CALIB_WINDOWS, max(2, len(subset) // (cal_len + 60)))
        windows = _build_calib_windows(subset, n_win, cal_len)
        if len(windows) < 1:
            continue

        # Simulate once per window, then search k by rescaling only
        cached = _presimulate_windows(windows, n_sims)
        k, score = _find_best_k(cached)
        total_iterations += 1
        _, avg_90, avg_50 = _score_k_on_cached(cached, k)

        if score < overall_best_score:
            overall_best_score = score
            overall_best_k = k
            overall_best_pct = int(frac * 100)
            overall_best_cov90 = round(avg_90, 4)
            overall_best_cov50 = round(avg_50, 4)
            best_n_windows = len(windows)
            best_cached = cached

        if _is_converged(avg_90, avg_50):
            converged = True
            break

    # If not converged, do a second pass: fine-grained search around
    # the current best using already-simulated paths (no new simulations).
    if not converged and best_cached is not None:
        total_iterations += 1
        spread = 0.5
        narrow_candidates = np.linspace(
            max(_K_MIN, overall_best_k - spread),
            min(_K_MAX, overall_best_k + spread),
            41
        )
        for k in narrow_candidates:
            sc, avg_90, avg_50 = _score_k_on_cached(best_cached, float(k))
            if sc < overall_best_score:
                overall_best_score = sc
                overall_best_k = float(k)
                overall_best_cov90 = round(avg_90, 4)
                overall_best_cov50 = round(avg_50, 4)
        if _is_converged(overall_best_cov90, overall_best_cov50):
            converged = True

    np.random.set_state(saved)

    meta = {
        "iterations": total_iterations,
        "converged": converged,
        "windows": best_n_windows,
        "lookback_pct": overall_best_pct,
        "avg_cov_90": overall_best_cov90,
        "avg_cov_50": overall_best_cov50,
    }
    return float(np.clip(overall_best_k, _K_MIN, _K_MAX)), meta


def _apply_calib(paths, k):
    """Scale only the volatility component, preserving drift."""
    if abs(k - 1.0) < 0.02:
        return paths
    start   = paths[:, 0:1]
    log_ret = np.log(paths / start)
    drift   = log_ret.mean(axis=0, keepdims=True)
    scaled  = drift + (log_ret - drift) * k
    return start * np.exp(scaled)


def _fetch_portfolio(portfolio, fetch_period):
    """Fetch data for every ticker, align dates, return synthetic close series."""
    tickers = [p["ticker"].upper().strip() for p in portfolio]
    weights = np.array([float(p["weight"]) for p in portfolio])
    w_sum = weights.sum()
    if w_sum <= 0:
        raise ValueError("Portfolio weights must be positive and sum to > 0.")
    weights = weights / w_sum

    raw = yf.download(tickers, period=fetch_period, auto_adjust=True,
                      progress=False, threads=True)
    if raw.empty:
        raise ValueError("No data returned for portfolio tickers.")

    if isinstance(raw.columns, pd.MultiIndex):
        close_df = raw["Close"]
    else:
        close_df = raw[["Close"]].copy()
        close_df.columns = tickers

    close_df = close_df.dropna(axis=1, how="all")
    missing = set(tickers) - set(close_df.columns)
    if missing:
        raise ValueError(f"Tickers not found or no data: {', '.join(sorted(missing))}")

    close_df = close_df.dropna()
    close_df = close_df[tickers]

    log_ret_df = np.log(close_df / close_df.shift(1)).dropna()
    port_log_ret = (log_ret_df.values * weights).sum(axis=1)
    cum_ret = np.concatenate([[0.0], np.cumsum(port_log_ret)])
    port_close = 100.0 * np.exp(cum_ret)

    port_series = pd.Series(port_close, index=close_df.index, name="Close")

    parts = [f"{t} {w*100:.0f}%" for t, w in zip(tickers, weights)]
    label = " / ".join(parts)
    return port_series, label


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

            portfolio     = params.get("portfolio")
            ticker        = params.get("ticker", "AAPL").upper().strip()
            n_simulations = min(int(params.get("simulations", 500)), 10000)
            test_days     = min(int(params.get("test_days", 252)), 504)
            manual_k      = params.get("manual_k")  # optional float override

            # Fetch enough history: test period + at least 1 year of training data
            total_years = test_days / 252 + 2
            if total_years <= 3:
                fetch_period = "3y"
            elif total_years <= 5:
                fetch_period = "5y"
            else:
                fetch_period = "10y"

            is_portfolio = portfolio and isinstance(portfolio, list) and len(portfolio) > 0

            if is_portfolio:
                try:
                    close, label = _fetch_portfolio(portfolio, fetch_period)
                except ValueError as ve:
                    self._send_error(400, str(ve))
                    return
                ticker = label
                if len(close) < test_days + 100:
                    self._send_error(400, "Insufficient overlapping history for portfolio backtest.")
                    return
            else:
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
            auto_k, calib_meta = _compute_calib_factor(train)
            calib_k = float(np.clip(manual_k, _K_MIN, _K_MAX)) if manual_k is not None else auto_k

            np.random.seed(None)
            paths, model = _simulate(log_returns, current_price, n_simulations, test_days)

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
                "auto_k":        round(auto_k, 3),
                "manual_k":      round(float(manual_k), 3) if manual_k is not None else None,
                "calibration":   calib_meta,
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
