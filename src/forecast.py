"""Forecasting methods, written from scratch. Each takes a history array y (length n) and returns the next value."""
import itertools
import numpy as np

M = 4  # quarterly seasonality


def naive(y): return y[-1]
def seasonal_naive(y): return y[-M]
def sma4(y): return y[-4:].mean()


def regression(y, window=16):
    """Log-linear trend plus quarter dummies fitted by least squares on the last `window` quarters."""
    n = len(y); w = min(window, n); idx = np.arange(n - w, n)
    X = np.c_[np.ones(w), idx - idx.mean(), np.eye(M)[idx % M][:, 1:]]
    b, *_ = np.linalg.lstsq(X, np.log(y[-w:]), rcond=None)
    xn = np.r_[1, n - idx.mean(), np.eye(M)[n % M][1:]]
    resid = np.log(y[-w:]) - X @ b
    return float(np.exp(xn @ b + 0.5 * resid.var()))


GRID = list(itertools.product([0.1, 0.3, 0.5, 0.8], [0.0, 0.1, 0.3], [0.0, 0.2, 0.5]))


def _hw(ly, a, b, g, phi=0.9):
    """Additive damped-trend Holt-Winters on log values. Returns (one-step SSE, next forecast)."""
    n = len(ly)
    l = ly[:M].mean(); tr = (ly[M:2 * M].mean() - ly[:M].mean()) / M if n >= 2 * M else 0.0
    s = list(ly[:M] - l)
    sse = 0.0
    for t in range(M, n):
        f = l + phi * tr + s[t - M]
        e = ly[t] - f; sse += e * e
        l_new = l + phi * tr + a * e
        tr = phi * tr + b * a * e
        s.append(s[t - M] + g * (1 - a) * e)
        l = l_new
    return sse, l + phi * tr + s[n - M]


def holt_winters(y):
    ly = np.log(y)
    best = min(((_hw(ly, *p)[0], p) for p in GRID), key=lambda z: z[0])[1]
    sse, f = _hw(ly, *best)
    sigma2 = sse / max(len(ly) - M, 1)
    return float(np.exp(f + 0.5 * sigma2))


def combo(y): return (holt_winters(y) + regression(y) + seasonal_naive(y)) / 3


METHODS = {"naive": naive, "seasonal naive": seasonal_naive, "4-quarter average": sma4,
           "trend + season regression": regression, "Holt-Winters (log, damped)": holt_winters, "combination of 3": combo}
