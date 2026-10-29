"""Pilot-based channel estimation: LS, LMMSE and interpolation helpers."""
from __future__ import annotations

import numpy as np


def ls_estimate(rx_pilots: np.ndarray, tx_pilots: np.ndarray) -> np.ndarray:
    """Least-squares estimate at pilot REs: H_p = Y_p / X_p."""
    return rx_pilots / tx_pilots


def interp_frequency(h_p: np.ndarray, pilot_sc: np.ndarray, n_sc: int) -> np.ndarray:
    """Linear interpolation (with edge extrapolation) of pilot estimates across frequency.

    h_p: (n_pilots, n_cols). Returns (n_sc, n_cols).
    """
    k = np.arange(n_sc)
    out = np.empty((n_sc, h_p.shape[1]), dtype=np.complex128)
    for c in range(h_p.shape[1]):
        re = np.interp(k, pilot_sc, h_p[:, c].real)
        im = np.interp(k, pilot_sc, h_p[:, c].imag)
        out[:, c] = re + 1j * im
        # linear extrapolation at the edges instead of holding the value
        if pilot_sc[0] > 0:
            slope = (h_p[1, c] - h_p[0, c]) / (pilot_sc[1] - pilot_sc[0])
            lo = k < pilot_sc[0]
            out[lo, c] = h_p[0, c] + slope * (k[lo] - pilot_sc[0])
        if pilot_sc[-1] < n_sc - 1:
            slope = (h_p[-1, c] - h_p[-2, c]) / (pilot_sc[-1] - pilot_sc[-2])
            hi = k > pilot_sc[-1]
            out[hi, c] = h_p[-1, c] + slope * (k[hi] - pilot_sc[-1])
    return out


def interp_time(h_cols: np.ndarray, pilot_symbols: list[int], n_symbols: int) -> np.ndarray:
    """Expand estimates at DMRS symbols to all OFDM symbols.

    One DMRS symbol -> hold. Several -> linear interpolation, nearest hold at edges.
    """
    if len(pilot_symbols) == 1:
        return np.repeat(h_cols, n_symbols, axis=1)
    t = np.arange(n_symbols)
    out = np.empty((h_cols.shape[0], n_symbols), dtype=np.complex128)
    for k in range(h_cols.shape[0]):
        out[k] = np.interp(t, pilot_symbols, h_cols[k].real) + 1j * np.interp(t, pilot_symbols, h_cols[k].imag)
    return out


def lmmse_weights(r_full: np.ndarray, pilot_sc: np.ndarray, noise_var: float) -> np.ndarray:
    """W such that h_full = W @ h_ls (pilot LS estimates).

    r_full: (n_sc, n_sc) frequency correlation of the channel.
    """
    r_hp = r_full[:, pilot_sc]
    r_pp = r_full[np.ix_(pilot_sc, pilot_sc)]
    a = r_pp + noise_var * np.eye(pilot_sc.size)
    return np.linalg.solve(a.T, r_hp.T).T


def robust_freq_correlation(n_sc: int, scs_hz: float, max_delay_s: float) -> np.ndarray:
    """Correlation for a uniform PDP over [0, max_delay] (no knowledge of the true PDP)."""
    k = np.arange(n_sc)
    df = (k[:, None] - k[None, :]) * scs_hz
    x = 2 * np.pi * df * max_delay_s
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(np.abs(x) < 1e-12, 1.0 + 0j, (1 - np.exp(-1j * x)) / (1j * x))
    return r


def nmse_db(h_hat: np.ndarray, h: np.ndarray) -> float:
    return float(10 * np.log10(np.sum(np.abs(h_hat - h) ** 2) / np.sum(np.abs(h) ** 2)))


class PilotEstimator:
    """Estimate the full (n_sc, n_symbols) SISO grid from DMRS pilots."""

    def __init__(self, method: str = "ls", r_full: np.ndarray | None = None):
        method = method.lower()
        if method not in ("ls", "lmmse"):
            raise ValueError(method)
        if method == "lmmse" and r_full is None:
            raise ValueError("lmmse needs a frequency correlation matrix")
        self.method = method
        self.r_full = r_full
        self._cache: dict[tuple, np.ndarray] = {}

    def estimate(self, y_grid: np.ndarray, x_pilots: np.ndarray, pilot_sc: np.ndarray,
                 pilot_symbols: list[int], noise_var: float) -> np.ndarray:
        n_sc, n_sym = y_grid.shape
        y_p = y_grid[np.ix_(pilot_sc, pilot_symbols)]
        h_ls = ls_estimate(y_p, x_pilots)
        if self.method == "ls":
            h_cols = interp_frequency(h_ls, pilot_sc, n_sc)
        else:
            key = (round(float(noise_var), 12), pilot_sc.size, int(pilot_sc[0]))
            w = self._cache.get(key)
            if w is None:
                w = lmmse_weights(self.r_full, pilot_sc, noise_var)
                self._cache[key] = w
            h_cols = w @ h_ls
        return interp_time(h_cols, pilot_symbols, n_sym)
