"""Pilot-based channel estimation: LS, LMMSE and interpolation helpers."""
from __future__ import annotations

import numpy as np


def ls_estimate(rx_pilots: np.ndarray, tx_pilots: np.ndarray) -> np.ndarray:
    """Least-squares estimate at pilot REs: H_p = Y_p / X_p."""
    return rx_pilots / tx_pilots


def linear_interp_matrix(pos: np.ndarray, n: int, extrapolate: bool = True) -> np.ndarray:
    """Matrix M (n, len(pos)) so that M @ v linearly interpolates samples v taken at ``pos``.

    Outside [pos[0], pos[-1]] the end segments are extended linearly (``extrapolate``)
    or the end value is held.
    """
    pos = np.asarray(pos, dtype=float)
    k = np.arange(n, dtype=float)
    m = np.zeros((n, pos.size))
    if pos.size == 1:
        m[:, 0] = 1.0
        return m
    seg = np.clip(np.searchsorted(pos, k, side="right") - 1, 0, pos.size - 2)
    t = (k - pos[seg]) / (pos[seg + 1] - pos[seg])
    if not extrapolate:
        t = np.clip(t, 0.0, 1.0)
    rows = np.arange(n)
    m[rows, seg] += 1 - t
    m[rows, seg + 1] += t
    return m


def interp_frequency(h_p: np.ndarray, pilot_pos: np.ndarray, n_sc: int) -> np.ndarray:
    """Linear interpolation across frequency with linear edge extrapolation.

    h_p: (..., n_pilots, n_cols). Returns (..., n_sc, n_cols).
    """
    return linear_interp_matrix(pilot_pos, n_sc, extrapolate=True) @ h_p


def interp_time(h_cols: np.ndarray, pilot_symbols: list[int], n_symbols: int) -> np.ndarray:
    """Expand estimates at DMRS symbols (..., n_sc, n_dmrs) to all symbols (..., n_sc, n_sym).

    One DMRS symbol -> hold; several -> linear interpolation, hold outside.
    """
    m = linear_interp_matrix(np.asarray(pilot_symbols), n_symbols, extrapolate=False)
    return h_cols @ m.T


def lmmse_weights(corr_fn, n_sc: int, pilot_pos: np.ndarray, noise_var: float) -> np.ndarray:
    """W such that h_full = W @ h_ls (pilot LS estimates).

    corr_fn(df_in_subcarriers) -> E[h(k + df) h(k)^*]; pilot positions may be fractional
    (e.g. OCC pair centres).
    """
    k = np.arange(n_sc, dtype=float)
    r_hp = corr_fn(k[:, None] - pilot_pos[None, :])
    r_pp = corr_fn(pilot_pos[:, None] - pilot_pos[None, :])
    a = r_pp + noise_var * np.eye(pilot_pos.size)
    return np.linalg.solve(a.T, r_hp.T).T


def pdp_correlation(delays_s: np.ndarray, powers: np.ndarray, scs_hz: float):
    """Correlation function (in subcarrier offsets) implied by a known power-delay profile."""
    def fn(dk):
        dk = np.asarray(dk, dtype=float)
        return np.tensordot(np.exp(-2j * np.pi * dk[..., None] * scs_hz * delays_s), powers, axes=([-1], [0]))
    return fn


def uniform_pdp_correlation(scs_hz: float, max_delay_s: float):
    """Robust correlation for a uniform PDP on [0, max_delay] (true PDP unknown)."""
    def fn(dk):
        x = 2 * np.pi * np.asarray(dk, dtype=float) * scs_hz * max_delay_s
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(np.abs(x) < 1e-12, 1.0 + 0j, (1 - np.exp(-1j * x)) / (1j * x))
    return fn


def nmse_db(h_hat: np.ndarray, h: np.ndarray) -> float:
    return float(10 * np.log10(np.sum(np.abs(h_hat - h) ** 2) / np.sum(np.abs(h) ** 2)))


class PilotEstimator:
    """Turn LS pilot estimates into a full (n_sc, n_symbols) channel estimate."""

    def __init__(self, method: str = "ls", corr_fn=None):
        method = method.lower()
        if method not in ("ls", "lmmse"):
            raise ValueError(method)
        if method == "lmmse" and corr_fn is None:
            raise ValueError("lmmse needs a correlation function")
        self.method = method
        self.corr_fn = corr_fn
        self._cache: dict[tuple, np.ndarray] = {}

    def estimate(self, pos: np.ndarray, h_ls: np.ndarray, n_sc: int, pilot_symbols: list[int],
                 n_symbols: int, noise_var: float) -> np.ndarray:
        if self.method == "ls":
            h_cols = interp_frequency(h_ls, pos, n_sc)
        else:
            key = (round(float(noise_var), 12), pos.size, float(pos[0]), n_sc)
            w = self._cache.get(key)
            if w is None:
                w = lmmse_weights(self.corr_fn, n_sc, pos, noise_var)
                self._cache[key] = w
            h_cols = w @ h_ls
        return interp_time(h_cols, pilot_symbols, n_symbols)
