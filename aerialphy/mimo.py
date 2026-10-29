"""Linear MIMO detection, precoding/beamforming and spectral-efficiency helpers.

Shapes follow numpy batching: H (..., n_rx, n_tx), y (..., n_rx).
"""
from __future__ import annotations

import numpy as np


def _herm(a: np.ndarray) -> np.ndarray:
    return np.swapaxes(a.conj(), -1, -2)


def zf_detect(h: np.ndarray, y: np.ndarray, noise_var: float):
    """Zero-forcing: x = (H^H H)^-1 H^H y. Returns (x_hat, post-equaliser noise var)."""
    hh = _herm(h)
    g = np.linalg.solve(hh @ h, hh)  # (..., n_tx, n_rx)
    x = (g @ y[..., None])[..., 0]
    nv = noise_var * np.sum(np.abs(g) ** 2, axis=-1)
    return x, nv


def mmse_detect(h: np.ndarray, y: np.ndarray, noise_var: float, es: float = 1.0):
    """Unbiased linear MMSE detection. Returns (x_hat, effective noise variance per stream)."""
    n_tx = h.shape[-1]
    hh = _herm(h)
    a = hh @ h + (noise_var / es) * np.eye(n_tx)
    w = np.linalg.solve(a, hh)
    x = (w @ y[..., None])[..., 0]
    # bias removal: diag(W H) = mu
    mu = np.real(np.einsum("...ij,...ji->...i", w, h))
    mu = np.clip(mu, 1e-12, None)
    x = x / mu
    # post-detection SINR = mu / (1 - mu) for unit-energy symbols
    nv = es * (1 - mu) / mu
    return x, np.clip(nv, 1e-12, None)


def ml_detect_bruteforce(h: np.ndarray, y: np.ndarray, const: np.ndarray) -> np.ndarray:
    """Exhaustive ML detection (only for tiny configs, used as a test oracle)."""
    n_tx = h.shape[-1]
    grids = np.meshgrid(*([const] * n_tx), indexing="ij")
    cand = np.stack([g.reshape(-1) for g in grids], axis=-1)  # (M^n, n_tx)
    hx = np.einsum("...rt,ct->...cr", h, cand)
    d = np.sum(np.abs(y[..., None, :] - hx) ** 2, axis=-1)
    return cand[np.argmin(d, axis=-1)]


# ---------------------------------------------------------------- beamforming

def svd_precoder(h: np.ndarray, n_streams: int) -> np.ndarray:
    """Right singular vectors of H for the strongest ``n_streams`` modes."""
    _, _, vh = np.linalg.svd(h)
    return _herm(vh)[..., :, :n_streams]


def water_filling(gains: np.ndarray, total_power: float) -> np.ndarray:
    """Water-filling power allocation over parallel channels with gain g_i (SNR per unit power)."""
    g = np.asarray(gains, dtype=float)
    order = np.argsort(g)[::-1]
    gs = g[order]
    p = np.zeros_like(gs)
    for k in range(gs.size, 0, -1):
        mu = (total_power + np.sum(1 / gs[:k])) / k
        pk = mu - 1 / gs[:k]
        if np.all(pk > 0):
            p[:k] = pk
            break
    out = np.zeros_like(p)
    out[order] = p
    return out


def capacity_open_loop(h: np.ndarray, snr: float) -> np.ndarray:
    """log2 det(I + snr/N_t H H^H): equal power, no CSI at transmitter."""
    n_rx, n_tx = h.shape[-2:]
    m = np.eye(n_rx) + (snr / n_tx) * (h @ _herm(h))
    return np.real(np.log2(np.linalg.det(m)))


def capacity_svd_waterfilling(h: np.ndarray, snr: float) -> np.ndarray:
    """Closed-loop capacity with SVD beamforming and water-filling (full CSIT)."""
    s = np.linalg.svd(h, compute_uv=False)
    lead = s.shape[:-1]
    s2 = (s ** 2).reshape(-1, s.shape[-1])
    out = np.empty(s2.shape[0])
    for i, g in enumerate(s2):
        p = water_filling(g * snr, 1.0)
        out[i] = np.sum(np.log2(1 + p * g * snr))
    return out.reshape(lead)


def se_single_stream(h: np.ndarray, w: np.ndarray, snr: float) -> np.ndarray:
    """SE of one stream with unit-norm precoder w and MRC combining: log2(1 + snr ||H w||^2)."""
    g = np.sum(np.abs((h @ w[..., None])[..., 0]) ** 2, axis=-1)
    return np.log2(1 + snr * g)


def mrt_precoder(h: np.ndarray) -> np.ndarray:
    """Dominant eigen-beam (MRT generalised to multi-rx: principal right singular vector)."""
    return svd_precoder(h, 1)[..., 0]


def dft_codebook(n_tx: int, oversample: int = 4) -> np.ndarray:
    """Columns are unit-norm DFT beams for a half-wavelength ULA (Type-I-like codebook)."""
    n_beams = n_tx * oversample
    k = np.arange(n_tx)[:, None]
    b = np.arange(n_beams)[None, :]
    return np.exp(2j * np.pi * k * b / n_beams) / np.sqrt(n_tx)


def codebook_select(h: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """Pick the beam maximising ||H w||^2 for each channel in the batch."""
    g = np.sum(np.abs(h @ codebook) ** 2, axis=-2)  # (..., n_beams)
    idx = np.argmax(g, axis=-1)
    return np.moveaxis(codebook[:, idx], 0, -1)


def ula_steering(n: int, theta_rad: np.ndarray) -> np.ndarray:
    k = np.arange(n)
    return np.exp(1j * np.pi * np.outer(np.sin(np.atleast_1d(theta_rad)), k))


def ricean_ula_channel(n_rx: int, n_tx: int, k_factor_db: float, rng: np.random.Generator,
                       batch: int = 1) -> np.ndarray:
    """LOS (ULA steering) + Rayleigh scatter channel, normalised E|h_ij|^2 = 1."""
    k = 10 ** (k_factor_db / 10)
    th_t = rng.uniform(-np.pi / 3, np.pi / 3, batch)
    th_r = rng.uniform(-np.pi / 3, np.pi / 3, batch)
    a_t = ula_steering(n_tx, th_t)
    a_r = ula_steering(n_rx, th_r)
    los = a_r[:, :, None] * a_t[:, None, :].conj()
    nlos = (rng.standard_normal((batch, n_rx, n_tx)) + 1j * rng.standard_normal((batch, n_rx, n_tx))) / np.sqrt(2)
    return np.sqrt(k / (k + 1)) * los + np.sqrt(1 / (k + 1)) * nlos
