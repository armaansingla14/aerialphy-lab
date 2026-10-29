"""Gray-mapped QAM modulation with hard and max-log LLR demapping (TS 38.211 5.1)."""
from __future__ import annotations

import numpy as np

_BITS_PER_SYMBOL = {"bpsk": 1, "qpsk": 2, "16qam": 4, "64qam": 6}


def bits_per_symbol(scheme: str) -> int:
    try:
        return _BITS_PER_SYMBOL[scheme.lower()]
    except KeyError as exc:
        raise ValueError(f"unknown modulation {scheme!r}") from exc


def _pam_levels(bits: np.ndarray) -> np.ndarray:
    """Recursive NR-style Gray PAM: b0 picks sign, following bits refine amplitude."""
    m = bits.shape[-1]
    if m == 1:
        return 1 - 2 * bits[..., 0].astype(float)
    inner = _pam_levels(bits[..., 1:])
    amp = 2 ** (m - 1) - inner  # reflect to keep Gray property
    return (1 - 2 * bits[..., 0].astype(float)) * amp


def constellation(scheme: str) -> np.ndarray:
    """Return the unit-average-energy constellation indexed by the integer bit label."""
    k = bits_per_symbol(scheme)
    labels = np.arange(2 ** k)
    bits = ((labels[:, None] >> np.arange(k - 1, -1, -1)) & 1).astype(np.uint8)
    return modulate(bits.reshape(-1), scheme)


def modulate(bits: np.ndarray, scheme: str) -> np.ndarray:
    k = bits_per_symbol(scheme)
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if bits.size % k:
        raise ValueError(f"bit count {bits.size} not a multiple of {k}")
    b = bits.reshape(-1, k)
    if scheme.lower() == "bpsk":
        return (1 - 2 * b[:, 0].astype(float)).astype(np.complex128)
    half = k // 2
    # 38.211 interleaves I/Q bits: b0,b2,b4 -> I ; b1,b3,b5 -> Q
    i = _pam_levels(b[:, 0::2][:, :half])
    q = _pam_levels(b[:, 1::2][:, :half])
    m = 2 ** half
    norm = np.sqrt(2 * (m * m - 1) / 3)
    return (i + 1j * q) / norm


def demodulate_llr(symbols: np.ndarray, scheme: str, noise_var) -> np.ndarray:
    """Max-log LLRs (positive => bit 0). ``noise_var`` may be scalar or per-symbol."""
    k = bits_per_symbol(scheme)
    const = constellation(scheme)
    labels = np.arange(const.size)
    bits = ((labels[:, None] >> np.arange(k - 1, -1, -1)) & 1).astype(bool)
    y = np.asarray(symbols).reshape(-1)
    nv = np.broadcast_to(np.asarray(noise_var, dtype=float), y.shape)
    d2 = np.abs(y[:, None] - const[None, :]) ** 2
    llr = np.empty((y.size, k))
    for j in range(k):
        d0 = d2[:, ~bits[:, j]].min(axis=1)
        d1 = d2[:, bits[:, j]].min(axis=1)
        llr[:, j] = (d1 - d0) / np.maximum(nv, 1e-12)
    return llr.reshape(-1)


def demodulate_hard(symbols: np.ndarray, scheme: str) -> np.ndarray:
    return (demodulate_llr(symbols, scheme, 1.0) < 0).astype(np.uint8)
