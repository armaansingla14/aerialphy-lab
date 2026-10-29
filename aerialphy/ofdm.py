"""CP-OFDM modulator / demodulator."""
from __future__ import annotations

import numpy as np

from .numerology import CarrierConfig


def _sc_indices(carrier: CarrierConfig) -> np.ndarray:
    """FFT bin indices of the occupied subcarriers, centred around DC."""
    n = carrier.n_sc
    k = np.arange(n) - n // 2
    return np.mod(k, carrier.fft_size)


def ofdm_modulate(grid: np.ndarray, carrier: CarrierConfig) -> np.ndarray:
    """grid: (..., n_sc, n_symbols) -> time samples (..., n_symbols * (fft + cp))."""
    nfft, cp = carrier.fft_size, carrier.cp_length
    lead = grid.shape[:-2]
    nsym = grid.shape[-1]
    freq = np.zeros(lead + (nfft, nsym), dtype=np.complex128)
    freq[..., _sc_indices(carrier), :] = grid
    t = np.fft.ifft(freq, axis=-2) * np.sqrt(nfft)
    t = np.concatenate([t[..., -cp:, :], t], axis=-2)
    return np.swapaxes(t, -1, -2).reshape(lead + (nsym * (nfft + cp),))


def ofdm_demodulate(x: np.ndarray, carrier: CarrierConfig, n_symbols: int | None = None) -> np.ndarray:
    nfft, cp = carrier.fft_size, carrier.cp_length
    nsym = n_symbols or carrier.n_symbols
    lead = x.shape[:-1]
    blk = x[..., : nsym * (nfft + cp)].reshape(lead + (nsym, nfft + cp))
    blk = blk[..., cp:]
    f = np.fft.fft(blk, axis=-1) / np.sqrt(nfft)
    f = np.swapaxes(f, -1, -2)
    return f[..., _sc_indices(carrier), :]
