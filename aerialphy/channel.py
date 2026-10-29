"""Fading channel models: AWGN, flat Rayleigh MIMO and TDL-like tapped delay lines.

The TDL power-delay profiles below are *adapted* from the shape of the 3GPP
TR 38.901 TDL-A / TDL-C tables (normalised delays scaled by a delay spread).
They are meant to give realistic frequency selectivity for experiments, not
to be a conformance-grade implementation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# (normalised delay, power dB)
_TDL_A = np.array([
    [0.0000, -13.4], [0.3819, 0.0], [0.4025, -2.2], [0.5868, -4.0], [0.4610, -6.0],
    [0.5375, -8.2], [0.6708, -9.9], [0.5750, -10.5], [0.7618, -7.5], [1.5375, -15.9],
    [1.8978, -6.6], [2.2242, -16.7], [2.1718, -12.4], [2.4942, -15.2], [2.5119, -10.8],
    [3.0582, -11.3], [4.0810, -12.7], [4.4579, -16.2], [4.5695, -18.3], [4.7966, -18.9],
    [5.0066, -16.6], [5.3043, -19.9], [9.6586, -29.7],
])
_TDL_C = np.array([
    [0.0000, -4.4], [0.2099, -1.2], [0.2219, -3.5], [0.2329, -5.2], [0.2176, -2.5],
    [0.6366, 0.0], [0.6448, -2.2], [0.6560, -3.9], [0.6584, -7.4], [0.7935, -7.1],
    [0.8213, -10.7], [0.9336, -11.1], [1.2285, -5.1], [1.3083, -6.8], [2.1704, -8.7],
    [2.7105, -13.2], [4.2589, -13.9], [4.6003, -13.9], [5.4902, -15.8], [5.6077, -17.1],
    [6.3065, -16.0], [6.6374, -15.7], [7.0427, -21.6], [8.6523, -22.8],
])


def awgn(x: np.ndarray, noise_var: float, rng: np.random.Generator) -> np.ndarray:
    n = (rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape)) * np.sqrt(noise_var / 2)
    return x + n


def snr_db_to_noise_var(snr_db: float, signal_power: float = 1.0) -> float:
    return signal_power / (10 ** (snr_db / 10))


def crandn(shape, rng: np.random.Generator) -> np.ndarray:
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)


def exp_correlation(n: int, rho: float) -> np.ndarray:
    idx = np.arange(n)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def _sqrtm_psd(r: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh(r)
    return (v * np.sqrt(np.clip(w, 0, None))) @ v.conj().T


@dataclass
class TDLChannel:
    """MIMO tapped-delay-line channel evaluated in the frequency domain.

    profile: "A", "C" (TDL-like) or "EXP" (exponential PDP, 8 taps).
    delay_spread_s: RMS-ish delay spread scaling for the normalised delays.
    doppler_hz: max Doppler; taps evolve over OFDM symbols via sum-of-sinusoids.
    rho_tx / rho_rx: exponential antenna correlation (Kronecker model).
    """

    profile: str = "A"
    delay_spread_s: float = 100e-9
    doppler_hz: float = 0.0
    n_rx: int = 1
    n_tx: int = 1
    rho_tx: float = 0.0
    rho_rx: float = 0.0
    n_sinusoids: int = 16

    def pdp(self) -> tuple[np.ndarray, np.ndarray]:
        p = self.profile.upper()
        if p == "A":
            tab = _TDL_A
        elif p == "C":
            tab = _TDL_C
        elif p == "EXP":
            k = np.arange(8)
            tab = np.stack([k * 0.5, -10 * np.log10(np.e) * k * 0.5], axis=1)
        elif p == "FLAT":
            tab = np.array([[0.0, 0.0]])
        else:
            raise ValueError(f"unknown profile {self.profile}")
        delays = tab[:, 0] * self.delay_spread_s
        powers = 10 ** (tab[:, 1] / 10)
        return delays, powers / powers.sum()

    def sample_taps(self, times_s: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Tap gains of shape (n_times, n_taps, n_rx, n_tx)."""
        _, powers = self.pdp()
        n_taps = powers.size
        nt = len(times_s)
        if self.doppler_hz <= 0:
            g = crandn((1, n_taps, self.n_rx, self.n_tx), rng)
            g = np.repeat(g, nt, axis=0)
        else:
            m = self.n_sinusoids
            shape = (n_taps, self.n_rx, self.n_tx, m)
            alpha = rng.uniform(0, 2 * np.pi, shape)
            phi = rng.uniform(0, 2 * np.pi, shape)
            fd = self.doppler_hz * np.cos(alpha)
            ph = 2 * np.pi * fd[None] * times_s[:, None, None, None, None] + phi[None]
            g = np.exp(1j * ph).sum(axis=-1) / np.sqrt(m)
        g = g * np.sqrt(powers)[None, :, None, None]
        if self.rho_rx > 0 or self.rho_tx > 0:
            rr = _sqrtm_psd(exp_correlation(self.n_rx, self.rho_rx))
            rt = _sqrtm_psd(exp_correlation(self.n_tx, self.rho_tx))
            g = rr @ g @ rt.T
        return g

    def frequency_response(self, freqs_hz: np.ndarray, times_s: np.ndarray,
                           rng: np.random.Generator) -> np.ndarray:
        """H of shape (n_rx, n_tx, n_freq, n_times)."""
        delays, _ = self.pdp()
        taps = self.sample_taps(times_s, rng)  # (T, L, R, X)
        steer = np.exp(-2j * np.pi * np.outer(freqs_hz, delays))  # (F, L)
        h = np.einsum("fl,tlrx->rxft", steer, taps)
        return h

    def freq_correlation(self, freqs_hz: np.ndarray) -> np.ndarray:
        """Frequency-domain correlation matrix E[h(f1) h(f2)^*] implied by the PDP."""
        delays, powers = self.pdp()
        df = freqs_hz[:, None] - freqs_hz[None, :]
        return np.einsum("l,ijl->ij", powers, np.exp(-2j * np.pi * df[..., None] * delays))


def apply_time_domain(x: np.ndarray, taps: np.ndarray, delays_samples: np.ndarray) -> np.ndarray:
    """Apply a static SISO multipath channel with integer-sample delays to x."""
    out = np.zeros(x.size + int(delays_samples.max()), dtype=np.complex128)
    for g, d in zip(taps, delays_samples):
        out[d:d + x.size] += g * x
    return out[: x.size]
