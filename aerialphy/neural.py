"""Small PyTorch models: channel estimator (MLP), grid denoiser (DnCNN-style CNN)
and a modulation classifier, plus synthetic data generators for them.

All training data is synthetic, drawn from the TDL-like channel models in
``aerialphy.channel``. Models are intentionally tiny so they train on a CPU in
a minute or two.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from .channel import TDLChannel, crandn
from .dmrs import DmrsConfig
from .estimation import interp_frequency, interp_time
from .link import subcarrier_freqs, symbol_times
from .modulation import constellation
from .numerology import CarrierConfig, Numerology


# ------------------------------------------------------------ channel-estimation data

@dataclass
class ChEstScenario:
    carrier: CarrierConfig = field(default_factory=lambda: CarrierConfig(Numerology(1), n_rb=12))
    dmrs: DmrsConfig = field(default_factory=DmrsConfig)
    profiles: tuple[str, ...] = ("A", "C", "EXP")
    delay_spread_s: tuple[float, float] = (30e-9, 300e-9)
    doppler_hz: tuple[float, float] = (0.0, 200.0)
    snr_db: tuple[float, float] = (0.0, 25.0)


def generate_chest_batch(sc: ChEstScenario, n: int, rng: np.random.Generator,
                         profile: str | None = None, snr_db: float | None = None,
                         delay_spread_s: float | None = None, doppler_hz: float | None = None):
    """SISO samples. Returns dict with LS pilots, LS-interpolated grid, true grid, SNR, pilot positions."""
    c, d = sc.carrier, sc.dmrs
    freqs, times = subcarrier_freqs(c), symbol_times(c)
    pilot_sc = d.pilot_subcarriers(c, 0)
    x_p = d.values(c, 0, 0)
    h_true = np.empty((n, c.n_sc, c.n_symbols), dtype=np.complex128)
    h_ls = np.empty((n, pilot_sc.size, len(d.symbols)), dtype=np.complex128)
    snrs = np.empty(n)
    for i in range(n):
        prof = profile or sc.profiles[rng.integers(len(sc.profiles))]
        ds = delay_spread_s if delay_spread_s is not None else rng.uniform(*sc.delay_spread_s)
        fd = doppler_hz if doppler_hz is not None else rng.uniform(*sc.doppler_hz)
        snr = snr_db if snr_db is not None else rng.uniform(*sc.snr_db)
        ch = TDLChannel(profile=prof, delay_spread_s=ds, doppler_hz=fd)
        h = ch.frequency_response(freqs, times, rng)[0, 0]
        nv = 10 ** (-snr / 10)
        y_p = h[np.ix_(pilot_sc, d.symbols)] * x_p + crandn(x_p.shape, rng) * np.sqrt(nv)
        h_true[i] = h
        h_ls[i] = y_p / x_p
        snrs[i] = snr
    return {"h_true": h_true, "h_ls": h_ls, "snr_db": snrs, "pilot_sc": pilot_sc}


def ls_interp_grid(h_ls: np.ndarray, pilot_sc: np.ndarray, n_sc: int, symbols: list[int], n_sym: int):
    return interp_time(interp_frequency(h_ls, pilot_sc, n_sc), symbols, n_sym)


def c2r(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.stack([x.real, x.imag], axis=1).astype(np.float32))


def r2c(t: torch.Tensor) -> np.ndarray:
    a = t.detach().cpu().numpy().astype(np.float64)
    return a[:, 0] + 1j * a[:, 1]


# ------------------------------------------------------------ models

class MLPChannelEstimator(nn.Module):
    """Maps LS pilots of one DMRS symbol (+ SNR) to the full-band response of that symbol."""

    def __init__(self, n_pilots: int, n_sc: int, hidden: int = 256):
        super().__init__()
        self.n_sc = n_sc
        self.net = nn.Sequential(
            nn.Linear(2 * n_pilots + 1, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 2 * n_sc),
        )

    def forward(self, pilots: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        # pilots: (B, 2, n_pilots)
        x = torch.cat([pilots.flatten(1), snr_db[:, None] / 20.0], dim=1)
        return self.net(x).view(-1, 2, self.n_sc)


class GridDenoiser(nn.Module):
    """DnCNN-style residual CNN refining an LS-interpolated (2, n_sc, n_sym) grid."""

    def __init__(self, width: int = 16, depth: int = 5):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(3, width, 3, padding=1), nn.ReLU()]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(width, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU()]
        layers += [nn.Conv2d(width, 2, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, grid: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        snr_map = (snr_db / 20.0)[:, None, None, None].expand(-1, 1, *grid.shape[2:])
        return grid - self.net(torch.cat([grid, snr_map], dim=1))


class ModulationClassifier(nn.Module):
    """1-D CNN over raw IQ samples (2, n) with global average pooling."""

    def __init__(self, n_classes: int = 4, width: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(8, width, 7, padding=3), nn.ReLU(),
            nn.Conv1d(width, width, 5, padding=2), nn.ReLU(),
            nn.Conv1d(width, width, 3, padding=1), nn.ReLU(),
        )
        self.head = nn.Linear(2 * width, n_classes)

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        # extra input channels: amplitude, power and the squared / fourth-power signal.
        # z^2 and z^4 collapse BPSK/QPSK onto (slowly rotating) points, which local
        # convolutions can detect despite the unknown phase and residual CFO.
        i, q = iq[:, 0:1], iq[:, 1:2]
        amp2 = i ** 2 + q ** 2
        z2r, z2i = i ** 2 - q ** 2, 2 * i * q
        z4r, z4i = z2r ** 2 - z2i ** 2, 2 * z2r * z2i
        x = torch.cat([iq, torch.sqrt(amp2 + 1e-9), amp2, z2r, z2i, z4r / 2, z4i / 2], dim=1)
        f = self.features(x)
        pooled = torch.cat([f.mean(dim=2), f.amax(dim=2)], dim=1)
        return self.head(pooled)


# ------------------------------------------------------------ modulation data

MOD_CLASSES = ("bpsk", "qpsk", "16qam", "64qam")


def generate_mod_batch(n: int, n_samples: int, rng: np.random.Generator,
                       snr_db: float | tuple[float, float] = (-5.0, 20.0), cfo_max: float = 1e-3):
    """Random-phase, small-CFO, AWGN-impaired symbol streams. Returns (iq (n,2,L), labels, snr)."""
    labels = rng.integers(0, len(MOD_CLASSES), n)
    out = np.empty((n, n_samples), dtype=np.complex128)
    snrs = np.empty(n)
    for i, lab in enumerate(labels):
        const = constellation(MOD_CLASSES[lab])
        s = const[rng.integers(0, const.size, n_samples)]
        snr = snr_db if np.isscalar(snr_db) else rng.uniform(*snr_db)
        cfo = rng.uniform(-cfo_max, cfo_max)
        ph = rng.uniform(0, 2 * np.pi) + 2 * np.pi * cfo * np.arange(n_samples)
        out[i] = s * np.exp(1j * ph) + crandn(n_samples, rng) * np.sqrt(10 ** (-snr / 10))
        snrs[i] = snr
    # per-example power normalisation (receiver AGC)
    out /= np.sqrt(np.mean(np.abs(out) ** 2, axis=1, keepdims=True))
    return np.stack([out.real, out.imag], axis=1).astype(np.float32), labels, snrs


def cumulant_features(iq: np.ndarray) -> np.ndarray:
    """Normalised fourth-order cumulants |C40|, C42 (classic feature-based baseline)."""
    z = iq[:, 0] + 1j * iq[:, 1]
    c21 = np.mean(np.abs(z) ** 2, axis=1)
    c20 = np.mean(z ** 2, axis=1)
    c40 = np.mean(z ** 4, axis=1) - 3 * c20 ** 2
    c42 = np.mean(np.abs(z) ** 4, axis=1) - np.abs(c20) ** 2 - 2 * c21 ** 2
    return np.stack([np.abs(c40) / c21 ** 2, c42 / c21 ** 2], axis=1)


def theoretical_cumulants() -> np.ndarray:
    feats = []
    for m in MOD_CLASSES:
        c = constellation(m)
        z = c[None, :]
        iq = np.stack([z.real, z.imag], axis=1)
        feats.append(cumulant_features(iq)[0])
    return np.array(feats)


def cumulant_classify(iq: np.ndarray, snr_db: np.ndarray | None = None) -> np.ndarray:
    """Nearest theoretical cumulant vector. Cumulants of Gaussian noise vanish, so an
    estimate of the SNR is used to undo the dilution caused by AGC normalisation."""
    f = cumulant_features(iq)
    if snr_db is not None:
        lin = 10 ** (np.asarray(snr_db) / 10)
        f = f * ((1 + lin) / lin)[:, None] ** 2
    ref = theoretical_cumulants()
    # C42 is phase/CFO invariant; |C40| degrades under residual CFO, so it is down-weighted
    w = np.array([0.25, 1.0])
    d = np.sum(w * (f[:, None, :] - ref[None, :, :]) ** 2, axis=2)
    return np.argmin(d, axis=1)
