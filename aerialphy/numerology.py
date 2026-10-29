"""5G NR numerology and resource-grid helpers (loosely following TS 38.211).

Only the parts needed for link-level simulation are modelled: subcarrier
spacing, slot timing, FFT sizing and a resource grid with RE allocation maps.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SUBCARRIERS_PER_RB = 12
SYMBOLS_PER_SLOT_NORMAL_CP = 14


@dataclass(frozen=True)
class Numerology:
    """NR numerology ``mu`` (0..4). SCS = 15 kHz * 2**mu."""

    mu: int = 1

    def __post_init__(self) -> None:
        if not 0 <= self.mu <= 4:
            raise ValueError(f"mu must be in 0..4, got {self.mu}")

    @property
    def scs_hz(self) -> float:
        return 15e3 * (2 ** self.mu)

    @property
    def slots_per_subframe(self) -> int:
        return 2 ** self.mu

    @property
    def slots_per_frame(self) -> int:
        return 10 * self.slots_per_subframe

    @property
    def slot_duration_s(self) -> float:
        return 1e-3 / self.slots_per_subframe

    @property
    def symbol_duration_s(self) -> float:
        """Useful symbol duration (without CP)."""
        return 1.0 / self.scs_hz


@dataclass
class CarrierConfig:
    """Carrier bandwidth part description."""

    numerology: Numerology = field(default_factory=Numerology)
    n_rb: int = 24
    n_symbols: int = SYMBOLS_PER_SLOT_NORMAL_CP

    @property
    def n_sc(self) -> int:
        return self.n_rb * SUBCARRIERS_PER_RB

    @property
    def fft_size(self) -> int:
        """Smallest power of two >= 1.2x occupied subcarriers (guard bands)."""
        n = 128
        while n < int(np.ceil(self.n_sc * 1.2)):
            n *= 2
        return n

    @property
    def sample_rate_hz(self) -> float:
        return self.fft_size * self.numerology.scs_hz

    @property
    def cp_length(self) -> int:
        """Normal CP length in samples (144/2048 of the FFT size)."""
        return int(round(self.fft_size * 144 / 2048))

    @property
    def bandwidth_hz(self) -> float:
        return self.n_sc * self.numerology.scs_hz


class ResourceGrid:
    """Complex RE grid of shape (n_ports, n_sc, n_symbols) with an RE-type map."""

    EMPTY, DATA, DMRS = 0, 1, 2

    def __init__(self, carrier: CarrierConfig, n_ports: int = 1):
        self.carrier = carrier
        self.n_ports = n_ports
        self.values = np.zeros((n_ports, carrier.n_sc, carrier.n_symbols), dtype=np.complex128)
        self.re_type = np.zeros((carrier.n_sc, carrier.n_symbols), dtype=np.int8)

    def mark(self, mask: np.ndarray, kind: int) -> None:
        if mask.shape != self.re_type.shape:
            raise ValueError("mask shape mismatch")
        if np.any(self.re_type[mask] != self.EMPTY):
            raise ValueError("RE collision while allocating")
        self.re_type[mask] = kind

    def data_mask(self) -> np.ndarray:
        return self.re_type == self.DATA

    def dmrs_mask(self) -> np.ndarray:
        return self.re_type == self.DMRS

    def num_data_re(self) -> int:
        return int(self.data_mask().sum())

    def map_data(self, symbols: np.ndarray, port: int = 0) -> None:
        mask = self.data_mask()
        if symbols.size != mask.sum():
            raise ValueError(f"expected {mask.sum()} data symbols, got {symbols.size}")
        # frequency-first mapping as in 38.211 (subcarrier index fastest)
        grid = self.values[port].T
        grid[mask.T] = symbols
        self.values[port] = grid.T

    def extract_data(self, grid: np.ndarray) -> np.ndarray:
        """Extract data REs from a (..., n_sc, n_symbols) grid, frequency first."""
        mask = self.data_mask()
        g = np.swapaxes(grid, -1, -2)
        return g[..., mask.T]
