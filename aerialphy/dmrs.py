"""DMRS pilot generation following the TS 38.211 pseudo-random (Gold) sequence."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .numerology import CarrierConfig, ResourceGrid

_NC = 1600


def gold_sequence(c_init: int, length: int) -> np.ndarray:
    """Length-``length`` binary sequence c(n) from TS 38.211 5.2.1."""
    total = length + _NC + 31
    x1 = np.zeros(total, dtype=np.uint8)
    x2 = np.zeros(total, dtype=np.uint8)
    x1[0] = 1
    for i in range(31):
        x2[i] = (c_init >> i) & 1
    for n in range(total - 31):
        x1[n + 31] = x1[n + 3] ^ x1[n]
        x2[n + 31] = x2[n + 3] ^ x2[n + 2] ^ x2[n + 1] ^ x2[n]
    return x1[_NC:_NC + length] ^ x2[_NC:_NC + length]


def dmrs_sequence(c_init: int, length: int) -> np.ndarray:
    c = gold_sequence(c_init, 2 * length).astype(float)
    return ((1 - 2 * c[0::2]) + 1j * (1 - 2 * c[1::2])) / np.sqrt(2)


def dmrs_c_init(slot: int, symbol: int, n_id: int = 0, n_scid: int = 0) -> int:
    return ((2 ** 17) * (14 * slot + symbol + 1) * (2 * n_id + 1) + 2 * n_id + n_scid) % (2 ** 31)


@dataclass
class DmrsConfig:
    """Type-1 (comb-2) DMRS. Each port uses a distinct comb offset (max 2 ports)."""

    symbols: list[int] = field(default_factory=lambda: [2, 11])
    n_id: int = 0

    def comb_offset(self, port: int) -> int:
        if port not in (0, 1):
            raise ValueError("comb-only DMRS supports ports 0 and 1")
        return port

    def pilot_subcarriers(self, carrier: CarrierConfig, port: int = 0) -> np.ndarray:
        return np.arange(self.comb_offset(port), carrier.n_sc, 2)

    def mask(self, carrier: CarrierConfig) -> np.ndarray:
        """All DMRS REs (both combs are reserved so data never collides)."""
        m = np.zeros((carrier.n_sc, carrier.n_symbols), dtype=bool)
        m[:, self.symbols] = True
        return m

    def values(self, carrier: CarrierConfig, slot: int, port: int = 0) -> np.ndarray:
        """Pilot values, shape (n_pilot_sc, n_dmrs_symbols)."""
        n = carrier.n_sc // 2
        cols = [dmrs_sequence(dmrs_c_init(slot, s, self.n_id), n) for s in self.symbols]
        return np.stack(cols, axis=1)


def build_grid(carrier: CarrierConfig, dmrs: DmrsConfig, n_ports: int, slot: int = 0) -> ResourceGrid:
    """Resource grid with DMRS inserted and the remaining REs marked as data."""
    grid = ResourceGrid(carrier, n_ports)
    dm = dmrs.mask(carrier)
    grid.mark(dm, ResourceGrid.DMRS)
    grid.mark(~dm, ResourceGrid.DATA)
    for p in range(n_ports):
        sc = dmrs.pilot_subcarriers(carrier, p)
        grid.values[p][np.ix_(sc, dmrs.symbols)] = dmrs.values(carrier, slot, p)
    return grid
