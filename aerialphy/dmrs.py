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
    """Type-1 DMRS: two CDM groups (comb offsets 0/1), length-2 FD-OCC -> up to 4 ports.

    Port p belongs to CDM group p // 2 and uses OCC w_f = [1, (-1)^(p % 2)] over
    pilot pairs k = 4n + 2k' + group, k' in {0, 1}, as in TS 38.211 Table 7.4.1.1.2-1.
    The full DMRS symbols are reserved (no data multiplexed on them).
    """

    symbols: list[int] = field(default_factory=lambda: [2, 11])
    n_id: int = 0

    @staticmethod
    def cdm_group(port: int) -> int:
        if not 0 <= port <= 3:
            raise ValueError("type-1 DMRS supports ports 0..3")
        return port // 2

    @staticmethod
    def occ(port: int) -> np.ndarray:
        return np.array([1.0, 1.0 if port % 2 == 0 else -1.0])

    def pilot_subcarriers(self, carrier: CarrierConfig, port: int = 0) -> np.ndarray:
        return np.arange(self.cdm_group(port), carrier.n_sc, 2)

    def mask(self, carrier: CarrierConfig) -> np.ndarray:
        m = np.zeros((carrier.n_sc, carrier.n_symbols), dtype=bool)
        m[:, self.symbols] = True
        return m

    def base_values(self, carrier: CarrierConfig, slot: int) -> np.ndarray:
        """Unspread pilot sequence r(m), shape (n_sc // 2, n_dmrs_symbols)."""
        n = carrier.n_sc // 2
        cols = [dmrs_sequence(dmrs_c_init(slot, s, self.n_id), n) for s in self.symbols]
        return np.stack(cols, axis=1)

    def values(self, carrier: CarrierConfig, slot: int, port: int = 0) -> np.ndarray:
        """Pilot values for ``port`` incl. OCC, shape (n_pilot_sc, n_dmrs_symbols)."""
        r = self.base_values(carrier, slot)
        w = np.tile(self.occ(port), r.shape[0] // 2 + 1)[: r.shape[0]]
        return r * w[:, None]

    def ls_estimates(self, y: np.ndarray, carrier: CarrierConfig, slot: int, port: int,
                     n_ports: int):
        """LS pilot estimates for ``port`` from one receive grid y (n_sc, n_symbols).

        If the port shares its CDM group with another active port, adjacent pilot
        pairs are despread with the OCC and the estimate is placed at the pair centre.
        Returns (subcarrier positions, estimates (n_pos, n_dmrs_symbols), noise scale).
        """
        sc = self.pilot_subcarriers(carrier, port)
        r = self.base_values(carrier, slot)
        z = y[np.ix_(sc, self.symbols)] / r
        shared = (port ^ 1) < n_ports
        if not shared:
            return sc.astype(float), z, 1.0
        npair = sc.size // 2
        z = z[: 2 * npair].reshape(npair, 2, -1)
        h = (z * self.occ(port)[None, :, None]).mean(axis=1)
        pos = sc[: 2 * npair].reshape(npair, 2).mean(axis=1)
        return pos, h, 0.5


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
