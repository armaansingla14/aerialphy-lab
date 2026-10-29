"""Slot-level PDSCH-like link simulation (frequency-domain channel per RE).

Chain: TB + CRC24A -> QC-LDPC -> QAM -> layer mapping -> grid with DMRS ->
MIMO TDL channel + AWGN -> per-port DMRS LS/LMMSE estimation (or perfect CSI)
-> ZF/MMSE detection -> max-log LLR -> min-sum decoding -> CRC check.

The channel is applied per resource element, which is exact for CP-OFDM as
long as the delay spread fits within the cyclic prefix (verified in tests by
running the full time-domain OFDM chain).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .channel import TDLChannel, crandn
from .coding import QCLDPC, crc_attach, crc_check
from .dmrs import DmrsConfig, build_grid
from .estimation import PilotEstimator, pdp_correlation
from .mimo import mmse_detect, zf_detect
from .modulation import bits_per_symbol, demodulate_llr, modulate
from .numerology import CarrierConfig

_CODES: dict[int, QCLDPC] = {}


def get_code(z: int = 32) -> QCLDPC:
    if z not in _CODES:
        _CODES[z] = QCLDPC(z=z)
    return _CODES[z]


@dataclass
class LinkConfig:
    carrier: CarrierConfig = field(default_factory=CarrierConfig)
    dmrs: DmrsConfig = field(default_factory=DmrsConfig)
    n_layers: int = 1
    n_rx: int = 1
    modulation: str = "qpsk"
    channel: TDLChannel | None = None
    estimator: str = "perfect"  # perfect | ls | lmmse
    detector: str = "mmse"  # zf | mmse
    coded: bool = True

    def make_channel(self) -> TDLChannel:
        ch = self.channel or TDLChannel(profile="A", delay_spread_s=100e-9)
        ch.n_rx, ch.n_tx = self.n_rx, self.n_layers
        return ch


@dataclass
class SlotResult:
    bit_errors: int
    n_bits: int
    tb_ok: bool | None
    nmse_db: float | None
    tx_bits: np.ndarray | None = None
    rx_bits: np.ndarray | None = None


def symbol_times(carrier: CarrierConfig) -> np.ndarray:
    t_sym = (carrier.fft_size + carrier.cp_length) / carrier.sample_rate_hz
    return np.arange(carrier.n_symbols) * t_sym


def subcarrier_freqs(carrier: CarrierConfig) -> np.ndarray:
    return (np.arange(carrier.n_sc) - carrier.n_sc // 2) * carrier.numerology.scs_hz


def tb_size(cfg: LinkConfig) -> tuple[int, int, int]:
    """(transport block bits excl. CRC, number of codewords, coded-bit capacity)."""
    grid = build_grid(cfg.carrier, cfg.dmrs, cfg.n_layers)
    cap = grid.num_data_re() * cfg.n_layers * bits_per_symbol(cfg.modulation)
    if not cfg.coded:
        return cap, 0, cap
    code = get_code()
    n_cw = cap // code.n
    return n_cw * code.k - 24, n_cw, cap


def transmit(cfg: LinkConfig, payload: np.ndarray | None, slot: int, rng: np.random.Generator):
    """Build the transmit grid. Returns (grid values (L, n_sc, n_sym), info, coded bits, ResourceGrid)."""
    grid = build_grid(cfg.carrier, cfg.dmrs, cfg.n_layers, slot)
    tbs, n_cw, cap = tb_size(cfg)
    if payload is None:
        payload = rng.integers(0, 2, tbs).astype(np.uint8)
    if payload.size != tbs:
        raise ValueError(f"payload must be {tbs} bits")
    if cfg.coded:
        code = get_code()
        cw = code.encode(crc_attach(payload).reshape(n_cw, code.k)).reshape(-1)
        pad = rng.integers(0, 2, cap - cw.size).astype(np.uint8)
        coded = np.concatenate([cw, pad])
    else:
        coded = payload
    syms = modulate(coded, cfg.modulation).reshape(-1, cfg.n_layers)  # layer mapping
    for layer in range(cfg.n_layers):
        grid.map_data(syms[:, layer], layer)
    return grid.values / np.sqrt(cfg.n_layers), payload, coded, grid


def receive(cfg: LinkConfig, y: np.ndarray, grid, slot: int, noise_var: float,
            h_true: np.ndarray | None = None, h_est: np.ndarray | None = None):
    """Estimate, detect and decode. y: (n_rx, n_sc, n_sym). Returns (payload, tb_ok, coded hard bits, h_hat)."""
    carrier = cfg.carrier
    if h_est is not None:
        h_hat = h_est
    elif cfg.estimator == "perfect":
        h_hat = h_true
    else:
        h_hat = estimate_channel(cfg, y, slot, noise_var)
    hd = np.moveaxis(grid.extract_data(h_hat), -1, 0)  # (n_re, n_rx, n_tx)
    yd = np.moveaxis(grid.extract_data(y), -1, 0)  # (n_re, n_rx)
    det = zf_detect if cfg.detector == "zf" else mmse_detect
    x_hat, nv = det(hd, yd, noise_var)
    llr = demodulate_llr(x_hat.reshape(-1), cfg.modulation, nv.reshape(-1))
    hard = (llr < 0).astype(np.uint8)
    if not cfg.coded:
        return hard, None, hard, h_hat
    code = get_code()
    tbs, n_cw, _ = tb_size(cfg)
    cw_llr = llr[: n_cw * code.n].reshape(n_cw, code.n)
    dec, _ = code.decode(cw_llr)
    info = code.info_bits(dec).reshape(-1)
    ok = crc_check(info)
    return info[:tbs], ok, hard, h_hat


_EST_CACHE: dict[tuple, PilotEstimator] = {}


def _estimator_for(cfg: LinkConfig) -> PilotEstimator:
    """Reuse estimators (and their cached LMMSE filters) across slots."""
    ch = cfg.make_channel()
    key = (cfg.estimator, ch.profile, ch.delay_spread_s, cfg.carrier.numerology.scs_hz)
    if key not in _EST_CACHE:
        corr = None
        if cfg.estimator == "lmmse":
            delays, powers = ch.pdp()
            corr = pdp_correlation(delays, powers, cfg.carrier.numerology.scs_hz)
        _EST_CACHE[key] = PilotEstimator(cfg.estimator, corr)
    return _EST_CACHE[key]


def estimate_channel(cfg: LinkConfig, y: np.ndarray, slot: int, noise_var: float) -> np.ndarray:
    carrier, dmrs = cfg.carrier, cfg.dmrs
    est = _estimator_for(cfg)
    h = np.empty((cfg.n_rx, cfg.n_layers, carrier.n_sc, carrier.n_symbols), dtype=np.complex128)
    for r in range(cfg.n_rx):
        for p in range(cfg.n_layers):
            pos, h_ls, scale = dmrs.ls_estimates(y[r], carrier, slot, p, cfg.n_layers)
            h[r, p] = est.estimate(pos, h_ls, carrier.n_sc, dmrs.symbols, carrier.n_symbols,
                                   noise_var * scale)
    return h


def simulate_slot(cfg: LinkConfig, snr_db: float, rng: np.random.Generator, slot: int = 0,
                  channel: TDLChannel | None = None) -> SlotResult:
    carrier = cfg.carrier
    ch = channel or cfg.make_channel()
    x, payload, coded, grid = transmit(cfg, None, slot, rng)
    h = ch.frequency_response(subcarrier_freqs(carrier), symbol_times(carrier), rng)
    h_eff = h / np.sqrt(cfg.n_layers)  # power split across layers
    y = np.einsum("rtks,tks->rks", h, x)
    noise_var = 10 ** (-snr_db / 10)
    y = y + crandn(y.shape, rng) * np.sqrt(noise_var)
    payload_hat, ok, hard, h_hat = receive(cfg, y, grid, slot, noise_var, h_true=h_eff)
    nm = None
    if cfg.estimator != "perfect":
        nm = float(10 * np.log10(np.sum(np.abs(h_hat - h_eff) ** 2) / np.sum(np.abs(h_eff) ** 2)))
    errs = int(np.sum(hard[: coded.size] != coded))
    return SlotResult(errs, int(coded.size), ok, nm, payload, payload_hat)
