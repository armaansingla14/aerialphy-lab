"""Simulated L2 (MAC) <-> DU high-PHY <-> RU low-PHY loop with a UE and air channel.

Message flow per slot (all over in-memory byte queues, so every message is
really encoded and decoded):

    MAC --SLOT.ind--      (from DU)
    MAC --DL_TTI.req + TX_DATA.req--> DU --U-plane DL--> RU --IQ time samples--> air --> UE
    MAC --UL_TTI.req--> DU;  UE --IQ--> air --> RU --U-plane UL--> DU
    DU  --RX_DATA.ind + CRC.ind--> MAC

The DU reuses the link-level chain in ``aerialphy.link`` (LDPC, QAM, DMRS,
LS/LMMSE estimation, MMSE detection). The RU performs CP-OFDM (I)FFT and
block-floating-point fronthaul (de)compression. The air interface applies a
static TDL channel in the time domain with integer-sample tap delays.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from aerialphy.channel import TDLChannel, crandn
from aerialphy.dmrs import DmrsConfig
from aerialphy.link import LinkConfig, receive, tb_size, transmit
from aerialphy.numerology import CarrierConfig, Numerology
from aerialphy.ofdm import ofdm_demodulate, ofdm_modulate

from . import fapi
from .fronthaul import FronthaulCodec

MOD_ORDER = {"qpsk": 2, "16qam": 4, "64qam": 6}


class Link:
    """Unidirectional in-memory transport that counts bytes and messages."""

    def __init__(self, name: str):
        self.name = name
        self.q: deque[bytes] = deque()
        self.bytes = 0
        self.messages = 0

    def send(self, data: bytes) -> None:
        self.q.append(bytes(data))
        self.bytes += len(data)
        self.messages += 1

    def drain(self) -> list[bytes]:
        out = list(self.q)
        self.q.clear()
        return out


def bits_to_bytes(bits: np.ndarray) -> bytes:
    return np.packbits(bits.astype(np.uint8)).tobytes()


def bytes_to_bits(data: bytes, n_bits: int) -> np.ndarray:
    b = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    out = np.zeros(n_bits, dtype=np.uint8)
    out[: min(n_bits, b.size)] = b[:n_bits]
    return out


# ---------------------------------------------------------------- DU (high PHY)

class DistributedUnit:
    def __init__(self, to_mac: Link, to_ru: Link, from_ru: Link, estimator: str = "lmmse"):
        self.to_mac, self.to_ru, self.from_ru = to_mac, to_ru, from_ru
        self.estimator = estimator
        self.cfg: LinkConfig | None = None
        self.fh: FronthaulCodec | None = None
        self.running = False
        self.pending_ul: dict[tuple[int, int], fapi.PuschPdu] = {}
        self.pending_dl: dict[tuple[int, int], fapi.PdschPdu] = {}
        self.proc_time_s = 0.0
        self.noise_var = 1e-2
        self.rng = np.random.default_rng(1234)

    def _err(self, msg_id: int, code: fapi.ErrorCode, sfn: int = 0, slot: int = 0) -> None:
        self.to_mac.send(fapi.encode(fapi.ErrorIndication(sfn, slot, msg_id, int(code))))

    def handle(self, raw: bytes) -> None:
        msg = fapi.decode(raw)
        t0 = time.perf_counter()
        if isinstance(msg, fapi.ParamRequest):
            caps = [fapi.TLV(0x01, 4), fapi.TLV(0x02, 273), fapi.TLV(0x03, 6)]  # max layers, max RB, max Qm
            self.to_mac.send(fapi.encode(fapi.ParamResponse(0, caps)))
        elif isinstance(msg, fapi.ConfigRequest):
            if msg.num_tx_ant not in (1, 2, 4) or msg.num_rx_ant not in (1, 2, 4):
                self.to_mac.send(fapi.encode(fapi.ConfigResponse(int(fapi.ErrorCode.MSG_INVALID_CONFIG))))
                return
            carrier = CarrierConfig(Numerology(msg.numerology), n_rb=msg.carrier_bw_rb)
            self.cfg = LinkConfig(carrier=carrier, dmrs=DmrsConfig(symbols=[msg.dmrs_type_a_pos, 11]),
                                  n_layers=1, n_rx=msg.num_rx_ant, estimator=self.estimator)
            self.fh = FronthaulCodec(msg.fh_comp_bitwidth)
            for t in msg.tlvs:
                if t.tag == 0x100:  # vendor TLV: expected UL noise variance (x1e6)
                    self.noise_var = t.value / 1e6
            self.to_mac.send(fapi.encode(fapi.ConfigResponse(0)))
        elif isinstance(msg, fapi.StartRequest):
            if self.cfg is None:
                self._err(fapi.MsgId.START_REQUEST, fapi.ErrorCode.MSG_INVALID_STATE)
                return
            self.running = True
        elif isinstance(msg, fapi.StopRequest):
            self.running = False
            self.to_mac.send(fapi.encode(fapi.StopIndication()))
        elif not self.running:
            self._err(msg.MSG_ID, fapi.ErrorCode.MSG_INVALID_STATE, getattr(msg, "sfn", 0), getattr(msg, "slot", 0))
        elif isinstance(msg, fapi.DlTtiRequest):
            for pdu in msg.pdsch:
                self.pending_dl[(msg.sfn, msg.slot, pdu.pdu_index)] = pdu
        elif isinstance(msg, fapi.TxDataRequest):
            for tx in msg.pdus:
                pdu = self.pending_dl.pop((msg.sfn, msg.slot, tx.pdu_index), None)
                if pdu is None:
                    self._err(msg.MSG_ID, fapi.ErrorCode.MSG_TX_ERR, msg.sfn, msg.slot)
                    continue
                self._dl_process(msg.sfn, msg.slot, pdu, tx.payload)
        elif isinstance(msg, fapi.UlTtiRequest):
            for pdu in msg.pusch:
                self.pending_ul[(msg.sfn, msg.slot)] = pdu
        self.proc_time_s += time.perf_counter() - t0

    def _pdu_cfg(self, pdu: fapi.PdschPdu) -> LinkConfig:
        c = self.cfg
        return LinkConfig(carrier=c.carrier, dmrs=c.dmrs, n_layers=pdu.num_layers, n_rx=c.n_rx,
                          modulation=fapi.MODULATION_ORDER[pdu.qam_mod_order], estimator=c.estimator,
                          channel=c.channel)

    def _dl_process(self, sfn: int, slot: int, pdu: fapi.PdschPdu, payload: bytes) -> None:
        cfg = self._pdu_cfg(pdu)
        tbs, _, _ = tb_size(cfg)
        x, _, _, _ = transmit(cfg, bytes_to_bits(payload, tbs), slot, self.rng)
        for port in range(cfg.n_layers):
            for pkt in self.fh.to_packets(x[port], 1, sfn, slot // 2, slot, port):
                self.to_ru.send(pkt)

    def ul_receive(self, sfn: int, slot: int) -> None:
        """Process UL U-plane packets for (sfn, slot) and indicate results to the MAC."""
        t0 = time.perf_counter()
        pdu = self.pending_ul.pop((sfn, slot), None)
        pkts = self.from_ru.drain()
        if pdu is None:
            return
        cfg = self._pdu_cfg(pdu)
        c = cfg.carrier
        by_port: dict[int, list[bytes]] = {}
        from .fronthaul import UPlanePacket
        for p in pkts:
            by_port.setdefault(UPlanePacket.decode(p).port, []).append(p)
        y = np.stack([self.fh.from_packets(by_port[r], c.n_sc, c.n_symbols) for r in range(cfg.n_rx)])
        from aerialphy.dmrs import build_grid
        grid = build_grid(c, cfg.dmrs, cfg.n_layers, slot)
        info, ok, _, h_hat = receive(cfg, y, grid, slot, self.noise_var)
        self.to_mac.send(fapi.encode(fapi.RxDataIndication(sfn, slot, [
            fapi.RxPdu(pdu.rnti, pdu.harq_process_id, bits_to_bytes(info))])))
        self.to_mac.send(fapi.encode(fapi.CrcIndication(sfn, slot, [
            fapi.CrcInfo(pdu.rnti, pdu.harq_process_id, 0 if ok else 1, 0, 0)])))
        self.proc_time_s += time.perf_counter() - t0


# ---------------------------------------------------------------- RU (low PHY)

class RadioUnit:
    def __init__(self, du: DistributedUnit, from_du: Link, to_du: Link):
        self.du, self.from_du, self.to_du = du, from_du, to_du

    def dl_transmit(self, n_ports: int) -> np.ndarray:
        """Decompress DL packets and return time-domain samples (n_ports, n_samples)."""
        from .fronthaul import UPlanePacket
        c = self.du.cfg.carrier
        by_port: dict[int, list[bytes]] = {}
        for p in self.from_du.drain():
            by_port.setdefault(UPlanePacket.decode(p).port, []).append(p)
        grids = np.stack([self.du.fh.from_packets(by_port[p], c.n_sc, c.n_symbols) for p in range(n_ports)])
        return ofdm_modulate(grids, c)

    def ul_capture(self, rx_time: np.ndarray, sfn: int, slot: int) -> None:
        c = self.du.cfg.carrier
        grids = ofdm_demodulate(rx_time, c)
        for r in range(grids.shape[0]):
            for pkt in self.du.fh.to_packets(grids[r], 0, sfn, slot // 2, slot, r):
                self.to_du.send(pkt)


# ---------------------------------------------------------------- air + UE

@dataclass
class AirChannel:
    channel: TDLChannel
    snr_db: float
    carrier: CarrierConfig
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    def propagate(self, x: np.ndarray, n_rx: int) -> np.ndarray:
        """x: (n_tx, n_samples). Static TDL taps with delays rounded to whole samples."""
        ch = self.channel
        ch.n_rx, ch.n_tx = n_rx, x.shape[0]
        delays, _ = ch.pdp()
        taps = ch.sample_taps(np.zeros(1), self.rng)[0]  # (L, R, X)
        d_samp = np.round(delays * self.carrier.sample_rate_hz).astype(int)
        y = np.zeros((n_rx, x.shape[1]), dtype=np.complex128)
        for r in range(n_rx):
            for t in range(x.shape[0]):
                h = np.zeros(d_samp.max() + 1, dtype=np.complex128)
                np.add.at(h, d_samp, taps[:, r, t])
                y[r] += np.convolve(x[t], h)[: x.shape[1]]
        nv = 10 ** (-self.snr_db / 10)
        return y + crandn(y.shape, self.rng) * np.sqrt(nv)


class SimUE:
    def __init__(self, rnti: int, cfg: LinkConfig, rng: np.random.Generator):
        self.rnti, self.cfg, self.rng = rnti, cfg, rng
        self.sent_ul: dict[tuple[int, int], np.ndarray] = {}

    def ul_transmit(self, sfn: int, slot: int, n_layers: int) -> np.ndarray:
        cfg = LinkConfig(carrier=self.cfg.carrier, dmrs=self.cfg.dmrs, n_layers=n_layers,
                         n_rx=self.cfg.n_rx, modulation=self.cfg.modulation)
        x, payload, _, _ = transmit(cfg, None, slot, self.rng)
        self.sent_ul[(sfn, slot)] = payload
        return ofdm_modulate(x, cfg.carrier)

    def dl_receive(self, y_time: np.ndarray, slot: int, n_layers: int, noise_var: float):
        cfg = LinkConfig(carrier=self.cfg.carrier, dmrs=self.cfg.dmrs, n_layers=n_layers,
                         n_rx=y_time.shape[0], modulation=self.cfg.modulation, estimator=self.cfg.estimator,
                         channel=self.cfg.channel)
        from aerialphy.dmrs import build_grid
        y = ofdm_demodulate(y_time, cfg.carrier)
        grid = build_grid(cfg.carrier, cfg.dmrs, n_layers, slot)
        info, ok, _, _ = receive(cfg, y, grid, slot, noise_var)
        return info, ok


# ---------------------------------------------------------------- MAC + loop

@dataclass
class LoopStats:
    dl_tb: int = 0
    dl_tb_ok: int = 0
    ul_tb: int = 0
    ul_crc_pass: int = 0
    ul_payload_match: int = 0
    fapi_bytes: int = 0
    fapi_msgs: int = 0
    fh_bytes: int = 0
    du_proc_s: float = 0.0
    errors: list[int] = field(default_factory=list)

    @property
    def dl_bler(self) -> float:
        return 1 - self.dl_tb_ok / max(self.dl_tb, 1)

    @property
    def ul_bler(self) -> float:
        return 1 - self.ul_crc_pass / max(self.ul_tb, 1)


def run_loop(n_slots: int = 10, snr_db: float = 10.0, n_rb: int = 24, mu: int = 1, n_ant: int = 2,
             n_layers: int = 1, modulation: str = "qpsk", profile: str = "A", delay_spread_s: float = 100e-9,
             fh_bitwidth: int = 9, estimator: str = "lmmse", seed: int = 0) -> LoopStats:
    rng = np.random.default_rng(seed)
    mac_to_du, du_to_mac = Link("p5/p7 down"), Link("p5/p7 up")
    du_to_ru, ru_to_du = Link("fh dl"), Link("fh ul")
    du = DistributedUnit(du_to_mac, du_to_ru, ru_to_du, estimator=estimator)
    ru = RadioUnit(du, du_to_ru, ru_to_du)
    stats = LoopStats()
    noise_var = 10 ** (-snr_db / 10)

    def mac_send(msg) -> None:
        mac_to_du.send(fapi.encode(msg))
        for raw in mac_to_du.drain():
            du.handle(raw)

    def mac_collect() -> list:
        return [fapi.decode(r) for r in du_to_mac.drain()]

    mac_send(fapi.ParamRequest())
    mac_send(fapi.ConfigRequest(phy_cell_id=1, numerology=mu, carrier_bw_rb=n_rb, num_tx_ant=n_ant,
                                num_rx_ant=n_ant, fh_comp_bitwidth=fh_bitwidth,
                                tlvs=[fapi.TLV(0x100, int(round(noise_var * 1e6)))]))
    resp = mac_collect()
    if not any(isinstance(m, fapi.ConfigResponse) and m.error_code == 0 for m in resp):
        raise RuntimeError(f"CONFIG failed: {resp}")
    ch = TDLChannel(profile=profile, delay_spread_s=delay_spread_s)
    du.cfg.channel = ch
    mac_send(fapi.StartRequest())
    ue_cfg = LinkConfig(carrier=du.cfg.carrier, dmrs=du.cfg.dmrs, n_rx=n_ant, modulation=modulation,
                        estimator=estimator, channel=ch)
    ue = SimUE(rnti=0x4601, cfg=ue_cfg, rng=rng)
    air = AirChannel(ch, snr_db, du.cfg.carrier, rng)
    probe = LinkConfig(carrier=du.cfg.carrier, dmrs=du.cfg.dmrs, n_layers=n_layers, modulation=modulation)
    tbs, _, _ = tb_size(probe)
    slots_per_frame = du.cfg.carrier.numerology.slots_per_frame
    qm = MOD_ORDER[modulation]
    for i in range(n_slots):
        sfn, slot = divmod(i, slots_per_frame)
        pdu_kw = dict(rnti=ue.rnti, bwp_size=n_rb, num_layers=n_layers, qam_mod_order=qm,
                      tb_size=(tbs + 7) // 8, dmrs_symbol_bitmap=(1 << 2) | (1 << 11), rb_size=n_rb)
        # ---- downlink
        dl_bits = rng.integers(0, 2, tbs).astype(np.uint8)
        mac_send(fapi.DlTtiRequest(sfn, slot, [fapi.PdschPdu(pdu_index=0, **pdu_kw)]))
        mac_send(fapi.TxDataRequest(sfn, slot, [fapi.TxPdu(0, bits_to_bytes(dl_bits))]))
        tx = ru.dl_transmit(n_layers)
        rx = air.propagate(tx, n_ant)
        info, ok = ue.dl_receive(rx, slot, n_layers, noise_var)
        stats.dl_tb += 1
        stats.dl_tb_ok += int(bool(ok) and np.array_equal(info, dl_bits))
        # ---- uplink
        mac_send(fapi.UlTtiRequest(sfn, slot, [fapi.PuschPdu(harq_process_id=i % 16, **pdu_kw)]))
        ul_tx = ue.ul_transmit(sfn, slot, n_layers)
        ru.ul_capture(air.propagate(ul_tx, n_ant), sfn, slot)
        du.ul_receive(sfn, slot)
        for m in mac_collect():
            if isinstance(m, fapi.CrcIndication):
                stats.ul_tb += len(m.crcs)
                stats.ul_crc_pass += sum(c.tb_crc_status == 0 for c in m.crcs)
            elif isinstance(m, fapi.RxDataIndication):
                for p in m.pdus:
                    got = bytes_to_bits(p.payload, tbs)
                    stats.ul_payload_match += int(np.array_equal(got, ue.sent_ul[(m.sfn, m.slot)]))
            elif isinstance(m, fapi.ErrorIndication):
                stats.errors.append(m.error_code)
    mac_send(fapi.StopRequest())
    mac_collect()
    stats.fapi_bytes = mac_to_du.bytes + du_to_mac.bytes
    stats.fapi_msgs = mac_to_du.messages + du_to_mac.messages
    stats.fh_bytes = du_to_ru.bytes + ru_to_du.bytes
    stats.du_proc_s = du.proc_time_s
    return stats
