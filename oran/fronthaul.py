"""Simplified O-RAN 7.2x-style fronthaul U-plane with block floating point compression.

Packet = eCPRI common header + U-plane application header + one section
carrying frequency-domain IQ for a contiguous PRB range of one OFDM symbol.
IQ is quantised to 16-bit fixed point and then BFP-compressed per PRB
(shared 4-bit exponent, ``iq_width``-bit signed mantissas), similar in spirit
to O-RAN WG4 CUS-plane block floating point. Bit layouts are simplified.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

ECPRI_HDR = struct.Struct(">BBH")  # revision/flags, message type, payload size
UPLANE_HDR = struct.Struct(">BBHB")  # dir|ver|filter, frame id, subframe|slot|symbol, port id
SECTION_HDR = struct.Struct(">HHHB")  # section id, start prb, num prb, udCompHdr
ECPRI_IQ_DATA = 0x00
FULL_SCALE = 2 ** 15 - 1


def _pack_bits(values: np.ndarray, width: int) -> bytes:
    v = (values.astype(np.int64) & ((1 << width) - 1)).astype(np.uint64)
    bits = ((v[:, None] >> np.arange(width - 1, -1, -1, dtype=np.uint64)) & 1).astype(np.uint8)
    return np.packbits(bits.reshape(-1)).tobytes()


def _unpack_bits(data: bytes, count: int, width: int) -> np.ndarray:
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))[: count * width].reshape(count, width)
    v = (bits.astype(np.int64) << np.arange(width - 1, -1, -1)).sum(axis=1)
    sign = 1 << (width - 1)
    return np.where(v & sign, v - (1 << width), v)


def bfp_compress(iq_int16: np.ndarray, iq_width: int) -> bytes:
    """iq_int16: (n_prb, 12) complex with integer parts in int16 range."""
    out = bytearray()
    for prb in iq_int16:
        v = np.empty(24, dtype=np.int64)
        v[0::2] = prb.real.astype(np.int64)
        v[1::2] = prb.imag.astype(np.int64)
        peak = int(np.max(np.abs(v)))
        exp = 0
        lim = (1 << (iq_width - 1)) - 1
        while (peak >> exp) > lim:
            exp += 1
        m = np.right_shift(v, exp) if exp else v
        out += bytes([exp]) + _pack_bits(m, iq_width)
    return bytes(out)


def bfp_decompress(data: bytes, n_prb: int, iq_width: int) -> np.ndarray:
    prb_bytes = 1 + (24 * iq_width + 7) // 8
    out = np.empty((n_prb, 12), dtype=np.complex128)
    for i in range(n_prb):
        chunk = data[i * prb_bytes:(i + 1) * prb_bytes]
        exp = chunk[0]
        m = _unpack_bits(chunk[1:], 24, iq_width) << exp
        out[i] = m[0::2] + 1j * m[1::2]
    return out


@dataclass
class UPlanePacket:
    direction: int  # 1 = DL, 0 = UL
    frame_id: int
    subframe: int
    slot: int
    symbol: int
    port: int
    section_id: int
    start_prb: int
    num_prb: int
    iq_width: int
    payload: bytes

    def encode(self) -> bytes:
        tsym = ((self.subframe & 0xF) << 10) | ((self.slot & 0x3F) << 4) | (self.symbol & 0xF)
        app = UPLANE_HDR.pack((self.direction << 7) | 0x10, self.frame_id & 0xFF, tsym, self.port)
        comp = ((self.iq_width & 0xF) << 4) | 0x1  # method 1 = BFP
        sec = SECTION_HDR.pack(self.section_id, self.start_prb, self.num_prb, comp)
        body = app + sec + self.payload
        return ECPRI_HDR.pack(0x10, ECPRI_IQ_DATA, len(body)) + body

    @classmethod
    def decode(cls, data: bytes) -> "UPlanePacket":
        _, mtype, size = ECPRI_HDR.unpack_from(data, 0)
        if mtype != ECPRI_IQ_DATA:
            raise ValueError("not an IQ data message")
        if size != len(data) - ECPRI_HDR.size:
            raise ValueError("eCPRI payload size mismatch")
        o = ECPRI_HDR.size
        b0, fid, tsym, port = UPLANE_HDR.unpack_from(data, o)
        o += UPLANE_HDR.size
        sid, sprb, nprb, comp = SECTION_HDR.unpack_from(data, o)
        o += SECTION_HDR.size
        return cls(direction=b0 >> 7, frame_id=fid, subframe=(tsym >> 10) & 0xF,
                   slot=(tsym >> 4) & 0x3F, symbol=tsym & 0xF, port=port, section_id=sid,
                   start_prb=sprb, num_prb=nprb, iq_width=comp >> 4, payload=bytes(data[o:]))


class FronthaulCodec:
    """Converts per-symbol frequency-domain IQ to/from U-plane packets."""

    def __init__(self, iq_width: int = 9, scale: float = 2 ** 12):
        if not 2 <= iq_width <= 16:
            raise ValueError("iq_width must be 2..16")
        self.iq_width = iq_width
        self.scale = scale  # floating-point unit amplitude -> integer units

    def to_packets(self, grid: np.ndarray, direction: int, frame: int, subframe: int, slot: int,
                   port: int) -> list[bytes]:
        n_sc, n_sym = grid.shape
        n_prb = n_sc // 12
        q = np.clip(np.round(grid * self.scale).real, -FULL_SCALE, FULL_SCALE) + \
            1j * np.clip(np.round(grid * self.scale).imag, -FULL_SCALE, FULL_SCALE)
        pkts = []
        for s in range(n_sym):
            payload = bfp_compress(q[:, s].reshape(n_prb, 12), self.iq_width)
            pkts.append(UPlanePacket(direction, frame, subframe, slot, s, port, 1, 0, n_prb,
                                     self.iq_width, payload).encode())
        return pkts

    def from_packets(self, packets: list[bytes], n_sc: int, n_sym: int) -> np.ndarray:
        grid = np.zeros((n_sc, n_sym), dtype=np.complex128)
        for raw in packets:
            p = UPlanePacket.decode(raw)
            iq = bfp_decompress(p.payload, p.num_prb, p.iq_width).reshape(-1)
            grid[p.start_prb * 12:(p.start_prb + p.num_prb) * 12, p.symbol] = iq / self.scale
        return grid
