"""FAPI-style (SCF 222 inspired) message definitions with binary encode/decode.

Messages are plain dataclasses. Each scalar field carries a ``struct`` format
in its metadata; nested PDU lists are count-prefixed and byte payloads are
length-prefixed. Every message is framed with a small header
(num_msgs u8, opaque u8, msg_id u16, body_len u32), little-endian, mirroring
the FAPI P5/P7 message header layout. Field sets are simplified subsets of
the real specification, chosen to drive the simulator.
"""
from __future__ import annotations

import enum
import struct
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, get_args, get_type_hints

HEADER = struct.Struct("<BBHI")


def u8(default: int = 0):
    return field(default=default, metadata={"fmt": "B"})


def u16(default: int = 0):
    return field(default=default, metadata={"fmt": "H"})


def u32(default: int = 0):
    return field(default=default, metadata={"fmt": "I"})


def i16(default: int = 0):
    return field(default=default, metadata={"fmt": "h"})


def f32(default: float = 0.0):
    return field(default=default, metadata={"fmt": "f"})


def pdu_list():
    return field(default_factory=list, metadata={"fmt": "list"})


def blob():
    return field(default=b"", metadata={"fmt": "bytes"})


class MsgId(enum.IntEnum):
    PARAM_REQUEST = 0x00
    PARAM_RESPONSE = 0x01
    CONFIG_REQUEST = 0x02
    CONFIG_RESPONSE = 0x03
    START_REQUEST = 0x04
    STOP_REQUEST = 0x05
    STOP_INDICATION = 0x06
    ERROR_INDICATION = 0x07
    DL_TTI_REQUEST = 0x80
    UL_TTI_REQUEST = 0x81
    SLOT_INDICATION = 0x82
    UL_DCI_REQUEST = 0x83
    TX_DATA_REQUEST = 0x84
    RX_DATA_INDICATION = 0x85
    CRC_INDICATION = 0x86


class ErrorCode(enum.IntEnum):
    MSG_OK = 0
    MSG_INVALID_STATE = 1
    MSG_INVALID_CONFIG = 2
    SFN_OUT_OF_SYNC = 3
    MSG_SLOT_ERR = 4
    MSG_BCH_MISSING = 5
    MSG_INVALID_SFN = 6
    MSG_UL_DCI_ERR = 7
    MSG_TX_ERR = 8


MODULATION_ORDER = {2: "qpsk", 4: "16qam", 6: "64qam"}


class Codec:
    """Recursive dataclass <-> bytes codec driven by field metadata."""

    @classmethod
    def encode_body(cls, obj: Any) -> bytes:
        out = bytearray()
        for f in fields(obj):
            fmt = f.metadata.get("fmt")
            val = getattr(obj, f.name)
            if fmt is None:
                continue
            if fmt == "list":
                out += struct.pack("<H", len(val))
                for item in val:
                    body = cls.encode_body(item)
                    out += struct.pack("<H", len(body)) + body
            elif fmt == "bytes":
                out += struct.pack("<I", len(val)) + bytes(val)
            else:
                out += struct.pack("<" + fmt, val)
        return bytes(out)

    @classmethod
    def decode_body(cls, typ: type, data: bytes, offset: int = 0):
        kwargs = {}
        hints = get_type_hints(typ)
        for f in fields(typ):
            fmt = f.metadata.get("fmt")
            if fmt is None:
                continue
            if fmt == "list":
                (n,) = struct.unpack_from("<H", data, offset)
                offset += 2
                item_t = get_args(hints[f.name])[0]
                items = []
                for _ in range(n):
                    (ln,) = struct.unpack_from("<H", data, offset)
                    offset += 2
                    item, end = cls.decode_body(item_t, data, offset)
                    if end != offset + ln:
                        raise ValueError(f"PDU length mismatch in {typ.__name__}.{f.name}")
                    items.append(item)
                    offset = end
                kwargs[f.name] = items
            elif fmt == "bytes":
                (n,) = struct.unpack_from("<I", data, offset)
                offset += 4
                kwargs[f.name] = bytes(data[offset:offset + n])
                offset += n
            else:
                (kwargs[f.name],) = struct.unpack_from("<" + fmt, data, offset)
                offset += struct.calcsize("<" + fmt)
        return typ(**kwargs), offset


_REGISTRY: dict[int, type] = {}


def message(msg_id: MsgId):
    def wrap(cls):
        cls = dataclass(cls)
        cls.MSG_ID = msg_id
        _REGISTRY[int(msg_id)] = cls
        return cls
    return wrap


def encode(msg: Any) -> bytes:
    if not is_dataclass(msg) or not hasattr(msg, "MSG_ID"):
        raise TypeError("not a FAPI message")
    body = Codec.encode_body(msg)
    return HEADER.pack(1, 0, int(msg.MSG_ID), len(body)) + body


def decode(data: bytes) -> Any:
    if len(data) < HEADER.size:
        raise ValueError("truncated header")
    _, _, msg_id, ln = HEADER.unpack_from(data, 0)
    if len(data) != HEADER.size + ln:
        raise ValueError(f"length mismatch: header says {ln}, got {len(data) - HEADER.size}")
    typ = _REGISTRY.get(msg_id)
    if typ is None:
        raise ValueError(f"unknown msg id 0x{msg_id:02x}")
    msg, end = Codec.decode_body(typ, data, HEADER.size)
    if end != len(data):
        raise ValueError("trailing bytes in message")
    return msg


# ------------------------------------------------------------------ PDUs

@dataclass
class TLV:
    tag: int = u16()
    value: int = u32()


@dataclass
class PdschPdu:
    rnti: int = u16()
    pdu_index: int = u16()
    bwp_start: int = u16()
    bwp_size: int = u16()
    num_layers: int = u8(1)
    qam_mod_order: int = u8(2)
    mcs_index: int = u8(0)
    rv_index: int = u8(0)
    tb_size: int = u32()
    dmrs_symbol_bitmap: int = u16()
    rb_start: int = u16()
    rb_size: int = u16()
    start_symbol: int = u8(0)
    num_symbols: int = u8(14)


@dataclass
class PuschPdu(PdschPdu):
    harq_process_id: int = u8(0)
    new_data: int = u8(1)


@dataclass
class TxPdu:
    pdu_index: int = u16()
    payload: bytes = blob()


@dataclass
class RxPdu:
    rnti: int = u16()
    harq_process_id: int = u8(0)
    payload: bytes = blob()


@dataclass
class CrcInfo:
    rnti: int = u16()
    harq_process_id: int = u8(0)
    tb_crc_status: int = u8(0)  # 0 = pass, 1 = fail
    ul_sinr_db_x10: int = i16(0)
    nmse_db_x10: int = i16(0)


# ------------------------------------------------------------------ messages

@message(MsgId.PARAM_REQUEST)
class ParamRequest:
    pass


@message(MsgId.PARAM_RESPONSE)
class ParamResponse:
    error_code: int = u8(0)
    tlvs: list[TLV] = pdu_list()


@message(MsgId.CONFIG_REQUEST)
class ConfigRequest:
    phy_cell_id: int = u16()
    numerology: int = u8(1)
    carrier_bw_rb: int = u16(24)
    num_tx_ant: int = u8(1)
    num_rx_ant: int = u8(1)
    dmrs_type_a_pos: int = u8(2)
    fh_comp_bitwidth: int = u8(9)
    tlvs: list[TLV] = pdu_list()


@message(MsgId.CONFIG_RESPONSE)
class ConfigResponse:
    error_code: int = u8(0)


@message(MsgId.START_REQUEST)
class StartRequest:
    pass


@message(MsgId.STOP_REQUEST)
class StopRequest:
    pass


@message(MsgId.STOP_INDICATION)
class StopIndication:
    pass


@message(MsgId.ERROR_INDICATION)
class ErrorIndication:
    sfn: int = u16()
    slot: int = u16()
    message_id: int = u8()
    error_code: int = u8()


@message(MsgId.SLOT_INDICATION)
class SlotIndication:
    sfn: int = u16()
    slot: int = u16()


@message(MsgId.DL_TTI_REQUEST)
class DlTtiRequest:
    sfn: int = u16()
    slot: int = u16()
    pdsch: list[PdschPdu] = pdu_list()


@message(MsgId.UL_TTI_REQUEST)
class UlTtiRequest:
    sfn: int = u16()
    slot: int = u16()
    pusch: list[PuschPdu] = pdu_list()


@message(MsgId.TX_DATA_REQUEST)
class TxDataRequest:
    sfn: int = u16()
    slot: int = u16()
    pdus: list[TxPdu] = pdu_list()


@message(MsgId.RX_DATA_INDICATION)
class RxDataIndication:
    sfn: int = u16()
    slot: int = u16()
    pdus: list[RxPdu] = pdu_list()


@message(MsgId.CRC_INDICATION)
class CrcIndication:
    sfn: int = u16()
    slot: int = u16()
    crcs: list[CrcInfo] = pdu_list()


def all_message_types() -> list[type]:
    return list(_REGISTRY.values())
