import numpy as np
import pytest

from oran import fapi
from oran.du_ru import bits_to_bytes, bytes_to_bits, run_loop
from oran.fronthaul import FronthaulCodec, UPlanePacket, bfp_compress, bfp_decompress

SAMPLE_MESSAGES = [
    fapi.ParamRequest(),
    fapi.ParamResponse(0, [fapi.TLV(1, 4), fapi.TLV(2, 273)]),
    fapi.ConfigRequest(phy_cell_id=500, numerology=1, carrier_bw_rb=273, num_tx_ant=4, num_rx_ant=4,
                       tlvs=[fapi.TLV(0x100, 1000)]),
    fapi.ConfigResponse(2),
    fapi.StartRequest(),
    fapi.StopRequest(),
    fapi.StopIndication(),
    fapi.ErrorIndication(1023, 19, 0x80, 3),
    fapi.SlotIndication(77, 5),
    fapi.DlTtiRequest(1, 2, [fapi.PdschPdu(rnti=0x4601, num_layers=2, qam_mod_order=4, tb_size=9000),
                             fapi.PdschPdu(rnti=0x4602, pdu_index=1)]),
    fapi.UlTtiRequest(3, 4, [fapi.PuschPdu(rnti=7, harq_process_id=15, new_data=0)]),
    fapi.UlTtiRequest(3, 4, []),
    fapi.TxDataRequest(5, 6, [fapi.TxPdu(0, bytes(range(256)) * 3)]),
    fapi.RxDataIndication(7, 8, [fapi.RxPdu(9, 1, b"\x00\xff" * 10)]),
    fapi.CrcIndication(9, 10, [fapi.CrcInfo(1, 2, 1, -35, -200), fapi.CrcInfo(3, 4, 0, 250, 10)]),
]


@pytest.mark.parametrize("msg", SAMPLE_MESSAGES, ids=lambda m: type(m).__name__)
def test_fapi_roundtrip(msg):
    raw = fapi.encode(msg)
    assert fapi.decode(raw) == msg
    _, _, mid, ln = fapi.HEADER.unpack_from(raw)
    assert mid == int(msg.MSG_ID) and ln == len(raw) - fapi.HEADER.size


@pytest.mark.parametrize("msg", SAMPLE_MESSAGES[1:], ids=lambda m: type(m).__name__)
def test_fapi_truncation_detected(msg):
    raw = fapi.encode(msg)
    with pytest.raises((ValueError, Exception)):
        fapi.decode(raw[:-1])


def test_fapi_unknown_id():
    raw = fapi.HEADER.pack(1, 0, 0x7F, 0)
    with pytest.raises(ValueError):
        fapi.decode(raw)


def test_every_registered_message_has_unique_id():
    ids = [t.MSG_ID for t in fapi.all_message_types()]
    assert len(ids) == len(set(ids)) == 14


@pytest.mark.parametrize("n_bits", [1, 8, 13, 1000])
def test_bit_byte_conversion(n_bits, rng):
    b = rng.integers(0, 2, n_bits).astype(np.uint8)
    assert np.array_equal(bytes_to_bits(bits_to_bytes(b), n_bits), b)


@pytest.mark.parametrize("width", [4, 6, 8, 9, 12, 14, 16])
def test_bfp_roundtrip_error_bound(width, rng):
    x = np.round(rng.standard_normal((5, 12)) * 3000) + 1j * np.round(rng.standard_normal((5, 12)) * 3000)
    y = bfp_decompress(bfp_compress(x, width), 5, width)
    peak = np.max(np.abs(np.concatenate([x.real, x.imag], axis=1)), axis=1)
    step = 2.0 ** np.ceil(np.log2(np.maximum(peak / (2 ** (width - 1) - 1), 1)))
    assert np.all(np.abs(y.real - x.real) < step[:, None] + 1e-9)
    if width == 16:
        assert np.allclose(x, y)


@pytest.mark.parametrize("width", [6, 9, 12, 16])
def test_fronthaul_packet_roundtrip(width, rng):
    codec = FronthaulCodec(width)
    g = (rng.standard_normal((48, 14)) + 1j * rng.standard_normal((48, 14))) / np.sqrt(2)
    pkts = codec.to_packets(g, 1, 3, 2, 5, 1)
    assert len(pkts) == 14
    p = UPlanePacket.decode(pkts[7])
    assert (p.direction, p.frame_id, p.subframe, p.slot, p.symbol, p.port) == (1, 3, 2, 5, 7, 1)
    back = codec.from_packets(pkts, 48, 14)
    evm = np.sqrt(np.mean(np.abs(back - g) ** 2))
    assert evm < 2.0 ** (-(width - 4))


def test_fronthaul_rejects_bad_width():
    with pytest.raises(ValueError):
        FronthaulCodec(1)


@pytest.mark.parametrize("n_ant,layers,mod", [(1, 1, "qpsk"), (2, 1, "16qam"), (2, 2, "qpsk")])
def test_du_ru_loop_high_snr(n_ant, layers, mod):
    s = run_loop(n_slots=3, snr_db=35.0, n_rb=12, n_ant=n_ant, n_layers=layers, modulation=mod,
                 profile="C", delay_spread_s=50e-9)
    assert s.dl_bler == 0 and s.ul_bler == 0
    assert s.ul_payload_match == 3
    assert not s.errors
    assert s.fapi_msgs > 3 * 4 and s.fh_bytes > 0


def test_du_ru_loop_low_snr_fails():
    s = run_loop(n_slots=2, snr_db=-8.0, n_rb=12, n_ant=1, modulation="16qam")
    assert s.ul_bler > 0 and s.dl_bler > 0
