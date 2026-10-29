import numpy as np
import pytest

from aerialphy.channel import TDLChannel
from aerialphy.link import LinkConfig, simulate_slot, tb_size
from aerialphy.numerology import CarrierConfig, Numerology


@pytest.mark.parametrize("mod", ["qpsk", "16qam", "64qam"])
@pytest.mark.parametrize("layers,n_rx", [(1, 1), (1, 2), (2, 2), (2, 4), (4, 4)])
@pytest.mark.parametrize("est", ["perfect", "ls", "lmmse"])
def test_link_high_snr_error_free(mod, layers, n_rx, est, rng):
    cfg = LinkConfig(carrier=CarrierConfig(Numerology(1), n_rb=12), n_layers=layers, n_rx=n_rx,
                     modulation=mod, estimator=est,
                     channel=TDLChannel(profile="FLAT"))
    r = simulate_slot(cfg, 45.0, rng)
    assert r.tb_ok
    assert np.array_equal(r.tx_bits, r.rx_bits)
    assert r.bit_errors == 0


@pytest.mark.parametrize("det", ["zf", "mmse"])
@pytest.mark.parametrize("profile", ["A", "C"])
def test_link_frequency_selective_decodes(det, profile, rng):
    cfg = LinkConfig(carrier=CarrierConfig(Numerology(1), n_rb=12), n_layers=1, n_rx=2,
                     estimator="lmmse", detector=det, channel=TDLChannel(profile=profile))
    oks = [simulate_slot(cfg, 20.0, rng, slot=s).tb_ok for s in range(3)]
    assert all(oks)


def test_link_low_snr_fails(rng):
    cfg = LinkConfig(carrier=CarrierConfig(Numerology(1), n_rb=12), modulation="64qam")
    assert not simulate_slot(cfg, -5.0, rng).tb_ok


@pytest.mark.parametrize("mod,layers", [("qpsk", 1), ("16qam", 2), ("64qam", 4)])
def test_tb_size_scales(mod, layers):
    cfg = LinkConfig(carrier=CarrierConfig(n_rb=24), n_layers=layers, modulation=mod)
    tbs, n_cw, cap = tb_size(cfg)
    assert n_cw >= 1 and tbs > 0 and n_cw * 384 <= cap


def test_uncoded_mode(rng):
    cfg = LinkConfig(carrier=CarrierConfig(n_rb=6), coded=False, channel=TDLChannel(profile="FLAT"))
    r = simulate_slot(cfg, 40.0, rng)
    assert r.tb_ok is None and r.bit_errors == 0
