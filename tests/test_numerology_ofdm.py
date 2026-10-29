import numpy as np
import pytest

from aerialphy.channel import TDLChannel, apply_time_domain, crandn
from aerialphy.link import subcarrier_freqs
from aerialphy.numerology import CarrierConfig, Numerology, ResourceGrid
from aerialphy.ofdm import ofdm_demodulate, ofdm_modulate


@pytest.mark.parametrize("mu,scs,slots", [(0, 15e3, 10), (1, 30e3, 20), (2, 60e3, 40), (3, 120e3, 80), (4, 240e3, 160)])
def test_numerology_table(mu, scs, slots):
    n = Numerology(mu)
    assert n.scs_hz == scs
    assert n.slots_per_frame == slots
    assert np.isclose(n.slot_duration_s * n.slots_per_subframe, 1e-3)


def test_invalid_numerology():
    with pytest.raises(ValueError):
        Numerology(5)


@pytest.mark.parametrize("n_rb", [6, 11, 24, 52, 106, 273])
def test_fft_size_covers_band(n_rb):
    c = CarrierConfig(n_rb=n_rb)
    assert c.fft_size >= c.n_sc
    assert c.fft_size & (c.fft_size - 1) == 0
    assert c.cp_length > 0


@pytest.mark.parametrize("mu", [0, 1, 2])
@pytest.mark.parametrize("n_rb", [6, 24, 52])
@pytest.mark.parametrize("n_ports", [1, 2])
def test_ofdm_roundtrip(mu, n_rb, n_ports, rng):
    c = CarrierConfig(Numerology(mu), n_rb=n_rb)
    g = crandn((n_ports, c.n_sc, c.n_symbols), rng)
    x = ofdm_modulate(g, c)
    assert x.shape == (n_ports, c.n_symbols * (c.fft_size + c.cp_length))
    assert np.allclose(ofdm_demodulate(x, c), g)


def test_ofdm_power_preserved(rng):
    c = CarrierConfig(n_rb=24)
    g = crandn((c.n_sc, c.n_symbols), rng)
    x = ofdm_modulate(g, c)
    useful = x.reshape(c.n_symbols, -1)[:, c.cp_length:]
    assert np.isclose(np.sum(np.abs(useful) ** 2), np.sum(np.abs(g) ** 2))


@pytest.mark.parametrize("profile", ["A", "C", "EXP"])
@pytest.mark.parametrize("ds", [30e-9, 60e-9, 100e-9])
def test_time_domain_multipath_equals_per_re_channel(profile, ds, rng):
    """With delays inside the CP, OFDM turns the channel into a per-RE multiply."""
    c = CarrierConfig(Numerology(1), n_rb=24)
    ch = TDLChannel(profile=profile, delay_spread_s=ds)
    delays, _ = ch.pdp()
    d = np.round(delays * c.sample_rate_hz).astype(int)
    assert d.max() < c.cp_length
    taps = ch.sample_taps(np.zeros(1), rng)[0, :, 0, 0]
    g = crandn((c.n_sc, c.n_symbols), rng)
    y = ofdm_demodulate(apply_time_domain(ofdm_modulate(g, c), taps, d), c)
    # frequency response for the rounded delays
    freqs = subcarrier_freqs(c)
    h = np.exp(-2j * np.pi * np.outer(freqs, d / c.sample_rate_hz)) @ taps
    assert np.allclose(y, h[:, None] * g, atol=1e-9)


def test_resource_grid_collision():
    c = CarrierConfig(n_rb=6)
    g = ResourceGrid(c)
    m = np.zeros((c.n_sc, c.n_symbols), bool)
    m[:, 0] = True
    g.mark(m, ResourceGrid.DMRS)
    with pytest.raises(ValueError):
        g.mark(m, ResourceGrid.DATA)


def test_resource_grid_frequency_first_mapping():
    c = CarrierConfig(n_rb=1, n_symbols=2)
    g = ResourceGrid(c)
    g.mark(np.ones((12, 2), bool), ResourceGrid.DATA)
    g.map_data(np.arange(24).astype(complex))
    assert g.values[0, 1, 0] == 1 and g.values[0, 0, 1] == 12
    assert np.array_equal(g.extract_data(g.values[0]), np.arange(24))
