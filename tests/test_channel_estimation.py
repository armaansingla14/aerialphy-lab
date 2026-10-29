import numpy as np
import pytest

from aerialphy.channel import TDLChannel, crandn, exp_correlation
from aerialphy.dmrs import DmrsConfig, build_grid, dmrs_sequence, gold_sequence
from aerialphy.estimation import (PilotEstimator, interp_frequency, interp_time, linear_interp_matrix,
                                  nmse_db, pdp_correlation, uniform_pdp_correlation)
from aerialphy.link import subcarrier_freqs, symbol_times
from aerialphy.numerology import CarrierConfig, Numerology


@pytest.mark.parametrize("profile", ["A", "C", "EXP", "FLAT"])
def test_pdp_normalised(profile):
    _, p = TDLChannel(profile=profile).pdp()
    assert np.isclose(p.sum(), 1.0)


@pytest.mark.parametrize("profile", ["A", "C", "EXP"])
@pytest.mark.parametrize("n_rx,n_tx", [(1, 1), (2, 2), (4, 2)])
def test_channel_average_gain(profile, n_rx, n_tx, rng):
    ch = TDLChannel(profile=profile, n_rx=n_rx, n_tx=n_tx)
    c = CarrierConfig(n_rb=4)
    p = np.mean([np.mean(np.abs(ch.frequency_response(subcarrier_freqs(c), np.zeros(1), rng)) ** 2)
                 for _ in range(400)])
    assert 0.85 < p < 1.15


def test_freq_correlation_matches_empirical(rng):
    ch = TDLChannel(profile="C", delay_spread_s=300e-9)
    c = CarrierConfig(n_rb=4)
    f = subcarrier_freqs(c)
    hs = np.stack([ch.frequency_response(f, np.zeros(1), rng)[0, 0, :, 0] for _ in range(3000)])
    emp = hs.T @ hs.conj() / hs.shape[0]
    assert np.max(np.abs(emp - ch.freq_correlation(f))) < 0.1


@pytest.mark.parametrize("doppler", [0.0, 100.0, 500.0])
def test_doppler_time_variation(doppler, rng):
    ch = TDLChannel(profile="FLAT", doppler_hz=doppler)
    t = np.linspace(0, 5e-3, 50)
    g = ch.sample_taps(t, rng)[:, 0, 0, 0]
    var = np.std(np.abs(g))
    if doppler == 0:
        assert var < 1e-12
    else:
        assert var > 0


def test_spatial_correlation(rng):
    ch = TDLChannel(profile="FLAT", n_rx=2, n_tx=1, rho_rx=0.9)
    g = np.stack([ch.sample_taps(np.zeros(1), rng)[0, 0, :, 0] for _ in range(4000)])
    c = np.abs(np.mean(g[:, 0] * g[:, 1].conj()))
    assert 0.8 < c < 1.0
    assert np.allclose(exp_correlation(3, 0.5)[0], [1, 0.5, 0.25])


def test_gold_sequence_known_prefix():
    # c_init = 0: x2 is all-zero so c(n) = x1(n + Nc) which is deterministic
    a = gold_sequence(0, 32)
    b = gold_sequence(0, 32)
    assert np.array_equal(a, b) and a.dtype == np.uint8
    assert not np.array_equal(gold_sequence(1, 64), gold_sequence(2, 64))


@pytest.mark.parametrize("c_init", [0, 1, 12345, 2 ** 31 - 1])
def test_dmrs_sequence_qpsk(c_init):
    s = dmrs_sequence(c_init, 100)
    assert np.allclose(np.abs(s), 1.0)


@pytest.mark.parametrize("n_ports", [1, 2, 3, 4])
def test_dmrs_grid(n_ports):
    c = CarrierConfig(n_rb=6)
    d = DmrsConfig()
    g = build_grid(c, d, n_ports)
    assert g.num_data_re() == c.n_sc * (c.n_symbols - len(d.symbols))
    for p in range(n_ports):
        assert np.allclose(np.abs(g.values[p][d.pilot_subcarriers(c, p)][:, d.symbols]), 1.0)


@pytest.mark.parametrize("n_ports", [2, 4])
def test_occ_separates_ports_on_flat_channel(n_ports, rng):
    c = CarrierConfig(n_rb=6)
    d = DmrsConfig()
    g = build_grid(c, d, n_ports, slot=3)
    h = crandn(n_ports, rng)
    y = np.einsum("p,pks->ks", h, g.values)
    for p in range(n_ports):
        _, est, _ = d.ls_estimates(y, c, 3, p, n_ports)
        assert np.allclose(est, h[p])


def test_interp_matrix_exact_on_lines():
    pos = np.array([1.0, 5.0, 9.0])
    m = linear_interp_matrix(pos, 12)
    v = 2 * pos + 1
    assert np.allclose(m @ v, 2 * np.arange(12) + 1)
    hold = linear_interp_matrix(pos, 12, extrapolate=False)
    assert np.isclose((hold @ v)[0], v[0]) and np.isclose((hold @ v)[-1], v[-1])


def test_interp_time_single_symbol():
    h = np.arange(4.0)[:, None] + 0j
    out = interp_time(h, [3], 14)
    assert out.shape == (4, 14) and np.allclose(out, h)


def _estimate_case(method, profile, snr_db, rng, n=30):
    c = CarrierConfig(Numerology(1), n_rb=12)
    d = DmrsConfig()
    ch = TDLChannel(profile=profile, delay_spread_s=100e-9)
    corr = pdp_correlation(*ch.pdp(), c.numerology.scs_hz) if method == "lmmse" else None
    est = PilotEstimator(method, corr)
    nv = 10 ** (-snr_db / 10)
    errs = []
    for _ in range(n):
        h = ch.frequency_response(subcarrier_freqs(c), symbol_times(c), rng)[0, 0]
        g = build_grid(c, d, 1)
        y = h * g.values[0] + crandn(h.shape, rng) * np.sqrt(nv)
        pos, hls, sc = d.ls_estimates(y, c, 0, 0, 1)
        errs.append(np.sum(np.abs(est.estimate(pos, hls, c.n_sc, d.symbols, c.n_symbols, nv * sc) - h) ** 2)
                    / np.sum(np.abs(h) ** 2))
    return 10 * np.log10(np.mean(errs))


@pytest.mark.parametrize("profile", ["A", "C", "EXP"])
@pytest.mark.parametrize("snr_db", [0, 10, 20])
def test_lmmse_beats_ls(profile, snr_db, rng):
    assert _estimate_case("lmmse", profile, snr_db, rng) < _estimate_case("ls", profile, snr_db, rng)


@pytest.mark.parametrize("method", ["ls", "lmmse"])
def test_estimation_improves_with_snr(method, rng):
    assert _estimate_case(method, "A", 25, rng) < _estimate_case(method, "A", 5, rng) - 5


def test_uniform_pdp_correlation_is_hermitian():
    fn = uniform_pdp_correlation(30e3, 1e-6)
    k = np.arange(10.0)
    r = fn(k[:, None] - k[None, :])
    assert np.allclose(r, r.conj().T) and np.allclose(np.diag(r), 1)


def test_nmse_zero_for_perfect():
    h = np.ones(5) + 1j
    assert nmse_db(h * 1.001, h) < -50


def test_lmmse_requires_correlation():
    with pytest.raises(ValueError):
        PilotEstimator("lmmse")
