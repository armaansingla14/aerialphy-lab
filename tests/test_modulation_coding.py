import numpy as np
import pytest

from aerialphy.coding import QCLDPC, crc_attach, crc_check, crc_bits
from aerialphy.modulation import (bits_per_symbol, constellation, demodulate_hard, demodulate_llr,
                                  modulate)

SCHEMES = ["bpsk", "qpsk", "16qam", "64qam"]


@pytest.mark.parametrize("scheme", SCHEMES)
def test_unit_energy(scheme):
    assert np.isclose(np.mean(np.abs(constellation(scheme)) ** 2), 1.0)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_constellation_unique(scheme):
    c = constellation(scheme)
    assert np.unique(np.round(c, 9)).size == c.size == 2 ** bits_per_symbol(scheme)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_gray_property(scheme):
    """Nearest neighbours differ in exactly one bit."""
    c = constellation(scheme)
    if c.size <= 2:
        return
    d = np.abs(c[:, None] - c[None, :])
    np.fill_diagonal(d, np.inf)
    dmin = d.min()
    for i in range(c.size):
        for j in np.nonzero(np.isclose(d[i], dmin))[0]:
            assert bin(i ^ j).count("1") == 1


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("n_sym", [1, 7, 500])
def test_mod_demod_noiseless(scheme, n_sym, rng):
    bits = rng.integers(0, 2, n_sym * bits_per_symbol(scheme)).astype(np.uint8)
    assert np.array_equal(demodulate_hard(modulate(bits, scheme), scheme), bits)


def test_qpsk_matches_38211():
    s = modulate(np.array([0, 0, 0, 1, 1, 0, 1, 1]), "qpsk") * np.sqrt(2)
    assert np.allclose(s, [1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j])


def test_16qam_matches_38211():
    # b = 0011 -> I = (1)(2-(-1)) = 3, Q = (1)(2-(-1)) = 3
    s = modulate(np.array([0, 0, 1, 1]), "16qam") * np.sqrt(10)
    assert np.isclose(s[0], 3 + 3j)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_llr_sign_consistent_with_hard(scheme, rng):
    y = crandn = (rng.standard_normal(200) + 1j * rng.standard_normal(200))
    llr = demodulate_llr(y, scheme, 0.1)
    assert np.array_equal((llr < 0).astype(np.uint8), demodulate_hard(y, scheme))


def test_bad_scheme():
    with pytest.raises(ValueError):
        modulate(np.zeros(4), "8psk")


@pytest.mark.parametrize("poly", ["crc24a", "crc24b", "crc16", "crc11"])
@pytest.mark.parametrize("n", [1, 40, 333])
def test_crc_detects_single_errors(poly, n, rng):
    b = rng.integers(0, 2, n).astype(np.uint8)
    cw = crc_attach(b, poly)
    assert crc_check(cw, poly)
    for pos in rng.choice(cw.size, size=min(5, cw.size), replace=False):
        bad = cw.copy()
        bad[pos] ^= 1
        assert not crc_check(bad, poly)


def test_crc_linear(rng):
    a = rng.integers(0, 2, 64).astype(np.uint8)
    b = rng.integers(0, 2, 64).astype(np.uint8)
    assert np.array_equal(crc_bits(a ^ b), crc_bits(a) ^ crc_bits(b))


@pytest.mark.parametrize("z", [8, 16, 32])
def test_ldpc_encode_valid(z, rng):
    code = QCLDPC(z=z)
    u = rng.integers(0, 2, (4, code.k))
    c = code.encode(u)
    assert code.syndrome_ok(c).all()
    assert np.array_equal(code.info_bits(c), u)
    assert 0.45 < code.rate < 0.6


@pytest.mark.parametrize("z", [16, 32])
def test_ldpc_noiseless_decode_one_iteration(z, rng):
    code = QCLDPC(z=z)
    c = code.encode(rng.integers(0, 2, (3, code.k)))
    hard, iters = code.decode(4.0 * (1 - 2 * c.astype(float)))
    assert np.array_equal(hard, c) and np.all(iters <= 1)


@pytest.mark.parametrize("n_flip", [1, 3, 6])
def test_ldpc_corrects_errors(n_flip, rng):
    code = QCLDPC(z=32)
    c = code.encode(rng.integers(0, 2, (1, code.k)))[0]
    llr = 2.0 * (1 - 2 * c.astype(float))
    idx = rng.choice(code.n, n_flip, replace=False)
    llr[idx] *= -0.5
    hard, _ = code.decode(llr)
    assert np.array_equal(hard[0], c)


def test_ldpc_coding_gain(rng):
    """At 2 dB Es/N0 QPSK, decoded BER must be well below raw BER."""
    code = QCLDPC(z=32)
    u = rng.integers(0, 2, (20, code.k))
    c = code.encode(u)
    x = modulate(c.reshape(-1), "qpsk")
    nv = 10 ** (-2 / 10)
    y = x + np.sqrt(nv / 2) * (rng.standard_normal(x.size) + 1j * rng.standard_normal(x.size))
    llr = demodulate_llr(y, "qpsk", nv).reshape(c.shape)
    raw = np.mean((llr < 0) != c)
    hard, _ = code.decode(llr)
    dec = np.mean(code.info_bits(hard) != u)
    assert dec < raw / 5
