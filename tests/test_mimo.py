import numpy as np
import pytest

from aerialphy.channel import crandn
from aerialphy.mimo import (capacity_open_loop, capacity_svd_waterfilling, codebook_select, dft_codebook,
                            ml_detect_bruteforce, mmse_detect, mrt_precoder, ricean_ula_channel,
                            se_single_stream, svd_precoder, water_filling, zf_detect)
from aerialphy.modulation import constellation, demodulate_hard, modulate


@pytest.mark.parametrize("n_rx,n_tx", [(1, 1), (2, 1), (2, 2), (4, 2), (4, 4), (8, 4)])
@pytest.mark.parametrize("det", [zf_detect, mmse_detect])
def test_detectors_noiseless(n_rx, n_tx, det, rng):
    h = crandn((50, n_rx, n_tx), rng)
    x = modulate(rng.integers(0, 2, 50 * n_tx * 2), "qpsk").reshape(50, n_tx)
    y = np.einsum("brt,bt->br", h, x)
    xh, _ = det(h, y, 1e-9)
    assert np.allclose(xh, x, atol=1e-5)


@pytest.mark.parametrize("n", [2, 4])
@pytest.mark.parametrize("scheme", ["qpsk", "16qam"])
def test_mmse_not_worse_than_zf(n, scheme, rng):
    trials, snr = 4000, 10.0
    h = crandn((trials, n, n), rng)
    bits = rng.integers(0, 2, trials * n * (2 if scheme == "qpsk" else 4))
    x = modulate(bits, scheme).reshape(trials, n)
    nv = 10 ** (-snr / 10)
    y = np.einsum("brt,bt->br", h, x) + crandn((trials, n), rng) * np.sqrt(nv)
    ber = {}
    for name, det in (("zf", zf_detect), ("mmse", mmse_detect)):
        xh, _ = det(h, y, nv)
        ber[name] = np.mean(demodulate_hard(xh.reshape(-1), scheme) != bits)
    assert ber["mmse"] <= ber["zf"]


def test_ml_beats_linear_on_2x2(rng):
    trials, nv = 3000, 10 ** (-1.0)
    h = crandn((trials, 2, 2), rng)
    bits = rng.integers(0, 2, trials * 4)
    x = modulate(bits, "qpsk").reshape(trials, 2)
    y = np.einsum("brt,bt->br", h, x) + crandn((trials, 2), rng) * np.sqrt(nv)
    ml = ml_detect_bruteforce(h, y, constellation("qpsk"))
    xm, _ = mmse_detect(h, y, nv)
    e_ml = np.mean(demodulate_hard(ml.reshape(-1), "qpsk") != bits)
    e_mm = np.mean(demodulate_hard(xm.reshape(-1), "qpsk") != bits)
    assert e_ml <= e_mm


def test_zf_noise_enhancement_matches_formula(rng):
    h = crandn((1, 3, 3), rng)
    _, nv = zf_detect(h, np.zeros((1, 3)), 0.5)
    expected = 0.5 * np.real(np.diag(np.linalg.inv(h[0].conj().T @ h[0])))
    assert np.allclose(nv[0], expected)


def test_mmse_sinr_reasonable(rng):
    h = crandn((200, 4, 4), rng)
    _, nv_m = mmse_detect(h, np.zeros((200, 4)), 0.1)
    _, nv_z = zf_detect(h, np.zeros((200, 4)), 0.1)
    # post-detection SINR of unbiased MMSE is never lower than ZF SNR
    assert np.all(1 / nv_m >= 1 / nv_z - 1e-9)


@pytest.mark.parametrize("gains,power", [([1.0, 0.5, 0.1], 1.0), ([10.0, 1.0], 0.05), ([2.0], 3.0), ([5, 4, 3, 2], 10.0)])
def test_water_filling(gains, power):
    p = water_filling(np.array(gains), power)
    assert np.isclose(p.sum(), power) and np.all(p >= 0)
    active = p > 0
    lvl = p[active] + 1 / np.array(gains)[active]
    assert np.allclose(lvl, lvl[0])


@pytest.mark.parametrize("n", [2, 4, 8])
@pytest.mark.parametrize("snr_db", [0, 10, 20])
def test_closed_loop_capacity_dominates(n, snr_db, rng):
    h = crandn((100, n, n), rng)
    snr = 10 ** (snr_db / 10)
    assert np.all(capacity_svd_waterfilling(h, snr) >= capacity_open_loop(h, snr) - 1e-9)


def test_svd_precoder_orthonormal(rng):
    h = crandn((10, 4, 4), rng)
    v = svd_precoder(h, 2)
    assert np.allclose(np.swapaxes(v.conj(), -1, -2) @ v, np.eye(2))


@pytest.mark.parametrize("n_tx", [2, 4, 8])
def test_mrt_maximises_gain(n_tx, rng):
    h = crandn((200, 2, n_tx), rng)
    w = mrt_precoder(h)
    cb = codebook_select(h, dft_codebook(n_tx))
    rnd = crandn((200, n_tx), rng)
    rnd /= np.linalg.norm(rnd, axis=1, keepdims=True)
    se_mrt, se_cb, se_rnd = (se_single_stream(h, x, 10.0) for x in (w, cb, rnd))
    assert np.all(se_mrt >= se_cb - 1e-9)
    assert se_cb.mean() > se_rnd.mean()


def test_dft_codebook_unit_norm():
    cb = dft_codebook(8, 4)
    assert cb.shape == (8, 32) and np.allclose(np.linalg.norm(cb, axis=0), 1)


@pytest.mark.parametrize("k_db", [-10, 0, 10])
def test_ricean_power(k_db, rng):
    h = ricean_ula_channel(4, 8, k_db, rng, batch=2000)
    assert 0.9 < np.mean(np.abs(h) ** 2) < 1.1
