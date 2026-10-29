import numpy as np
import pytest
import torch

from aerialphy.neural import (MOD_CLASSES, ChEstScenario, GridDenoiser, MLPChannelEstimator,
                              ModulationClassifier, c2r, cumulant_classify, generate_chest_batch,
                              generate_mod_batch, ls_interp_grid, r2c)
from aerialphy.numerology import CarrierConfig, Numerology
from aerialphy.train import denoiser_predict, mlp_predict, train_chest, train_modclass


@pytest.fixture
def small_sc():
    return ChEstScenario(carrier=CarrierConfig(Numerology(1), n_rb=4))


def test_chest_batch_shapes(small_sc, rng):
    b = generate_chest_batch(small_sc, 5, rng)
    c = small_sc.carrier
    assert b["h_true"].shape == (5, c.n_sc, c.n_symbols)
    assert b["h_ls"].shape == (5, c.n_sc // 2, 2)


def test_c2r_roundtrip(rng):
    x = rng.standard_normal((3, 4, 5)) + 1j * rng.standard_normal((3, 4, 5))
    assert np.allclose(r2c(c2r(x)), x, atol=1e-6)


def test_ls_interp_grid_noiseless_flat(small_sc, rng):
    b = generate_chest_batch(small_sc, 3, rng, profile="EXP", snr_db=200.0, delay_spread_s=1e-9,
                             doppler_hz=0.0)
    c = small_sc.carrier
    g = ls_interp_grid(b["h_ls"], b["pilot_sc"], c.n_sc, small_sc.dmrs.symbols, c.n_symbols)
    assert np.allclose(g, b["h_true"], atol=1e-3)


@pytest.mark.parametrize("batch", [1, 4])
def test_model_shapes(batch):
    mlp = MLPChannelEstimator(24, 48)
    assert mlp(torch.zeros(batch, 2, 24), torch.zeros(batch)).shape == (batch, 2, 48)
    den = GridDenoiser().eval()
    assert den(torch.zeros(batch, 2, 48, 14), torch.zeros(batch)).shape == (batch, 2, 48, 14)
    clf = ModulationClassifier()
    assert clf(torch.zeros(batch, 2, 128)).shape == (batch, len(MOD_CLASSES))


def test_short_training_runs_and_predicts(small_sc, rng):
    mlp, den, hist = train_chest(small_sc, steps=20, batch_size=8, log_every=10, verbose=False)
    assert len(hist) == 2
    b = generate_chest_batch(small_sc, 4, rng)
    assert mlp_predict(mlp, b, small_sc).shape == b["h_true"].shape
    assert denoiser_predict(den, b, small_sc).shape == b["h_true"].shape


def test_mod_batch_normalised(rng):
    iq, lab, snr = generate_mod_batch(16, 64, rng)
    p = np.mean(iq[:, 0] ** 2 + iq[:, 1] ** 2, axis=1)
    assert np.allclose(p, 1, atol=1e-5) and set(lab) <= set(range(4))


def test_cumulant_baseline_separates_bpsk_qpsk_without_cfo(rng):
    iq, lab, snr = generate_mod_batch(400, 512, rng, snr_db=20.0, cfo_max=0.0)
    pred = cumulant_classify(iq, snr)
    sel = lab < 2
    assert np.mean(pred[sel] == lab[sel]) > 0.95


def test_modclass_training_learns(rng):
    torch.manual_seed(0)
    model = train_modclass(steps=60, batch_size=64, n_samples=128, verbose=False)
    iq, lab, _ = generate_mod_batch(400, 128, rng, snr_db=20.0)
    with torch.no_grad():
        acc = (model(torch.from_numpy(iq)).argmax(1).numpy() == lab).mean()
    assert acc > 0.4  # better than 0.25 chance after a very short run
