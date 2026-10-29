"""Training loops for the small neural models (CPU, synthetic data)."""
from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

from .neural import (ChEstScenario, GridDenoiser, MLPChannelEstimator, ModulationClassifier,
                     c2r, generate_chest_batch, generate_mod_batch, ls_interp_grid)


def _pilot_tensor(h_ls: np.ndarray) -> torch.Tensor:
    """(B, n_pilots, n_dmrs) complex -> (B * n_dmrs, 2, n_pilots) float."""
    x = np.moveaxis(h_ls, 2, 1).reshape(-1, h_ls.shape[1])
    return c2r(x)


def mlp_predict(model: MLPChannelEstimator, batch: dict, sc: ChEstScenario) -> np.ndarray:
    from .estimation import interp_time
    from .neural import r2c
    n_dm = len(sc.dmrs.symbols)
    pil = _pilot_tensor(batch["h_ls"])
    snr = torch.from_numpy(np.repeat(batch["snr_db"], n_dm).astype(np.float32))
    with torch.no_grad():
        cols = r2c(model(pil, snr)).reshape(-1, n_dm, sc.carrier.n_sc)
    return interp_time(np.swapaxes(cols, 1, 2), sc.dmrs.symbols, sc.carrier.n_symbols)


def denoiser_predict(model: GridDenoiser, batch: dict, sc: ChEstScenario) -> np.ndarray:
    from .neural import r2c
    c = sc.carrier
    g = ls_interp_grid(batch["h_ls"], batch["pilot_sc"], c.n_sc, sc.dmrs.symbols, c.n_symbols)
    model.eval()
    with torch.no_grad():
        out = model(c2r(g), torch.from_numpy(batch["snr_db"].astype(np.float32)))
    return r2c(out)


def train_chest(sc: ChEstScenario, steps: int = 1500, batch_size: int = 32, seed: int = 0,
                log_every: int = 250, verbose: bool = True):
    """Train both the MLP estimator and the CNN denoiser on the same synthetic stream."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    c = sc.carrier
    n_p = sc.dmrs.pilot_subcarriers(c, 0).size
    mlp = MLPChannelEstimator(n_p, c.n_sc)
    den = GridDenoiser()
    opt_m = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    opt_d = torch.optim.Adam(den.parameters(), lr=1e-3)
    sched_m = torch.optim.lr_scheduler.CosineAnnealingLR(opt_m, steps)
    sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, steps)
    mse = nn.MSELoss()
    t0 = time.time()
    hist = []
    for step in range(1, steps + 1):
        b = generate_chest_batch(sc, batch_size, rng)
        n_dm = len(sc.dmrs.symbols)
        # MLP: per DMRS symbol column
        pil = _pilot_tensor(b["h_ls"])
        snr_col = torch.from_numpy(np.repeat(b["snr_db"], n_dm).astype(np.float32))
        tgt_col = c2r(np.moveaxis(b["h_true"][:, :, sc.dmrs.symbols], 2, 1).reshape(-1, c.n_sc))
        loss_m = mse(mlp(pil, snr_col), tgt_col)
        opt_m.zero_grad()
        loss_m.backward()
        opt_m.step()
        sched_m.step()
        # denoiser: full grid
        den.train()
        g = ls_interp_grid(b["h_ls"], b["pilot_sc"], c.n_sc, sc.dmrs.symbols, c.n_symbols)
        loss_d = mse(den(c2r(g), torch.from_numpy(b["snr_db"].astype(np.float32))), c2r(b["h_true"]))
        opt_d.zero_grad()
        loss_d.backward()
        opt_d.step()
        sched_d.step()
        if step % log_every == 0:
            hist.append((step, loss_m.item(), loss_d.item()))
            if verbose:
                print(f"  step {step:5d}  mlp {loss_m.item():.4f}  denoiser {loss_d.item():.4f}  ({time.time() - t0:.0f}s)")
    return mlp.eval(), den.eval(), hist


def train_modclass(steps: int = 1500, batch_size: int = 128, n_samples: int = 256, seed: int = 0,
                   verbose: bool = True):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 100)
    model = ModulationClassifier()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    ce = nn.CrossEntropyLoss()
    t0 = time.time()
    for step in range(1, steps + 1):
        iq, lab, _ = generate_mod_batch(batch_size, n_samples, rng)
        loss = ce(model(torch.from_numpy(iq)), torch.from_numpy(lab))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if verbose and step % 250 == 0:
            print(f"  step {step:5d}  ce {float(loss):.4f}  ({time.time() - t0:.0f}s)")
    return model.eval()
