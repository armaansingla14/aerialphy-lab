"""Channel-estimation NMSE: LS vs LMMSE vs neural (MLP) vs CNN denoiser.

Trains the two neural models on a synthetic stream of random TDL-like channels
(profiles A/C/EXP, delay spread 30-300 ns, Doppler 0-200 Hz, SNR 0-25 dB), then
evaluates all estimators on fresh samples over a grid of channel conditions.

    python experiments/chest_nmse.py [--steps 2000] [--samples 300]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from common import md_table, write_csv, write_md

from aerialphy.channel import TDLChannel
from aerialphy.estimation import PilotEstimator, pdp_correlation, uniform_pdp_correlation
from aerialphy.neural import ChEstScenario, generate_chest_batch, ls_interp_grid
from aerialphy.train import denoiser_predict, mlp_predict, train_chest


def nmse(est: np.ndarray, true: np.ndarray) -> float:
    """Mean over samples of per-sample normalised squared error (linear)."""
    e = np.sum(np.abs(est - true) ** 2, axis=(1, 2)) / np.sum(np.abs(true) ** 2, axis=(1, 2))
    return float(np.mean(e))


def lmmse_batch(b: dict, sc: ChEstScenario, corr_fn) -> np.ndarray:
    c = sc.carrier
    out = np.empty_like(b["h_true"])
    for i in range(out.shape[0]):
        nv = 10 ** (-b["snr_db"][i] / 10)
        est = PilotEstimator("lmmse", corr_fn)
        out[i] = est.estimate(b["pilot_sc"].astype(float), b["h_ls"][i], c.n_sc, sc.dmrs.symbols,
                              c.n_symbols, nv)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    sc = ChEstScenario()
    t0 = time.time()
    print(f"training MLP estimator + CNN denoiser for {args.steps} steps")
    mlp, den, _ = train_chest(sc, steps=args.steps)
    train_s = time.time() - t0
    n_params = {"mlp": sum(p.numel() for p in mlp.parameters()),
                "denoiser": sum(p.numel() for p in den.parameters())}
    rng = np.random.default_rng(2024)
    c = sc.carrier
    rows = []
    for profile in ("A", "C", "EXP"):
        for ds in (100e-9, 300e-9):
            ch = TDLChannel(profile=profile, delay_spread_s=ds)
            genie = pdp_correlation(*ch.pdp(), c.numerology.scs_hz)
            robust = uniform_pdp_correlation(c.numerology.scs_hz, c.cp_length / c.sample_rate_hz)
            for snr in (0.0, 5.0, 10.0, 15.0, 20.0, 25.0):
                b = generate_chest_batch(sc, args.samples, rng, profile=profile, snr_db=snr,
                                         delay_spread_s=ds, doppler_hz=100.0)
                ls = ls_interp_grid(b["h_ls"], b["pilot_sc"], c.n_sc, sc.dmrs.symbols, c.n_symbols)
                res = {
                    "ls": nmse(ls, b["h_true"]),
                    "lmmse_robust": nmse(lmmse_batch(b, sc, robust), b["h_true"]),
                    "lmmse_genie": nmse(lmmse_batch(b, sc, genie), b["h_true"]),
                    "mlp": nmse(mlp_predict(mlp, b, sc), b["h_true"]),
                    "cnn_denoiser": nmse(denoiser_predict(den, b, sc), b["h_true"]),
                }
                row = {"profile": profile, "delay_spread_ns": int(ds * 1e9), "snr_db": snr}
                row.update({f"nmse_db_{k}": 10 * np.log10(v) for k, v in res.items()})
                rows.append(row)
                print(", ".join(f"{k}={v}" if not isinstance(v, float) else f"{k}={v:.2f}" for k, v in row.items()))
    write_csv("chest_nmse.csv", rows)
    methods = ["ls", "lmmse_robust", "lmmse_genie", "mlp", "cnn_denoiser"]
    # summary averaged (in linear domain) over all conditions, and per SNR
    summary = []
    for snr in sorted({r["snr_db"] for r in rows}):
        sel = [r for r in rows if r["snr_db"] == snr]
        s = {"snr_db": snr}
        for m in methods:
            s[m] = 10 * np.log10(np.mean([10 ** (r[f"nmse_db_{m}"] / 10) for r in sel]))
        summary.append(s)
    overall = {"snr_db": "all"}
    for m in methods:
        overall[m] = 10 * np.log10(np.mean([10 ** (r[f"nmse_db_{m}"] / 10) for r in rows]))
    summary.append(overall)
    wins = {m: sum(r[f"nmse_db_{m}"] < r["nmse_db_ls"] for r in rows) for m in methods[1:]}
    n_cond = len(rows)
    fmt = {m: ".2f" for m in methods}
    text = (
        "# Channel-estimation NMSE (simulated, SISO, DMRS type 1, 2 DMRS symbols)\n\n"
        f"12 PRB at 30 kHz SCS. {n_cond} channel conditions x {args.samples} test samples = "
        f"{n_cond * args.samples} samples per method. Doppler fixed at 100 Hz for evaluation.\n"
        f"Training: {args.steps} steps of batch 32 on random channels ({train_s:.0f} s on CPU). "
        f"Parameters: MLP {n_params['mlp']}, CNN denoiser {n_params['denoiser']}.\n"
        f"Command: `python experiments/chest_nmse.py --steps {args.steps} --samples {args.samples}`\n\n"
        "lmmse_genie uses the true PDP and delay spread; lmmse_robust assumes a uniform PDP over the CP.\n\n"
        "## NMSE in dB, averaged over profiles A/C/EXP and delay spreads 100/300 ns\n\n"
        + md_table(summary, ["snr_db"] + methods, fmt)
        + "\n\n## Conditions (out of "
        + str(n_cond) + ") where each method beats LS\n\n"
        + "\n".join(f"- {m}: {wins[m]}" for m in wins)
        + "\n\n## Full table\n\n"
        + md_table(rows, None, {f"nmse_db_{m}": ".2f" for m in methods} | {"snr_db": ".0f"})
    )
    write_md("chest_nmse.md", text)


if __name__ == "__main__":
    main()
