"""Modulation classification accuracy vs SNR: 1-D CNN vs fourth-order cumulant baseline.

    python experiments/modclass.py [--steps 1500] [--per-snr 2000]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from common import md_table, write_csv, write_md

from aerialphy.neural import MOD_CLASSES, cumulant_classify, generate_mod_batch
from aerialphy.train import train_modclass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--per-snr", type=int, default=2000)
    ap.add_argument("--samples", type=int, default=256)
    args = ap.parse_args()
    torch.set_num_threads(4)
    t0 = time.time()
    model = train_modclass(steps=args.steps, n_samples=args.samples)
    train_s = time.time() - t0
    rng = np.random.default_rng(99)
    rows = []
    conf = np.zeros((4, 4), dtype=int)
    for snr in (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0):
        iq, lab, snrs = generate_mod_batch(args.per_snr, args.samples, rng, snr_db=snr)
        with torch.no_grad():
            pred = np.concatenate([model(torch.from_numpy(iq[i:i + 250])).argmax(1).numpy()
                                   for i in range(0, len(iq), 250)])
        base = cumulant_classify(iq, snrs)
        if snr >= 10:
            np.add.at(conf, (lab, pred), 1)
        rows.append({"snr_db": snr, "cnn_accuracy": float(np.mean(pred == lab)),
                     "cumulant_accuracy": float(np.mean(base == lab)), "n_test": args.per_snr})
        print(rows[-1])
    write_csv("modclass.csv", rows)
    conf_rows = [{"true \\ pred": MOD_CLASSES[i], **{MOD_CLASSES[j]: int(conf[i, j]) for j in range(4)}}
                 for i in range(4)]
    n_params = sum(p.numel() for p in model.parameters())
    text = (
        "# Modulation classification (simulated)\n\n"
        f"Classes: {', '.join(MOD_CLASSES)}. {args.samples} symbols per example with random phase, "
        f"residual CFO up to 1e-3 cycles/symbol, AWGN, AGC normalisation. CNN: {n_params} parameters, "
        f"trained {args.steps} steps x 128 examples on SNR uniform in [-5, 20] dB ({train_s:.0f} s CPU).\n"
        "Baseline: nearest theoretical normalised cumulants (|C40|, C42) with SNR-corrected dilution.\n"
        f"Command: `python experiments/modclass.py --steps {args.steps} --per-snr {args.per_snr}`\n\n"
        + md_table(rows, None, {"cnn_accuracy": ".3f", "cumulant_accuracy": ".3f", "snr_db": ".0f"})
        + "\n\n## CNN confusion matrix (SNR 10/15/20 dB pooled)\n\n" + md_table(conf_rows)
    )
    write_md("modclass.md", text)


if __name__ == "__main__":
    main()
