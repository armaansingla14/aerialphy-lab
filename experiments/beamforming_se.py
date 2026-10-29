"""Spectral efficiency with and without beamforming (Shannon bounds on simulated channels).

    python experiments/beamforming_se.py [--draws 5000]
"""
from __future__ import annotations

import argparse

import numpy as np

from common import md_table, write_csv, write_md

from aerialphy.channel import crandn
from aerialphy.mimo import (capacity_open_loop, capacity_svd_waterfilling, codebook_select, dft_codebook,
                            mrt_precoder, ricean_ula_channel, se_single_stream)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=5000)
    args = ap.parse_args()
    rng = np.random.default_rng(7)
    rows = []
    channels = {
        "rayleigh": lambda nr, nt: crandn((args.draws, nr, nt), rng),
        "ricean_k10dB_ula": lambda nr, nt: ricean_ula_channel(nr, nt, 10.0, rng, args.draws),
    }
    for chname, gen in channels.items():
        for n_rx, n_tx in ((1, 4), (2, 4), (1, 8), (2, 8), (4, 4)):
            h = gen(n_rx, n_tx)
            cb = dft_codebook(n_tx, 4)
            w_cb = codebook_select(h, cb)
            w_mrt = mrt_precoder(h)
            e1 = np.zeros(n_tx, dtype=complex)
            e1[0] = 1
            w_single = np.broadcast_to(e1, (args.draws, n_tx))
            for snr_db in (-5, 0, 5, 10, 20):
                snr = 10 ** (snr_db / 10)
                row = {
                    "channel": chname, "mimo": f"{n_rx}x{n_tx}", "snr_db": snr_db,
                    "se_single_antenna": float(np.mean(se_single_stream(h, w_single, snr))),
                    "se_open_loop": float(np.mean(capacity_open_loop(h, snr))),
                    "se_dft_codebook": float(np.mean(se_single_stream(h, w_cb, snr))),
                    "se_eigen_bf": float(np.mean(se_single_stream(h, w_mrt, snr))),
                    "se_svd_waterfill": float(np.mean(capacity_svd_waterfilling(h, snr))),
                }
                rows.append(row)
    write_csv("beamforming_se.csv", rows)
    cols = ["channel", "mimo", "snr_db", "se_single_antenna", "se_open_loop", "se_dft_codebook", "se_eigen_bf",
            "se_svd_waterfill"]
    fmt = {c: ".2f" for c in cols[3:]}
    text = (
        "# Spectral efficiency with and without beamforming (simulated, bits/s/Hz)\n\n"
        f"Mean over {args.draws} channel draws per row; Shannon rates log2(1+SNR) per stream.\n"
        "Command: `python experiments/beamforming_se.py --draws " + str(args.draws) + "`\n\n"
        "- se_single_antenna: one transmit antenna, MRC at the receiver (no beamforming)\n"
        "- se_open_loop: all antennas, equal power, no CSI at transmitter\n"
        "- se_dft_codebook: one stream on the best beam of an oversampled DFT codebook\n"
        "- se_eigen_bf: one stream on the dominant right singular vector (full CSIT)\n"
        "- se_svd_waterfill: SVD multi-stream with water-filling (full CSIT)\n\n"
        + md_table(rows, cols, fmt)
    )
    write_md("beamforming_se.md", text)
    print(text[:3000])


if __name__ == "__main__":
    main()
