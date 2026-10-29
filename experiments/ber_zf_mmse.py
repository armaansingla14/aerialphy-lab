"""Uncoded BER vs SNR for ZF and MMSE detection over a TDL-A-like MIMO channel.

Every SNR point uses the same channel/noise realisations for both detectors
(same seed), so the comparison is paired. Perfect CSI and LMMSE-estimated CSI
are both evaluated.

    python experiments/ber_zf_mmse.py [--slots 30]
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from common import md_table, write_csv, write_md

from aerialphy.channel import TDLChannel
from aerialphy.link import LinkConfig, simulate_slot
from aerialphy.numerology import CarrierConfig, Numerology


def run(slots: int, snrs: list[float]) -> list[dict]:
    rows = []
    carrier = CarrierConfig(Numerology(1), n_rb=24)
    cases = [(2, 2, "qpsk"), (2, 2, "16qam"), (4, 4, "qpsk"), (4, 4, "16qam")]
    for n_tx, n_rx, mod in cases:
        for est in ("perfect", "lmmse"):
            for snr in snrs:
                row = {"mimo": f"{n_rx}x{n_tx}", "modulation": mod, "csi": est, "snr_db": snr}
                for det in ("zf", "mmse"):
                    rng = np.random.default_rng(1000 + int(snr * 10))
                    cfg = LinkConfig(carrier=carrier, n_layers=n_tx, n_rx=n_rx, modulation=mod, estimator=est,
                                     detector=det, coded=False,
                                     channel=TDLChannel(profile="A", delay_spread_s=100e-9, doppler_hz=50.0))
                    err = bits = 0
                    for s in range(slots):
                        r = simulate_slot(cfg, snr, rng, slot=s % 20)
                        err += r.bit_errors
                        bits += r.n_bits
                    row[f"ber_{det}"] = err / bits
                row["bits_per_point"] = bits
                rows.append(row)
                print(f"{row['mimo']} {mod:5s} {est:7s} {snr:5.1f} dB  ZF {row['ber_zf']:.3e}  MMSE {row['ber_mmse']:.3e}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", type=int, default=30)
    args = ap.parse_args()
    t0 = time.time()
    rows = run(args.slots, [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
    write_csv("ber_zf_mmse.csv", rows)
    perfect = [r for r in rows if r["csi"] == "perfect"]
    for r in rows:
        r["mmse_gain"] = r["ber_zf"] / r["ber_mmse"] if r["ber_mmse"] > 0 else float("nan")
    text = ("# Uncoded BER: ZF vs MMSE (simulated)\n\n"
            f"TDL-A-like channel, 100 ns delay spread, 50 Hz Doppler, 24 PRB, 30 kHz SCS, "
            f"{args.slots} slots per point. Command: `python experiments/ber_zf_mmse.py --slots {args.slots}`\n\n"
            "## Perfect CSI\n\n"
            + md_table(perfect, ["mimo", "modulation", "snr_db", "ber_zf", "ber_mmse", "mmse_gain"],
                       {"ber_zf": ".3e", "ber_mmse": ".3e", "mmse_gain": ".2f", "snr_db": ".0f"})
            + "\n\n## LMMSE-estimated CSI (DMRS)\n\n"
            + md_table([r for r in rows if r["csi"] == "lmmse"],
                       ["mimo", "modulation", "snr_db", "ber_zf", "ber_mmse", "mmse_gain"],
                       {"ber_zf": ".3e", "ber_mmse": ".3e", "mmse_gain": ".2f", "snr_db": ".0f"})
            + f"\n\nRuntime: {time.time() - t0:.0f} s\n")
    write_md("ber_zf_mmse.md", text)


if __name__ == "__main__":
    main()
