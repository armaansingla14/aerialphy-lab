"""Run the simulated MAC <-> DU <-> RU <-> UE loop and record BLER, message and fronthaul load.

    python experiments/du_ru_loop.py [--slots 40]
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from common import md_table, write_csv, write_md

from oran import fapi
from oran.du_ru import run_loop
from oran.fronthaul import FronthaulCodec


def codec_benchmark(n: int = 20000) -> dict:
    msgs = [fapi.UlTtiRequest(1, 2, [fapi.PuschPdu(rnti=i, tb_size=1000) for i in range(4)]),
            fapi.CrcIndication(1, 2, [fapi.CrcInfo(i, 0, 0, 100, -150) for i in range(4)])]
    t0 = time.perf_counter()
    for i in range(n):
        fapi.decode(fapi.encode(msgs[i % 2]))
    dt = time.perf_counter() - t0
    return {"roundtrips_per_s": n / dt, "us_per_roundtrip": dt / n * 1e6}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", type=int, default=40)
    args = ap.parse_args()
    rows = []
    for mod, layers in (("qpsk", 1), ("16qam", 1), ("16qam", 2)):
        for snr in (0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0):
            t0 = time.time()
            s = run_loop(n_slots=args.slots, snr_db=snr, n_rb=24, n_ant=2, n_layers=layers, modulation=mod,
                         profile="A", delay_spread_s=100e-9, fh_bitwidth=9, seed=int(snr))
            rows.append({"modulation": mod, "layers": layers, "snr_db": snr, "dl_bler": s.dl_bler,
                         "ul_bler": s.ul_bler, "ul_payload_match": s.ul_payload_match, "slots": args.slots,
                         "fapi_bytes_per_slot": s.fapi_bytes / args.slots,
                         "fh_bytes_per_slot": s.fh_bytes / args.slots,
                         "du_ms_per_slot": 1e3 * s.du_proc_s / args.slots, "errors": len(s.errors)})
            print(rows[-1], f"{time.time() - t0:.1f}s")
    write_csv("du_ru_loop.csv", rows)
    # fronthaul compression width sweep (EVM on unit-power random IQ + BLER at a fixed SNR)
    rng = np.random.default_rng(5)
    g = (rng.standard_normal((288, 14)) + 1j * rng.standard_normal((288, 14))) / np.sqrt(2)
    fh_rows = []
    for w in (4, 5, 6, 8, 9, 12, 16):
        codec = FronthaulCodec(w)
        pk = codec.to_packets(g, 0, 0, 0, 0, 0)
        back = codec.from_packets(pk, 288, 14)
        evm = np.sqrt(np.mean(np.abs(back - g) ** 2) / np.mean(np.abs(g) ** 2))
        s = run_loop(n_slots=args.slots // 2, snr_db=24.0, n_rb=24, n_ant=2, modulation="16qam",
                     fh_bitwidth=w, seed=1)
        fh_rows.append({"iq_width": w, "evm_pct": 100 * evm, "bytes_per_symbol": len(pk[0]),
                        "ul_bler_16qam_24dB": s.ul_bler, "dl_bler_16qam_24dB": s.dl_bler})
        print(fh_rows[-1])
    write_csv("fronthaul_bfp.csv", fh_rows)
    bench = codec_benchmark()
    fmt = {"dl_bler": ".3f", "ul_bler": ".3f", "fapi_bytes_per_slot": ".0f", "fh_bytes_per_slot": ".0f",
           "du_ms_per_slot": ".1f", "snr_db": ".0f"}
    text = (
        "# Simulated DU/RU loop (FAPI-style L2/L1 messages + BFP fronthaul)\n\n"
        f"24 PRB, 30 kHz SCS, 2 RU antennas, TDL-A-like channel (100 ns), LMMSE estimation, MMSE detection, "
        f"{args.slots} slots per row (each slot carries one DL and one UL transport block).\n"
        f"Command: `python experiments/du_ru_loop.py --slots {args.slots}`\n\n"
        + md_table(rows, ["modulation", "layers", "snr_db", "dl_bler", "ul_bler", "ul_payload_match",
                          "fapi_bytes_per_slot", "fh_bytes_per_slot", "du_ms_per_slot", "errors"], fmt)
        + "\n\n`du_ms_per_slot` is Python wall-clock time spent in the DU model on this machine, not a "
          "real-time figure.\n\n## Fronthaul BFP mantissa width sweep\n\n"
        + md_table(fh_rows, None, {"evm_pct": ".3f", "ul_bler_16qam_24dB": ".3f", "dl_bler_16qam_24dB": ".3f"})
        + f"\n\n## FAPI codec\n\nEncode+decode round trip of 4-PDU UL_TTI/CRC messages: "
          f"{bench['us_per_roundtrip']:.1f} us ({bench['roundtrips_per_s']:.0f}/s, pure Python).\n"
    )
    write_md("du_ru_loop.md", text)


if __name__ == "__main__":
    main()
