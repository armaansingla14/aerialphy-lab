# Simulated DU/RU loop (FAPI-style L2/L1 messages + BFP fronthaul)

24 PRB, 30 kHz SCS, 2 RU antennas, TDL-A-like channel (100 ns), LMMSE estimation, MMSE detection, 30 slots per row (each slot carries one DL and one UL transport block).
Command: `python experiments/du_ru_loop.py --slots 30`

| modulation | layers | snr_db | dl_bler | ul_bler | ul_payload_match | fapi_bytes_per_slot | fh_bytes_per_slot | du_ms_per_slot | errors |
|---|---|---|---|---|---|---|---|---|---|
| qpsk | 1 | 0 | 0.700 | 0.867 | 4 | 1030 | 28896 | 132.5 | 0 |
| qpsk | 1 | 4 | 0.200 | 0.200 | 24 | 1030 | 28896 | 134.1 | 0 |
| qpsk | 1 | 8 | 0.000 | 0.067 | 28 | 1030 | 28896 | 143.3 | 0 |
| qpsk | 1 | 12 | 0.000 | 0.000 | 30 | 1030 | 28896 | 142.3 | 0 |
| qpsk | 1 | 16 | 0.000 | 0.000 | 30 | 1030 | 28896 | 135.6 | 0 |
| qpsk | 1 | 20 | 0.000 | 0.000 | 30 | 1030 | 28896 | 92.0 | 0 |
| qpsk | 1 | 24 | 0.000 | 0.000 | 30 | 1030 | 28896 | 142.5 | 0 |
| 16qam | 1 | 0 | 1.000 | 1.000 | 0 | 1912 | 28896 | 263.9 | 0 |
| 16qam | 1 | 4 | 0.933 | 0.967 | 1 | 1912 | 28896 | 239.5 | 0 |
| 16qam | 1 | 8 | 0.633 | 0.400 | 18 | 1912 | 28896 | 178.8 | 0 |
| 16qam | 1 | 12 | 0.233 | 0.167 | 25 | 1912 | 28896 | 244.3 | 0 |
| 16qam | 1 | 16 | 0.033 | 0.033 | 29 | 1912 | 28896 | 266.5 | 0 |
| 16qam | 1 | 20 | 0.000 | 0.000 | 30 | 1912 | 28896 | 167.5 | 0 |
| 16qam | 1 | 24 | 0.000 | 0.000 | 30 | 1912 | 28896 | 93.9 | 0 |
| 16qam | 2 | 0 | 1.000 | 1.000 | 0 | 3676 | 38528 | 400.0 | 0 |
| 16qam | 2 | 4 | 1.000 | 1.000 | 0 | 3676 | 38528 | 351.0 | 0 |
| 16qam | 2 | 8 | 1.000 | 1.000 | 0 | 3676 | 38528 | 352.9 | 0 |
| 16qam | 2 | 12 | 0.933 | 0.867 | 4 | 3676 | 38528 | 342.0 | 0 |
| 16qam | 2 | 16 | 0.767 | 0.800 | 6 | 3676 | 38528 | 264.4 | 0 |
| 16qam | 2 | 20 | 0.267 | 0.267 | 22 | 3676 | 38528 | 170.5 | 0 |
| 16qam | 2 | 24 | 0.067 | 0.133 | 26 | 3676 | 38528 | 177.0 | 0 |

`du_ms_per_slot` is Python wall-clock time spent in the DU model on this machine, not a real-time figure.

## Fronthaul BFP mantissa width sweep

| iq_width | evm_pct | bytes_per_symbol | ul_bler_16qam_24dB | dl_bler_16qam_24dB |
|---|---|---|---|---|
| 4 | 23.601 | 328 | 0.000 | 0.000 |
| 5 | 11.866 | 400 | 0.000 | 0.000 |
| 6 | 5.909 | 472 | 0.000 | 0.000 |
| 8 | 1.468 | 616 | 0.000 | 0.000 |
| 9 | 0.727 | 688 | 0.000 | 0.000 |
| 12 | 0.078 | 904 | 0.000 | 0.000 |
| 16 | 0.010 | 1192 | 0.000 | 0.000 |

## FAPI codec

Encode+decode round trip of 4-PDU UL_TTI/CRC messages: 752.3 us (1329/s, pure Python).
