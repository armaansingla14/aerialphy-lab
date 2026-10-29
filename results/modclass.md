# Modulation classification (simulated)

Classes: bpsk, qpsk, 16qam, 64qam. 256 symbols per example with random phase, residual CFO up to 1e-3 cycles/symbol, AWGN, AGC normalisation. CNN: 10340 parameters, trained 1500 steps x 128 examples on SNR uniform in [-5, 20] dB (167 s CPU).
Baseline: nearest theoretical normalised cumulants (|C40|, C42) with SNR-corrected dilution.
Command: `python experiments/modclass.py --steps 1500 --per-snr 2000`

| snr_db | cnn_accuracy | cumulant_accuracy | n_test |
|---|---|---|---|
| -5 | 0.336 | 0.267 | 2000 |
| 0 | 0.502 | 0.540 | 2000 |
| 5 | 0.615 | 0.733 | 2000 |
| 10 | 0.768 | 0.741 | 2000 |
| 15 | 0.775 | 0.762 | 2000 |
| 20 | 0.757 | 0.750 | 2000 |

## CNN confusion matrix (SNR 10/15/20 dB pooled)

| true \ pred | bpsk | qpsk | 16qam | 64qam |
|---|---|---|---|---|
| bpsk | 1449 | 0 | 0 | 0 |
| qpsk | 0 | 1513 | 0 | 0 |
| 16qam | 0 | 72 | 1293 | 139 |
| 64qam | 1 | 61 | 1125 | 347 |
