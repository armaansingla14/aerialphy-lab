# Spectral efficiency with and without beamforming (simulated, bits/s/Hz)

Mean over 5000 channel draws per row; Shannon rates log2(1+SNR) per stream.
Command: `python experiments/beamforming_se.py --draws 5000`

- se_single_antenna: one transmit antenna, MRC at the receiver (no beamforming)
- se_open_loop: all antennas, equal power, no CSI at transmitter
- se_dft_codebook: one stream on the best beam of an oversampled DFT codebook
- se_eigen_bf: one stream on the dominant right singular vector (full CSIT)
- se_svd_waterfill: SVD multi-stream with water-filling (full CSIT)

| channel | mimo | snr_db | se_single_antenna | se_open_loop | se_dft_codebook | se_eigen_bf | se_svd_waterfill |
|---|---|---|---|---|---|---|---|
| rayleigh | 1x4 | -5 | 0.36 | 0.38 | 0.79 | 1.12 | 1.12 |
| rayleigh | 1x4 | 0 | 0.85 | 0.95 | 1.69 | 2.20 | 2.20 |
| rayleigh | 1x4 | 5 | 1.71 | 1.95 | 2.97 | 3.60 | 3.60 |
| rayleigh | 1x4 | 10 | 2.90 | 3.30 | 4.48 | 5.16 | 5.16 |
| rayleigh | 1x4 | 20 | 5.88 | 6.46 | 7.73 | 8.44 | 8.44 |
| rayleigh | 2x4 | -5 | 0.66 | 0.75 | 1.17 | 1.52 | 1.54 |
| rayleigh | 2x4 | 0 | 1.44 | 1.83 | 2.28 | 2.77 | 2.99 |
| rayleigh | 2x4 | 5 | 2.61 | 3.69 | 3.71 | 4.27 | 5.21 |
| rayleigh | 2x4 | 10 | 4.05 | 6.26 | 5.28 | 5.87 | 8.04 |
| rayleigh | 2x4 | 20 | 7.25 | 12.47 | 8.57 | 9.17 | 14.45 |
| rayleigh | 1x8 | -5 | 0.35 | 0.39 | 0.99 | 1.77 | 1.77 |
| rayleigh | 1x8 | 0 | 0.84 | 0.98 | 2.01 | 3.10 | 3.10 |
| rayleigh | 1x8 | 5 | 1.68 | 2.01 | 3.38 | 4.63 | 4.63 |
| rayleigh | 1x8 | 10 | 2.86 | 3.38 | 4.94 | 6.25 | 6.25 |
| rayleigh | 1x8 | 20 | 5.83 | 6.57 | 8.21 | 9.55 | 9.55 |
| rayleigh | 2x8 | -5 | 0.67 | 0.77 | 1.35 | 2.14 | 2.32 |
| rayleigh | 2x8 | 0 | 1.46 | 1.91 | 2.55 | 3.56 | 4.42 |
| rayleigh | 2x8 | 5 | 2.63 | 3.90 | 4.02 | 5.13 | 7.21 |
| rayleigh | 2x8 | 10 | 4.08 | 6.60 | 5.61 | 6.76 | 10.34 |
| rayleigh | 2x8 | 20 | 7.29 | 12.92 | 8.91 | 10.07 | 16.89 |
| rayleigh | 4x4 | -5 | 1.13 | 1.43 | 1.65 | 2.00 | 2.15 |
| rayleigh | 4x4 | 0 | 2.21 | 3.34 | 2.94 | 3.38 | 4.21 |
| rayleigh | 4x4 | 5 | 3.61 | 6.53 | 4.47 | 4.94 | 7.33 |
| rayleigh | 4x4 | 10 | 5.18 | 10.91 | 6.08 | 6.57 | 11.50 |
| rayleigh | 4x4 | 20 | 8.46 | 22.08 | 9.38 | 9.87 | 22.22 |
| ricean_k10dB_ula | 1x4 | -5 | 0.39 | 0.40 | 1.11 | 1.17 | 1.17 |
| ricean_k10dB_ula | 1x4 | 0 | 0.97 | 1.00 | 2.22 | 2.31 | 2.31 |
| ricean_k10dB_ula | 1x4 | 5 | 1.99 | 2.04 | 3.65 | 3.75 | 3.75 |
| ricean_k10dB_ula | 1x4 | 10 | 3.36 | 3.44 | 5.23 | 5.33 | 5.33 |
| ricean_k10dB_ula | 1x4 | 20 | 6.53 | 6.63 | 8.51 | 8.62 | 8.62 |
| ricean_k10dB_ula | 2x4 | -5 | 0.70 | 0.72 | 1.73 | 1.78 | 1.78 |
| ricean_k10dB_ula | 2x4 | 0 | 1.56 | 1.64 | 3.06 | 3.12 | 3.12 |
| ricean_k10dB_ula | 2x4 | 5 | 2.83 | 3.09 | 4.59 | 4.66 | 4.68 |
| ricean_k10dB_ula | 2x4 | 10 | 4.34 | 5.03 | 6.21 | 6.28 | 6.62 |
| ricean_k10dB_ula | 2x4 | 20 | 7.59 | 10.35 | 9.52 | 9.58 | 12.22 |
| ricean_k10dB_ula | 1x8 | -5 | 0.39 | 0.40 | 1.71 | 1.81 | 1.81 |
| ricean_k10dB_ula | 1x8 | 0 | 0.96 | 1.00 | 3.03 | 3.16 | 3.16 |
| ricean_k10dB_ula | 1x8 | 5 | 1.98 | 2.05 | 4.57 | 4.70 | 4.70 |
| ricean_k10dB_ula | 1x8 | 10 | 3.34 | 3.44 | 6.19 | 6.32 | 6.32 |
| ricean_k10dB_ula | 1x8 | 20 | 6.52 | 6.64 | 9.49 | 9.63 | 9.63 |
| ricean_k10dB_ula | 2x8 | -5 | 0.70 | 0.72 | 2.48 | 2.54 | 2.54 |
| ricean_k10dB_ula | 2x8 | 0 | 1.55 | 1.65 | 3.95 | 4.02 | 4.03 |
| ricean_k10dB_ula | 2x8 | 5 | 2.82 | 3.13 | 5.54 | 5.62 | 5.82 |
| ricean_k10dB_ula | 2x8 | 10 | 4.33 | 5.15 | 7.18 | 7.26 | 8.31 |
| ricean_k10dB_ula | 2x8 | 20 | 7.58 | 10.66 | 10.50 | 10.58 | 14.51 |
| ricean_k10dB_ula | 4x4 | -5 | 1.17 | 1.22 | 2.49 | 2.53 | 2.53 |
| ricean_k10dB_ula | 4x4 | 0 | 2.30 | 2.53 | 3.96 | 4.01 | 4.01 |
| ricean_k10dB_ula | 4x4 | 5 | 3.74 | 4.47 | 5.56 | 5.60 | 5.77 |
| ricean_k10dB_ula | 4x4 | 10 | 5.32 | 7.19 | 7.20 | 7.24 | 8.27 |
| ricean_k10dB_ula | 4x4 | 20 | 8.61 | 15.57 | 10.51 | 10.56 | 16.13 |
