# aerialphy-lab

MIMO and 5G NR physical-layer experiments in Python/PyTorch, loosely following 3GPP
Rel-16/17 concepts, plus a small O-RAN style DU/RU prototype driven by FAPI-like messages.

Everything here is link-level **simulation** on synthetic channels. Nothing was run on
radio hardware or real fronthaul equipment, and the code does not claim 3GPP conformance.

## Features

- **NR numerology and resource grid**: mu = 0..4, SCS, slot timing, FFT/CP sizing,
  RE allocation with frequency-first mapping (`aerialphy/numerology.py`).
- **CP-OFDM** modulator/demodulator (`ofdm.py`).
- **Modulation**: BPSK/QPSK/16-QAM/64-QAM with TS 38.211 Gray mapping, hard and max-log LLR demapping.
- **Channel coding**: CRC24A/24B/16/11 (38.212 polynomials) and a simplified regular (3,6)
  quasi-cyclic LDPC code (Z = 32, n = 384, k = 196) with a batched normalised min-sum
  decoder. It mirrors the protograph + lifting structure of NR LDPC but is **not** BG1/BG2.
- **Channels**: AWGN, TDL-A/TDL-C-like power-delay profiles (adapted from the TR 38.901 table
  shapes), exponential PDP, sum-of-sinusoids Doppler, Kronecker antenna correlation, and a
  Ricean ULA channel for beamforming studies.
- **DMRS**: Gold-sequence pilots (38.211 5.2.1), type-1 comb with two CDM groups and
  length-2 FD-OCC (up to 4 ports).
- **Channel estimation**: LS with linear interpolation, LMMSE with the true PDP ("genie")
  or a robust uniform PDP, and two small neural estimators:
  - an MLP that maps one DMRS symbol's LS pilots (+ SNR) to the full-band response,
  - a DnCNN-style residual CNN denoiser that refines the LS-interpolated grid.
- **MIMO**: ZF and unbiased MMSE detection with post-detection noise for LLRs, brute-force ML
  (as a test oracle), SVD precoding, water-filling, eigen-beamforming and a DFT
  (Type-I-like) codebook.
- **Modulation classifier**: 1-D CNN on IQ samples, compared to a fourth-order cumulant baseline.
- **Slot-level link** (`link.py`): TB + CRC -> LDPC -> QAM -> layers -> DMRS grid -> TDL MIMO
  channel -> estimation -> ZF/MMSE -> LLR -> decoding -> CRC.
- **O-RAN DU/RU prototype** (`oran/`):
  - `fapi.py`: dataclass messages (PARAM, CONFIG, START/STOP, SLOT.indication, DL_TTI.request,
    UL_TTI.request, TX_DATA.request, RX_DATA.indication, CRC.indication, ERROR.indication)
    with a metadata-driven little-endian binary codec and FAPI-style header.
  - `fronthaul.py`: 7.2x-like U-plane packets (eCPRI header + section header) with
    per-PRB block floating point compression.
  - `du_ru.py`: MAC -> DU (high PHY) -> RU (OFDM + BFP) -> air (time-domain multipath) -> UE
    loop for both DL and UL, with every message actually encoded and decoded.
- **Test suite**: 294 pytest cases, most of them parametrized across numerologies,
  bandwidths, antenna/layer counts, modulations, channel profiles and estimators.

## Layout

```
aerialphy/      numerology, ofdm, modulation, coding, channel, dmrs, estimation, mimo,
                link, neural (models + data), train (training loops)
oran/           fapi (messages + codec), fronthaul (U-plane + BFP), du_ru (simulated loop)
experiments/    ber_zf_mmse.py, chest_nmse.py, beamforming_se.py, modclass.py, du_ru_loop.py
results/        CSV and Markdown outputs written by the experiments
tests/          pytest suite
matlab/         independent MATLAB/Octave reference scripts (not executed here)
```

## Running

```
pip install -r requirements.txt
python -m pytest -q                         # 294 tests, ~25 s on CPU

cd experiments
python ber_zf_mmse.py --slots 100           # ~11 min
python chest_nmse.py --steps 2000 --samples 300   # ~6 min incl. training
python beamforming_se.py --draws 5000       # seconds
python modclass.py --steps 1500 --per-snr 2000    # ~3-5 min
python du_ru_loop.py --slots 30             # ~7 min
```

All runs are CPU-only; the neural models are tiny (7.8k to 177k parameters).

## Measured results

All numbers below were produced by the scripts above on synthetic data; the full tables are in
`results/`. Runtimes are for this development machine.

### ZF vs MMSE detection (uncoded BER, perfect CSI)

`python experiments/ber_zf_mmse.py --slots 100` (TDL-A-like, 100 ns delay spread, 50 Hz Doppler,
24 PRB at 30 kHz, 100 slots per point, same channel and noise realisations for both detectors).

| MIMO | Mod | SNR (dB) | BER ZF | BER MMSE |
|---|---|---|---|---|
| 2x2 | QPSK | 10 | 7.52e-02 | 5.33e-02 |
| 2x2 | QPSK | 20 | 8.98e-03 | 6.15e-03 |
| 2x2 | 16-QAM | 20 | 3.49e-02 | 3.04e-02 |
| 4x4 | QPSK | 10 | 1.26e-01 | 5.51e-02 |
| 4x4 | QPSK | 20 | 1.91e-02 | 6.89e-03 |
| 4x4 | 16-QAM | 20 | 6.81e-02 (LMMSE CSI) | 4.84e-02 (LMMSE CSI) |

MMSE had lower BER than ZF at every SNR point and configuration tested: the BER ratio was
1.12-1.50x for 2x2 and 1.2-2.8x for 4x4. With DMRS-based LMMSE channel estimates instead of
perfect CSI the ordering is the same (full table in `results/ber_zf_mmse.md`).

### Channel-estimation NMSE: neural vs LS (and LMMSE)

`python experiments/chest_nmse.py --steps 2000 --samples 300`: SISO, 12 PRB, 2 DMRS symbols,
36 channel conditions (profiles A/C/EXP x delay spread 100/300 ns x SNR 0..25 dB) x 300 test
samples = 10,800 samples per method. NMSE in dB, averaged over profiles and delay spreads:

| SNR (dB) | LS | LMMSE (robust) | LMMSE (genie PDP) | MLP | CNN denoiser |
|---|---|---|---|---|---|
| 0 | 0.37 | -6.67 | -10.84 | -9.31 | -6.79 |
| 5 | -4.59 | -10.85 | -14.76 | -13.05 | -11.23 |
| 10 | -9.36 | -15.28 | -18.86 | -15.86 | -14.95 |
| 15 | -14.79 | -20.32 | -23.51 | -17.87 | -18.84 |
| 20 | -19.59 | -24.57 | -27.50 | -18.79 | -21.93 |
| 25 | -24.30 | -28.42 | -30.96 | -19.24 | -24.47 |
| all | -5.74 | -12.45 | -16.45 | -14.09 | -12.47 |

- Both neural estimators beat LS at low and medium SNR (MLP: 9.7 dB better at 0 dB SNR).
  The MLP beat LS in 29 of 36 conditions and the CNN denoiser in 34 of 36.
- The MLP has an error floor around -19 dB and is **worse than LS at 20-25 dB**. The CNN
  denoiser roughly matches LS there.
- LMMSE with the true PDP stays best at every SNR. The neural models beat the robust LMMSE
  (no PDP knowledge) only at low SNR.

### Spectral efficiency with and without beamforming

`python experiments/beamforming_se.py --draws 5000` (Shannon rates, mean over 5000 draws), SNR = 10 dB:

| Channel | MIMO | Single TX antenna | Open loop (no CSIT) | DFT codebook beam | Eigen-beam | SVD + water-filling |
|---|---|---|---|---|---|---|
| Rayleigh | 1x4 | 2.90 | 3.30 | 4.48 | 5.16 | 5.16 |
| Rayleigh | 1x8 | 2.86 | 3.38 | 4.94 | 6.25 | 6.25 |
| Rayleigh | 4x4 | 5.18 | 10.91 | 6.08 | 6.57 | 11.50 |
| Ricean K=10 dB ULA | 1x8 | 3.34 | 3.44 | 6.19 | 6.32 | 6.32 |
| Ricean K=10 dB ULA | 2x8 | 4.33 | 5.15 | 7.18 | 7.26 | 8.31 |

(bits/s/Hz). With eight TX antennas, single-stream beamforming roughly doubles SE compared
with one antenna (2.86 -> 6.25 for Rayleigh, 3.34 -> 6.32 for Ricean). On the LOS-dominated
Ricean channel the DFT codebook gets within 0.13 bit/s/Hz of full-CSI eigen-beamforming.

### Modulation classification

`python experiments/modclass.py --steps 1500 --per-snr 2000` (BPSK/QPSK/16QAM/64QAM, 256
symbols, random phase plus residual CFO):

| SNR (dB) | -5 | 0 | 5 | 10 | 15 | 20 |
|---|---|---|---|---|---|---|
| CNN accuracy | 0.336 | 0.502 | 0.615 | 0.768 | 0.775 | 0.757 |
| Cumulant baseline | 0.267 | 0.540 | 0.733 | 0.741 | 0.762 | 0.750 |

At 10 dB and above the CNN was 1-3 points better than the cumulant baseline. At 0-5 dB it was
worse. Both methods mostly confuse 16-QAM with 64-QAM; with 256 symbols the CNN separates
BPSK and QPSK perfectly at 10 dB and above.

### DU/RU loop

`python experiments/du_ru_loop.py --slots 30` (24 PRB, 2 antennas, LMMSE + MMSE, 9-bit BFP):

| Mod / layers | SNR (dB) | DL BLER | UL BLER | FAPI bytes/slot | Fronthaul bytes/slot |
|---|---|---|---|---|---|
| QPSK / 1 | 4 | 0.200 | 0.200 | 1030 | 28896 |
| QPSK / 1 | 8 | 0.000 | 0.067 | 1030 | 28896 |
| 16-QAM / 1 | 12 | 0.233 | 0.167 | 1912 | 28896 |
| 16-QAM / 1 | 20 | 0.000 | 0.000 | 1912 | 28896 |
| 16-QAM / 2 | 24 | 0.067 | 0.133 | 3676 | 38528 |

BFP compression EVM on unit-power IQ: 23.6 % (4-bit), 5.9 % (6-bit), 0.73 % (9-bit),
0.078 % (12-bit). At 24 dB SNR with 16-QAM, no block errors occurred in 15 slots for any
width from 4 to 16 bits. The pure-Python FAPI codec took about 0.75 ms per encode+decode
round trip for 4-PDU messages.

## Limitations / what is simulated

- All channels, noise and impairments are synthetic. The TDL profiles are adapted from
  38.901 table shapes, not a conformance implementation. The link applies the channel per RE
  (a test checks that this matches time-domain multipath when delays fit inside the CP). The
  DU/RU loop uses real time-domain convolution.
- The LDPC code is a simplified QC-LDPC, not the NR base graphs. There is no rate matching,
  HARQ combining, scrambling or LBRM.
- DMRS symbols are fully reserved (no data FDM'd on DMRS symbols). Up to 4 ports are supported.
- The FAPI messages are a simplified subset inspired by SCF FAPI. Field sets and IDs are not
  byte-compatible with any vendor L1. The fronthaul packets are 7.2x-like but simplified.
- The DU timing (`du_ms_per_slot`) is Python wall-clock time, nowhere near real time.
- The MATLAB scripts in `matlab/` are reference implementations that were **not run** here.

## License

MIT, see `LICENSE`.
