"""Channel coding: CRC (TS 38.212 polynomials) and a simplified QC-LDPC code.

The LDPC code is a regular (3, 6) quasi-cyclic construction with lifting size
Z and pseudo-random circulant shifts. It mirrors the *structure* of the NR
base graphs (protograph + circulant lifting, layered min-sum decoding) but is
not the 38.212 BG1/BG2 code.
"""
from __future__ import annotations

import numpy as np

CRC_POLYS = {
    "crc24a": 0x1864CFB,
    "crc24b": 0x1800063,
    "crc16": 0x11021,
    "crc11": 0xE21,
}


def crc_bits(bits: np.ndarray, poly_name: str = "crc24a") -> np.ndarray:
    poly = CRC_POLYS[poly_name]
    L = poly.bit_length() - 1
    reg = 0
    mask = (1 << L) - 1
    for b in np.asarray(bits, dtype=np.uint8):
        top = ((reg >> (L - 1)) & 1) ^ int(b)
        reg = (reg << 1) & mask
        if top:
            reg ^= poly & mask
    return np.array([(reg >> (L - 1 - i)) & 1 for i in range(L)], dtype=np.uint8)


def crc_attach(bits: np.ndarray, poly_name: str = "crc24a") -> np.ndarray:
    return np.concatenate([np.asarray(bits, dtype=np.uint8), crc_bits(bits, poly_name)])


def crc_check(bits_with_crc: np.ndarray, poly_name: str = "crc24a") -> bool:
    L = CRC_POLYS[poly_name].bit_length() - 1
    b = np.asarray(bits_with_crc, dtype=np.uint8)
    return bool(np.array_equal(crc_bits(b[:-L], poly_name), b[-L:]))


def _gf2_systematic(h: np.ndarray):
    """Row-reduce H over GF(2). Returns (P, perm) such that H[:, perm] ~ [P | I]."""
    h = h.copy().astype(np.uint8)
    m, n = h.shape
    perm = np.arange(n)
    row = 0
    pivots = []
    for col in range(n - 1, -1, -1):
        if row >= m:
            break
        r = m - 1 - row
        cand = np.nonzero(h[: r + 1, col])[0]
        if cand.size == 0:
            continue
        piv = cand[-1]
        h[[piv, r]] = h[[r, piv]]
        others = np.nonzero(h[:, col])[0]
        others = others[others != r]
        h[others] ^= h[r]
        pivots.append(col)
        row += 1
    rank = row
    piv_cols = pivots[::-1]  # ascending row order m-rank..m-1
    rest = [c for c in range(n) if c not in set(piv_cols)]
    perm = np.array(rest + piv_cols)
    hp = h[m - rank:, :][:, perm]
    k = n - rank
    return hp[:, :k], perm, rank


class QCLDPC:
    """Regular (3,6) QC-LDPC with lifting size z: n = 12 z, design rate 1/2."""

    def __init__(self, z: int = 32, seed: int = 7, max_iter: int = 20, alpha: float = 0.8):
        rng = np.random.default_rng(seed)
        mb, nb = 6, 12
        # protograph: each column touches 3 of the 6 rows, each row has degree 6
        proto = np.zeros((mb, nb), dtype=bool)
        for c in range(nb):
            rows = [(c + j * 2) % mb for j in range(3)]
            proto[rows, c] = True
        shifts = rng.integers(0, z, size=(mb, nb))
        self.z = z
        self.proto = proto
        self.shifts = np.where(proto, shifts, -1)
        h = np.zeros((mb * z, nb * z), dtype=np.uint8)
        eye = np.eye(z, dtype=np.uint8)
        for i in range(mb):
            for j in range(nb):
                if proto[i, j]:
                    h[i * z:(i + 1) * z, j * z:(j + 1) * z] = np.roll(eye, shifts[i, j], axis=1)
        self.H = h
        self.n = nb * z
        p, perm, rank = _gf2_systematic(h)
        self._P = p
        self._perm = perm
        self.k = self.n - rank
        self.max_iter = max_iter
        self.alpha = alpha
        rows, cols = np.nonzero(h)
        order = np.lexsort((cols, rows))
        self._edge_var = cols[order].reshape(h.shape[0], -1)  # (m, dc) regular rows

    @property
    def rate(self) -> float:
        return self.k / self.n

    def encode(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=np.uint8).reshape(-1, self.k)
        parity = (u.astype(np.int64) @ self._P.T.astype(np.int64)) % 2
        c_perm = np.concatenate([u, parity.astype(np.uint8)], axis=1)
        c = np.empty_like(c_perm)
        c[:, self._perm] = c_perm
        return c

    def info_bits(self, c: np.ndarray) -> np.ndarray:
        return np.asarray(c)[..., self._perm[: self.k]]

    def syndrome_ok(self, c: np.ndarray) -> np.ndarray:
        c = np.atleast_2d(c)
        return np.all((c.astype(np.int64) @ self.H.T.astype(np.int64)) % 2 == 0, axis=1)

    def decode(self, llr: np.ndarray):
        """Normalised min-sum (flooding), batched. llr > 0 means bit 0.

        Returns (hard codeword bits, iterations used per codeword).
        """
        llr = np.atleast_2d(np.asarray(llr, dtype=float))
        b = llr.shape[0]
        ev = self._edge_var
        r = np.zeros((b,) + ev.shape)
        total = llr.copy()
        iters = np.full(b, self.max_iter)
        done = np.zeros(b, dtype=bool)
        hard = (total < 0).astype(np.uint8)
        for it in range(self.max_iter):
            q = total[:, ev] - r
            sgn = np.where(q < 0, -1.0, 1.0)
            prod = np.prod(sgn, axis=2, keepdims=True)
            aq = np.abs(q)
            idx = np.argmin(aq, axis=2)
            min1 = np.take_along_axis(aq, idx[..., None], axis=2)
            aq2 = aq.copy()
            np.put_along_axis(aq2, idx[..., None], np.inf, axis=2)
            min2 = aq2.min(axis=2, keepdims=True)
            mag = np.where(np.arange(ev.shape[1])[None, None, :] == idx[..., None], min2, min1)
            r_new = self.alpha * prod * sgn * mag
            r = np.where(done[:, None, None], r, r_new)
            total = llr.copy()
            np.add.at(total, (slice(None), ev.reshape(-1)), r.reshape(b, -1))
            hard = (total < 0).astype(np.uint8)
            ok = self.syndrome_ok(hard)
            newly = ok & ~done
            iters[newly] = it + 1
            done |= ok
            if done.all():
                break
        return hard, iters
