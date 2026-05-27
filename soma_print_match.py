"""3D soma-print matcher for Larry within-EX volumes.

Reads 3D Cellpose centroids produced by exvivo_segmentation.py and
produces an N_A x N_B cost matrix + outer Hungarian one-to-one matches,
to feed downstream registration (find_landmarks.py).

Algorithm:
  - Per centroid, take k nearest neighbors in (z, y, x).
  - Soma-print = the k displacement vectors (each 3-d) sorted by
    magnitude.  (Polar-angle sorting has no clean 3D analog; magnitude
    sort gives a deterministic order that helps the inner Hungarian, and
    the inner Hungarian itself is the rotation-tolerant matcher.)
  - Per-volume normalize each soma-print by the median NN distance.
  - For each (i, j) pair, inner-Hungarian over the k x k vector-pair
    cost matrix; entry D[i, j] = mean assigned-pair distance.
  - Outer rectangular Hungarian over D -> min(N_A, N_B) matches.

Inputs (defaults; override with --a-cent / --b-cent / --out-dir):
  results/larry_within_exvivo_3d/A_centroids_3d.npy   (N, 3) z,y,x
  results/larry_within_exvivo_3d/B_centroids_3d.npy

Outputs (in --out-dir, default = results/larry_within_exvivo_3d/):
  distance_matrix_3d.npy        (N_A, N_B) float32
  similarity_matrix_3d.npy      (N_A, N_B) float32
  assignments_3d.npz            matches (M, 2), distances (M,)
  match_summary_3d.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
K_SOMA = int(os.environ.get("MATCH3D_K", "10"))


def soma_print_3d(points: np.ndarray, k: int) -> np.ndarray:
    """(N, 3) -> (N, k, 3) k-NN displacement vectors, sorted by magnitude."""
    n = len(points)
    diff = points[:, None, :] - points[None, :, :]        # (N, N, 3)
    d2 = np.einsum("ijk,ijk->ij", diff, diff)
    np.fill_diagonal(d2, np.inf)
    idx = np.argsort(d2, axis=1)[:, :k]                   # (N, k)
    out = np.zeros((n, k, 3), dtype=np.float32)
    for i in range(n):
        vecs = points[idx[i]] - points[i]                 # (k, 3)
        mag = np.linalg.norm(vecs, axis=1)
        out[i] = vecs[np.argsort(mag)]
    return out


def median_nn_distance(sp: np.ndarray) -> float:
    """sp: (N, k, 3).  Median over cells of the smallest NN magnitude."""
    mag = np.linalg.norm(sp, axis=2)                      # (N, k)
    return float(np.median(mag.min(axis=1)))


# Worker-side globals so sp_a / sp_b are pickled to each worker exactly once
# (via Pool initializer) instead of once per task.
_W_SP_A: np.ndarray | None = None
_W_SP_B: np.ndarray | None = None


def _worker_init(sp_a: np.ndarray, sp_b: np.ndarray) -> None:
    global _W_SP_A, _W_SP_B
    _W_SP_A = sp_a
    _W_SP_B = sp_b


def _worker_rows(row_indices: list[int]):
    """Compute a batch of rows of D and return [(i, row_i), ...]."""
    sp_a, sp_b = _W_SP_A, _W_SP_B
    nb = len(sp_b)
    out = []
    for i in row_indices:
        diff = sp_a[i][None, :, None, :] - sp_b[:, None, :, :]
        costs = np.linalg.norm(diff, axis=-1)             # (N_B, k, k)
        row = np.empty(nb, dtype=np.float32)
        for j in range(nb):
            c_ij = costs[j]
            r, c = linear_sum_assignment(c_ij)
            row[j] = c_ij[r, c].mean()
        out.append((i, row))
    return out


def full_distance_matrix(sp_a: np.ndarray, sp_b: np.ndarray) -> np.ndarray:
    """Inner-Hungarian mean-cost matrix (N_A, N_B), parallelized over rows.

    Worker count read from $MATCH3D_WORKERS (default: cpu_count()); set to 1
    to force the serial path.  Results are bitwise identical regardless of
    worker count because each row is computed with the same code.
    """
    na, nb = len(sp_a), len(sp_b)
    D = np.empty((na, nb), dtype=np.float32)

    n_workers = int(os.environ.get("MATCH3D_WORKERS", cpu_count() or 1))
    n_workers = max(1, min(n_workers, na))

    if n_workers == 1:
        for i in tqdm(range(na)):
            diff = sp_a[i][None, :, None, :] - sp_b[:, None, :, :]
            costs = np.linalg.norm(diff, axis=-1)         # (N_B, k, k)
            for j in range(nb):
                c_ij = costs[j]
                r, c = linear_sum_assignment(c_ij)
                D[i, j] = c_ij[r, c].mean()
        return D

    chunk_size = max(1, na // (n_workers * 4))
    chunks = [list(range(i, min(i + chunk_size, na)))
              for i in range(0, na, chunk_size)]
    print(f"  parallel: workers={n_workers}  rows={na}  "
          f"chunks={len(chunks)}  chunk_size~{chunk_size}", flush=True)

    with Pool(n_workers, initializer=_worker_init,
              initargs=(sp_a, sp_b)) as pool:
        with tqdm(total=na) as pbar:
            for batch in pool.imap_unordered(_worker_rows, chunks):
                for i, row in batch:
                    D[i] = row
                pbar.update(len(batch))
    return D


def to_similarity(d, scale: float = 1.0) -> np.ndarray:
    return 100.0 / (1.0 + np.asarray(d, dtype=np.float64) / scale)


def main(a_cent: str | Path | None = None,
         b_cent: str | Path | None = None,
         out_dir: str | Path | None = None) -> None:
    a_path = Path(a_cent) if a_cent else RES / "A_centroids_3d.npy"
    b_path = Path(b_cent) if b_cent else RES / "B_centroids_3d.npy"
    out = Path(out_dir) if out_dir else RES
    out.mkdir(parents=True, exist_ok=True)

    A_cent = np.load(a_path)                              # (N_A, 3) z,y,x
    B_cent = np.load(b_path)                              # (N_B, 3)
    print(f"[load] A={len(A_cent)} from {a_path}")
    print(f"[load] B={len(B_cent)} from {b_path}  k={K_SOMA}")

    print(f"[soma] computing 3D soma-print")
    t0 = time.time()
    sp_a = soma_print_3d(A_cent, K_SOMA)
    sp_b = soma_print_3d(B_cent, K_SOMA)
    print(f"[soma] done in {time.time()-t0:.1f}s")

    L_a = median_nn_distance(sp_a) or 1.0
    L_b = median_nn_distance(sp_b) or 1.0
    print(f"[soma] median NN distance  A={L_a:.2f}  B={L_b:.2f}")
    sp_a /= L_a
    sp_b /= L_b

    print(f"[hung] full cost matrix {len(A_cent)} x {len(B_cent)}")
    D = full_distance_matrix(sp_a, sp_b)
    S = to_similarity(D).astype(np.float32)
    np.save(out / "distance_matrix_3d.npy", D)
    np.save(out / "similarity_matrix_3d.npy", S)

    r, c = linear_sum_assignment(D)
    matches_rc = np.stack([r, c], axis=1).astype(np.int32)
    match_d = D[r, c]
    print(f"[hung] {len(r)} matches, "
          f"median d̄={np.median(match_d):.3f}")
    np.savez_compressed(out / "assignments_3d.npz",
                        matches=matches_rc, distances=match_d)

    summary = {
        "n_A": int(len(A_cent)),
        "n_B": int(len(B_cent)),
        "a_cent": str(a_path),
        "b_cent": str(b_path),
        "k_soma": K_SOMA,
        "median_nn_distance": {"A": L_a, "B": L_b},
        "n_matches": int(len(r)),
        "match_d_stats": {
            "min": float(match_d.min()),
            "median": float(np.median(match_d)),
            "mean": float(match_d.mean()),
            "max": float(match_d.max()),
        },
    }
    (out / "match_summary_3d.json").write_text(json.dumps(summary, indent=2))
    print(f"[done] wrote {out / 'match_summary_3d.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--a-cent", help="path to A centroids .npy (z,y,x)")
    ap.add_argument("--b-cent", help="path to B centroids .npy (z,y,x)")
    ap.add_argument("--out-dir", help="output directory for matrices + assignments")
    args = ap.parse_args()
    main(a_cent=args.a_cent, b_cent=args.b_cent, out_dir=args.out_dir)
