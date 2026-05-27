"""Shared 3D affine RANSAC utilities for find_landmarks.py and warp_volume.py."""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT     = Path(__file__).resolve().parent
RES      = ROOT / "results"
DATA_DIR = ROOT / "data"

INLIER_VOX = 15.0
RANSAC_ITERS = 5000
RANSAC_SEED = 0


# ----------------------------------------------------------------------
# Affine math (3D, 12 DoF)
# ----------------------------------------------------------------------

def fit_affine_3d(src: np.ndarray, dst: np.ndarray):
    n = len(src)
    X = np.zeros((3 * n, 12), dtype=np.float64)
    y = np.zeros(3 * n, dtype=np.float64)
    for axis in range(3):
        X[axis::3, axis * 4:axis * 4 + 3] = src
        X[axis::3, axis * 4 + 3] = 1.0
        y[axis::3] = dst[:, axis]
    p, *_ = np.linalg.lstsq(X, y, rcond=None)
    M = p.reshape(3, 4)[:, :3]
    t = p.reshape(3, 4)[:, 3]
    return M, t


def _non_coplanar(p: np.ndarray, tol: float = 1e-6) -> bool:
    a, b, c, d = p
    return abs(np.linalg.det(np.stack([b - a, c - a, d - a]))) > tol


def ransac_affine_3d(src: np.ndarray, dst: np.ndarray,
                     n_iter: int = RANSAC_ITERS,
                     inlier_vox: float = INLIER_VOX,
                     seed: int = RANSAC_SEED):
    rng = np.random.default_rng(seed)
    n = len(src)
    if n < 4:
        return None
    best_inliers = None
    best_count = -1
    for _ in range(n_iter):
        idx = rng.choice(n, size=4, replace=False)
        if not _non_coplanar(src[idx]):
            continue
        try:
            M, t = fit_affine_3d(src[idx], dst[idx])
        except np.linalg.LinAlgError:
            continue
        if not (np.all(np.isfinite(M)) and np.all(np.isfinite(t))):
            continue
        pred = src @ M.T + t
        res = np.linalg.norm(pred - dst, axis=1)
        inliers = res < inlier_vox
        cnt = int(inliers.sum())
        if cnt > best_count:
            best_count = cnt
            best_inliers = inliers
    if best_count < 4:
        return None
    M, t = fit_affine_3d(src[best_inliers], dst[best_inliers])
    return M, t, best_inliers


