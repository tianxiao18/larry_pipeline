"""3D affine registration of Larry_2A_1 -> Larry_2A_8 over similarity
thresholds, using full-z Cellpose centroids.

  - per-threshold global RANSAC affine (12 DoF in 3D),
  - per-threshold 2-region K-means regional RANSAC (separate affine per
    spatial cluster, K-means clusters source centroids in (z, y, x)).

Also serves as the shared module for RANSAC + MIP utilities used by
find_landmarks.py and warp_volume.py.

Inputs (results/larry_within_exvivo_3d/):
  A_centroids_3d.npy  (N_A, 3) z,y,x
  B_centroids_3d.npy  (N_B, 3)
  assignments_3d.npz  matches (M, 2), distances (M,)

Outputs (results/larry_within_exvivo_3d/):
  registration_fits_3d.json            global fits
  registration_fits_3d_regional.json   2-region fits
  registration_thresholds_3d.png       XY+XZ MIP scatter overlays
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

ROOT     = Path(__file__).resolve().parent
RES      = ROOT / "results"
DATA_DIR = ROOT / "data"
MIP_DIR  = RES / "mips"
MIP_DIR.mkdir(parents=True, exist_ok=True)

S_THRESHOLDS = [None, 30.0, 40.0, 50.0, 60.0, 67.0, 71.0]
K_REGIONS = 2
CLUSTER_COLORS = ["lime", "magenta"]
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


def warp_points_regional(pts: np.ndarray, regions: list[dict]) -> np.ndarray:
    """Piecewise affine in 3D: assign each point to nearest src_center,
    then apply that region's (M, t)."""
    centers = np.stack([r["src_center"] for r in regions])    # (K, 3)
    d = np.linalg.norm(pts[:, None, :] - centers[None, :, :], axis=2)
    assign = np.argmin(d, axis=1)
    out = np.zeros_like(pts, dtype=np.float64)
    for k, r in enumerate(regions):
        m = assign == k
        if not m.any():
            continue
        out[m] = pts[m] @ r["M"].T + r["t"]
    return out


# ----------------------------------------------------------------------
# MIPs
# ----------------------------------------------------------------------

def stretch(img: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.5) -> np.ndarray:
    lo, hi = np.percentile(img, [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1
    return np.clip((img.astype(np.float32) - lo) / (hi - lo), 0, 1)


def colorize(gray: np.ndarray, rgb: tuple) -> np.ndarray:
    out = np.zeros((*gray.shape, 3), dtype=np.float32)
    for c in range(3):
        out[..., c] = gray * rgb[c]
    return out


def load_or_compute_mips(label: str, tif_name: str):
    xy_path = MIP_DIR / f"{label}_xy_mip.npy"
    xz_path = MIP_DIR / f"{label}_xz_mip.npy"
    xy = np.load(xy_path) if xy_path.exists() else None
    if xz_path.exists():
        xz = np.load(xz_path)
    else:
        import tifffile
        with tifffile.TiffFile(str(DATA_DIR / tif_name)) as tf:
            n = len(tf.pages)
            h, w = tf.pages[0].shape
            xz = np.zeros((n, w), dtype=np.uint16)
            xy_acc = np.zeros((h, w), dtype=np.uint16) if xy is None else None
            for i in range(n):
                page = tf.pages[i].asarray()
                np.maximum(xz[i], page.max(axis=0), out=xz[i])
                if xy_acc is not None:
                    np.maximum(xy_acc, page, out=xy_acc)
        np.save(xz_path, xz)
        if xy is None:
            xy = xy_acc
            np.save(xy_path, xy)
    return xy, xz


# ----------------------------------------------------------------------
# Panel rendering
# ----------------------------------------------------------------------

def add_panel(ax_xy, ax_xz, warped_zyx, B_cent, B_xy_s, B_xz_s,
              title_main: str, title_sub: str, region_assign=None):
    """Scatter overlay panel (XY + XZ).  If region_assign given, color
    warped points by cluster."""
    ax_xy.imshow(colorize(B_xy_s, (1.0, 0.0, 0.8)), origin="upper")
    if region_assign is None:
        ax_xy.scatter(warped_zyx[:, 2], warped_zyx[:, 1], s=2, c="lime",
                      alpha=0.6, linewidths=0)
    else:
        for k in range(int(region_assign.max()) + 1):
            m = region_assign == k
            ax_xy.scatter(warped_zyx[m, 2], warped_zyx[m, 1], s=2,
                          color=CLUSTER_COLORS[k], alpha=0.7, linewidths=0)
    ax_xy.scatter(B_cent[:, 2], B_cent[:, 1], s=2, c="magenta",
                  alpha=0.4, linewidths=0)
    ax_xy.set_title(f"{title_main}\n{title_sub}", color="white", fontsize=9)
    ax_xy.set_xticks([]); ax_xy.set_yticks([])
    ax_xy.set_facecolor("black")
    ax_xy.text(0.02, 0.97, "XY MIP", transform=ax_xy.transAxes,
               color="yellow", fontsize=10, fontweight="bold", va="top",
               bbox=dict(facecolor="black", edgecolor="yellow", pad=2))

    ax_xz.imshow(colorize(B_xz_s, (1.0, 0.0, 0.8)), origin="upper",
                 aspect="auto")
    if region_assign is None:
        ax_xz.scatter(warped_zyx[:, 2], warped_zyx[:, 0], s=2, c="lime",
                      alpha=0.6, linewidths=0)
    else:
        for k in range(int(region_assign.max()) + 1):
            m = region_assign == k
            ax_xz.scatter(warped_zyx[m, 2], warped_zyx[m, 0], s=2,
                          color=CLUSTER_COLORS[k], alpha=0.7, linewidths=0)
    ax_xz.scatter(B_cent[:, 2], B_cent[:, 0], s=2, c="magenta",
                  alpha=0.4, linewidths=0)
    ax_xz.set_xticks([]); ax_xz.set_yticks([])
    ax_xz.set_facecolor("black")
    ax_xz.text(0.02, 0.97, "XZ MIP\n(x →, z ↓)", transform=ax_xz.transAxes,
               color="cyan", fontsize=10, fontweight="bold", va="top",
               bbox=dict(facecolor="black", edgecolor="cyan", pad=2))


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    A_cent = np.load(RES / "A_centroids_3d.npy")  # (N_A, 3) z,y,x
    B_cent = np.load(RES / "B_centroids_3d.npy")
    a = np.load(RES / "assignments_3d.npz")
    matches, dists = a["matches"], a["distances"]

    src_all = A_cent[matches[:, 0]]
    dst_all = B_cent[matches[:, 1]]
    sims = 100.0 / (1.0 + dists)

    A_xy, A_xz = load_or_compute_mips("Larry_2A_1", "Larry_2A_1_488_4x.tif")
    B_xy, B_xz = load_or_compute_mips("Larry_2A_8", "Larry_2A_8_488_4x.tif")
    B_xy_s = stretch(B_xy)
    B_xz_s = stretch(B_xz)

    panels = []  # list of (main_title, sub_title, warped_pts, region_assign or None)

    # Baseline: identity (no warp)
    panels.append(("Baseline (no registration)",
                   f"A_cent in own frame  ·  no transform",
                   A_cent.astype(np.float64), None))

    # --- Global RANSAC per threshold ---
    fit_log: dict = {}
    for s_thr in S_THRESHOLDS:
        if s_thr is None:
            mask = np.ones(len(matches), dtype=bool); tag = "all"
        else:
            mask = sims >= s_thr; tag = f"s>={s_thr:g}"
        n_pts = int(mask.sum())
        if n_pts < 4:
            fit_log[tag] = {"n": n_pts, "note": "too few for 3D affine"}
            continue
        src = src_all[mask]; dst = dst_all[mask]
        result = ransac_affine_3d(src, dst)
        if result is None:
            fit_log[tag] = {"n": n_pts, "note": "RANSAC failed"}
            continue
        M, off, inl = result
        n_inl = int(inl.sum())
        pred = src[inl] @ M.T + off
        res = np.linalg.norm(pred - dst[inl], axis=1)
        med_res = float(np.median(res))
        max_res = float(res.max())
        sz = float(np.linalg.norm(M[:, 0]))
        sy = float(np.linalg.norm(M[:, 1]))
        sx = float(np.linalg.norm(M[:, 2]))
        det = float(np.linalg.det(M))
        fit_log[tag] = dict(n=n_pts, n_inliers=n_inl,
                            median_residual_vox=med_res,
                            max_residual_vox=max_res,
                            det=det, sx=sx, sy=sy, sz=sz,
                            M=M.tolist(), t=off.tolist())
        warped = A_cent @ M.T + off
        panels.append((f"{tag}  ·  n={n_pts}  ·  inliers={n_inl}",
                       f"med res = {med_res:.2f} vox  ·  det = {det:.3f}",
                       warped, None))

    (RES / "registration_fits_3d.json").write_text(json.dumps(fit_log, indent=2))

    # --- 2-region RANSAC per threshold ---
    regional_log: dict = {}
    for s_thr in S_THRESHOLDS:
        if s_thr is None:
            mask = np.ones(len(matches), dtype=bool); tag = "all"
        else:
            mask = sims >= s_thr; tag = f"s>={s_thr:g}"
        src = src_all[mask]; dst = dst_all[mask]
        n_pts = int(mask.sum())
        if n_pts < K_REGIONS * 4:
            regional_log[tag] = {"n": n_pts, "note": "too few for regional fit"}
            continue

        km = KMeans(n_clusters=K_REGIONS, random_state=0, n_init=10)
        labels = km.fit_predict(src)

        regions = []
        region_info = []
        ok = True
        for k in range(K_REGIONS):
            idx = labels == k
            if idx.sum() < 4:
                ok = False; break
            result = ransac_affine_3d(src[idx], dst[idx])
            if result is None:
                ok = False; break
            M, off, inl = result
            pred = src[idx][inl] @ M.T + off
            res = np.linalg.norm(pred - dst[idx][inl], axis=1)
            regions.append({"M": M, "t": off,
                            "src_center": src[idx].mean(axis=0)})
            region_info.append(dict(
                n=int(idx.sum()), n_inliers=int(inl.sum()),
                median_residual_vox=float(np.median(res)),
                det=float(np.linalg.det(M)),
                M=M.tolist(), t=off.tolist(),
                src_center=regions[-1]["src_center"].tolist()))
        if not ok:
            regional_log[tag] = {"note": "RANSAC failed in one region"}
            continue

        regional_log[tag] = region_info

        # Warp ALL A centroids piecewise by nearest cluster center in src space
        warped = warp_points_regional(A_cent, regions)
        centers = np.stack([r["src_center"] for r in regions])
        d = np.linalg.norm(A_cent[:, None, :] - centers[None, :, :], axis=2)
        assign = np.argmin(d, axis=1)
        n_inl_total = sum(r["n_inliers"] for r in region_info)
        sub = "  ".join(
            f"R{k}: inl={region_info[k]['n_inliers']}  "
            f"med={region_info[k]['median_residual_vox']:.1f}"
            for k in range(K_REGIONS))
        panels.append((f"2-region  {tag}  ·  n={n_pts}  ·  inl={n_inl_total}",
                       sub, warped, assign))

    (RES / "registration_fits_3d_regional.json").write_text(
        json.dumps(regional_log, indent=2))

    if not panels:
        print("[warn] no panels rendered")
        return

    # --- Figure: 2 rows (XY MIP, XZ MIP) per panel ---
    n = len(panels)
    cols = 4
    grid_rows = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(5.5 * cols, 11 * grid_rows), facecolor="black")
    for i, (tt, sub, warped, region_assign) in enumerate(panels):
        gr = i // cols
        gc = i % cols
        ax_xy = fig.add_subplot(2 * grid_rows, cols, 2 * gr * cols + gc + 1)
        ax_xz = fig.add_subplot(2 * grid_rows, cols,
                                (2 * gr + 1) * cols + gc + 1)
        add_panel(ax_xy, ax_xz, warped, B_cent, B_xy_s, B_xz_s,
                  tt, sub, region_assign)

    fig.suptitle(
        "3D RANSAC affine  ·  global & 2-region K-means  ·  "
        "filtered by similarity s = 100/(1+d̄)  ·  "
        "green/colored = warped A centroids   magenta = B centroids   "
        f"(inlier={INLIER_VOX:g} vox, iters={RANSAC_ITERS})",
        color="white", fontsize=14, y=1.00,
    )
    out = RES / "registration_thresholds_3d.png"
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"wrote {out}")
    print(f"wrote {RES / 'registration_fits_3d.json'}")
    print(f"wrote {RES / 'registration_fits_3d_regional.json'}")


if __name__ == "__main__":
    main()
