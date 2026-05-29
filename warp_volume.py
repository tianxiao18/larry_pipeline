"""Seam-free piecewise warp of A -> B (pipeline stage 4).

Same per-blob (affine + TPS-residual) fits as the per-blob RANSAC affines, but
instead of a HARD per-voxel Voronoi label (which makes the two blobs' transforms
collide at their boundary -> visible stitch/seam), this
evaluates BOTH blobs' full warp field everywhere and BLENDS them with smooth
inverse-distance (Gaussian) weights -> a continuous displacement field, no seam.

Ported from larry_register_erdem_v4_blobs_3d.py (soft partition-of-unity blend):
  * both fields evaluated on a coarse grid, trilinearly upsampled  -> low-pass
  * TPS residual magnitude clipped to MAX_RES_VOX                  -> no smear
  * weight_k(b) = exp(-d_k(b)/sigma), normalized over blobs        -> no seam

Output:
  results/Larry_2A_1_warped_to_2A_8.tif
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import map_coordinates

from ransac_affine import DATA_DIR, RES, ransac_affine_3d
from find_landmarks import (
    VIA_CSV, parse_via_csv, rasterize_blobs, assign_centroids_to_blobs,
    TPS_SMOOTHING, TPS_MAX_CTRL,
)

A_TIF   = DATA_DIR / "Larry_2A_1_488_4x.tif"
B_TIF   = DATA_DIR / "Larry_2A_8_488_4x.tif"
OUT_TIF = RES / "Larry_2A_1_warped_to_2A_8.tif"
OUT_PNG = RES / "warp_mips.png"

GZ, GY, GX  = 48, 144, 144     # coarse warp grid (eval here, trilinear upsample)
SLAB        = 8                # output z-slices per chunk
BLEND_SIGMA = 150.0            # Gaussian blend scale at blob boundary (vox, xy)
MAX_RES_VOX = 60.0             # clip TPS residual magnitude


def closest_xy_dist(grid_pts, anchors):
    """Min xy-distance from each (z,y,x) grid point to a set of anchors."""
    out = np.empty(len(grid_pts), dtype=np.float64)
    chunk = 4096
    for s in range(0, len(grid_pts), chunk):
        e = min(s + chunk, len(grid_pts))
        d = np.linalg.norm(grid_pts[s:e, None, 1:3] - anchors[None, :, 1:3], axis=2)
        out[s:e] = d.min(axis=1)
    return out


def main() -> None:
    t_all = time.time()

    # --- per-blob affines (forward A->B) -> invert to B->A ---
    fits = json.loads((RES / "registration_via_blobs_3d_fits.json").read_text())
    inv_affines = {}
    for r in fits:
        if "note" in r:
            print(f"  blob {r['blob']}: SKIP ({r['note']})")
            continue
        M = np.array(r["M"]); t = np.array(r["t"])
        inv_affines[r["blob"]] = (np.linalg.inv(M), t)
        print(f"  blob {r['blob']}: inl={r['n_inliers']}/{r['n_matches']}  det={r['det']:.3f}")

    # --- B frame shape ---
    with tifffile.TiffFile(str(B_TIF)) as tf:
        Bz = len(tf.pages); By, Bx = tf.pages[0].shape
    print(f"[frame] B = ({Bz}, {By}, {Bx})")

    # --- per-blob TPS (built from RANSAC inliers, exactly as warp_volume.py) ---
    A_cent = np.load(RES / "A_centroids_3d.npy")
    B_cent = np.load(RES / "B_centroids_3d.npy")
    asn = np.load(RES / "assignments_3d.npz")
    src_all = A_cent[asn["matches"][:, 0]]
    dst_all = B_cent[asn["matches"][:, 1]]
    polys = parse_via_csv(VIA_CSV)
    with tifffile.TiffFile(str(A_TIF)) as tf:
        A_shape = tf.pages[0].shape
    A_blob_mask = rasterize_blobs(polys["Larry_2A_1_xy_mip_clean.png"], A_shape)
    match_src_label = assign_centroids_to_blobs(A_cent, A_blob_mask)[asn["matches"][:, 0]]
    rng = np.random.default_rng(0)

    tps_per_blob = {}      # blob -> RBFInterpolator(A-space approx -> A target)
    B_anchors = {}         # blob -> (n,3) B-side z,y,x inlier centroids
    for k, (M_inv_k, t_k) in inv_affines.items():
        idx = match_src_label == k
        if idx.sum() < 4:
            continue
        result = ransac_affine_3d(src_all[idx], dst_all[idx])
        if result is None:
            continue
        _, _, inl = result
        sc, dc = src_all[idx][inl], dst_all[idx][inl]
        B_anchors[k] = dc.astype(np.float64)
        if len(sc) > TPS_MAX_CTRL:
            sub = rng.choice(len(sc), TPS_MAX_CTRL, replace=False)
            sc, dc = sc[sub], dc[sub]
        dc_a = (dc - t_k[None, :]) @ M_inv_k.T          # B-anchors -> A-space (affine)
        tps_per_blob[k] = RBFInterpolator(
            dc_a, sc, kernel="thin_plate_spline", smoothing=TPS_SMOOTHING)
        print(f"[tps] blob {k}: {len(B_anchors[k])} anchors, TPS ready")

    blobs = sorted(inv_affines)

    # --- coarse grid over the whole B frame ---
    gz = np.linspace(0, Bz - 1, GZ)
    gy = np.linspace(0, By - 1, GY)
    gx = np.linspace(0, Bx - 1, GX)
    GZG, GYG, GXG = np.meshgrid(gz, gy, gx, indexing="ij")
    flat = np.stack([GZG.ravel(), GYG.ravel(), GXG.ravel()], axis=1)   # (G,3) z,y,x

    # --- per-blob full src field (affine-inv + clipped TPS residual) on grid ---
    src_fields = {}
    for k in blobs:
        M_inv_k, t_k = inv_affines[k]
        aff = (flat - t_k[None, :]) @ M_inv_k.T                        # A-space affine
        if k in tps_per_blob:
            CHUNK = 100_000
            corrected = np.empty_like(aff)
            for s in range(0, len(aff), CHUNK):
                e = min(s + CHUNK, len(aff))
                corrected[s:e] = tps_per_blob[k](aff[s:e])
            resid = corrected - aff
            rgn = np.linalg.norm(resid, axis=1)
            n_clip = int((rgn > MAX_RES_VOX).sum())
            if n_clip:
                scale = np.where(rgn > MAX_RES_VOX, MAX_RES_VOX / np.maximum(rgn, 1e-9), 1.0)
                resid *= scale[:, None]
            src_fields[k] = aff + resid
            print(f"[grid] blob {k}: resid p50={np.median(rgn):.2f} "
                  f"p95={np.percentile(rgn,95):.2f} max={rgn.max():.2f} clip={n_clip}/{len(rgn)}")
        else:
            src_fields[k] = aff

    # --- smooth inverse-distance blend weights (partition of unity) ---
    print(f"[blend] sigma={BLEND_SIGMA} vox")
    weights = {}
    wsum = np.zeros(len(flat))
    for k in blobs:
        d = closest_xy_dist(flat, B_anchors[k])
        w = np.exp(-d / BLEND_SIGMA)
        weights[k] = w
        wsum += w
    wsum += 1e-12
    src = sum(src_fields[k] * (weights[k] / wsum)[:, None] for k in blobs)
    src_grid = src.reshape(GZ, GY, GX, 3)

    # --- warp slab-by-slab: trilinear-upsample src grid, sample A ---
    print(f"[load] {A_TIF.name}")
    A_f = tifffile.imread(str(A_TIF)).astype(np.float32)
    out_xy_mip = np.zeros((By, Bx), dtype=np.float32)
    print(f"[warp] writing {OUT_TIF.name}")
    with tifffile.TiffWriter(str(OUT_TIF), bigtiff=True) as tw:
        for z0 in range(0, Bz, SLAB):
            z1 = min(z0 + SLAB, Bz); t0 = time.time()
            zz, yy, xx = np.meshgrid(
                np.arange(z0, z1, dtype=np.float32),
                np.arange(By, dtype=np.float32),
                np.arange(Bx, dtype=np.float32),
                indexing="ij")
            nz = (zz / (Bz - 1)) * (GZ - 1)
            ny = (yy / (By - 1)) * (GY - 1)
            nx = (xx / (Bx - 1)) * (GX - 1)
            sz = map_coordinates(src_grid[..., 0], [nz, ny, nx], order=1, mode="nearest")
            sy = map_coordinates(src_grid[..., 1], [nz, ny, nx], order=1, mode="nearest")
            sx = map_coordinates(src_grid[..., 2], [nz, ny, nx], order=1, mode="nearest")
            samp = map_coordinates(A_f, [sz, sy, sx], order=1, mode="constant", cval=0.0)
            samp = np.clip(samp, 0, 65535).astype(np.uint16)
            tw.write(samp, photometric="minisblack", compression="zlib",
                     compressionargs={"level": 4})
            np.maximum(out_xy_mip, samp.astype(np.float32).max(0), out=out_xy_mip)
            print(f"  z={z0:3d}-{z1-1:3d}/{Bz-1}  slab {time.time()-t0:.1f}s", flush=True)

    np.save(RES / "Larry_2A_1_warped_to_2A_8_xy_mip.npy", out_xy_mip)
    print(f"[done] {OUT_TIF}  ({OUT_TIF.stat().st_size/1e9:.2f} GB)  "
          f"wall {time.time()-t_all:.1f}s")

    # --- visualization: XY MIPs (A source, B target, blended warp, overlay) ---
    print(f"[viz] {OUT_PNG.name}")
    A_mip = A_f.max(0)
    B_mip = tifffile.imread(str(B_TIF)).max(0).astype(np.float32)
    norm = lambda im: np.clip(im / (np.percentile(im, 99.5) + 1e-6), 0, 1)
    Wn, Bn = norm(out_xy_mip), norm(B_mip)
    fig, ax = plt.subplots(1, 4, figsize=(22, 6))
    for a, im, ttl in zip(ax[:3], [norm(A_mip), Bn, Wn],
                          ["A (source)", "B (target)", "A blend-warped -> B"]):
        a.imshow(im, cmap="gray"); a.set_title(ttl); a.axis("off")
    ov = np.zeros((*Bn.shape, 3)); ov[..., 1] = Bn; ov[..., 0] = Wn
    ax[3].imshow(ov); ax[3].set_title("overlay (B=green, warped=red)"); ax[3].axis("off")
    fig.suptitle("seam-free piecewise warp (per-blob affine+TPS, Gaussian "
                 f"distance blend sigma={BLEND_SIGMA:.0f} vox)")
    fig.tight_layout(); fig.savefig(OUT_PNG, dpi=130)
    print(f"[save] {OUT_PNG}")


if __name__ == "__main__":
    main()
