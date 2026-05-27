"""Warp the full-resolution Larry_2A_1 volume into Larry_2A_8's frame
using the per-blob RANSAC affine and save as a BigTIFF in the same
format/dtype as the input TIFFs.

Output shape = B's shape (191, 1982, 1981) uint16.

For each output voxel (z_b, y_b, x_b):
  - Look up B's VIA blob mask at (y_b, x_b) -> blob_id in {0, 1, 2}
  - Apply inverse of that blob's affine (M, t):  p_a = M_inv @ (p_b - t)
  - Sample A at p_a via trilinear map_coordinates
  - blob_id == 0 (outside both polygons) -> output 0

Processed slab-by-slab along z to keep memory under control.

Output:
  results/larry_within_exvivo_3d/Larry_2A_1_warped_to_2A_8.tif
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import distance_transform_edt, map_coordinates
from scipy.interpolate import RBFInterpolator

from ransac_affine import DATA_DIR, RES, ransac_affine_3d
from find_landmarks import (
    VIA_CSV, parse_via_csv, rasterize_blobs, assign_centroids_to_blobs,
    TPS_SMOOTHING, TPS_MAX_CTRL,
)

A_TIF = DATA_DIR / "Larry_2A_1_488_4x.tif"
B_TIF = DATA_DIR / "Larry_2A_8_488_4x.tif"
OUT_TIF = RES / "Larry_2A_1_warped_to_2A_8.tif"

SLAB = 16   # output z-slices processed per chunk; balances memory vs speed


def main() -> None:
    print(f"[load] reading A volume {A_TIF.name}")
    t0 = time.time()
    A_vol = tifffile.imread(str(A_TIF))   # (nz_A, nyA, nxA) uint16
    print(f"  A_vol: {A_vol.shape} {A_vol.dtype}  ({time.time()-t0:.1f}s)")

    print(f"[load] reading B header for shape")
    with tifffile.TiffFile(str(B_TIF)) as tf:
        nz_B = len(tf.pages)
        h_B, w_B = tf.pages[0].shape
    print(f"  B shape: ({nz_B}, {h_B}, {w_B})")

    # Load saved per-blob affines + rasterize B's VIA blob mask
    print(f"[fits] loading registration_via_blobs_3d_fits.json")
    fits = json.loads((RES / "registration_via_blobs_3d_fits.json").read_text())
    inv_affines = {}
    for r in fits:
        if "note" in r:
            print(f"  blob {r['blob']}: SKIP ({r['note']})")
            continue
        M = np.array(r["M"])
        t = np.array(r["t"])
        inv_affines[r["blob"]] = (np.linalg.inv(M), t)
        print(f"  blob {r['blob']}: inl={r['n_inliers']}/{r['n_matches']}  "
              f"det={r['det']:.3f}")

    polys = parse_via_csv(VIA_CSV)
    B_blob_mask = rasterize_blobs(polys["Larry_2A_8_xy_mip_clean.png"],
                                  (h_B, w_B))
    # No cropping: extend blob labels to cover the whole xy frame by
    # assigning every "outside" pixel to its nearest VIA blob (xy Voronoi
    # via Euclidean distance transform on the inverted mask).
    print(f"[extend] filling outside-VIA pixels via nearest-blob Voronoi")
    _, idx = distance_transform_edt(B_blob_mask == 0, return_indices=True)
    B_blob_mask = B_blob_mask[idx[0], idx[1]]
    assert (B_blob_mask > 0).all(), "every xy pixel must now have a blob"

    # Per-blob TPS correction fields (B-indexed, coarse grid)
    COARSE_Z = 16; COARSE_XY = 16; EVAL_CHUNK = 50_000
    A_cent_3d = np.load(RES / "A_centroids_3d.npy")
    B_cent_3d = np.load(RES / "B_centroids_3d.npy")
    asn = np.load(RES / "assignments_3d.npz")
    src_all = A_cent_3d[asn["matches"][:, 0]]
    dst_all = B_cent_3d[asn["matches"][:, 1]]
    with tifffile.TiffFile(str(A_TIF)) as tf:
        A_shape = tf.pages[0].shape
    A_blob_mask_tps = rasterize_blobs(polys["Larry_2A_1_xy_mip_clean.png"], A_shape)
    match_src_label = assign_centroids_to_blobs(A_cent_3d, A_blob_mask_tps)[asn["matches"][:, 0]]
    rng = np.random.default_rng(0)
    tps_disp_fields = {}   # blob_id -> (dz, dy, dx) coarse arrays on B grid
    for k, (M_inv_k, t_k) in inv_affines.items():
        idx = match_src_label == k
        if idx.sum() < 4:
            continue
        result = ransac_affine_3d(src_all[idx], dst_all[idx])
        if result is None:
            continue
        _, _, inl = result
        sc, dc = src_all[idx][inl], dst_all[idx][inl]
        if len(sc) > TPS_MAX_CTRL:
            sub = rng.choice(len(sc), TPS_MAX_CTRL, replace=False)
            sc, dc = sc[sub], dc[sub]
        # TPS maps affine-inverse(dst) -> src  (correction in A-space)
        dc_a = (dc - t_k[None, :]) @ M_inv_k.T
        tps_k = RBFInterpolator(dc_a, sc, kernel="thin_plate_spline", smoothing=TPS_SMOOTHING)
        zc = np.arange(0, nz_B + COARSE_Z,  COARSE_Z,  dtype=np.float64)
        yc = np.arange(0, h_B  + COARSE_XY, COARSE_XY, dtype=np.float64)
        xc = np.arange(0, w_B  + COARSE_XY, COARSE_XY, dtype=np.float64)
        nzc, nyc, nxc = len(zc), len(yc), len(xc)
        ZC, YC, XC = np.meshgrid(zc, yc, xc, indexing="ij")
        gb = np.stack([ZC.ravel(), YC.ravel(), XC.ravel()], axis=1)
        ga_approx = (gb - t_k[None, :]) @ M_inv_k.T
        ga_tps = np.empty_like(ga_approx)
        for i0 in range(0, len(ga_approx), EVAL_CHUNK):
            i1 = min(len(ga_approx), i0 + EVAL_CHUNK)
            ga_tps[i0:i1] = tps_k(ga_approx[i0:i1])
        disp = ga_tps - ga_approx
        tps_disp_fields[k] = (
            disp[:, 0].reshape(nzc, nyc, nxc).astype(np.float32),
            disp[:, 1].reshape(nzc, nyc, nxc).astype(np.float32),
            disp[:, 2].reshape(nzc, nyc, nxc).astype(np.float32),
        )
        print(f"[tps] blob {k}: coarse correction field ({nzc}×{nyc}×{nxc}) ready", flush=True)

    # Output writer (BigTIFF, same dtype as input)
    print(f"[warp] writing {OUT_TIF}")
    nz_A, ny_A, nx_A = A_vol.shape
    out_vol = np.zeros((nz_B, h_B, w_B), dtype=np.uint16)

    # Precompute y/x meshgrid for one slab; z varies per slab
    yy, xx = np.meshgrid(np.arange(h_B, dtype=np.float64),
                         np.arange(w_B, dtype=np.float64), indexing="ij")
    by = np.clip(np.round(yy).astype(int), 0, h_B - 1)
    bx = np.clip(np.round(xx).astype(int), 0, w_B - 1)
    blob_per_xy = B_blob_mask[by, bx]              # (h_B, w_B) int

    masks_for_blob = {k: (blob_per_xy == k) for k in [1, 2]}

    for z0 in range(0, nz_B, SLAB):
        z1 = min(nz_B, z0 + SLAB)
        zs = np.arange(z0, z1, dtype=np.float64)
        t_slab = time.time()
        for k, (M_inv, t) in inv_affines.items():
            xy_mask = masks_for_blob[k]
            if not xy_mask.any():
                continue
            # Build coords for all pixels in this slab that fall in blob k
            n_xy = int(xy_mask.sum())
            # broadcast: (n_z, n_xy)
            zb = np.broadcast_to(zs[:, None], (z1 - z0, n_xy)).reshape(-1)
            yb = np.broadcast_to(yy[xy_mask][None, :], (z1 - z0, n_xy)).reshape(-1)
            xb = np.broadcast_to(xx[xy_mask][None, :], (z1 - z0, n_xy)).reshape(-1)
            p_b = np.stack([zb, yb, xb], axis=1)         # (N, 3)
            # Inverse affine to A coords
            p_a = (p_b - t[None, :]) @ M_inv.T
            za = p_a[:, 0]
            ya = p_a[:, 1]
            xa = p_a[:, 2]
            # TPS correction on top of affine
            if k in tps_disp_fields:
                dz_f, dy_f, dx_f = tps_disp_fields[k]
                q = np.stack([zb / COARSE_Z, yb / COARSE_XY, xb / COARSE_XY])
                za = za + map_coordinates(dz_f, q, order=1, mode="nearest")
                ya = ya + map_coordinates(dy_f, q, order=1, mode="nearest")
                xa = xa + map_coordinates(dx_f, q, order=1, mode="nearest")
            # Sample
            sampled = map_coordinates(
                A_vol, np.stack([za, ya, xa]),
                order=1, mode="constant", cval=0.0,
            ).astype(np.uint16)
            # Write into output slab via flat indexing
            slab_view = out_vol[z0:z1]
            # broadcast back: for each z in slab, for each in-mask (y, x)
            zi_arr = np.repeat(np.arange(z1 - z0), n_xy)
            yx_y = np.tile(yy[xy_mask].astype(int), z1 - z0)
            yx_x = np.tile(xx[xy_mask].astype(int), z1 - z0)
            slab_view[zi_arr, yx_y, yx_x] = sampled
        print(f"  z={z0}-{z1-1}/{nz_B-1}  slab {time.time()-t_slab:.1f}s",
              flush=True)

    print(f"[save] writing BigTIFF")
    tifffile.imwrite(
        str(OUT_TIF), out_vol,
        bigtiff=True,
        photometric="minisblack",
        compression="zlib",
        compressionargs={"level": 4},
        metadata={"axes": "ZYX"},
    )
    print(f"[done] wrote {OUT_TIF}  "
          f"({OUT_TIF.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
