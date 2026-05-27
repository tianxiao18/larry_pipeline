"""3D registration with per-blob RANSAC, where the 2 blobs come from VIA
manual polygons exported as CSV.

Each centroid is assigned to a blob by looking up its (y, x) inside the
rasterized polygon mask of its volume.  Matches inherit the A-side
centroid's blob label.  Then RANSAC affine is fit independently per blob.

Inputs:
  results/larry_within_exvivo_3d/via_export_csv.csv   VIA polygons (2 per volume)
  results/larry_within_exvivo_3d/A_centroids_3d.npy
  results/larry_within_exvivo_3d/B_centroids_3d.npy
  results/larry_within_exvivo_3d/assignments_3d.npz
  Larry_2A_1_488_4x.tif, Larry_2A_8_488_4x.tif  (for XY/XZ MIPs)

Outputs:
  registration_via_blobs_3d_fits.json      M, t, metrics per-blob RANSAC fit
  registration_via_blobs_3d_inliers.json   per-blob RANSAC inlier (A,B) pairs
  registration_via_blobs_3d_sparse.json    FPS-selected sparse landmark pairs
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from skimage.draw import polygon as sk_polygon
from scipy.interpolate import RBFInterpolator

from ransac_affine import RES, ransac_affine_3d, load_or_compute_mips

TPS_SMOOTHING = 1.0   # RBF regularization (0 = exact interpolation through inliers)
TPS_MAX_CTRL  = 400   # subsample inliers if more than this (O(N^3) solve)
SPARSE_TOTAL  = 100   # total FPS landmarks across both blobs for sparse refits
SPARSE_MIN_CTRL = 40  # per-blob floor; chosen from the sweep elbow analysis
VIA_CSV = RES / "via_export_csv.csv"

# Map VIA filename -> (volume_tag, MIP shape (H, W) from raw MIP)
FILENAME_TO_TAG = {
    "Larry_2A_1_xy_mip_clean.png": "A",
    "Larry_2A_8_xy_mip_clean.png": "B",
}


def parse_via_csv(path: Path) -> dict[str, list[dict]]:
    """Returns {filename: [{"region_id": int, "xs": ndarray, "ys": ndarray}, ...]}"""
    polys: dict[str, list[dict]] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fn = row["filename"]
            shape = json.loads(row["region_shape_attributes"])
            if shape.get("name") != "polygon":
                continue
            polys.setdefault(fn, []).append({
                "region_id": int(row["region_id"]),
                "xs": np.array(shape["all_points_x"], dtype=np.float64),
                "ys": np.array(shape["all_points_y"], dtype=np.float64),
            })
    for fn in polys:
        polys[fn].sort(key=lambda p: p["region_id"])
    return polys


def rasterize_blobs(polys: list[dict], shape: tuple[int, int]) -> np.ndarray:
    """Rasterize VIA polygons into a uint8 label image (0=bg, 1, 2, ...)."""
    H, W = shape
    mask = np.zeros((H, W), dtype=np.uint8)
    for k, p in enumerate(polys[:2]):
        rr, cc = sk_polygon(p["ys"], p["xs"], shape=(H, W))
        mask[rr, cc] = k + 1
    return mask


def assign_centroids_to_blobs(cent: np.ndarray, blob_mask: np.ndarray) -> np.ndarray:
    """cent: (N, 3) (z, y, x).  Returns int labels 1/2 (0 = outside both blobs)."""
    H, W = blob_mask.shape
    ys = np.clip(np.round(cent[:, 1]).astype(int), 0, H - 1)
    xs = np.clip(np.round(cent[:, 2]).astype(int), 0, W - 1)
    return blob_mask[ys, xs].astype(np.int32)


def fps_indices(pts: np.ndarray, k: int, rng) -> np.ndarray:
    """Farthest-point sampling. Returns k indices spread across pts (N, D)."""
    n = len(pts)
    if k >= n:
        return np.arange(n)
    chosen = np.empty(k, dtype=np.int64)
    start = int(rng.integers(0, n))
    chosen[0] = start
    dists = np.linalg.norm(pts - pts[start], axis=1)
    for i in range(1, k):
        j = int(np.argmax(dists))
        chosen[i] = j
        dists = np.minimum(dists, np.linalg.norm(pts - pts[j], axis=1))
    return chosen


def lstsq_affine_3d(src: np.ndarray, dst: np.ndarray):
    """Closed-form least-squares affine: dst ≈ src @ M.T + t."""
    A = np.hstack([src, np.ones((len(src), 1))])
    sol, *_ = np.linalg.lstsq(A, dst, rcond=None)
    return sol[:3].T, sol[3]


def main() -> None:
    A_cent = np.load(RES / "A_centroids_3d.npy")
    B_cent = np.load(RES / "B_centroids_3d.npy")
    a = np.load(RES / "assignments_3d.npz")
    matches, dists = a["matches"], a["distances"]
    print(f"[load] A={len(A_cent)} B={len(B_cent)} matches={len(matches)}")

    polys = parse_via_csv(VIA_CSV)
    print(f"[via] parsed {sum(len(v) for v in polys.values())} polygons "
          f"from {len(polys)} files")

    # MIPs are computed only to provide the (H, W) canvas shape that the
    # VIA polygons were drawn on; the XZ MIP and pixel values are unused.
    A_xy, _ = load_or_compute_mips("Larry_2A_1", "Larry_2A_1_488_4x.tif")
    B_xy, _ = load_or_compute_mips("Larry_2A_8", "Larry_2A_8_488_4x.tif")

    A_blob_mask = rasterize_blobs(polys["Larry_2A_1_xy_mip_clean.png"], A_xy.shape)
    B_blob_mask = rasterize_blobs(polys["Larry_2A_8_xy_mip_clean.png"], B_xy.shape)
    print(f"[via] A blob sizes: 1={int((A_blob_mask==1).sum())}, "
          f"2={int((A_blob_mask==2).sum())}, "
          f"outside={int((A_blob_mask==0).sum())}")
    print(f"[via] B blob sizes: 1={int((B_blob_mask==1).sum())}, "
          f"2={int((B_blob_mask==2).sum())}, "
          f"outside={int((B_blob_mask==0).sum())}")

    # Assign every A centroid + every B centroid to a blob
    A_labels = assign_centroids_to_blobs(A_cent, A_blob_mask)
    B_labels = assign_centroids_to_blobs(B_cent, B_blob_mask)
    print(f"[assign] A: in blob 1: {int((A_labels==1).sum())},  "
          f"in blob 2: {int((A_labels==2).sum())},  "
          f"outside: {int((A_labels==0).sum())}")
    print(f"[assign] B: in blob 1: {int((B_labels==1).sum())},  "
          f"in blob 2: {int((B_labels==2).sum())},  "
          f"outside: {int((B_labels==0).sum())}")

    # Match → blob: take the A-side label
    match_src_label = A_labels[matches[:, 0]]
    src = A_cent[matches[:, 0]]
    dst = B_cent[matches[:, 1]]

    regions = []
    region_info = []
    inlier_pairs = []
    for k in [1, 2]:
        idx = match_src_label == k
        n_matches = int(idx.sum())
        print(f"[ransac] blob {k}: {n_matches} matches")
        if n_matches < 4:
            print(f"   too few matches; skipping")
            region_info.append({"blob": k, "n_matches": n_matches,
                                "note": "too few"})
            regions.append(None)
            continue
        result = ransac_affine_3d(src[idx], dst[idx])
        if result is None:
            region_info.append({"blob": k, "n_matches": n_matches,
                                "note": "RANSAC failed"})
            regions.append(None)
            continue
        M, off, inl = result
        n_inl = int(inl.sum())
        pred = src[idx][inl] @ M.T + off
        res = np.linalg.norm(pred - dst[idx][inl], axis=1)
        info = dict(
            blob=k,
            n_matches=n_matches,
            n_inliers=n_inl,
            median_residual_vox=float(np.median(res)),
            max_residual_vox=float(res.max()),
            det=float(np.linalg.det(M)),
            sx=float(np.linalg.norm(M[:, 2])),
            sy=float(np.linalg.norm(M[:, 1])),
            sz=float(np.linalg.norm(M[:, 0])),
            M=M.tolist(), t=off.tolist(),
        )
        region_info.append(info)
        regions.append({"M": M, "t": off, "blob": k,
                        "src_inl": src[idx][inl], "dst_inl": dst[idx][inl]})

        global_idx = np.where(match_src_label == k)[0][inl]
        for gi, a_zyx, b_zyx in zip(global_idx, src[idx][inl], dst[idx][inl]):
            inlier_pairs.append([
                int(gi),
                [int(round(float(a_zyx[0]))), int(round(float(a_zyx[1]))), int(round(float(a_zyx[2])))],
                [int(round(float(b_zyx[0]))), int(round(float(b_zyx[1]))), int(round(float(b_zyx[2])))],
            ])
        print(f"           inl={n_inl}/{n_matches}  "
              f"med_res={info['median_residual_vox']:.2f}vox  "
              f"det={info['det']:.3f}  "
              f"scales=({info['sx']:.3f},{info['sy']:.3f},{info['sz']:.3f})")

    # Sparse-100 (FPS) refits: spatially-distributed subset of RANSAC inliers.
    # Per-blob budget is proportional to that blob's inlier count, summing to
    # SPARSE_TOTAL. Affine refit is closed-form LSQ; TPS uses same kwargs as
    # the full-control-point fit so the only variable is # of control points.
    rng = np.random.default_rng(0)
    valid_inl = sum(len(r["src_inl"]) for r in regions if r is not None)
    for r, info in zip(regions, region_info):
        if r is None:
            info["n_sparse_ctrl"] = 0
            continue
        n_inl = len(r["src_inl"])
        prop = int(round(SPARSE_TOTAL * n_inl / valid_inl)) if valid_inl else 0
        budget = min(n_inl, max(SPARSE_MIN_CTRL, prop))
        idx = fps_indices(r["src_inl"], budget, rng)
        sc, dc = r["src_inl"][idx], r["dst_inl"][idx]
        M_s, t_s = lstsq_affine_3d(sc, dc)
        tps_s = RBFInterpolator(sc, dc, kernel="thin_plate_spline",
                                smoothing=TPS_SMOOTHING)
        # Held-out evaluation: residuals on inliers NOT used as control points.
        # In-sample residuals underestimate TPS error because the fit interpolates
        # through (or very close to) its control points. Held-out gives the true
        # generalization error and is what the sweep script reports.
        held = np.ones(len(r["src_inl"]), dtype=bool)
        held[idx] = False
        src_h, dst_h = r["src_inl"][held], r["dst_inl"][held]
        res_aff = np.linalg.norm(src_h @ M_s.T + t_s - dst_h, axis=1)
        res_tps = np.linalg.norm(tps_s(src_h) - dst_h, axis=1)
        r["sparse_idx"] = idx
        info["n_sparse_ctrl"] = int(len(idx))
        info["sparse_aff_med_residual_vox"] = float(np.median(res_aff))
        info["sparse_aff_max_residual_vox"] = float(res_aff.max())
        info["sparse_tps_med_residual_vox"] = float(np.median(res_tps))
        info["sparse_tps_max_residual_vox"] = float(res_tps.max())
        print(f"[sparse] blob {r['blob']}: ctrl={len(idx)}  "
              f"held-out aff_med_res={np.median(res_aff):.2f}vox  "
              f"held-out tps_med_res={np.median(res_tps):.2f}vox")

    (RES / "registration_via_blobs_3d_fits.json").write_text(
        json.dumps(region_info, indent=2))
    (RES / "registration_via_blobs_3d_inliers.json").write_text(
        json.dumps(inlier_pairs, indent=2))
    print(f"[save] {len(inlier_pairs)} RANSAC inlier pairs -> "
          f"registration_via_blobs_3d_inliers.json")

    # FPS-selected sparse landmarks (A↔B) used for the sparse affine/TPS refits
    sparse_pairs = []
    for r in regions:
        if r is None:
            continue
        sc = r["src_inl"][r["sparse_idx"]]
        dc = r["dst_inl"][r["sparse_idx"]]
        for s, d in zip(sc, dc):
            sparse_pairs.append({
                "blob": int(r["blob"]),
                "A_zyx": [float(s[0]), float(s[1]), float(s[2])],
                "B_zyx": [float(d[0]), float(d[1]), float(d[2])],
            })
    (RES / "registration_via_blobs_3d_sparse.json").write_text(
        json.dumps(sparse_pairs, indent=2))
    print(f"[save] {len(sparse_pairs)} sparse FPS landmarks -> "
          f"registration_via_blobs_3d_sparse.json")



if __name__ == "__main__":
    main()
