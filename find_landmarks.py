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
  registration_via_blobs_3d.png
  registration_via_blobs_3d_fits.json      M, t, metrics per-blob RANSAC fit
  registration_via_blobs_3d_inliers.json   per-blob RANSAC inlier (A,B) pairs
  registration_via_blobs_3d_sparse.json    FPS-selected sparse landmark pairs
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.draw import polygon as sk_polygon
from scipy.interpolate import RBFInterpolator

from ransac_affine import (
    RES, INLIER_VOX, RANSAC_ITERS,
    ransac_affine_3d, warp_points_regional,
    stretch, colorize, load_or_compute_mips,
)

CLUSTER_COLORS = ["lime", "deepskyblue"]
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

    A_xy, A_xz = load_or_compute_mips("Larry_2A_1", "Larry_2A_1_488_4x.tif")
    B_xy, B_xz = load_or_compute_mips("Larry_2A_8", "Larry_2A_8_488_4x.tif")

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
        r["M_sparse"] = M_s
        r["t_sparse"] = t_s
        r["tps_sparse"] = tps_s
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

    # Save combined inlier correspondences for volume warping in TPS script
    # valid = [r for r in regions if r is not None]
    # if valid:
    #     np.save(RES / "tps_ctrl_src.npy",
    #             np.concatenate([r["src_inl"] for r in valid]))
    #     np.save(RES / "tps_ctrl_dst.npy",
    #             np.concatenate([r["dst_inl"] for r in valid]))
    #     print(f"[tps] saved {sum(len(r['src_inl']) for r in valid)} "
    #           f"combined inlier control points")

    # Warp all A centroids by their blob's affine; centroids outside both
    # blobs stay at identity (treated as "no warp" or skipped from plot).
    warped = A_cent.copy().astype(np.float64)
    warped_sparse_aff = A_cent.copy().astype(np.float64)
    warped_sparse_tps = A_cent.copy().astype(np.float64)
    for r in regions:
        if r is None:
            continue
        m = A_labels == r["blob"]
        warped[m] = A_cent[m] @ r["M"].T + r["t"]
        warped_sparse_aff[m] = A_cent[m] @ r["M_sparse"].T + r["t_sparse"]
        warped_sparse_tps[m] = r["tps_sparse"](A_cent[m])

    # TPS on top of affine: fit RBF from inlier src -> dst per blob,
    # subsample control points if needed to keep the O(N^3) solve tractable.
    warped_tps = A_cent.copy().astype(np.float64)
    for r in regions:
        if r is None:
            continue
        m = A_labels == r["blob"]
        sc, dc = r["src_inl"], r["dst_inl"]
        if len(sc) > TPS_MAX_CTRL:
            idx_sub = rng.choice(len(sc), TPS_MAX_CTRL, replace=False)
            sc, dc = sc[idx_sub], dc[idx_sub]
        tps = RBFInterpolator(sc, dc, kernel="thin_plate_spline",
                              smoothing=TPS_SMOOTHING)
        warped_tps[m] = tps(A_cent[m])

    B_xy_s = stretch(B_xy)
    B_xz_s = stretch(B_xz)
    A_xy_s = stretch(A_xy)
    A_xz_s = stretch(A_xz)

    fig, axes = plt.subplots(2, 5, figsize=(35, 14), facecolor="black")

    # Left col: VIA blob overlay on A
    ax = axes[0, 0]
    ax.imshow(colorize(A_xy_s, (0.0, 1.0, 0.2)), origin="upper")
    for k in [1, 2]:
        m_pix = A_blob_mask == k
        fill = np.zeros((*A_blob_mask.shape, 4), dtype=np.float32)
        col = matplotlib.colors.to_rgb(CLUSTER_COLORS[k - 1])
        fill[m_pix] = (*col, 0.25)
        ax.imshow(fill, origin="upper")
        m_cent = A_labels == k
        ax.scatter(A_cent[m_cent, 2], A_cent[m_cent, 1], s=2,
                   c=CLUSTER_COLORS[k - 1], alpha=0.7, linewidths=0,
                   label=f"blob {k} A (n={int(m_cent.sum())})")
    m_out = A_labels == 0
    if m_out.any():
        ax.scatter(A_cent[m_out, 2], A_cent[m_out, 1], s=2,
                   c="dimgray", alpha=0.3, linewidths=0,
                   label=f"outside ({int(m_out.sum())})")
    ax.legend(facecolor="black", labelcolor="white", fontsize=8, loc="upper right")
    ax.set_title("VIA blob masks on Larry_2A_1 (XY)", color="white", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor("black")
    ax.text(0.02, 0.97, "XY MIP", transform=ax.transAxes, color="yellow",
            fontsize=10, fontweight="bold", va="top",
            bbox=dict(facecolor="black", edgecolor="yellow", pad=2))

    ax = axes[1, 0]
    ax.imshow(colorize(A_xz_s, (0.0, 1.0, 0.2)), origin="upper", aspect="auto")
    for k in [1, 2]:
        m_cent = A_labels == k
        ax.scatter(A_cent[m_cent, 2], A_cent[m_cent, 0], s=2,
                   c=CLUSTER_COLORS[k - 1], alpha=0.7, linewidths=0)
    if m_out.any():
        ax.scatter(A_cent[m_out, 2], A_cent[m_out, 0], s=2,
                   c="dimgray", alpha=0.3, linewidths=0)
    ax.set_title("VIA blob assignment of A centroids (XZ)",
                 color="white", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor("black")
    ax.text(0.02, 0.97, "XZ MIP\n(x →, z ↓)", transform=ax.transAxes,
            color="cyan", fontsize=10, fontweight="bold", va="top",
            bbox=dict(facecolor="black", edgecolor="cyan", pad=2))

    # Right col: warped A over B
    ax = axes[0, 1]
    ax.imshow(colorize(B_xy_s, (1.0, 0.0, 0.8)), origin="upper")
    for k in [1, 2]:
        m = A_labels == k
        if not m.any():
            continue
        ax.scatter(warped[m, 2], warped[m, 1], s=3, c=CLUSTER_COLORS[k - 1],
                   alpha=0.55, linewidths=0)
    ax.scatter(B_cent[:, 2], B_cent[:, 1], s=2, c="magenta", alpha=0.4,
               linewidths=0)
    parts = []
    for r in region_info:
        if "note" in r:
            parts.append(f"B{r['blob']}: {r['note']}")
        else:
            parts.append(f"B{r['blob']}: inl={r['n_inliers']}/{r['n_matches']}  "
                         f"med={r['median_residual_vox']:.1f}vox  "
                         f"det={r['det']:.3f}")
    ax.set_title("Warped A over B (XY)\n" + "  ".join(parts),
                 color="white", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor("black")
    ax.text(0.02, 0.97, "XY MIP", transform=ax.transAxes, color="yellow",
            fontsize=10, fontweight="bold", va="top",
            bbox=dict(facecolor="black", edgecolor="yellow", pad=2))

    ax = axes[1, 1]
    ax.imshow(colorize(B_xz_s, (1.0, 0.0, 0.8)), origin="upper", aspect="auto")
    for k in [1, 2]:
        m = A_labels == k
        if not m.any():
            continue
        ax.scatter(warped[m, 2], warped[m, 0], s=3, c=CLUSTER_COLORS[k - 1],
                   alpha=0.55, linewidths=0)
    ax.scatter(B_cent[:, 2], B_cent[:, 0], s=2, c="magenta", alpha=0.4,
               linewidths=0)
    ax.set_title("Warped A over B (XZ)", color="white", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor("black")
    ax.text(0.02, 0.97, "XZ MIP\n(x →, z ↓)", transform=ax.transAxes,
            color="cyan", fontsize=10, fontweight="bold", va="top",
            bbox=dict(facecolor="black", edgecolor="cyan", pad=2))

    # Col 2: TPS-warped A over B
    ax = axes[0, 2]
    ax.imshow(colorize(B_xy_s, (1.0, 0.0, 0.8)), origin="upper")
    for k in [1, 2]:
        m = A_labels == k
        if not m.any():
            continue
        ax.scatter(warped_tps[m, 2], warped_tps[m, 1], s=3,
                   c=CLUSTER_COLORS[k - 1], alpha=0.55, linewidths=0)
    ax.scatter(B_cent[:, 2], B_cent[:, 1], s=2, c="magenta", alpha=0.4,
               linewidths=0)
    ax.set_title(f"TPS-warped A over B (XY)\nctrl≤{TPS_MAX_CTRL}  smooth={TPS_SMOOTHING}",
                 color="white", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor("black")
    ax.text(0.02, 0.97, "XY MIP", transform=ax.transAxes, color="yellow",
            fontsize=10, fontweight="bold", va="top",
            bbox=dict(facecolor="black", edgecolor="yellow", pad=2))

    ax = axes[1, 2]
    ax.imshow(colorize(B_xz_s, (1.0, 0.0, 0.8)), origin="upper", aspect="auto")
    for k in [1, 2]:
        m = A_labels == k
        if not m.any():
            continue
        ax.scatter(warped_tps[m, 2], warped_tps[m, 0], s=3,
                   c=CLUSTER_COLORS[k - 1], alpha=0.55, linewidths=0)
    ax.scatter(B_cent[:, 2], B_cent[:, 0], s=2, c="magenta", alpha=0.4,
               linewidths=0)
    ax.set_title("TPS-warped A over B (XZ)", color="white", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor("black")
    ax.text(0.02, 0.97, "XZ MIP\n(x →, z ↓)", transform=ax.transAxes,
            color="cyan", fontsize=10, fontweight="bold", va="top",
            bbox=dict(facecolor="black", edgecolor="cyan", pad=2))

    # Col 3: sparse-FPS affine refit (~SPARSE_TOTAL landmarks across blobs)
    parts_s_aff = []
    for r in region_info:
        if "note" in r:
            parts_s_aff.append(f"B{r['blob']}: {r['note']}")
        else:
            parts_s_aff.append(f"B{r['blob']}: ctrl={r['n_sparse_ctrl']}  "
                               f"med={r['sparse_aff_med_residual_vox']:.1f}vox")
    ax = axes[0, 3]
    ax.imshow(colorize(B_xy_s, (1.0, 0.0, 0.8)), origin="upper")
    for k in [1, 2]:
        m = A_labels == k
        if not m.any():
            continue
        ax.scatter(warped_sparse_aff[m, 2], warped_sparse_aff[m, 1], s=3,
                   c=CLUSTER_COLORS[k - 1], alpha=0.55, linewidths=0)
    ax.scatter(B_cent[:, 2], B_cent[:, 1], s=2, c="magenta", alpha=0.4,
               linewidths=0)
    # Mark the FPS-selected source landmarks used for the sparse fit
    for r in regions:
        if r is None:
            continue
        sc = r["src_inl"][r["sparse_idx"]]
        ax.scatter(sc[:, 2], sc[:, 1], s=18, marker="x", c="white",
                   linewidths=0.8, alpha=0.9)
    ax.set_title(f"Sparse-{SPARSE_TOTAL} affine over B (XY)\n" +
                 "  ".join(parts_s_aff),
                 color="white", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor("black")
    ax.text(0.02, 0.97, "XY MIP", transform=ax.transAxes, color="yellow",
            fontsize=10, fontweight="bold", va="top",
            bbox=dict(facecolor="black", edgecolor="yellow", pad=2))

    ax = axes[1, 3]
    ax.imshow(colorize(B_xz_s, (1.0, 0.0, 0.8)), origin="upper", aspect="auto")
    for k in [1, 2]:
        m = A_labels == k
        if not m.any():
            continue
        ax.scatter(warped_sparse_aff[m, 2], warped_sparse_aff[m, 0], s=3,
                   c=CLUSTER_COLORS[k - 1], alpha=0.55, linewidths=0)
    ax.scatter(B_cent[:, 2], B_cent[:, 0], s=2, c="magenta", alpha=0.4,
               linewidths=0)
    for r in regions:
        if r is None:
            continue
        sc = r["src_inl"][r["sparse_idx"]]
        ax.scatter(sc[:, 2], sc[:, 0], s=18, marker="x", c="white",
                   linewidths=0.8, alpha=0.9)
    ax.set_title(f"Sparse-{SPARSE_TOTAL} affine over B (XZ)",
                 color="white", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor("black")
    ax.text(0.02, 0.97, "XZ MIP\n(x →, z ↓)", transform=ax.transAxes,
            color="cyan", fontsize=10, fontweight="bold", va="top",
            bbox=dict(facecolor="black", edgecolor="cyan", pad=2))

    # Col 4: sparse-FPS TPS (same landmarks as col 3)
    parts_s_tps = []
    for r in region_info:
        if "note" in r:
            parts_s_tps.append(f"B{r['blob']}: {r['note']}")
        else:
            parts_s_tps.append(f"B{r['blob']}: ctrl={r['n_sparse_ctrl']}  "
                               f"med={r['sparse_tps_med_residual_vox']:.1f}vox")
    ax = axes[0, 4]
    ax.imshow(colorize(B_xy_s, (1.0, 0.0, 0.8)), origin="upper")
    for k in [1, 2]:
        m = A_labels == k
        if not m.any():
            continue
        ax.scatter(warped_sparse_tps[m, 2], warped_sparse_tps[m, 1], s=3,
                   c=CLUSTER_COLORS[k - 1], alpha=0.55, linewidths=0)
    ax.scatter(B_cent[:, 2], B_cent[:, 1], s=2, c="magenta", alpha=0.4,
               linewidths=0)
    for r in regions:
        if r is None:
            continue
        sc = r["src_inl"][r["sparse_idx"]]
        ax.scatter(sc[:, 2], sc[:, 1], s=18, marker="x", c="white",
                   linewidths=0.8, alpha=0.9)
    ax.set_title(f"Sparse-{SPARSE_TOTAL} TPS over B (XY)\n" +
                 "  ".join(parts_s_tps),
                 color="white", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor("black")
    ax.text(0.02, 0.97, "XY MIP", transform=ax.transAxes, color="yellow",
            fontsize=10, fontweight="bold", va="top",
            bbox=dict(facecolor="black", edgecolor="yellow", pad=2))

    ax = axes[1, 4]
    ax.imshow(colorize(B_xz_s, (1.0, 0.0, 0.8)), origin="upper", aspect="auto")
    for k in [1, 2]:
        m = A_labels == k
        if not m.any():
            continue
        ax.scatter(warped_sparse_tps[m, 2], warped_sparse_tps[m, 0], s=3,
                   c=CLUSTER_COLORS[k - 1], alpha=0.55, linewidths=0)
    ax.scatter(B_cent[:, 2], B_cent[:, 0], s=2, c="magenta", alpha=0.4,
               linewidths=0)
    for r in regions:
        if r is None:
            continue
        sc = r["src_inl"][r["sparse_idx"]]
        ax.scatter(sc[:, 2], sc[:, 0], s=18, marker="x", c="white",
                   linewidths=0.8, alpha=0.9)
    ax.set_title(f"Sparse-{SPARSE_TOTAL} TPS over B (XZ)",
                 color="white", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor("black")
    ax.text(0.02, 0.97, "XZ MIP\n(x →, z ↓)", transform=ax.transAxes,
            color="cyan", fontsize=10, fontweight="bold", va="top",
            bbox=dict(facecolor="black", edgecolor="cyan", pad=2))

    fig.suptitle(
        f"3D VIA-blob: full affine (col 2) · full TPS≤{TPS_MAX_CTRL} (col 3) · "
        f"sparse-{SPARSE_TOTAL} affine (col 4) · sparse-{SPARSE_TOTAL} TPS (col 5)  ·  "
        f"{len(A_cent)} A centroids, {len(matches)} matches  ·  "
        f"inlier={INLIER_VOX:g} vox  ·  iters={RANSAC_ITERS}",
        color="white", fontsize=14, y=1.00,
    )
    out = RES / "registration_via_blobs_3d.png"
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"[done] wrote {out}")
    print(f"[done] wrote {RES / 'registration_via_blobs_3d_fits.json'}")


if __name__ == "__main__":
    main()
