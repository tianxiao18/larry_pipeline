"""End-to-end pipeline: two ex-vivo volumes -> landmark correspondences -> warped volume.

Runs each stage in order. Skip a stage by removing it from `STAGES`, or run a
single stage with `python run_pipeline.py <stage_name>`.

    Stage 1  exvivo_segmentation    Cellpose 3D -> per-volume centroids + masks
    Stage 2  soma_print_match       k-NN soma-print + Hungarian -> 1:1 matches
    Stage 3  find_landmarks         Per-blob RANSAC affine on matches -> inliers
    Stage 4  warp_volume            Seam-free per-blob affine+TPS (Gaussian blend) -> BigTIFF
"""
from __future__ import annotations

import importlib
import sys
import time


# (module_name, one-line description shown at runtime)
STAGES: list[tuple[str, str]] = [
    ("exvivo_segmentation",
     "Cellpose 3D on each volume; writes A/B_centroids_3d.npy and A/B_mask_3d.npz."),
    ("soma_print_match",
     "For each centroid, build a k-NN soma-print, then inner+outer Hungarian "
     "to produce assignments_3d.npz (one-to-one A<->B matches)."),
    ("find_landmarks",
     "Assign each centroid to a VIA-drawn blob, fit a RANSAC affine per blob, "
     "and dump inlier (A,B) landmark pairs to JSON."),
    ("warp_volume",
     "Evaluate both blobs' affine+TPS warp fields over B's frame, blend them "
     "with Gaussian distance weights (seam-free), then trilinearly sample A -> "
     "writes a BigTIFF in B's shape. (warp_volume_hard.py = old hard-Voronoi version.)"),
]


def run_stage(module_name: str, description: str) -> None:
    print("\n" + "=" * 78)
    print(f"[stage] {module_name}")
    print(f"        {description}")
    print("=" * 78)
    t0 = time.time()
    mod = importlib.import_module(module_name)
    mod.main()
    print(f"[stage] {module_name} done in {time.time() - t0:.1f}s")


def main() -> None:
    requested = sys.argv[1:]
    if not requested:
        for name, desc in STAGES:
            run_stage(name, desc)
        return

    name_to_desc = dict(STAGES)
    for arg in requested:
        if arg in name_to_desc:
            run_stage(arg, name_to_desc[arg])
        else:
            raise SystemExit(
                f"unknown stage {arg!r}; choose from {[s for s, _ in STAGES]}"
            )


if __name__ == "__main__":
    main()
