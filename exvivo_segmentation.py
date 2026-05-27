"""Cellpose 3D detection on Larry within-exvivo volumes.

Runs `cpsam` with do_3D=True on the full z-stack of each volume and writes
3D centroids + compressed 3D mask. No MIP — full-volume segmentation so
that downstream registration can use real (z, y, x) coordinates.

Outputs (cellinvariance/some2.0/results/larry_within_exvivo_3d/):
  A_centroids_3d.npy   (N, 3) float32, columns = (z, y, x)
  B_centroids_3d.npy
  A_mask_3d.npz        compressed label volume, key 'mask' (uint16 if <65k cells)
  B_mask_3d.npz
  meta.json
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import center_of_mass
from tqdm import trange as tqdm_trange

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VOLUMES = [
    ("A", "Larry_2A_1", DATA_DIR / "Larry_2A_1_488_4x.tif"),
    ("B", "Larry_2A_8", DATA_DIR / "Larry_2A_8_488_4x.tif"),
]

PCT_LO = 1.0
PCT_HI = 99.5
CELLPOSE_PARAMS = dict(
    diameter=None,
    cellprob_threshold=0.0,
    flow_threshold=0.4,
    min_size=15,
    do_3D=True,
    anisotropy=float(os.environ.get("CP3D_ANISOTROPY", "1.0")),
)

CP_BATCH_SIZE = int(os.environ.get("CP3D_BATCH_SIZE", "64"))


def load_volume(tif: Path, label: str) -> np.ndarray:
    import tifffile
    t0 = time.time()
    vol = tifffile.imread(str(tif))
    if vol.dtype != np.uint16:
        vol = vol.astype(np.uint16, copy=False)
    print(f"[load:{label}] {vol.shape} loaded in {time.time()-t0:.1f}s",
          flush=True)
    return vol


def percentile_norm_u8(vol: np.ndarray) -> np.ndarray:
    a, b = np.percentile(vol, [PCT_LO, PCT_HI])
    if b <= a:
        return np.zeros_like(vol, dtype=np.uint8)
    x = np.clip((vol.astype(np.float32) - a) / (b - a), 0.0, 1.0)
    return (x * 255.0 + 0.5).astype(np.uint8)


def centroids_3d(mask: np.ndarray) -> np.ndarray:
    labels = np.unique(mask)
    labels = labels[labels != 0]
    if len(labels) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    coms = center_of_mass(mask > 0, labels=mask, index=labels)
    return np.asarray(coms, dtype=np.float32)  # (z, y, x)


def segment_volume(vol_u8: np.ndarray, label: str, model):
    t0 = time.time()
    print(f"[seg:{label}] cellpose do_3D=True on {vol_u8.shape} "
          f"anisotropy={CELLPOSE_PARAMS['anisotropy']} "
          f"batch_size={CP_BATCH_SIZE}", flush=True)
    masks, _flows, _styles = model.eval(
        vol_u8, normalize=True, z_axis=0, channel_axis=None,
        batch_size=CP_BATCH_SIZE,
        **CELLPOSE_PARAMS,
    )
    masks = np.asarray(masks)
    nlabels = int(masks.max())
    dtype = np.uint16 if nlabels < 65535 else np.uint32
    masks = masks.astype(dtype, copy=False)
    print(f"[seg:{label}] -> {nlabels} cells in {time.time()-t0:.1f}s "
          f"mask dtype={masks.dtype}", flush=True)
    return masks, centroids_3d(masks)


def _patch_cellpose_trange() -> None:
    """Retarget cellpose's internal trange to a normal terminal tqdm.

    Cellpose pipes its trange through TqdmToLogger with mininterval=30, so the
    tiling / dynamics progress is effectively invisible. Replacing the symbol
    in the relevant modules restores per-batch progress bars.
    """
    import cellpose.core as cp_core
    from cellpose import models as cp_models

    def trange(*args, **kwargs):
        kwargs.pop("file", None)
        return tqdm_trange(*args, **kwargs)

    cp_core.trange = trange
    cp_models.trange = trange
    try:
        import cellpose.dynamics as dyn
        dyn.trange = trange
    except ImportError:
        pass


def main() -> None:
    model = None  # lazy: only load Cellpose if a volume actually needs segmenting

    summary = {"params": CELLPOSE_PARAMS, "volumes": {}}
    for tag, name, tif in VOLUMES:
        cent_path = OUT_DIR / f"{tag}_centroids_3d.npy"
        mask_path = OUT_DIR / f"{tag}_mask_3d.npz"
        if cent_path.exists() and mask_path.exists():
            cent = np.load(cent_path)
            with np.load(mask_path) as npz:
                mask_shape = list(npz["mask"].shape)
                mask_dtype = str(npz["mask"].dtype)
            print(f"[skip:{name}] cached centroids ({len(cent)}) + mask exist",
                  flush=True)
            summary["volumes"][tag] = {
                "name": name, "tif": tif.name,
                "shape": mask_shape, "n_cells": int(len(cent)),
                "mask_dtype": mask_dtype,
            }
            continue

        if model is None:
            from cellpose import models
            _patch_cellpose_trange()
            use_gpu = True
            model = models.CellposeModel(gpu=use_gpu, pretrained_model="cpsam")
            print(f"[init] cellpose loaded gpu={use_gpu} "
                  f"device={getattr(model, 'device', '?')}", flush=True)

        vol = load_volume(tif, name)
        vol_u8 = percentile_norm_u8(vol)
        del vol
        mask, cent = segment_volume(vol_u8, name, model)
        del vol_u8

        np.save(OUT_DIR / f"{tag}_centroids_3d.npy", cent)
        np.savez_compressed(OUT_DIR / f"{tag}_mask_3d.npz", mask=mask)
        print(f"[save:{name}] wrote centroids ({len(cent)}) + mask",
              flush=True)

        summary["volumes"][tag] = {
            "name": name,
            "tif": tif.name,
            "shape": list(mask.shape),
            "n_cells": int(len(cent)),
            "mask_dtype": str(mask.dtype),
        }
        del mask, cent

    (OUT_DIR / "meta.json").write_text(json.dumps(summary, indent=2))
    print(f"[done] wrote {OUT_DIR / 'meta.json'}", flush=True)


if __name__ == "__main__":
    main()
