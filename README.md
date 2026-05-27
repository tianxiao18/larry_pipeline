# larry_pipeline

Register two ex-vivo light-sheet volumes (`Larry_2A_1` -> `Larry_2A_8`) by
matching Cellpose-segmented cells, fitting per-blob affines, and warping A
into B's frame.

## Stages

| Stage | Script | Output |
| --- | --- | --- |
| 1. Cellpose-SAM 3D segmentation | `exvivo_segmentation.py` | `A/B_centroids_3d.npy`, `A/B_mask_3d.npz` |
| 2. Soma-print matching (k-NN + Hungarian) | `soma_print_match.py` | `assignments_3d.npz` |
| 3. Per-blob RANSAC affine on matches | `find_landmarks.py` | `registration_via_blobs_3d_*.json`, `.png` |
| 4. Inverse-affine + TPS volume warp | `warp_volume.py` | `Larry_2A_1_warped_to_2A_8.tif` |

`ransac_affine.py` is a shared module imported by stages 3 and 4.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py                  # all 4 stages
python run_pipeline.py find_landmarks   # single stage
```

Stage 1 skips any volume whose outputs already exist. Cellpose downloads the
`cpsam` checkpoint on first run and uses GPU if available.

## Inputs

```
data/    Larry_2A_1_488_4x.tif, Larry_2A_8_488_4x.tif
results/ via_export_csv.csv     # 2 polygons per volume from VIA, keyed by
                                # Larry_2A_{1,8}_xy_mip_clean.png
```

Everything else under `results/` is created by the pipeline.

## Env vars

`CP3D_ANISOTROPY` (1.0), `CP3D_BATCH_SIZE` (64) — stage 1 Cellpose.
`MATCH3D_K` (10), `MATCH3D_WORKERS` (cpu_count) — stage 2.
