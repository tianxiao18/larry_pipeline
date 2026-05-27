# larry_pipeline

End-to-end registration between two ex-vivo light-sheet volumes of the same
sample (`Larry_2A_1` -> `Larry_2A_8`). Produces per-blob affine + TPS
correspondences and a warped BigTIFF of A in B's frame.

## Pipeline

| Stage | Script | Output |
| --- | --- | --- |
| 1. Cellpose-SAM 3D segmentation | `exvivo_segmentation.py` | `A/B_centroids_3d.npy`, `A/B_mask_3d.npz` |
| 2. Soma-print matching (k-NN + Hungarian) | `soma_print_match.py` | `assignments_3d.npz` |
| 3. Per-blob RANSAC affine on matches | `find_landmarks.py` | `registration_via_blobs_3d_*.json`, `.png` |
| 4. Inverse-affine + TPS volume warp | `warp_volume.py` | `Larry_2A_1_warped_to_2A_8.tif` |

`ransac_affine.py` is a shared module imported by stages 3 and 4.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Cellpose 4.x will pull a GPU build of torch if CUDA is available; otherwise it
falls back to CPU (slow on full volumes). The first run downloads the `cpsam`
checkpoint.

## Layout

Everything (code, inputs, outputs) lives inside `larry_pipeline/`:

```
larry_pipeline/
  *.py
  README.md
  requirements.txt
  data/
    Larry_2A_1_488_4x.tif         # volume A
    Larry_2A_8_488_4x.tif         # volume B
  results/
    via_export_csv.csv            # hand-drawn 2-polygon VIA export keyed by
                                  # Larry_2A_1_xy_mip_clean.png and
                                  # Larry_2A_8_xy_mip_clean.png
    mips/                         # XY/XZ MIP cache (auto)
    ...                           # all stage outputs land here
```

The two TIFs in `data/` and `results/via_export_csv.csv` are the only inputs;
everything else is created by the pipeline.

The VIA CSV is hand-drawn before stage 3 — two polygon blobs per volume's XY
MIP, exported from [VGG Image Annotator](https://www.robots.ox.ac.uk/~vgg/software/via/).
All other files are created by the pipeline.

## Run

Full pipeline (stages 1–4 in order):

```bash
python run_pipeline.py
```

Single stage:

```bash
python run_pipeline.py find_landmarks
```

Tunables (env vars):

| Var | Default | Used by |
| --- | --- | --- |
| `CP3D_ANISOTROPY` | `1.0` | stage 1 (Cellpose z:xy ratio) |
| `CP3D_BATCH_SIZE` | `64` | stage 1 |
| `MATCH3D_K` | `10` | stage 2 (k-NN soma-print size) |
| `MATCH3D_WORKERS` | `cpu_count()` | stage 2 (set to `1` to force serial) |
