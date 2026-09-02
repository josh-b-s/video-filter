# Edge-art video pipeline
## For the client (no Python required)

If you received a zipped `EdgeArtPicker` folder:

1. Unzip it anywhere, double-click `EdgeArtPicker.exe`.
2. See "What non-technical users do" in `build_exe.md` for the
   click-by-click walkthrough.
3. Finished videos land in the `output/` folder next to the exe.

If something looks wrong (crash on startup, GPU not being used,
missing files), send it back to the dev team - `build_exe.md` covers
the most common causes.

## Layout

```
config.py                  global settings (edit this for defaults)
core/
  job_store.py              per-video points/crop-trim/toggles/status, in video_jobs.json
  face_flatten.py           face detection + flattening only
  edges.py                  edge_detect() + recolor() only
  video_io.py                trim/crop, FPS resampling, chunk concatenation, frame reads
  sam2_segment.py           SAM2/EdgeTAM video + single-image mask prediction
  pipeline.py                orchestrates the above for one video (trim/crop -> resample -> SAM2 -> chunks -> stitch)
gui/
  point_picker.py           the clickable canvas widget + live mask preview
  crop_trim_picker.py       crop-rectangle drag + trim-range sliders, per video
  queue_worker.py           background QThread that processes the queue
  main_window.py            wires video list + tabs (points/crop-trim) + queue together
app.py                      GUI entry point
process_queue_cli.py        headless: process already-picked videos, no GUI
install_dependencies.py     installs requirements.txt + torch, with a choice of CUDA build
build_exe.md                packaging app.py into a standalone .exe for the client
tools/tune_edges.py         optional advanced slider tuner (python -m tools.tune_edges <file>)
```

## One-time setup

```
python install_dependencies.py --cuda cu118
```

Installs `requirements.txt`, then torch/torchvision with the CUDA
build you choose (`--help` for other options: `--cpu`, `--auto`, other
CUDA versions). `cu118` is the safe default if unsure - runs on older
GPUs/drivers too, unlike newer CUDA builds which need newer drivers
and just fall back to CPU otherwise. EdgeTAM is vendored under
`EdgeTAM/` (not on PyPI) and installs automatically as part of this
step.

## Normal workflow (GUI)

```
python app.py
```

1. Choose the folder of videos.
2. For each video: in the **Points** tab, click fg/bg points on the
   person; in the **Crop / Trim** tab, drag a crop rectangle and set
   trim start/end with the sliders. Toggle "Remove face" and "Use
   EdgeTAM/SAM2 segmentation" as needed (SAM2 off grays out the Points
   tab, since points do nothing without it). Click **Confirm & Queue
   → Next video**.
3. That video starts processing in the background immediately while
   you keep picking the next one.
4. Outputs land in `output/`.

Picked points, crop/trim range, and per-video toggles persist to
`video_jobs.json` next to `config.py`, so closing and reopening the
app resumes where you left off (anything left "queued"/"processing"
from a previous session is automatically re-queued on startup).

## Packaging as a standalone .exe

See `build_exe.md`.

## Headless / render-box use

If someone already picked points via the GUI and you just need to run
the actual processing on a separate (faster/GPU) machine, copy the
video folder + `video_jobs.json` over and run:

```
python process_queue_cli.py
```
