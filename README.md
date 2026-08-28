# Edge-art video pipeline

Restructured for one-responsibility-per-file, plus a GUI for
non-technical people to pick SAM2 prompt points per video and queue
them for background processing.

## Layout

```
config.py                  global settings (edit this for defaults)
core/
  job_store.py              per-video points/face-toggle/status, in video_jobs.json
  face_flatten.py           face detection + flattening only
  edges.py                  edge_detect() + recolor() only
  video_io.py                FPS resampling, chunk concatenation, frame reads
  sam2_segment.py           SAM2/EdgeTAM video + single-image mask prediction
  pipeline.py                orchestrates the above for one video (SAM2 -> chunks -> stitch)
gui/
  point_picker.py           the clickable canvas widget + live mask preview
  queue_worker.py           background QThread that processes the queue
  main_window.py            wires video list + canvas + queue together
app.py                      GUI entry point
process_queue_cli.py        headless: process already-picked videos, no GUI
tools/tune_edges.py         optional advanced slider tuner (python -m tools.tune_edges <file>)
```

## Normal workflow (GUI)

```
python app.py
```

1. Choose the folder of videos.
2. For each video: click fg/bg points on the person, toggle "Remove
   face" (on by default), click **Confirm & Queue → Next video**.
3. That video starts processing in the background immediately while
   you keep picking the next one.
4. Outputs land in `output/`.

Picked points and per-video settings persist to `video_jobs.json` next
to `config.py`, so closing and reopening the app resumes where you
left off (anything left "queued"/"processing" from a previous session
is automatically re-queued on startup).

## Packaging as a standalone .exe

See `build_exe.md`.

## Headless / render-box use

If someone already picked points via the GUI and you just need to run
the actual processing on a separate (faster/GPU) machine, copy the
video folder + `video_jobs.json` over and run:

```
python process_queue_cli.py
```

## What changed from the old single-file scripts

- `canny.py` split into `core/face_flatten.py`, `core/edges.py`,
  `core/video_io.py`, `core/pipeline.py`.
- `sam2_segment.py` moved into `core/`, plus a small single-image
  prediction helper added for the GUI's live preview.
- `pick_prompt_points.py` and `test_sam2.py` are superseded by the GUI
  (`gui/point_picker.py` does live-preview point picking; queued jobs
  get processed automatically, so there's no separate "preview mask on
  N sample frames" step - you see the live mask while picking instead).
- `config.prompt_points_path` (one video at a time) replaced by
  `core/job_store.py` (one entry per video, so a whole folder can be
  picked and queued without overwriting each other).
- `use_face_blur` is no longer a single global switch - it's a
  per-video toggle in the job store, defaulting to
  `config.use_face_blur_default` (on).
