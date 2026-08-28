# Building the standalone .exe

The GUI (`app.py`) is what non-technical teammates should launch - a
double-clickable app, no terminal, no Python install required.

## One-time setup (on a Windows machine, in your normal dev venv)

```bat
pip install -r requirements.txt
pip install pyinstaller
```

## Build

From the project root:

```bat
pyinstaller --noconfirm --windowed --name EdgeArtPicker ^
  --add-data "config.py;." ^
  --add-data "models;models" ^
  --add-data "configs;configs" ^
  --collect-all mediapipe ^
  --collect-all torch ^
  --collect-all sam2 ^
  app.py
```

Notes:
- `--windowed` suppresses the console window (this is a GUI app).
- `--collect-all` is needed for mediapipe/torch/sam2 because they load
  data files and native extensions PyInstaller can't always find by
  static analysis alone. If the built exe crashes on startup with an
  import or "file not found" error, that's almost always a missing
  `--collect-all` for whichever package it's complaining about.
- The face landmarker `.task` model file and the EdgeTAM checkpoint
  (`.pt`) are large binaries - bundling them into the exe works but
  makes it huge. Simpler: ship them as separate files next to the exe
  and leave `config.face_landmarker_model_path` /
  `config.sam2_checkpoint_path` as relative paths - PyInstaller's
  `--add-data` above already keeps `models/` and `configs/` alongside
  the exe rather than embedding them.
- GPU support: whichever machine runs the exe needs a CUDA-capable
  torch install matching its GPU/driver - torch itself doesn't need
  rebuilding per machine, but if you're distributing to teammates with
  different GPUs, test on a representative machine first. Without
  CUDA, everything still works, just on CPU (much slower for SAM2).

## Output

`dist/EdgeArtPicker/EdgeArtPicker.exe` - zip up the whole
`dist/EdgeArtPicker/` folder (not just the .exe) and send that; it
contains the DLLs/data the exe needs alongside it.

## What non-technical users do

1. Unzip, double-click `EdgeArtPicker.exe`.
2. "Choose video folder..." → pick the folder of clips.
3. For each video: click on the person (left click = keep, right
   click = exclude), watch the red mask preview, tick/untick "Remove
   face", click **Confirm & Queue → Next video**.
4. Processing happens in the background while you keep picking the
   next video. The log at the bottom shows progress; the list on the
   left shows a status icon per video (queued / processing / done /
   error).
5. Finished videos land in the `output/` folder next to the exe.
