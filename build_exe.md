# Building the standalone .exe

The GUI (`app.py`) is what non-technical teammates should launch - a
double-clickable app, no terminal, no Python install required.

## One-time setup (on a Windows machine, in your normal dev venv)

```bat
python install_dependencies.py --cuda cu118
pip install pyinstaller
```

(See `install_dependencies.py` for CUDA version options. Pick a
moderate build like cu118, not the newest one, if this exe needs to
run on client machines with an unknown or older GPU driver - newer
CUDA builds need newer drivers and just fall back to CPU rather than
stepping down to an older CUDA version on their own.)

## Build

From the project root:

```bat
pyinstaller --noconfirm --clean --windowed --name EdgeArtPicker ^
  --add-data "config.py;." ^
  --collect-all mediapipe ^
  --collect-all torch ^
  --collect-all sam2 ^
  --collect-all hydra ^
  --collect-all omegaconf ^
  --collect-all timm ^
  app.py

xcopy /E /I models dist\EdgeArtPicker\models
xcopy /E /I EdgeTAM\checkpoints dist\EdgeArtPicker\EdgeTAM\checkpoints
```

**Then copy the model/checkpoint folders manually, after the build:**

```bat
xcopy /E /I models dist\EdgeArtPicker\models
xcopy /E /I EdgeTAM\checkpoints dist\EdgeArtPicker\EdgeTAM\checkpoints
```

Why manually instead of via `--add-data`: since PyInstaller 6, `--onedir`
builds put the `.exe` directly in `dist\EdgeArtPicker\`, but anything
added via `--add-data` (or `--collect-all`) lands inside
`dist\EdgeArtPicker\_internal\` instead. `config.py`'s paths
(`"models/..."`, `"EdgeTAM/checkpoints/edgetam.pt"`) are relative to
the exe's working directory - `dist\EdgeArtPicker\` - not to
`_internal\`, so anything placed via `--add-data` at those paths is
one folder too deep for the app to find. Copying them there directly,
after the build, keeps the folder structure matching what `config.py`
actually expects. Redo the `xcopy` steps after every rebuild, since
`--clean` wipes `dist\` each time.

`--clean` wipes PyInstaller's cache and the `build/` folder before
each run, so intermediate files don't pile up across repeated builds.

After a build, `build/` is safe to delete (or just add it to
`.gitignore` - it's regenerated every time). Keep `EdgeArtPicker.spec`
though: PyInstaller writes this on first run and it's the reusable
build recipe (equivalent to the whole command above) - later builds
can just run `pyinstaller EdgeArtPicker.spec` instead of retyping all
the flags. Only `dist/EdgeArtPicker/` is what you actually zip and
send to the client.

Notes:
- `--windowed` suppresses the console window (this is a GUI app).
- `--collect-all` is needed for mediapipe/torch/sam2/hydra/omegaconf/
  timm because they load data files and native extensions PyInstaller
  can't always find by static analysis alone. None of these are
  imported by your own code directly - `sam2/__init__.py` imports
  `hydra`, and EdgeTAM's backbone loads `timm` via a Hydra *target
  string* (`sam2.modeling.backbones.timm.TimmBackbone`), resolved at
  runtime rather than a normal `import` statement - so PyInstaller's
  static analysis can't see either dependency coming, even with
  `--collect-all sam2` in place. If the built exe crashes on startup
  or on first SAM2 use with an import or "file not found" error,
  that's almost always a missing `--collect-all` for whichever package
  it's complaining about - check the traceback for the module name and
  add it to the list. Any Hydra-instantiated component (backbones,
  heads, etc.) is a likely source of one of these, since Hydra
  target-string imports are invisible to PyInstaller by design.
- The face landmarker `.task` model file and the EdgeTAM checkpoint
  (`.pt`) are large binaries, copied in manually via `xcopy` after the
  build rather than through `--add-data` - see the note above the
  build command for why. Don't forget this step after every rebuild;
  a build that "succeeds" but skips this will crash the moment SAM2 or
  face-removal actually runs, not at startup.
- `config.sam2_model_config_path` (e.g. `"configs/edgetam.yaml"`) is a
  Hydra config path resolved relative to the vendored `sam2` package's
  own install location (`EdgeTAM/sam2/configs/`), not a filesystem
  path relative to the exe - so it doesn't need its own `--add-data`
  line. `--collect-all sam2` already bundles that whole package,
  configs included.
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
3. For each video, in the **Points** tab: click on the person (left
   click = keep, right click = exclude), watch the red mask preview.
   In the **Crop / Trim** tab: drag a rectangle to crop, use the
   sliders to set where the clip starts/ends.
4. Tick/untick "Remove face" and "Use EdgeTAM/SAM2 segmentation" as
   needed for that video (unchecking SAM2 grays out the Points tab,
   since points do nothing without it), then click
   **Confirm & Queue → Next video**.
5. Processing happens in the background while you keep picking the
   next video. The log at the bottom shows progress; the list on the
   left shows a status icon per video (queued / processing / done /
   error).
6. Finished videos land in the `output/` folder next to the exe.