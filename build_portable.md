# Packaging as a portable folder (no PyInstaller)

Alternative to `build_exe.md`'s PyInstaller build. Ships a real,
minimal Python interpreter alongside your actual source files instead
of freezing everything into a single compiled exe - the client still
gets a double-click launcher with no Python install required, but
nothing here needs a `--collect-all` guess-and-check cycle, since it's
just running your .py files normally, the same way `python app.py`
already does from your dev machine.

Trade-off versus the PyInstaller `.exe`: the client can see/open the
`.py` source files if they look in the folder (not an issue if hiding
source doesn't matter to you). Folder size is similar either way -
you're shipping all of torch/mediapipe/sam2's weight regardless of
packaging method.

## One-time setup (per Python/dependency version)

1. Download the **"Windows embeddable package (64-bit)"** for your
   project's Python version (currently 3.10.11) from
   https://www.python.org/downloads/windows/ - save it as
   `embed_python.zip` in the project root.
2. From the project root:
   ```bat
   build_portable.bat
   ```
   This extracts the embeddable Python into `portable\python\`,
   enables site-packages (off by default in the embeddable
   distribution), bootstraps `pip`, runs
   `install_dependencies.py --cuda cu118` to install everything
   (`requirements.txt` + a CUDA torch build), and runs
   `warm_hf_cache.py` to pre-download timm's backbone weights into
   `hf_cache\` - needs a real internet connection for this step (the
   client's machine won't need one; see "Offline operation" below).
3. Re-run this only when you bump the Python version, need a
   different CUDA build, or EdgeTAM's timm backbone changes - not
   every time you change your own code.

## Offline operation

Without this, the client's machine needs internet access the first
time SAM2 runs, since EdgeTAM's `TimmBackbone` fetches its weights
from Hugging Face Hub at runtime. `warm_hf_cache.py` pre-downloads
that weight into `hf_cache\` during your build (step 2 above, with
your internet connection, not the client's), and `app.py` points
`HF_HOME` at that same folder and sets `HF_HUB_OFFLINE=1` before
anything imports `timm`/`huggingface_hub` - so the shipped package
never makes a network call at all.

If you ever see a `HFValidationError` or a "couldn't find model" error
from a client, the likely cause is `warm_hf_cache.py`'s `MODEL_NAME`
not matching what your vendored EdgeTAM actually requests - confirm
with:
```bat
findstr /s /n "create_model" EdgeTAM\sam2\modeling\backbones\timm.py
```
and update `MODEL_NAME` in `warm_hf_cache.py` to match exactly, then
re-run `build_portable.bat` and re-package.

## Packaging for the client (every release)

```bat
package_for_client.bat
```

Assembles `dist_portable\EdgeArtPicker\` - copies the portable Python
plus `app.py`, `config.py`, `core/`, `gui/`, `tools/`, `EdgeTAM/`,
`models/`, and `hf_cache/` (the pre-downloaded backbone weights - see
"Offline operation" above), and writes `EdgeArtPicker.bat` as the
launcher. Zip that folder and send it.

Run this every time you have new code to ship - it's fast since it's
just copying files, not reinstalling dependencies.

## What the client does

1. Unzip anywhere.
2. Double-click `EdgeArtPicker.bat`.
3. Everything else is the same as the PyInstaller build - see "What
   non-technical users do" in `build_exe.md` for the click-by-click
   walkthrough (that section applies regardless of which packaging
   method built the app).

`EdgeArtPicker.bat` uses `pythonw.exe`, not `python.exe` - same effect
as PyInstaller's `--windowed` flag, no console window flashes open.
It also `cd`s into its own folder first (`%~dp0`) before launching, so
`config.py`'s relative paths (`models/...`, `EdgeTAM/checkpoints/...`,
`output/...`) resolve correctly no matter where the client unzips the
folder or what shortcut they use to launch it.

## Notes

- **GPU support** works the same as the PyInstaller build - whichever
  CUDA build you chose in `build_portable.bat` (`--cuda cu118` by
  default) determines what GPUs it can use; falls back to CPU
  automatically if no compatible GPU/driver is present.
- **Antivirus / SmartScreen**: a `.bat` that launches a bundled
  `python.exe` running torch/mediapipe can occasionally get flagged by
  overzealous antivirus or Windows SmartScreen on a machine that's
  never seen this software before, since it's an unsigned executable
  doing a lot of native/ML work. If the client reports a security
  warning, "Run anyway" / an antivirus exclusion for the folder is the
  fix - this isn't unique to this packaging method, the PyInstaller
  exe can trigger the same warnings, just worth knowing to expect it.
- `portable\`, `dist_portable\`, and `hf_cache\` are all build
  output - add them to `.gitignore` alongside `build\`/`dist\`/
  `*.spec` from the PyInstaller approach.
