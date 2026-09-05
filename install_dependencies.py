"""
Dependency installer for the client machine.

Installs everything from requirements.txt, then installs torch/
torchvision separately with a CUDA build you choose - because torch
is the one package where "which build" actually matters (CPU-only vs
a specific CUDA version), unlike everything else in requirements.txt
which is platform-agnostic.

Usage:
    python install_dependencies.py

You'll be prompted to pick a torch build. If you already know what
you want, skip the prompt:

    python install_dependencies.py --cuda cu118
    python install_dependencies.py --cuda cu121
    python install_dependencies.py --cpu
    python install_dependencies.py --auto      (detect GPU, pick a build)

Recommendation for "should work on whatever machine we hand this to":
pick a moderate CUDA version (cu118 or cu121), not the newest one.
Older CUDA builds run on OLDER drivers too (they degrade to CPU
gracefully if there's no compatible GPU) - a bleeding-edge CUDA build
requires a bleeding-edge driver and does NOT step down to an older
CUDA version on its own, just straight to CPU. --auto below detects
whether an NVIDIA GPU is present at all but can't tell you the client's
driver version from here, so it can't pick the newest safe CUDA build
for them - that's a judgment call you make once, not something to
automate per-machine.
"""

import argparse
import subprocess
import sys

# Keep this list in sync with whatever CUDA builds PyTorch currently
# publishes at https://pytorch.org/get-started/locally/ - torch drops
# old CUDA builds and adds new ones over time, so this WILL need
# updating on the next torch upgrade rather than left as fixed forever.
TORCH_INDEX_URLS = {
    "cu118": "https://download.pytorch.org/whl/cu118",
    "cu121": "https://download.pytorch.org/whl/cu121",
    "cu124": "https://download.pytorch.org/whl/cu124",
    "cpu": "https://download.pytorch.org/whl/cpu",
}

TORCH_PACKAGES = ["torch", "torchvision"]


def detect_nvidia_gpu() -> bool:
    """Best-effort check for an NVIDIA GPU via nvidia-smi. Returns
    False (not an error) if nvidia-smi isn't found - that just means
    either no NVIDIA GPU or no driver installed, both of which mean
    "install CPU torch"."""
    try:
        subprocess.run(
            ["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def prompt_for_choice() -> str:
    print("\nWhich torch build do you want to install?")
    print("  1) CPU only (no GPU, or unsure)")
    print("  2) CUDA 11.8  (recommended default - works on most NVIDIA GPUs/drivers)")
    print("  3) CUDA 12.1")
    print("  4) CUDA 12.4  (newest - needs a newer NVIDIA driver)")
    choice = input("Enter 1-4: ").strip()
    return {"1": "cpu", "2": "cu118", "3": "cu121", "4": "cu124"}.get(choice, "cu118")


def install_requirements() -> None:
    print("\nInstalling requirements.txt (everything except torch/torchvision)...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True
    )


def install_torch(build: str) -> None:
    index_url = TORCH_INDEX_URLS[build]
    print(f"\nInstalling torch/torchvision ({build}) from {index_url} ...")
    # --force-reinstall --no-deps: without --force-reinstall, pip sees
    # any already-installed version satisfying the (unpinned) torch
    # requirement and silently skips this install entirely - even if
    # that installed version is the plain CPU wheel pulled in earlier
    # by EdgeTAM's unpinned "torch>=2.3.1" dependency. This is exactly
    # what happened here: the CUDA install step ran and printed
    # "Requirement already satisfied", never actually installing the
    # CUDA build. --no-deps avoids redundantly reinstalling everything
    # torch/torchvision depend on.
    subprocess.run(
        [sys.executable, "-m", "pip", "install", *TORCH_PACKAGES, "--index-url", index_url,
         "--force-reinstall", "--no-deps"],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--cuda", choices=[k for k in TORCH_INDEX_URLS if k != "cpu"],
                        help="Install a specific CUDA build of torch.")
    group.add_argument("--cpu", action="store_true", help="Install CPU-only torch.")
    group.add_argument("--auto", action="store_true",
                        help="Detect an NVIDIA GPU and default to cu118 if found, else CPU. "
                             "Does not know the driver version - cu118 is chosen for broad "
                             "compatibility, not because it's the newest option.")
    args = parser.parse_args()

    if args.cuda:
        build = args.cuda
    elif args.cpu:
        build = "cpu"
    elif args.auto:
        build = "cu118" if detect_nvidia_gpu() else "cpu"
        print(f"Auto-detected: {'NVIDIA GPU found' if build != 'cpu' else 'no NVIDIA GPU found'} "
              f"-> installing {build} build.")
    else:
        build = prompt_for_choice()

    install_requirements()
    install_torch(build)

    print("\nDone. Run the app with: python app.py")


if __name__ == "__main__":
    main()