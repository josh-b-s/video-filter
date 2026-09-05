r"""
Pre-downloads the timm backbone weights EdgeTAM's TimmBackbone depends
on (repvit_m1.dist_in1k, per the model id seen in app logs) into a
local "hf_cache" folder - so the packaged app can run with zero
internet access at runtime (see app.py's HF_HOME/HF_HUB_OFFLINE setup,
which points at this same folder and forces offline mode).

Run this ONCE during packaging, with a real internet connection - the
client never runs this. Safe to re-run; an already-cached model is a
no-op.

IMPORTANT: confirm the exact model identifier and any create_model
kwargs against your vendored copy before relying on this:
    findstr /s /n "create_model" EdgeTAM\sam2\modeling\backbones\timm.py
The exact kwargs (features_only, num_classes, etc.) don't change which
weight file gets cached, but the model NAME must match exactly or this
warms the wrong cache entry.

Usage:
    python warm_hf_cache.py
"""

import os

# Must be set before timm/huggingface_hub get imported - this is the
# folder package_for_client.bat later copies into the client's
# package, and the same folder app.py points HF_HOME at when the app
# actually runs.
HF_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_cache")
os.environ["HF_HOME"] = HF_CACHE_DIR

import timm

# Confirm this matches your vendored EdgeTAM/sam2/modeling/backbones/timm.py
# exactly (see the findstr command above) before relying on this script.
MODEL_NAME = "repvit_m1.dist_in1k"


def main() -> None:
    print(f"Downloading {MODEL_NAME} into {HF_CACHE_DIR} ...")
    timm.create_model(MODEL_NAME, pretrained=True)
    print("Done. This hf_cache folder is what package_for_client.bat "
          "copies into the client's package for offline use.")


if __name__ == "__main__":
    main()
