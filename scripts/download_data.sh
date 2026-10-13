#!/usr/bin/env bash
# Fetch both datasets and build the deterministic corruption manifests.
set -euo pipefail

ROOT="${1:-data}"
mkdir -p "$ROOT"

echo "==> Oxford-IIIT Pet (Tasks 1-3)"
# torchvision downloads, verifies the checksum and extracts
python - "$ROOT" <<'PY'
import sys
from torchvision.datasets import OxfordIIITPet
root = sys.argv[1]
for split in ("trainval", "test"):
    ds = OxfordIIITPet(root=root, split=split, download=True)
    print(f"    {split}: {len(ds)} images")
PY

echo "==> FS2K (Task 4)"
if [ ! -d "$ROOT/FS2K" ]; then
  cat <<'MSG'
    FS2K needs a manual download (the authors host it on Google Drive):
        https://github.com/DengPingFan/FS2K
    Unpack it so that the layout is:
        data/FS2K/anno_train.json
        data/FS2K/anno_test.json
        data/FS2K/photo/...
        data/FS2K/sketch/...
MSG
else
  echo "    found $ROOT/FS2K"
fi

echo "==> Building corruption manifests"
python -m src.data.manifest --root "$ROOT" --out "$ROOT/manifests"
echo "Done."
