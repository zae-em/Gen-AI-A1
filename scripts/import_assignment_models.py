"""Download the pinned assignment models and their original report evidence."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "AbdullahMasood73/genai-assignment-1"
BUNDLE = ROOT / "imports" / "genai-assignment-1"
MODEL_NAMES = {
    "universal": "task1_universal_ae.onnx",
    "classifier": "task2_classifier.onnx",
    "salt": "task2_specialist_salt_pepper.onnx",
    "blur": "task2_specialist_blur.onnx",
    "occlusion": "task2_specialist_occlusion.onnx",
    "soft": "task3_soft_moe.onnx",
    "gan": "task4_generator.onnx",
}


def git_hash(data):
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def fetch(entry, commit):
    target = BUNDLE / entry["path"]
    if target.is_file() and git_hash(target.read_bytes()) == entry["sha"]:
        return
    url = f"https://raw.githubusercontent.com/{REPOSITORY}/{commit}/{entry['path']}"
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "RestorationBench"})
            with urllib.request.urlopen(request, timeout=90) as response:
                data = response.read()
            if len(data) != entry["size"] or git_hash(data) != entry["sha"]:
                raise ValueError(f"Download integrity check failed: {entry['path']}")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".download")
            temporary.write_bytes(data)
            temporary.replace(target)
            return
        except Exception:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def main():
    tree = json.loads((ROOT / "docs/upstream-tree.json").read_text(encoding="utf-8-sig"))
    commit = tree["sha"]
    references = {"README.md", "backend/app.py", "pyproject.toml",
                  "requirements.txt", "requirements-train.txt", "requirements-lock.txt"}
    entries = [item for item in tree["tree"] if item["type"] == "blob" and
               (item["path"].startswith(("artifacts/", "report/", "backend/samples/", "docs/",
                                         "restoration/", "scripts/", "notebooks/"))
                or item["path"] in references)]
    entries.sort(key=lambda item: (not item["path"].startswith("artifacts/models/"), item["path"]))
    print(f"Downloading {len(entries)} files ({sum(item['size'] for item in entries) / 1e6:.1f} MB), commit {commit}", flush=True)
    failures = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fetch, entry, commit): entry for entry in entries}
        for count, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
            except Exception as error:
                failures.append({"path": futures[future]["path"], "error": str(error)})
                print(f"Failed: {failures[-1]}", flush=True)
            if count % 25 == 0 or count == len(entries):
                print(f"Downloaded {count}/{len(entries)} files", flush=True)
    if failures:
        raise RuntimeError(f"Incomplete import: {failures}")
    manifest = json.loads((BUNDLE / "artifacts/models/manifest.json").read_text())
    installed = {}
    for name, filename in MODEL_NAMES.items():
        source = BUNDLE / "artifacts/models" / f"{name}.onnx"
        entry = manifest["models"][name]
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != entry["sha256"] or not entry["verified"] or entry["synthetic"]:
            raise ValueError(f"Model verification failed: {name}")
        destination = ROOT / "models/onnx" / filename
        shutil.copy2(source, destination)
        installed[name] = {"source": entry["file"], "installed_as": filename,
                           "sha256": digest, "epoch": entry["epoch"]}
    provenance = {"repository": f"https://github.com/{REPOSITORY}", "commit": commit,
                  "file_count": len(entries), "models": installed,
                  "note": "Original upstream report and experiment results are preserved in imports/genai-assignment-1."}
    (ROOT / "models/onnx/import-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print("Installed all seven trained models; SHA-256 checks passed.", flush=True)


if __name__ == "__main__":
    main()
