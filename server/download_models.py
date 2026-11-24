"""Downloads every model the local server needs (about 5 GB), skipping files already present."""

from __future__ import annotations

import os
import shutil
import sys
import urllib.request
from pathlib import Path

FORCING_DIR = Path(os.environ.get("FORCING_DIR", "D:/visavis/AvatarForcing"))
DATA_DIR = Path(os.environ.get("VISAVIS_DATA", "D:/visavis/data"))
MODELS = FORCING_DIR / "pretrained_dir"
HF = "https://huggingface.co"

# Avatar Forcing's own weights are published on Google Drive by its authors.
DRIVE = {
    "flow_transformer.pth": "1lchvHiDka0bBQ29bdY6lHxCyNGbGzvS_",
    "motion_autoencoder.pth": "1a0ZqX9GMrkb3JFQgOBmUB0V-Bpwtv4Hp",
}

# (repository, file in repository, destination)
FILES: list[tuple[str, str, Path]] = []

# wav2vec 2.0: turns the voices into the features that drive the avatar's face
for name in [
    "config.json",
    "preprocessor_config.json",
    "feature_extractor_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "model.safetensors",
    "pytorch_model.bin",
]:
    FILES.append(("facebook/wav2vec2-base-960h", name, MODELS / "wav2vec2-base-960h" / name))

# Whisper used to understand the user's speech
STT = DATA_DIR / "models/whisper-stt"
for name in [
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "normalizer.json",
    "added_tokens.json",
    "special_tokens_map.json",
    "model.safetensors",
]:
    FILES.append(("openai/whisper-large-v3-turbo", name, STT / name))

# Piper voices
VOICES = DATA_DIR / "voices"
for path in [
    "fr/fr_FR/siwis/medium/fr_FR-siwis-medium",
    "fr/fr_FR/tom/medium/fr_FR-tom-medium",
    "en/en_US/lessac/medium/en_US-lessac-medium",
    "en/en_US/ryan/medium/en_US-ryan-medium",
]:
    stem = path.rsplit("/", 1)[1]
    FILES.append(("rhasspy/piper-voices", f"{path}.onnx", VOICES / f"{stem}.onnx"))
    FILES.append(("rhasspy/piper-voices", f"{path}.onnx.json", VOICES / f"{stem}.onnx.json"))


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
    with urllib.request.urlopen(request) as response, open(partial, "wb") as out:
        shutil.copyfileobj(response, out, length=1 << 20)
    partial.replace(target)


def main() -> int:
    failed = 0
    for repo, name, target in FILES:
        if target.exists() and target.stat().st_size > 0:
            continue
        print(f"{repo}/{name}", flush=True)
        try:
            download(f"{HF}/{repo}/resolve/main/{name}", target)
        except Exception as error:
            failed += 1
            print(f"  ECHEC : {error}", flush=True)

    for name, file_id in DRIVE.items():
        target = MODELS / name
        if target.exists() and target.stat().st_size > 0:
            continue
        print(f"Google Drive : {name}", flush=True)
        try:
            import gdown

            gdown.download(id=file_id, output=str(target), quiet=True)
            if not target.exists():
                raise RuntimeError("fichier non reçu")
        except Exception as error:
            failed += 1
            print(f"  ECHEC : {error}", flush=True)

    # The example portrait shipped with Avatar Forcing, used until you add your own.
    avatars = DATA_DIR / "avatars"
    avatars.mkdir(parents=True, exist_ok=True)
    if not any(avatars.iterdir()):
        for image in sorted((FORCING_DIR / "data").glob("*.jpg"))[:1]:
            shutil.copy(image, avatars / image.name)

    print("Terminé." if not failed else f"{failed} fichier(s) en échec.", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
