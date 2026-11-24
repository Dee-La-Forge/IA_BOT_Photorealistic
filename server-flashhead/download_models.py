"""Downloads every model the local server needs (about 10 GB), skipping files already present."""

from __future__ import annotations

import os
import shutil
import sys
import urllib.request
from pathlib import Path

FLASHHEAD_DIR = Path(os.environ.get("FLASHHEAD_DIR", "D:/visavis/SoulX-FlashHead"))
DATA_DIR = Path(os.environ.get("VISAVIS_DATA", "D:/visavis/data"))
MODELS = FLASHHEAD_DIR / "models"
HF = "https://huggingface.co"

# (repository, file in repository, destination)
FILES: list[tuple[str, str, Path]] = []

# SoulX-FlashHead (Lite): the talking-head video model and its video autoencoder
for name in [
    "Model_Lite/config.json",
    "Model_Lite/diffusion_pytorch_model.safetensors",
    "VAE_LTX/config.json",
    "VAE_LTX/diffusion_pytorch_model.safetensors",
]:
    FILES.append(("Soul-AILab/SoulX-FlashHead-1_3B", name, MODELS / "SoulX-FlashHead-1_3B" / name))

# wav2vec 2.0: turns the avatar's voice into the features that drive its face
for name in [
    "config.json",
    "preprocessor_config.json",
    "feature_extractor_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "model.safetensors",
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

    # Example portraits shipped with SoulX-FlashHead, used until you add your own.
    avatars = DATA_DIR / "avatars"
    avatars.mkdir(parents=True, exist_ok=True)
    if not any(avatars.iterdir()):
        for image in sorted((FLASHHEAD_DIR / "examples").glob("*.png"))[:2]:
            shutil.copy(image, avatars / image.name)

    print("Terminé." if not failed else f"{failed} fichier(s) en échec.", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
