"""Talking-head engine: SoulX-FlashHead (Lite) generating the whole face in real time.

From one portrait, the model produces 512x512 video in step with a voice: lips, head
motion, blinks and expression. It works in slices of 24 frames (0.96 s at 25 fps).

Per avatar, a short idle clip (the model driven by silence) is generated once and cached;
the browser loops it while the avatar is not speaking.
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

# The model's own torch.compile calls need Triton, which Windows lacks: run uncompiled.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import cv2
import numpy as np
import torch

FLASHHEAD_DIR = Path(os.environ.get("FLASHHEAD_DIR", "D:/visavis/SoulX-FlashHead")).resolve()
DATA_DIR = Path(os.environ.get("VISAVIS_DATA", "D:/visavis/data")).resolve()
AVATAR_SRC_DIR = DATA_DIR / "avatars"
AVATAR_CACHE_DIR = DATA_DIR / "avatar-cache"
REF_DIR = DATA_DIR / "avatar-refs"

FPS = 25
SIZE = 512
AUDIO_RATE = 16000
SLICE_FRAMES = 24  # frames produced per generation step
SLICE_SAMPLES = SLICE_FRAMES * AUDIO_RATE // FPS
IDLE_SLICES = 8  # 7.7 s of idle motion, played forward then backward
JPEG_QUALITY = 90
CACHE_VERSION = 10
SEED = 42

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

# FlashHead resolves its config and modules relative to its own folder.
sys.path.insert(0, str(FLASHHEAD_DIR))
os.chdir(FLASHHEAD_DIR)


@dataclass
class Patch:
    """One generated frame, as a JPEG to draw at (x, y) on the avatar canvas."""

    x: int
    y: int
    jpeg: bytes


class Avatar:
    def __init__(self, avatar_id: str, cache: Path):
        self.id = avatar_id
        meta = json.loads((cache / "meta.json").read_text())
        self.width, self.height, self.count = meta["width"], meta["height"], meta["frames"]


class TalkStream:
    """Generates one stretch of continuous video for an avatar, slice by slice."""

    def __init__(self, engine: "Engine", avatar_id: str):
        self.engine = engine
        self.avatar_id = avatar_id
        self.motion = None  # last frames of the previous slice, which the next one continues
        self.audio = np.zeros(engine.context_samples, dtype=np.float32)

    @torch.no_grad()
    def frames(self, audio: np.ndarray) -> list[Patch]:
        """Turns SLICE_SAMPLES of 16 kHz audio into the SLICE_FRAMES frames that go with it."""
        from flash_head.inference import get_audio_embedding, run_pipeline

        engine = self.engine
        audio = np.asarray(audio, dtype=np.float32)[:SLICE_SAMPLES]
        audio = np.pad(audio, (0, SLICE_SAMPLES - len(audio)))
        self.audio = np.concatenate([self.audio[SLICE_SAMPLES:], audio])
        with engine.lock:
            pipeline = engine.pipeline
            pipeline.reset_person_name(self.avatar_id)
            if self.motion is not None:
                pipeline.latent_motion_frames = self.motion
            embedding = get_audio_embedding(
                pipeline, self.audio, engine.embed_end - engine.frame_num, engine.embed_end
            )
            video = run_pipeline(pipeline, embedding)[engine.motion_frames :]
            self.motion = pipeline.latent_motion_frames
            rgb = video.to(torch.uint8).cpu().numpy()
        patches = []
        for frame in rgb:
            ok, jpeg = cv2.imencode(
                ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            )
            patches.append(Patch(0, 0, jpeg.tobytes()))
        return patches


class Engine:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.avatars: dict[str, Avatar] = {}
        self.pipeline = None

    # ---- model ------------------------------------------------------------
    def load(self) -> None:
        if self.pipeline is not None:
            return
        from flash_head.inference import get_infer_params, get_pipeline

        self.pipeline = get_pipeline(
            world_size=1,
            ckpt_dir="models/SoulX-FlashHead-1_3B",
            model_type="lite",
            wav2vec_dir="models/wav2vec2-base-960h",
        )
        self.pipeline.rank = 1  # silences its per-step timing prints
        params = get_infer_params()
        self.frame_num = params["frame_num"]
        self.motion_frames = params["motion_frames_num"]
        if self.frame_num - self.motion_frames != SLICE_FRAMES or params["tgt_fps"] != FPS:
            raise RuntimeError("Unexpected FlashHead frame layout")
        self.context_samples = params["cached_audio_duration"] * AUDIO_RATE
        self.embed_end = params["cached_audio_duration"] * FPS

    # ---- avatars ----------------------------------------------------------
    def list_sources(self) -> dict[str, Path]:
        AVATAR_SRC_DIR.mkdir(parents=True, exist_ok=True)
        return {
            p.stem: p
            for p in sorted(AVATAR_SRC_DIR.iterdir())
            if p.suffix.lower() in VIDEO_EXTS | IMAGE_EXTS
        }

    def is_prepared(self, avatar_id: str, source: Path) -> bool:
        meta = AVATAR_CACHE_DIR / avatar_id / "meta.json"
        try:
            data = json.loads(meta.read_text())
        except (OSError, json.JSONDecodeError):
            return False
        return data.get("version") == CACHE_VERSION and data.get("source") == _stamp(source)

    def prepare_all(self, log=print) -> None:
        """Registers every portrait in the avatars folder and builds missing idle clips."""
        self.load()
        from flash_head.inference import get_base_data

        sources = self.list_sources()
        if REF_DIR.exists():
            shutil.rmtree(REF_DIR)
        REF_DIR.mkdir(parents=True)
        usable: dict[str, Path] = {}
        for avatar_id, source in sources.items():
            try:
                cv2.imwrite(str(REF_DIR / f"{avatar_id}.png"), _reference_image(source))
                usable[avatar_id] = source
            except Exception as error:  # one bad file must not stop the server
                log(f"Avatar « {avatar_id} » ignoré : {error}")
        if not usable:
            return

        with self.lock:
            get_base_data(self.pipeline, REF_DIR.as_posix(), base_seed=SEED, use_face_crop=False)
            # FlashHead names each portrait after its path; keep the bare file name.
            for name in ("cond_image_dict", "cond_image_tensor_dict", "ref_img_latent_dict"):
                table = getattr(self.pipeline, name)
                setattr(self.pipeline, name, {Path(key).name: value for key, value in table.items()})

        for avatar_id, source in usable.items():
            if not self.is_prepared(avatar_id, source):
                log(f"Préparation de l'avatar « {avatar_id} » (une seule fois)…")
                self._build_idle(avatar_id, source)
            self.avatars[avatar_id] = Avatar(avatar_id, AVATAR_CACHE_DIR / avatar_id)

    def _build_idle(self, avatar_id: str, source: Path) -> None:
        cache = AVATAR_CACHE_DIR / avatar_id
        if cache.exists():
            shutil.rmtree(cache)
        cache.mkdir(parents=True)
        stream = self.stream(avatar_id)
        silence = np.zeros(SLICE_SAMPLES, dtype=np.float32)
        jpegs = [patch.jpeg for _ in range(IDLE_SLICES) for patch in stream.frames(silence)]
        with open(cache / "frames.bin", "wb") as f:
            f.write(struct.pack("<I", len(jpegs)))
            for jpeg in jpegs:
                f.write(struct.pack("<I", len(jpeg)))
                f.write(jpeg)
        (cache / "thumb.jpg").write_bytes(jpegs[0])
        (cache / "meta.json").write_text(
            json.dumps(
                {
                    "version": CACHE_VERSION,
                    "source": _stamp(source),
                    "width": SIZE,
                    "height": SIZE,
                    "frames": len(jpegs),
                }
            )
        )

    # ---- generation -------------------------------------------------------
    def stream(self, avatar_id: str) -> TalkStream:
        return TalkStream(self, avatar_id)


def _stamp(source: Path) -> str:
    stat = source.stat()
    return f"{source.name}:{stat.st_size}:{int(stat.st_mtime)}"


def _reference_image(source: Path) -> np.ndarray:
    """A square SIZE x SIZE portrait cut from an image, or from a video's first frame."""
    if source.suffix.lower() in IMAGE_EXTS:
        # np.fromfile copes with accented file names, which cv2.imread does not on Windows.
        image = cv2.imdecode(np.fromfile(str(source), dtype=np.uint8), cv2.IMREAD_COLOR)
    else:
        capture = cv2.VideoCapture(str(source))
        ok, image = capture.read()
        capture.release()
        if not ok:
            image = None
    if image is None:
        raise RuntimeError(f"Cannot read {source}")
    height, width = image.shape[:2]
    side = min(height, width)
    # Portraits keep their top (the head); wide pictures keep their middle.
    left = (width - side) // 2
    image = image[:side, left : left + side]
    return cv2.resize(image, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
