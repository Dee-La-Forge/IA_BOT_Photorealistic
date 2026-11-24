"""Avatar engine: portraits on disk, and the Avatar Forcing network that animates them.

An avatar is one portrait. During a call the network generates its video continuously,
10 frames at a time: it speaks with the voice it is given and, in between, listens to
the user's voice and face (see forcing.py).
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import threading
from pathlib import Path

import cv2
import numpy as np

from forcing import (  # noqa: F401  (re-exported for main.py)
    AUDIO_RATE,
    BLOCK_FRAMES,
    FPS,
    PRIME_FRAMES,
    SAMPLES_PER_FRAME,
    SIZE,
    FaceTracker,
    ForcingModel,
    crop_square,
    ForcingStream,
)

DATA_DIR = Path(os.environ.get("VISAVIS_DATA", "D:/visavis/data")).resolve()
AVATAR_SRC_DIR = DATA_DIR / "avatars"
AVATAR_CACHE_DIR = DATA_DIR / "avatar-cache"

JPEG_QUALITY = 90
CACHE_VERSION = 21
MAX_HEIGHT = 720  # the full picture is shown at most this tall
FEATHER = 0.08  # share of the face square over which a frame blends into the picture

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def encode_jpeg(frame: np.ndarray) -> bytes:
    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return jpeg.tobytes()


class Avatar:
    def __init__(self, avatar_id: str, cache: Path):
        self.id = avatar_id
        meta = json.loads((cache / "meta.json").read_text())
        self.width, self.height = meta["width"], meta["height"]
        self.count = 1  # the browser shows the picture until the live video starts
        self.portrait = cv2.imread(str(cache / "portrait.png"))  # the face square, as the network sees it
        # Where that square sits in the full picture, and what surrounds it there.
        self.x, self.y, self.side = meta["square"]
        full = cv2.imread(str(cache / "full.jpg"))
        under = full[self.y : self.y + self.side, self.x : self.x + self.side].astype(np.float32)
        mask = _feather(self.side, self.x, self.y, self.width, self.height)
        # Only the strips along the edges are ever blended: kept apart so that the rest
        # of each frame is copied untouched (blending whole frames is too slow).
        band = max(2, int(self.side * FEATHER))
        middle = slice(band, self.side - band)
        self.strips = [
            (region, under[region], mask[region])
            for region in (
                (slice(0, band), slice(None)),
                (slice(self.side - band, self.side), slice(None)),
                (middle, slice(0, band)),
                (middle, slice(self.side - band, self.side)),
            )
            if float(mask[region].min()) < 1.0
        ]

    def set_back(self, frame: np.ndarray) -> np.ndarray:
        """A generated frame, resized to its place in the full picture and blended at the edges."""
        frame = cv2.resize(frame, (self.side, self.side), interpolation=cv2.INTER_LINEAR)
        for region, under, mask in self.strips:
            frame[region] = (under + (frame[region].astype(np.float32) - under) * mask).astype(np.uint8)
        return frame


class Engine:
    def __init__(self) -> None:
        self.lock = threading.Lock()  # one generation step on the GPU at a time
        self.avatars: dict[str, Avatar] = {}
        self.model: ForcingModel | None = None

    def load(self) -> None:
        if self.model is None:
            self.model = ForcingModel()

    # ---- avatars ----------------------------------------------------------
    def list_sources(self) -> dict[str, Path]:
        AVATAR_SRC_DIR.mkdir(parents=True, exist_ok=True)
        return {
            p.stem: p
            for p in sorted(AVATAR_SRC_DIR.iterdir())
            if p.suffix.lower() in VIDEO_EXTS | IMAGE_EXTS
        }

    def is_prepared(self, avatar_id: str, source: Path) -> bool:
        try:
            data = json.loads((AVATAR_CACHE_DIR / avatar_id / "meta.json").read_text())
        except (OSError, json.JSONDecodeError):
            return False
        return data.get("version") == CACHE_VERSION and data.get("source") == _stamp(source)

    def prepare_all(self, log=print) -> None:
        """Crops every portrait in the avatars folder to the face square the network expects."""
        self.load()
        for avatar_id, source in self.list_sources().items():
            try:
                if not self.is_prepared(avatar_id, source):
                    log(f"Préparation de l'avatar « {avatar_id} »…")
                    self._prepare(avatar_id, source)
                self.avatars[avatar_id] = Avatar(avatar_id, AVATAR_CACHE_DIR / avatar_id)
            except Exception as error:  # one bad file must not stop the server
                log(f"Avatar « {avatar_id} » ignoré : {error}")

    def _prepare(self, avatar_id: str, source: Path) -> None:
        cache = AVATAR_CACHE_DIR / avatar_id
        if cache.exists():
            shutil.rmtree(cache)
        cache.mkdir(parents=True)
        image = _read_image(source)
        square = self.model.face_square(image)
        if square is None:
            raise RuntimeError("No face found in the portrait")
        cv2.imwrite(str(cache / "portrait.png"), crop_square(image, square))
        # The whole picture, at display size; the face square is animated inside it.
        scale = min(1.0, MAX_HEIGHT / image.shape[0])
        width = int(image.shape[1] * scale) // 2 * 2
        height = int(image.shape[0] * scale) // 2 * 2
        full = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        left, top, side = (int(round(v * scale)) for v in square)
        side = min(side, width - left, height - top)
        jpeg = encode_jpeg(full)
        (cache / "full.jpg").write_bytes(jpeg)
        (cache / "thumb.jpg").write_bytes(jpeg)
        (cache / "frames.bin").write_bytes(struct.pack("<II", 1, len(jpeg)) + jpeg)
        (cache / "meta.json").write_text(
            json.dumps(
                {
                    "version": CACHE_VERSION,
                    "source": _stamp(source),
                    "width": width,
                    "height": height,
                    "square": [left, top, side],
                }
            )
        )

    # ---- generation -------------------------------------------------------
    def stream(self, avatar: Avatar) -> ForcingStream:
        with self.lock:
            return self.model.stream(avatar.portrait)

    def step(
        self,
        avatar: Avatar,
        stream: ForcingStream,
        voice: np.ndarray,
        heard: np.ndarray,
        faces: list[np.ndarray] | None,
    ) -> list[bytes]:
        """One generation step, returned as JPEG frames."""
        import time

        t0 = time.perf_counter()
        with self.lock:
            t1 = time.perf_counter()
            frames = stream.step(voice, heard, faces)
        t2 = time.perf_counter()
        steady = stream.steady(frames)
        t3 = time.perf_counter()
        jpegs = [encode_jpeg(avatar.set_back(frame)) for frame in steady]
        if os.environ.get("VISAVIS_TIMINGS") == "1":
            print(f"[etapes] attente {1e3*(t1-t0):.0f} generation {1e3*(t2-t1):.0f} buste {1e3*(t3-t2):.0f} jpeg {1e3*(time.perf_counter()-t3):.0f} ms", flush=True)
        return jpegs

    def tracker(self) -> FaceTracker:
        return FaceTracker(self.model)


def _feather(side: int, left: int, top: int, width: int, height: int) -> np.ndarray:
    """Blend weights for a face square set into a picture: 1 inside, fading out toward each
    edge that lies inside the picture (an edge on the picture's own border needs no fade)."""
    band = max(2, int(side * FEATHER))
    ramp = np.ones(side, dtype=np.float32)
    fade = (1 - np.cos(np.linspace(0, np.pi, band, dtype=np.float32))) / 2
    near = max(2, int(side * 0.02))  # closer than this to the border counts as on it
    across, down = ramp.copy(), ramp.copy()
    if left > near:
        across[:band] = fade
    if width - (left + side) > near:
        across[-band:] = fade[::-1]
    if top > near:
        down[:band] = fade
    if height - (top + side) > near:
        down[-band:] = fade[::-1]
    return (down[:, None] * across[None, :])[..., None]


def _stamp(source: Path) -> str:
    stat = source.stat()
    return f"{source.name}:{stat.st_size}:{int(stat.st_mtime)}"


def _read_image(source: Path) -> np.ndarray:
    """The portrait: an image file, or the first frame of a video."""
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
    return image
