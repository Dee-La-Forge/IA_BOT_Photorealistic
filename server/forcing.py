"""Streaming wrapper around Avatar Forcing (Ki et al., CVPR 2026).

Avatar Forcing animates a portrait from two audio tracks: the avatar's own voice (it
speaks) and the user's voice (it listens and reacts). The published code only runs
offline on whole files; this module drives the same network block by block, 10 frames
(0.4 s) at a time, for as long as a conversation lasts.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

FORCING_DIR = Path(os.environ.get("FORCING_DIR", "D:/visavis/AvatarForcing")).resolve()
os.environ.setdefault("TORCH_HOME", str(FORCING_DIR.parent / "cache" / "torch"))

FPS = 25
SIZE = 512
AUDIO_RATE = 16000
SAMPLES_PER_FRAME = AUDIO_RATE // FPS
BLOCK_FRAMES = 10  # frames produced per step
BLOCK_SAMPLES = BLOCK_FRAMES * SAMPLES_PER_FRAME
PRIME_FRAMES = 50  # the first step produces 2 s at once
PRIME_SAMPLES = PRIME_FRAMES * SAMPLES_PER_FRAME
CONTEXT = 2  # frames of the previous block that each new block starts from
MAX_FRAMES = 65536  # positions the network can address: about 43 minutes
# Level a voice is brought to before the audio encoder (which expects normalized input).
SPEECH_LEVEL = 0.7

# Combination of the 20 motion components that zooms the whole picture, and the one that
# bobs it up and down, measured on generated video (see README). Removed from the motion:
# they read as camera moves rather than as a person moving.
CAMERA_DIRECTIONS = Path(__file__).with_name("forcing_camera.npy")
# Combination of the components that moves the eyes sideways, fitted on generated video of
# two portraits by measuring where the iris sits in the eye. Left alone, the avatar looks
# past the user while it talks.
GAZE_DIRECTIONS = Path(__file__).with_name("forcing_gaze.npy")
# Share of the sideways gaze brought back onto the camera: while speaking, while silent.
GAZE_HOLD = float(os.environ.get("FORCING_GAZE_HOLD", "0.9"))
GAZE_HOLD_QUIET = float(os.environ.get("FORCING_GAZE_HOLD_QUIET", "0.5"))
# Denoising steps per block: fewer is faster, at some cost in motion quality.
NFE = int(os.environ.get("FORCING_NFE", "7"))
# How strongly the avatar reacts to the user (their voice and face) while listening.
LISTEN_GUIDANCE = float(os.environ.get("FORCING_LISTEN", "2.0"))
# Two ways of removing camera-like motion: on the picture (shoulders held still) or on
# the motion components themselves (which also damps the head).
LOCK_BODY = os.environ.get("FORCING_LOCK_BODY", "1") == "1"
HOLD_CAMERA = os.environ.get("FORCING_HOLD_CAMERA", "0") == "1"
# How much larger than generated the avatar's movements are shown while it listens.
LIVELINESS = float(os.environ.get("FORCING_LIVELINESS", "1.9"))
# Per frame, how fast the filtered pose follows the generated one (1 = no filtering).
SLOW_SHARE = float(os.environ.get("FORCING_SLOW_SHARE", "0.2"))
SOFT_SHARE = float(os.environ.get("FORCING_SOFT_SHARE", "0.4"))
# The motion components that carry the head's position and turn (measured: each moves
# the whole picture strongly and the mouth hardly at all). The others carry expression.
POSE_COMPONENTS = [1, 2, 3, 4, 10, 13, 15, 16]
# Share of the head's distance from the portrait's own pose that each block takes back.
POSE_PULL = float(os.environ.get("FORCING_POSE_PULL", "0.10"))
# Share of the pose's distance from rest that each block takes back (see _steady).
DRIFT_PULL = float(os.environ.get("FORCING_DRIFT_PULL", "0.04"))


class ForcingModel:
    """The network, loaded once."""

    def __init__(self, device: int = 0) -> None:
        sys.path.insert(0, str(FORCING_DIR))
        cwd = os.getcwd()
        os.chdir(FORCING_DIR)  # its config uses relative paths
        try:
            from omegaconf import OmegaConf

            from models.avatarforcing.AvatarForcing import AvatarForcing
            from models.avatarforcing.flow_transformer import precompute_freqs_cis

            opt = OmegaConf.load("configs/inference.yaml")
            opt.rank, opt.ngpus = device, 1
            self.opt = opt
            self.device = torch.device(f"cuda:{device}")
            net = AvatarForcing(opt).to(self.device)
            for path in ("pretrained_dir/motion_autoencoder.pth", "pretrained_dir/flow_transformer.pth"):
                state = torch.load(path, map_location=self.device, weights_only=True)
                with torch.no_grad():
                    for name, param in net.named_parameters():
                        if name in state:
                            param.copy_(state[name])
            net.eval()
            # The published position table stops at 1024 frames (41 s): extend it.
            for module in net.flow_transformer.modules():
                if hasattr(module, "freqs_cis") and hasattr(module, "head_dim"):
                    module.freqs_cis = precompute_freqs_cis(module.head_dim, MAX_FRAMES).to(self.device)
            self.net = net
        finally:
            os.chdir(cwd)

        import face_alignment

        self.detector = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D, flip_input=False
        ).face_detector

        with torch.inference_mode():
            self.directions = net.motion_autoencoder.dec.direction(None).float()  # 512 x 20
        self.gaze = None
        if GAZE_DIRECTIONS.exists():
            basis, _ = np.linalg.qr(np.load(GAZE_DIRECTIONS).T)
            self.gaze = torch.from_numpy(basis).float().to(self.device)  # 20 x k, orthonormal
        self.camera = None
        if CAMERA_DIRECTIONS.exists():
            basis, _ = np.linalg.qr(np.load(CAMERA_DIRECTIONS).T)
            self.camera = torch.from_numpy(basis).float().to(self.device)  # 20 x k, orthonormal

    def face_square(self, image: np.ndarray, pad_ratio: float = 1.0) -> tuple[int, int, int] | None:
        """Left, top and side of the square around the face, as the network was trained on."""
        height, width = image.shape[:2]
        scale = 360.0 / height
        small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        boxes = self.detector.detect_from_image(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        boxes = [b for b in boxes if b[4] > 0.95]
        if not boxes:
            return None
        x1, y1, x2, y2 = (int(v / scale) for v in boxes[0][:4])
        half = int(max(y2 - y1, x2 - x1) / 2 * (1 + pad_ratio))
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        left, top = max(cx - half, 0), max(cy - half, 0)
        right, bottom = min(cx + half, width), min(cy + half, height)
        half = min(right - left, bottom - top) // 2
        cx, cy = left + (right - left) // 2, top + (bottom - top) // 2
        return cx - half, cy - half, 2 * half

    def crop_portrait(self, image: np.ndarray) -> np.ndarray:
        """Cuts the avatar's portrait down to the face square (BGR in, 512 x 512 BGR out)."""
        square = self.face_square(image)
        if square is None:
            raise RuntimeError("No face found in the portrait")
        return crop_square(image, square)

    def stream(self, portrait: np.ndarray, **options) -> "ForcingStream":
        return ForcingStream(self, portrait, **options)


def crop_square(image: np.ndarray, square: tuple[int, int, int]) -> np.ndarray:
    left, top, side = square
    return cv2.resize(image[top : top + side, left : left + side], (SIZE, SIZE), interpolation=cv2.INTER_AREA)


class FaceTracker:
    """Follows the user's face in their webcam picture, so the avatar can react to it."""

    REFRESH = 16  # pictures between two face detections; the square is reused in between

    def __init__(self, model: ForcingModel) -> None:
        self.model = model
        self.square: tuple[int, int, int] | None = None
        self.seen = 0

    def crop(self, image: np.ndarray) -> np.ndarray | None:
        """The user's face as a 512 x 512 picture, or None when no face is in view."""
        if self.square is None or self.seen % self.REFRESH == 0:
            self.square = self.model.face_square(image)
        self.seen += 1
        if self.square is None:
            return None
        left, top, side = self.square
        if side < 32 or top + side > image.shape[0] or left + side > image.shape[1]:
            self.square = None
            return None
        return crop_square(image, self.square)


class BodyLock:
    """Holds the avatar's shoulders still in the picture, leaving the head free to move.

    The network moves a face by warping the whole portrait, so part of every head
    movement comes out as the entire picture sliding, turning or zooming, which looks
    like a camera moving. Here the shoulders are followed from frame to frame and each
    frame is shifted back so that they stay where they were in the first one. What is
    left is the head moving on a still body.
    """

    MAX_SHIFT = 12.0  # pixels
    MAX_SCALE = 0.025  # log scale
    MAX_TURN = 0.03  # radians

    def __init__(self, frame: np.ndarray, zoom: float = 1.0) -> None:
        # `zoom` slightly above 1 hides the edges that shifting the picture uncovers; it is
        # not needed when the frame is set back into a larger picture that covers them.
        self.zoom = zoom
        self.reference = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height = frame.shape[0]
        shoulders = np.zeros_like(self.reference)
        shoulders[int(0.78 * height) :] = 255  # below the chin in a face-centred portrait
        self.points = cv2.goodFeaturesToTrack(self.reference, 150, 0.01, 5, mask=shoulders)
        self.where = None if self.points is None else self.points.copy()
        self.state = np.zeros(4)  # log scale, turn, shift x, shift y (about the centre)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if self.points is None or len(self.points) < 12:
            return frame  # nothing distinctive to follow on this portrait
        size = frame.shape[0]
        centre = size / 2
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, status, _ = cv2.calcOpticalFlowPyrLK(
            self.reference, gray, self.points, self.where.copy(),
            winSize=(25, 25), maxLevel=3, flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
        )
        good = status.ravel() == 1
        move = None
        if good.sum() >= 10:
            move, _ = cv2.estimateAffinePartial2D(
                self.points[good], found[good], method=cv2.RANSAC, ransacReprojThreshold=2.0
            )
        if move is None:
            self.where = self.points.copy()  # lost: search from the resting position again
        else:
            self.where = cv2.transform(self.points, move).astype(np.float32)
            scale = float(np.hypot(move[0, 0], move[1, 0]))
            turn = float(np.arctan2(move[1, 0], move[0, 0]))
            shift = move[:, :2] @ np.array([centre, centre]) + move[:, 2] - centre
            measured = np.array([np.log(scale), turn, shift[0], shift[1]])
            self.state += 0.5 * (measured - self.state)  # tracking noise must not jitter the picture
        limit = np.array([self.MAX_SCALE, self.MAX_TURN, self.MAX_SHIFT, self.MAX_SHIFT])
        log_scale, turn, dx, dy = np.clip(self.state, -limit, limit)
        # Each output pixel is read where the shoulders have carried it, slightly zoomed in.
        k = np.exp(log_scale) / self.zoom
        a, b = k * np.cos(turn), k * np.sin(turn)
        matrix = np.array(
            [[a, -b, centre + dx - (a - b) * centre], [b, a, centre + dy - (a + b) * centre]]
        )
        return cv2.warpAffine(
            frame, matrix, (size, size),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE,
        )


class VoiceTrack:
    """The recent past of one voice, kept at the level the audio encoder expects."""

    def __init__(self) -> None:
        self.audio = np.zeros(PRIME_SAMPLES, dtype=np.float32)
        self.level = 0.05  # running loudness of the speech heard so far
        self.rng = np.random.default_rng(0)

    def push(self, audio: np.ndarray) -> None:
        audio = np.asarray(audio, dtype=np.float32)
        rms = float(np.sqrt(np.mean(audio**2))) if len(audio) else 0.0
        if rms > 0.01:  # speech, not background: follow its loudness slowly
            self.level += 0.1 * (rms - self.level)
        self.audio = np.concatenate([self.audio, audio])[-PRIME_SAMPLES:]

    def window(self) -> np.ndarray:
        """The last 2 s, scaled so that speech sits at SPEECH_LEVEL, over a faint noise floor.

        Pure digital silence is something the audio encoder never saw in training.
        """
        scaled = self.audio * (SPEECH_LEVEL / max(self.level, 0.02))
        return scaled + self.rng.standard_normal(len(scaled)).astype(np.float32) * 0.003


class ForcingStream:
    """One continuous stretch of video for a portrait."""

    def __init__(
        self,
        model: ForcingModel,
        portrait: np.ndarray,
        nfe: int = NFE,
        voice_guidance: float = 2.0,
        listen_guidance: float = LISTEN_GUIDANCE,
        seed: int = 25,
        drift_pull: float = DRIFT_PULL,
        gaze_hold: float = GAZE_HOLD,
        gaze_hold_quiet: float = GAZE_HOLD_QUIET,
        hold_camera: bool = HOLD_CAMERA,
        lock_body: bool = LOCK_BODY,
        liveliness: float = LIVELINESS,
    ) -> None:
        self.model = model
        self.portrait = portrait
        self.net = net = model.net
        self.voice_guidance, self.listen_guidance = voice_guidance, listen_guidance
        self.avatar, self.user = VoiceTrack(), VoiceTrack()
        self.position = 0  # frames generated so far
        self.tail = None  # latents of the last block
        self.anchor = None  # resting value of the camera components
        self.last_face = None  # the user's face motion at the end of the previous block
        self.drift_pull, self.hold_camera = drift_pull, hold_camera
        self.gaze_hold, self.gaze_hold_quiet = gaze_hold, gaze_hold_quiet
        self.eyes = gaze_hold_quiet  # current strength of the gaze hold
        self.lock_body, self.body = lock_body, None
        self.liveliness = liveliness
        self.slow = None  # the pose with its frame-to-frame flicker filtered out
        self.soft = None  # the same, filtered more lightly
        self.centre = None  # where the pose has been hovering over the last few seconds
        self.quiet = 0.0  # 1 while the avatar is silent, 0 while it speaks
        device = model.device
        torch.manual_seed(seed)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            rgb = cv2.cvtColor(portrait, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
            reference = torch.from_numpy(rgb).permute(2, 0, 1)[None].to(device)
            self.identity, weights, features = net.encode_image_into_latent(reference)
            self.own = weights.float()  # the portrait's own pose: head straight, eyes on the camera
            self.rest = net.motion_autoencoder.dec.direction(weights)
            self.features = [f.repeat_interleave(BLOCK_FRAMES, dim=0) for f in features]
            net.denoising_step_list = torch.tensor(
                np.linspace(net.opt.num_train_timestep, 0, nfe - 1).tolist()
            )
            net.initialize_kv_cache(batch_size=1, dtype=torch.float32, device=device)
        self.cache = net.kv_cache
        self.steps = net.denoising_step_list

    # ---- audio ------------------------------------------------------------
    def _features(self, track: VoiceTrack, frames: int) -> torch.Tensor:
        """Audio features of the last `frames` frames, encoded with 2 s of context."""
        audio = torch.from_numpy(track.window())[None].to(self.model.device)
        return self.net.audio_encoder.inference(audio, seq_len=PRIME_FRAMES)[:, -frames:]

    def _watched(self, faces: list[np.ndarray] | None, frames: int) -> torch.Tensor:
        """The user's face motion over the block, from however many pictures arrived."""
        net, device = self.net, self.model.device
        if not faces:
            # No camera on the user: their motion is left unset, only their voice is heard.
            self.last_face = None
            return torch.zeros(1, frames, self.rest.shape[-1], device=device)
        rgb = np.stack([cv2.cvtColor(face, cv2.COLOR_BGR2RGB) for face in faces]).astype(np.float32)
        pictures = torch.from_numpy(rgb / 127.5 - 1.0).permute(0, 3, 1, 2).to(device)
        motion = net.motion_autoencoder.enc.enc_motion(pictures)
        motion = net.motion_autoencoder.dec.direction(motion).float()  # K x 512
        path = motion if self.last_face is None else torch.cat([self.last_face, motion])
        self.last_face = motion[-1:]
        if len(path) == 1:
            return path[None].repeat(1, frames, 1)
        # The webcam sends fewer pictures than the video has frames: glide between them.
        return F.interpolate(path.T[None], size=frames, mode="linear", align_corners=True)[0].T[None]

    # ---- generation -------------------------------------------------------
    @torch.inference_mode()
    def step(
        self, avatar_audio: np.ndarray, user_audio: np.ndarray, user_faces: list[np.ndarray] | None = None
    ) -> np.ndarray:
        """Advances the video by one step and returns its frames (uint8 BGR, N x 512 x 512 x 3).

        The first call takes 2 s of each voice and returns 50 frames; every later call
        takes 0.4 s of each and returns 10 frames. `user_faces` are the pictures of the
        user's face seen since the previous step (any number, FaceTracker crops).
        """
        net, first = self.net, self.position == 0
        frames = PRIME_FRAMES if first else BLOCK_FRAMES
        context = 0 if first else CONTEXT
        if self.position + frames >= MAX_FRAMES:
            raise RuntimeError("Stream too long; start a new one")
        self.avatar.push(avatar_audio)
        self.user.push(user_audio)
        net.kv_cache, net.denoising_step_list = self.cache, self.steps  # this stream's state

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            voice = self._features(self.avatar, frames + context)
            heard = self._features(self.user, frames + context)
            watched = self._watched(user_faces, frames + context)
            noise = torch.randn(1, frames, net.opt.dim_w, device=self.model.device)
            x = noise if first else torch.cat([self.tail[:, -CONTEXT:], noise], dim=1)
            start = 0 if first else self.position - CONTEXT
            condition, rest, timing = net.prepare_cfg_condition(
                voice, heard, watched, self.rest, seq_len=frames, context_len=context
            )
            for index, timestep in enumerate(self.steps):
                x = net.solve_cfg(
                    B=1, index=index, current_timestep=timestep, x_t=x,
                    precomputed_c=condition, precomputed_wr=rest, precomputed_adaLN=timing,
                    start_pos=start, context_len=context, use_kv_cache=not first,
                    a_cfg_scale=self.voice_guidance, u_cfg_scale=self.listen_guidance,
                )
            # Steady the new frames before they become the history the next block continues.
            speaking = float(np.sqrt(np.mean(np.square(avatar_audio)))) > 0.01
            motion = self._steady(x[:, -frames:].float(), speaking)
            x = torch.cat([x[:, :-frames], motion.to(x.dtype)], dim=1)
            net.update_kv_cache(
                final_latents=x, precomputed_c=condition, precomputed_wr=rest,
                precomputed_adaLN=timing, start_pos=start,
            )
            self.tail = x
            self.position += frames
            shown = self._enliven(motion, speaking)
            images = []
            for i in range(0, frames, BLOCK_FRAMES):
                latents = (self.identity.unsqueeze(1) + shown[:, i : i + BLOCK_FRAMES]).reshape(BLOCK_FRAMES, -1)
                image, _ = net.motion_autoencoder.dec(latents, alpha=None, feats=self.features)
                images.append(image)
        image = torch.cat(images).float().clamp(-1, 1)
        rgb = ((image + 1) * 127.5).permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()
        return rgb[..., ::-1]

    def steady(self, frames: np.ndarray) -> list[np.ndarray]:
        """Holds the shoulders still across the frames of a step (CPU work, after `step`)."""
        if not self.lock_body:
            return [np.ascontiguousarray(frame) for frame in frames]
        if self.body is None:
            # Locked onto the portrait itself, so the frames line up with the full picture.
            self.body = BodyLock(self.portrait)
        return [self.body.apply(np.ascontiguousarray(frame)) for frame in frames]

    def _enliven(self, motion: torch.Tensor, speaking: bool) -> torch.Tensor:
        """Shapes the movement shown while the avatar is not the one speaking.

        The network is lively when it talks and subdued when it listens, and its motion
        flickers a little from frame to frame, which shows as a trembling picture on a
        face that is otherwise still. So, while the avatar is silent, the flicker is
        filtered out and the slow part of its movement (head, gaze, expression) is shown
        larger than generated. While it speaks, the motion is shown as generated, so the
        lips stay sharp. Only the displayed motion is changed, never the history the
        network continues from; every change of regime is spread over frames.
        """
        directions = self.model.directions
        away = motion[0] @ directions - self.anchor
        slow, soft, centre = [], [], []
        for row in away:
            self.slow = row if self.slow is None else self.slow + SLOW_SHARE * (row - self.slow)
            self.soft = row if self.soft is None else self.soft + SOFT_SHARE * (row - self.soft)
            self.centre = row if self.centre is None else self.centre + 0.01 * (row - self.centre)
            slow.append(self.slow)
            soft.append(self.soft)
            centre.append(self.centre)
        slow, soft, centre = torch.stack(slow), torch.stack(soft), torch.stack(centre)
        before = self.quiet
        # Back to the raw motion within one block when speech starts: lips must not lag.
        self.quiet = before + float(np.clip((0.0 if speaking else 1.0) - before, -1.0, 0.3))
        if before == 0.0 and self.quiet == 0.0:
            return motion
        quiet = torch.linspace(before, self.quiet, len(away), device=away.device)[:, None]
        # Movements are enlarged around where the pose is hovering, not around the resting
        # pose: enlarging a standing offset would just hold the head askew.
        shown = (1 - quiet) * away + quiet * (soft + (self.liveliness - 1) * (slow - centre))
        return motion + ((shown - away) @ directions.T)[None]

    def _steady(self, motion: torch.Tensor, speaking: bool) -> torch.Tensor:
        """Keeps the avatar facing the user over a long conversation.

        Left alone, the network's pose drifts: within seconds the head has turned aside,
        the gaze has left the camera, and after half a minute the face has grown in the
        picture. With every block the pose is pulled back a little: the head toward the
        portrait's own pose (straight, eyes on the camera), the expression toward where
        it rested at the start of the call (the portrait's own expression may be a smile
        or an open mouth, which is no resting face).

        The gaze gets a firmer hand: its sideways part is brought back to the portrait's
        own (eyes on the camera) in every frame, almost fully while the avatar speaks, so
        that it addresses the user, and half way while it listens, so that it can still
        glance away.
        """
        directions = self.model.directions
        weights = motion[0].float() @ directions  # T x 20
        first = self.anchor is None
        if first:
            self.anchor = weights.float().mean(dim=0, keepdim=True)
            self.anchor[:, POSE_COMPONENTS] = self.own[:, POSE_COMPONENTS]
            self.pull = torch.full_like(self.anchor, self.drift_pull)
            self.pull[:, POSE_COMPONENTS] = POSE_PULL
        away = weights - self.anchor
        camera = self.model.camera
        hold = camera is not None and self.hold_camera
        held = (away @ camera) @ camera.T if hold else torch.zeros_like(away)
        # Ramped over the block, so that the correction itself never shows as a jump.
        ramp = torch.linspace(1 / len(away), 1, len(away), device=away.device)[:, None]
        pull = self.pull * (0.0 if first else 1.0)
        correction = held + pull * ramp * (away - held).mean(dim=0, keepdim=True)
        gaze = self.model.gaze
        if gaze is not None:
            before = self.eyes
            target = self.gaze_hold if speaking else self.gaze_hold_quiet
            self.eyes = before + float(np.clip(target - before, -0.2, 0.2))
            hold = torch.linspace(before, self.eyes, len(away), device=away.device)[:, None]
            aside = ((weights - self.own) @ gaze) @ gaze.T  # how far the eyes are off the camera
            correction = correction + hold * aside
        return motion - (correction @ directions.T)[None]
