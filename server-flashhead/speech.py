"""Local speech: voice activity detection, speech-to-text and text-to-speech."""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path

import numpy as np
import torch
import torchaudio

DATA_DIR = Path(os.environ.get("VISAVIS_DATA", "D:/visavis/data")).resolve()
VOICE_DIR = DATA_DIR / "voices"
STT_DIR = DATA_DIR / "models" / "whisper-stt"

MIC_RATE = 16000
VAD_WINDOW = 512  # samples (32 ms), the size Silero VAD expects at 16 kHz

# Piper voices offered per language: (label, file stem).
VOICES: dict[str, list[tuple[str, str]]] = {
    "fr": [("Siwis (femme)", "fr_FR-siwis-medium"), ("Tom (homme)", "fr_FR-tom-medium")],
    "en": [("Lessac (woman)", "en_US-lessac-medium"), ("Ryan (man)", "en_US-ryan-medium")],
}

# Phrases Whisper tends to invent when it is given silence or noise.
HALLUCINATIONS = re.compile(
    r"sous-titres|amara\.org|merci d'avoir regardé|thanks? for watching|subtitles by|"
    r"abonnez-vous|subscribe",
    re.IGNORECASE,
)


class VoiceActivity:
    """Silero VAD: probability that a 32 ms window contains speech."""

    def __init__(self) -> None:
        from silero_vad import load_silero_vad

        self.model = load_silero_vad()

    def reset(self) -> None:
        self.model.reset_states()

    def probability(self, window: np.ndarray) -> float:
        with torch.no_grad():
            return float(self.model(torch.from_numpy(window), MIC_RATE).item())


class Transcriber:
    """Whisper speech-to-text running on the local GPU."""

    def __init__(self) -> None:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.processor = WhisperProcessor.from_pretrained(STT_DIR)
        self.model = (
            WhisperForConditionalGeneration.from_pretrained(STT_DIR, torch_dtype=self.dtype)
            .to(self.device)
            .eval()
        )
        self.lock = threading.Lock()

    @torch.no_grad()
    def transcribe(self, audio: np.ndarray, lang: str) -> str:
        features = self.processor(
            audio, sampling_rate=MIC_RATE, return_tensors="pt"
        ).input_features.to(self.device, dtype=self.dtype)
        with self.lock:
            tokens = self.model.generate(
                features, language=lang, task="transcribe", max_new_tokens=200
            )
        text = self.processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()
        if not re.search(r"\w", text) or HALLUCINATIONS.search(text):
            return ""
        return text


class Speaker:
    """Piper text-to-speech (runs on the CPU)."""

    def __init__(self) -> None:
        self.voices: dict[str, object] = {}
        self.lock = threading.Lock()

    def available_id(self, voice_id: str) -> bool:
        return (VOICE_DIR / f"{voice_id}.onnx").exists()

    def available(self, lang: str) -> list[dict[str, str]]:
        return [
            {"id": stem, "name": label}
            for label, stem in VOICES.get(lang, [])
            if (VOICE_DIR / f"{stem}.onnx").exists()
        ]

    def _voice(self, voice_id: str):
        if voice_id not in self.voices:
            from piper import PiperVoice

            self.voices[voice_id] = PiperVoice.load(str(VOICE_DIR / f"{voice_id}.onnx"))
        return self.voices[voice_id]

    def synthesize(self, text: str, voice_id: str) -> tuple[np.ndarray, int]:
        """Returns mono float32 audio and its sample rate."""
        with self.lock:
            voice = self._voice(voice_id)
            parts = [chunk.audio_float_array for chunk in voice.synthesize(text)]
            rate = voice.config.sample_rate
        if not parts:
            return np.zeros(0, dtype=np.float32), rate
        return np.concatenate(parts).astype(np.float32), rate


def resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate or len(audio) == 0:
        return audio
    tensor = torch.from_numpy(audio).unsqueeze(0)
    return torchaudio.functional.resample(tensor, source_rate, target_rate).squeeze(0).numpy()
