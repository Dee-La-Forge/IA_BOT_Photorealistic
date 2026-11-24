"""Vis-à-vis local server: microphone in, talking avatar out, nothing leaves the machine.

    mic PCM ──► Silero VAD ──► Whisper ──► Ollama ──► Piper ──► MuseTalk ──► browser
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import struct
import time
from collections import deque
from contextlib import asynccontextmanager

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from engine import AVATAR_CACHE_DIR, AVATAR_SRC_DIR, FPS, SLICE_FRAMES, SLICE_SAMPLES, Engine
from speech import MIC_RATE, VAD_WINDOW, VOICES, Speaker, Transcriber, VoiceActivity, resample

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:latest")
OLLAMA_CONTEXT = int(os.environ.get("OLLAMA_CONTEXT", "8192"))
PLAY_RATE = 24000  # sample rate of the voice sent to the browser
TIMINGS = os.environ.get("VISAVIS_TIMINGS") == "1"  # print where each reply spends its time

engine = Engine()
speaker = Speaker()
transcriber: Transcriber
vad: VoiceActivity

SENTENCE_END = re.compile(r"[.!?…]+[\"»)]?(?=\s)|\n+")
NOT_SPOKEN = re.compile(r"[*_#`~<>|\[\]{}]|[\U0001F000-\U0001FAFF☀-➿️]")

GREETINGS = {
    "fr": "Bonjour ! Ça me fait plaisir de te voir. De quoi as-tu envie de parler ?",
    "en": "Hi! It's really nice to see you. What would you like to talk about?",
}


def system_prompt(lang: str) -> str:
    language = "French" if lang == "fr" else "English"
    return "\n".join(
        [
            "You are a warm, curious and attentive companion in a live face-to-face video call.",
            "Everything you write is turned into speech and spoken aloud by a video avatar, so "
            "write only the words to be said: plain spoken sentences, with no markdown, lists, "
            "emojis or stage directions.",
            "Talk the way people do in a real conversation: usually one to three short sentences, "
            "then let the other person speak. Go longer only when they ask for an explanation "
            "or a story.",
            "The user's words reach you through speech recognition, so they can contain "
            "transcription mistakes. Work out what was most likely meant, and ask when it is "
            "really unclear.",
            f"Speak {language}. You have already greeted the user at the start of the call, "
            "so do not greet them again.",
            *(["Address the user informally, with « tu », never « vous »."] if lang == "fr" else []),
        ]
    )


async def preload_llm() -> None:
    """Asks Ollama to load the model now, so the first reply does not wait for it."""
    body = {
        "model": OLLAMA_MODEL,
        "messages": [],
        "keep_alive": "10m",
        "options": {"num_ctx": OLLAMA_CONTEXT},
    }
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            await client.post(f"{OLLAMA_URL}/api/chat", json=body)
    except Exception as error:  # reported to the user when a reply is actually needed
        print(f"Ollama injoignable pour le préchargement : {error}", flush=True)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global transcriber, vad
    print("Chargement des modèles…", flush=True)
    engine.load()
    transcriber = Transcriber()
    vad = VoiceActivity()
    engine.prepare_all(log=lambda line: print(line, flush=True))
    if not engine.avatars:
        print(f"Aucun portrait d'avatar dans {AVATAR_SRC_DIR}", flush=True)
    # The first generation is slow (GPU warm-up): do it now rather than in the first reply.
    for avatar in list(engine.avatars.values())[:1]:
        engine.stream(avatar.id).frames(np.zeros(SLICE_SAMPLES, dtype=np.float32))
    transcriber.transcribe(np.zeros(MIC_RATE, dtype=np.float32), "fr")
    for voices in VOICES.values():
        for _, voice_id in voices:
            if speaker.available_id(voice_id):
                speaker.synthesize("Bonjour.", voice_id)
    await preload_llm()
    print("Serveur prêt.", flush=True)
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_methods=["GET"],
)


@app.get("/api/config")
def config():
    return {
        "fps": FPS,
        "model": OLLAMA_MODEL,
        "avatars": [
            {"id": a.id, "width": a.width, "height": a.height, "frames": a.count}
            for a in engine.avatars.values()
        ],
        "voices": {lang: speaker.available(lang) for lang in VOICES},
    }


def voice_for(avatar_id: str, lang: str, installed: list[str]) -> str:
    """The voice an avatar speaks with: set per avatar in avatars/voices.json, else the first."""
    try:
        chosen = json.loads((AVATAR_SRC_DIR / "voices.json").read_text(encoding="utf-8"))
        voice = chosen.get(avatar_id, {}).get(lang)
    except (OSError, ValueError, AttributeError):
        voice = None
    return voice if voice in installed else installed[0]


def avatar_file(avatar_id: str, name: str, media_type: str) -> FileResponse:
    if avatar_id not in engine.avatars:
        raise HTTPException(404)
    return FileResponse(AVATAR_CACHE_DIR / avatar_id / name, media_type=media_type)


@app.get("/api/avatars/{avatar_id}/frames")
def avatar_frames(avatar_id: str):
    return avatar_file(avatar_id, "frames.bin", "application/octet-stream")


@app.get("/api/avatars/{avatar_id}/thumb")
def avatar_thumb(avatar_id: str):
    return avatar_file(avatar_id, "thumb.jpg", "image/jpeg")


class Session:
    """One browser tab in a call."""

    def __init__(self, ws: WebSocket) -> None:
        self.ws = ws
        self.send_lock = asyncio.Lock()
        self.lang = "fr"
        self.avatar = None
        self.history: list[dict[str, str]] = []
        self.turn = 0
        self.task: asyncio.Task | None = None
        self.muted = False
        self.barge_in = True
        # Frame clock: the browser reports which frame it is showing.
        self.tick_index = 0.0
        self.tick_time = time.monotonic()
        self.next_index = 0
        self.lead = 22  # frames of head start given to generation
        # Speech being produced: the video stream and the voice not yet turned into frames.
        self.stream = None
        self.voice = ""
        self.voice_buffer = np.zeros(0, dtype=np.float32)
        # Voice activity state
        self.pending = np.zeros(0, dtype=np.int16)
        self.preroll: deque[np.ndarray] = deque(maxlen=10)
        self.speech: list[np.ndarray] = []
        self.in_speech = False
        self.run = 0
        self.voiced = 0
        self.silence = 0
        self.barged = False

    # ---- transport --------------------------------------------------------
    async def send(self, message: dict) -> None:
        async with self.send_lock:
            await self.ws.send_text(json.dumps(message, ensure_ascii=False))

    async def send_chunk(self, header: dict, payload: bytes) -> None:
        head = json.dumps(header).encode()
        async with self.send_lock:
            await self.ws.send_bytes(struct.pack("<I", len(head)) + head + payload)

    def client_index(self) -> float:
        return self.tick_index + (time.monotonic() - self.tick_time) * FPS

    def avatar_busy(self) -> bool:
        running = self.task is not None and not self.task.done()
        return running or self.client_index() < self.next_index

    # ---- microphone -------------------------------------------------------
    def reset_listening(self) -> None:
        self.preroll.clear()
        self.speech = []
        self.in_speech = False
        self.run = self.voiced = self.silence = 0
        self.barged = False
        vad.reset()

    async def on_audio(self, data: bytes) -> None:
        if self.muted or self.avatar is None:
            return
        if not self.barge_in and self.avatar_busy():
            # Half-duplex: ignore the microphone while the avatar has the floor.
            if self.in_speech or self.run:
                self.reset_listening()
            self.pending = np.zeros(0, dtype=np.int16)
            return

        self.pending = np.concatenate([self.pending, np.frombuffer(data, dtype=np.int16)])
        while len(self.pending) >= VAD_WINDOW:
            window = self.pending[:VAD_WINDOW].astype(np.float32) / 32768.0
            self.pending = self.pending[VAD_WINDOW:]
            probability = vad.probability(window)

            if not self.in_speech:
                self.preroll.append(window)
                self.run = self.run + 1 if probability > 0.5 else 0
                if self.run >= 3:
                    self.in_speech = True
                    self.speech = list(self.preroll)
                    self.voiced, self.silence, self.barged = self.run, 0, False
                    await self.send({"type": "listening", "active": True})
                continue

            self.speech.append(window)
            if probability > 0.5:
                self.voiced += 1
                self.silence = 0
            elif probability < 0.35:
                self.silence += 1

            # About 400 ms of speech over the avatar counts as cutting in.
            if not self.barged and self.voiced >= 12 and self.avatar_busy():
                self.barged = True
                await self.interrupt()

            too_long = len(self.speech) * VAD_WINDOW > 30 * MIC_RATE
            if self.silence >= 22 or too_long:  # ~700 ms of silence ends the turn
                audio = np.concatenate(self.speech)
                voiced = self.voiced
                self.reset_listening()
                await self.send({"type": "listening", "active": False})
                if voiced >= 8:
                    asyncio.create_task(self.on_utterance(audio))

    async def on_utterance(self, audio: np.ndarray) -> None:
        text = await asyncio.to_thread(transcriber.transcribe, audio, self.lang)
        if text:
            await self.start_turn(text)

    # ---- turns ------------------------------------------------------------
    async def interrupt(self) -> None:
        self.turn += 1
        task, self.task = self.task, None
        if task and not task.done():
            task.cancel()
        self.next_index = 0
        await self.send({"type": "interrupt"})

    async def start_turn(self, text: str) -> None:
        if self.avatar is None:
            return
        if self.avatar_busy():
            await self.interrupt()
        self.turn += 1
        self.task = asyncio.create_task(self.run_turn(self.turn, text))

    async def greet(self) -> None:
        self.turn += 1
        # The model may have been unloaded since startup: reload it while the greeting plays.
        self.preload = asyncio.create_task(preload_llm())
        self.task = asyncio.create_task(self.run_greeting(self.turn))

    async def run_greeting(self, turn: int) -> None:
        text = GREETINGS[self.lang]
        self.begin_speech()
        try:
            await self.send({"type": "assistant", "text": text, "turn": turn})
            await self.speak(turn, text)
            await self.emit(turn, flush=True)
        except Exception as error:
            if not isinstance(error, asyncio.CancelledError):
                await self.report(error)
            raise
        finally:
            await self.send({"type": "turn_end", "turn": turn})

    async def run_turn(self, turn: int, text: str) -> None:
        await self.send({"type": "user", "text": text})
        self.history.append({"role": "user", "content": text})
        await self.send({"type": "thinking", "turn": turn})

        sentences: asyncio.Queue[str | Exception | None] = asyncio.Queue()
        producer = asyncio.create_task(self.produce(sentences))
        spoken: list[str] = []
        began = time.monotonic()
        self.begin_speech()
        try:
            while True:
                item = await sentences.get()
                if not spoken and TIMINGS:
                    print(f"[délais] première phrase du modèle : {time.monotonic() - began:.2f} s", flush=True)
                if item is None:
                    await self.emit(turn, flush=True)
                    break
                if isinstance(item, Exception):
                    raise item
                spoken.append(item)
                await self.send({"type": "assistant", "text": item, "turn": turn})
                await self.speak(turn, item)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            await self.report(error)
        finally:
            producer.cancel()
            if spoken:
                self.history.append({"role": "assistant", "content": " ".join(spoken)})
            try:
                await self.send({"type": "turn_end", "turn": turn})
            except Exception:
                pass

    async def report(self, error: Exception) -> None:
        print(f"Erreur pendant le tour : {error!r}", flush=True)
        code = "ollama" if isinstance(error, (httpx.HTTPError, OllamaError)) else "server"
        try:
            await self.send({"type": "error", "code": code, "detail": str(error)[:300]})
        except Exception:
            pass

    async def produce(self, out: asyncio.Queue) -> None:
        """Streams the language model's reply into `out`, one sentence at a time."""
        try:
            body = {
                "model": OLLAMA_MODEL,
                "stream": True,
                # No silent reasoning phase: in a spoken exchange it only adds seconds of delay.
                "think": False,
                "keep_alive": "10m",
                # A small context keeps the model's memory use modest; the default can be huge.
                "options": {"num_ctx": OLLAMA_CONTEXT},
                "messages": [
                    {"role": "system", "content": system_prompt(self.lang)},
                    *self.history[-40:],
                ],
            }
            buffer = ""
            async with httpx.AsyncClient(timeout=httpx.Timeout(None, connect=5)) as client:
                async with client.stream("POST", f"{OLLAMA_URL}/api/chat", json=body) as response:
                    if response.status_code != 200:
                        detail = (await response.aread()).decode(errors="replace")
                        raise OllamaError(f"{response.status_code}: {detail[:200]}")
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        data = json.loads(line)
                        buffer += data.get("message", {}).get("content", "")
                        while match := SENTENCE_END.search(buffer):
                            sentence = buffer[: match.end()].strip()
                            buffer = buffer[match.end() :]
                            if sentence:
                                await out.put(sentence)
                        if data.get("done"):
                            break
            if buffer.strip():
                await out.put(buffer.strip())
            await out.put(None)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            await out.put(error)

    def begin_speech(self) -> None:
        """Starts a fresh stretch of video for a new turn."""
        self.stream = engine.stream(self.avatar.id)
        self.voice_buffer = np.zeros(0, dtype=np.float32)

    async def speak(self, turn: int, text: str) -> None:
        """Synthesizes one sentence and streams the frames it fills."""
        spoken = NOT_SPOKEN.sub("", text).strip()
        if not re.search(r"\w", spoken):
            return
        began = time.monotonic()
        audio, rate = await asyncio.to_thread(speaker.synthesize, spoken, self.voice)
        if TIMINGS:
            print(f"[délais] voix {time.monotonic() - began:.2f} s", flush=True)
        if turn != self.turn or len(audio) == 0:
            return
        self.voice_buffer = np.concatenate([self.voice_buffer, resample(audio, rate, PLAY_RATE)])
        await self.emit(turn, flush=False)

    async def emit(self, turn: int, flush: bool) -> None:
        """Sends every complete slice of buffered voice with its video.

        The model works in fixed slices, so the end of a sentence waits for the next
        one; `flush` pads the last slice with silence at the end of a turn.
        """
        per_slice = SLICE_FRAMES * PLAY_RATE // FPS
        if flush and len(self.voice_buffer) % per_slice:
            padding = per_slice - len(self.voice_buffer) % per_slice
            self.voice_buffer = np.pad(self.voice_buffer, (0, padding))
        while len(self.voice_buffer) >= per_slice and turn == self.turn:
            piece, self.voice_buffer = self.voice_buffer[:per_slice], self.voice_buffer[per_slice:]
            started = time.monotonic()
            earliest = int(self.client_index()) + self.lead
            fresh = earliest > self.next_index  # not continuing straight on from the last slice
            start = max(self.next_index, earliest)
            self.next_index = start + SLICE_FRAMES
            patches = await asyncio.to_thread(
                self.stream.frames, resample(piece, PLAY_RATE, MIC_RATE)
            )
            if turn != self.turn:
                break
            if fresh:
                # Adapt the head start to how long a slice really takes to generate.
                observed = (time.monotonic() - started) * FPS
                self.lead = int(min(50, max(10, 0.5 * self.lead + 0.5 * (observed + 5))))
                if TIMINGS:
                    print(
                        f"[délais] images {observed / FPS:.2f} s, avance {self.lead} images",
                        flush=True,
                    )
            header = {
                "type": "chunk",
                "turn": turn,
                "start": start,
                "rate": PLAY_RATE,
                "samples": per_slice,
                "patches": [{"x": p.x, "y": p.y, "size": len(p.jpeg)} for p in patches],
            }
            pcm = (np.clip(piece, -1.0, 1.0) * 32767).astype(np.int16)
            await self.send_chunk(header, pcm.tobytes() + b"".join(p.jpeg for p in patches))


class OllamaError(RuntimeError):
    pass


@app.websocket("/ws")
async def websocket(ws: WebSocket) -> None:
    await ws.accept()
    session = Session(ws)
    try:
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message.get("bytes") is not None:
                await session.on_audio(message["bytes"])
                continue
            data = json.loads(message["text"])
            kind = data.get("type")
            if kind == "tick":
                session.tick_index = float(data["index"])
                session.tick_time = time.monotonic()
            elif kind == "start":
                lang = data.get("lang")
                session.lang = lang if lang in VOICES else "fr"
                voices = [v["id"] for v in speaker.available(session.lang)]
                avatar_id = data.get("avatar")
                if not voices or avatar_id not in engine.avatars:
                    await session.send({"type": "error", "code": "setup"})
                    continue
                session.voice = voice_for(avatar_id, session.lang, voices)
                session.avatar = engine.avatars[avatar_id]
                session.barge_in = bool(data.get("bargeIn", True))
                session.tick_index = float(data.get("index", 0))
                session.tick_time = time.monotonic()
                session.reset_listening()
                await session.send({"type": "ready"})
                await session.greet()
            elif kind == "text":
                text = str(data.get("text", "")).strip()[:2000]
                if text:
                    await session.start_turn(text)
            elif kind == "interrupt":
                await session.interrupt()
            elif kind == "mute":
                session.muted = bool(data.get("on"))
                session.reset_listening()
            elif kind == "bargeIn":
                session.barge_in = bool(data.get("on"))
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        session.turn += 1
        if session.task and not session.task.done():
            session.task.cancel()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("VISAVIS_PORT", "8765")))
