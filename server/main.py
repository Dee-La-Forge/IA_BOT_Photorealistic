"""Vis-à-vis local server: microphone in, talking avatar out, nothing leaves the machine.

    mic PCM ──► Silero VAD ──► Whisper ──► Ollama ──► Piper ──► Avatar Forcing ──► browser
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import struct
import time
from collections import deque
from contextlib import asynccontextmanager

import cv2
import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from engine import AVATAR_CACHE_DIR, AVATAR_SRC_DIR, BLOCK_FRAMES, FPS, PRIME_FRAMES, Engine
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
        warm = engine.stream(avatar)
        for frames in (PRIME_FRAMES, BLOCK_FRAMES):
            silence = np.zeros(frames * MIC_RATE // FPS, dtype=np.float32)
            engine.step(avatar, warm, silence, silence, None)
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
        self.lead = 8  # frames the video is kept ahead of what the browser is showing
        # The avatar's video is generated without pause for the whole call.
        self.stream = None
        self.video: asyncio.Task | None = None
        self.voice = ""
        self.voice_queue = np.zeros(0, dtype=np.float32)  # speech not yet turned into frames
        self.speech_end = 0  # frame at which the speech sent so far stops
        self.heard = np.zeros(0, dtype=np.float32)  # what the microphone picked up lately
        self.tracker = None
        self.faces: list[np.ndarray] = []  # the user's face, as seen since the last block
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
        return running or len(self.voice_queue) > 0 or self.client_index() < self.speech_end

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

        samples = np.frombuffer(data, dtype=np.int16)
        # The avatar listens too: keep the last second for its next block of video.
        self.heard = np.concatenate([self.heard, samples.astype(np.float32) / 32768.0])[-MIC_RATE:]
        self.pending = np.concatenate([self.pending, samples])
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
        self.voice_queue = np.zeros(0, dtype=np.float32)
        self.speech_end = 0
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
        try:
            await self.send({"type": "assistant", "text": text, "turn": turn})
            await self.speak(turn, text)
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
        try:
            while True:
                item = await sentences.get()
                if not spoken and TIMINGS:
                    print(f"[délais] première phrase du modèle : {time.monotonic() - began:.2f} s", flush=True)
                if item is None:
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

    async def speak(self, turn: int, text: str) -> None:
        """Synthesizes one sentence and queues it for the video loop to voice."""
        spoken = NOT_SPOKEN.sub("", text).strip()
        if not re.search(r"\w", spoken):
            return
        began = time.monotonic()
        audio, rate = await asyncio.to_thread(speaker.synthesize, spoken, self.voice)
        if TIMINGS:
            print(f"[délais] voix {time.monotonic() - began:.2f} s", flush=True)
        if turn != self.turn or len(audio) == 0:
            return
        self.voice_queue = np.concatenate([self.voice_queue, resample(audio, rate, PLAY_RATE)])

    # ---- video ------------------------------------------------------------
    def on_face(self, image: str) -> None:
        """A webcam picture from the browser: the avatar reacts to the user's face."""
        try:
            data = base64.b64decode(image.split(",", 1)[-1])
            picture = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        except (ValueError, cv2.error):
            return
        if picture is None or self.tracker is None:
            return
        face = self.tracker.crop(picture)
        if face is not None:
            self.faces = [*self.faces, face][-4:]

    async def run_video(self) -> None:
        """Generates the avatar's video for the whole call, a block at a time.

        Each block takes the next stretch of the avatar's own voice (silence when it has
        nothing to say) together with what the user just said and showed, so the same
        loop makes the avatar speak, listen and wait.
        """
        try:
            self.stream = await asyncio.to_thread(engine.stream, self.avatar)
            first = True
            while True:
                ahead = self.next_index - self.client_index()
                if not first and ahead > self.lead:
                    await asyncio.sleep(min(0.1, (ahead - self.lead) / FPS))
                    continue
                frames = PRIME_FRAMES if first else BLOCK_FRAMES
                wanted = frames * PLAY_RATE // FPS
                piece, self.voice_queue = self.voice_queue[:wanted], self.voice_queue[wanted:]
                voiced = len(piece) > 0
                piece = np.pad(piece, (0, wanted - len(piece)))
                heard_wanted = frames * MIC_RATE // FPS
                heard = self.heard[-heard_wanted:]
                heard = np.pad(heard, (heard_wanted - len(heard), 0))
                self.heard = np.zeros(0, dtype=np.float32)
                faces, self.faces = self.faces, []

                began = time.monotonic()
                jpegs = await asyncio.to_thread(
                    engine.step, self.avatar, self.stream,
                    resample(piece, PLAY_RATE, MIC_RATE), heard, faces,
                )
                # Placed straight after the previous block, or just ahead of the browser
                # if generation fell behind (the browser then holds its last frame).
                soonest = int(self.client_index()) + 2
                if not first and soonest > self.next_index:
                    self.lead = min(30, self.lead + 3)
                start = max(self.next_index, soonest)
                self.next_index = start + frames
                if voiced:
                    self.speech_end = self.next_index
                if TIMINGS and not first:
                    took = time.monotonic() - began
                    print(f"[délais] bloc {took * 1000:.0f} ms, avance {ahead:.0f} images", flush=True)
                header = {
                    "type": "chunk",
                    "start": start,
                    "rate": PLAY_RATE,
                    "samples": wanted if voiced else 0,
                    "voice": voiced,
                    "patches": [
                        {"x": self.avatar.x, "y": self.avatar.y, "size": len(jpeg)} for jpeg in jpegs
                    ],
                }
                pcm = (np.clip(piece, -1.0, 1.0) * 32767).astype(np.int16).tobytes() if voiced else b""
                await self.send_chunk(header, pcm + b"".join(jpegs))
                first = False
        except asyncio.CancelledError:
            raise
        except Exception as error:
            await self.report(error)


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
                session.tracker = engine.tracker()
                if session.video:
                    session.video.cancel()
                session.video = asyncio.create_task(session.run_video())
                await session.send({"type": "ready"})
                await session.greet()
            elif kind == "face":
                await asyncio.to_thread(session.on_face, str(data.get("image", "")))
            elif kind == "camera" and not data.get("on"):
                session.faces = []
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
        for task in (session.task, session.video):
            if task and not task.done():
                task.cancel()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("VISAVIS_PORT", "8765")))
