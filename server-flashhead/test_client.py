"""End-to-end check without a browser: plays the part of the web page.

Connects to the running server, greets, sends one typed turn, then one spoken
turn (a synthesized voice sent as microphone audio), and prints timings.
"""

from __future__ import annotations

import asyncio
import json
import struct
import sys
import time
import urllib.request

import numpy as np
import websockets

SERVER = "127.0.0.1:8765"
FPS = 25


async def main() -> int:
    config = json.load(urllib.request.urlopen(f"http://{SERVER}/api/config"))
    print("config:", config["model"], [a["id"] for a in config["avatars"]], flush=True)
    avatar = config["avatars"][0]["id"]
    frames = urllib.request.urlopen(f"http://{SERVER}/api/avatars/{avatar}/frames").read()
    print("frames pack:", struct.unpack("<I", frames[:4])[0], "frames,", len(frames) // 1024, "KiB")

    origin = time.monotonic()
    index = lambda: (time.monotonic() - origin) * FPS  # noqa: E731
    state = {"sent": 0.0, "first": None, "late": 0, "frames": 0, "audio": 0.0, "end": 0}

    async with websockets.connect(f"ws://{SERVER}/ws", max_size=None) as ws:
        await ws.send(json.dumps({"type": "start", "lang": "fr", "avatar": avatar, "index": index()}))

        async def ticks():
            while True:
                await asyncio.sleep(0.25)
                await ws.send(json.dumps({"type": "tick", "index": index()}))

        ticker = asyncio.create_task(ticks())

        async def until_turn_end(label: str):
            state.update(first=None, late=0, frames=0, audio=0.0)
            while True:
                message = await asyncio.wait_for(ws.recv(), timeout=120)
                if isinstance(message, bytes):
                    size = struct.unpack("<I", message[:4])[0]
                    header = json.loads(message[4 : 4 + size])
                    if state["first"] is None:
                        state["first"] = time.monotonic() - state["sent"]
                    if header["start"] < index():
                        state["late"] += 1
                    state["frames"] += len(header["patches"])
                    state["audio"] += header["samples"] / header["rate"]
                    state["end"] = header["start"] + len(header["patches"])
                    continue
                data = json.loads(message)
                if data["type"] in ("user", "assistant", "error"):
                    print(f"  [{data['type']}] {data.get('text') or data}", flush=True)
                if data["type"] == "error":
                    return False
                if data["type"] == "turn_end":
                    first = state["first"]
                    print(
                        f"{label}: first audio+video after "
                        f"{first if first is None else round(first, 2)} s, {state['frames']} frames, "
                        f"{state['audio']:.1f} s audio, {state['late']} late chunk(s)",
                        flush=True,
                    )
                    return state["frames"] > 0

        state["sent"] = time.monotonic()
        ok = await until_turn_end("greeting")

        # Let the greeting finish playing, as a real listener would.
        await asyncio.sleep(max(0.0, (state["end"] - index()) / FPS) + 0.3)
        state["sent"] = time.monotonic()
        await ws.send(json.dumps({"type": "text", "text": "Quelle est la capitale de l'Italie ?"}))
        ok = await until_turn_end("typed turn") and ok

        # Spoken turn: synthesize a question and stream it as microphone audio in real time.
        sys.path.insert(0, ".")
        from speech import Speaker, resample

        audio, rate = Speaker().synthesize(
            "Et peux-tu me dire combien font douze fois douze ?", "fr_FR-tom-medium"
        )
        pcm = (resample(audio, rate, 16000) * 32767).astype(np.int16)
        pcm = np.concatenate([np.zeros(8000, np.int16), pcm, np.zeros(24000, np.int16)])
        await asyncio.sleep(max(0.0, (state["end"] - index()) / FPS) + 0.3)
        for i in range(0, len(pcm), 1024):
            await ws.send(pcm[i : i + 1024].tobytes())
            await asyncio.sleep(1024 / 16000)
            if i + 1024 >= len(pcm) - 24000 and state["sent"] < 0:
                pass
        state["sent"] = time.monotonic() - 24000 / 16000  # moment the speech ended
        ok = await until_turn_end("spoken turn (measured from end of speech)") and ok
        ticker.cancel()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
