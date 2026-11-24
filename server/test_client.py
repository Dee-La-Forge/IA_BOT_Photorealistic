"""End-to-end check without a browser: plays the part of the web page.

Connects to the running server and holds a short call: greeting, a silent stretch, one
typed turn, one spoken turn (a synthesized voice sent as microphone audio). Prints how
long the avatar takes to start speaking and whether its video kept up with real time.
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

    origin = time.monotonic()
    index = lambda: (time.monotonic() - origin) * FPS  # noqa: E731
    seen = {"frames": 0, "end": None, "gaps": 0, "gap_frames": 0, "late": 0, "voice": None, "voice_end": 0}
    texts: list[str] = []

    async with websockets.connect(f"ws://{SERVER}/ws", max_size=None) as ws:
        await ws.send(json.dumps({"type": "start", "lang": "fr", "avatar": avatar, "index": index()}))

        async def ticks():
            while True:
                await asyncio.sleep(0.25)
                await ws.send(json.dumps({"type": "tick", "index": index()}))

        async def receive():
            async for message in ws:
                if isinstance(message, bytes):
                    size = struct.unpack("<I", message[:4])[0]
                    header = json.loads(message[4 : 4 + size])
                    count = len(header["patches"])
                    if seen["end"] is not None and header["start"] != seen["end"]:
                        seen["gaps"] += 1
                        seen["gap_frames"] += header["start"] - seen["end"]
                    if header["start"] < index():
                        seen["late"] += 1
                    seen["end"] = header["start"] + count
                    seen["frames"] += count
                    if header["voice"]:
                        if seen["voice"] is None:
                            seen["voice"] = time.monotonic()
                        seen["voice_end"] = seen["end"]
                    continue
                data = json.loads(message)
                if data["type"] in ("user", "assistant", "error"):
                    texts.append(f"  [{data['type']}] {data.get('text') or data}")

        tasks = [asyncio.create_task(ticks()), asyncio.create_task(receive())]

        async def reply(label: str, since: float) -> bool:
            """Waits for the avatar to speak, then for it to finish."""
            seen["voice"] = None
            for _ in range(300):
                await asyncio.sleep(0.1)
                if seen["voice"] is not None:
                    break
            if seen["voice"] is None:
                print(f"{label}: l'avatar n'a pas parlé", flush=True)
                return False
            delay = seen["voice"] - since
            while index() < seen["voice_end"] + 5:  # until the speech has played out
                await asyncio.sleep(0.2)
            print("\n".join(texts), flush=True)
            texts.clear()
            print(f"{label}: voix après {delay:.2f} s", flush=True)
            return True

        ok = await reply("accueil", time.monotonic())

        await asyncio.sleep(4)  # nobody speaks: the avatar must still be generated
        ok = await reply_after_text(ws, reply) and ok

        sys.path.insert(0, ".")
        from speech import Speaker, resample

        audio, rate = Speaker().synthesize(
            "Et peux-tu me dire combien font douze fois douze ?", "fr_FR-tom-medium"
        )
        pcm = (resample(audio, rate, 16000) * 32767).astype(np.int16)
        pcm = np.concatenate([np.zeros(8000, np.int16), pcm, np.zeros(24000, np.int16)])
        for i in range(0, len(pcm), 1024):
            await ws.send(pcm[i : i + 1024].tobytes())
            await asyncio.sleep(1024 / 16000)
        ok = await reply("tour parlé (depuis la fin de la phrase)", time.monotonic() - 1.5) and ok

        for task in tasks:
            task.cancel()
        duration = time.monotonic() - origin
        print(
            f"vidéo: {seen['frames']} images en {duration:.0f} s d'appel "
            f"({seen['frames'] / duration:.1f} images/s), {seen['gaps']} trou(s) "
            f"totalisant {seen['gap_frames']} images, {seen['late']} bloc(s) en retard",
            flush=True,
        )
    return 0 if ok else 1


async def reply_after_text(ws, reply) -> bool:
    since = time.monotonic()
    await ws.send(json.dumps({"type": "text", "text": "Quelle est la capitale de l'Italie ?"}))
    return await reply("tour écrit", since)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
