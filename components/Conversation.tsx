"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { strings, type Lang, type Strings } from "@/lib/i18n";
import { AvatarPlayer } from "@/lib/player";

const SERVER = process.env.NEXT_PUBLIC_VISAVIS_SERVER ?? "http://127.0.0.1:8765";
const SOCKET = `${SERVER.replace(/^http/, "ws")}/ws`;

type Phase = "setup" | "connecting" | "live";
type ErrorCode = keyof Strings["errors"];

interface AvatarInfo {
  id: string;
  width: number;
  height: number;
  frames: number;
}

interface VoiceInfo {
  id: string;
  name: string;
}

interface Config {
  fps: number;
  model: string;
  avatars: AvatarInfo[];
  voices: Record<Lang, VoiceInfo[]>;
}

interface Line {
  id: number;
  role: "user" | "assistant";
  text: string;
  turn?: number;
}

function formatTime(seconds: number) {
  return `${Math.floor(seconds / 60)}:${(seconds % 60).toString().padStart(2, "0")}`;
}

export default function Conversation() {
  const [lang, setLang] = useState<Lang>("fr");
  const t = strings[lang];

  const [phase, setPhase] = useState<Phase>("setup");
  const [config, setConfig] = useState<Config | null>(null);
  const [configState, setConfigState] = useState<"loading" | "ready" | "error">("loading");
  const [avatarId, setAvatarId] = useState("");
  const [bargeIn, setBargeIn] = useState(true);
  const [error, setError] = useState<ErrorCode | "">("");

  const [lines, setLines] = useState<Line[]>([]);
  const [draft, setDraft] = useState("");
  const [thinking, setThinking] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const [hearing, setHearing] = useState(false);
  const [micOn, setMicOn] = useState(true);
  const [camOn, setCamOn] = useState(false);
  const [elapsed, setElapsed] = useState(0);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const playerRef = useRef<AvatarPlayer | null>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const micRef = useRef<{ stream: MediaStream; context: AudioContext } | null>(null);
  const camRef = useRef<MediaStream | null>(null);
  const selfVideoRef = useRef<HTMLVideoElement | null>(null);
  const faceTimerRef = useRef<ReturnType<typeof setInterval> | undefined>(undefined);
  const faceCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const transcriptRef = useRef<HTMLDivElement | null>(null);
  const tickRef = useRef<ReturnType<typeof setInterval> | undefined>(undefined);
  const lineIdRef = useRef(0);
  const closingRef = useRef(false);

  const avatar = config?.avatars.find((a) => a.id === avatarId);
  const voices = config?.voices[lang] ?? [];

  // ---- Local server configuration ----------------------------------------
  const loadConfig = useCallback(() => {
    setConfigState("loading");
    fetch(`${SERVER}/api/config`)
      .then((res) => res.json())
      .then((json: Config) => {
        setConfig(json);
        setAvatarId((current) =>
          json.avatars.some((a) => a.id === current) ? current : (json.avatars[0]?.id ?? ""),
        );
        setError("");
        setConfigState("ready");
      })
      .catch(() => {
        setError("server");
        setConfigState("error");
      });
  }, []);

  useEffect(loadConfig, [loadConfig]);

  // ---- Session lifecycle -------------------------------------------------
  const stopCamera = useCallback(() => {
    clearInterval(faceTimerRef.current);
    const socket = socketRef.current;
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: "camera", on: false }));
    }
    camRef.current?.getTracks().forEach((track) => track.stop());
    camRef.current = null;
    if (selfVideoRef.current) selfVideoRef.current.srcObject = null;
    setCamOn(false);
  }, []);

  const teardown = useCallback(() => {
    closingRef.current = true;
    clearInterval(tickRef.current);
    socketRef.current?.close();
    socketRef.current = null;
    playerRef.current?.stop();
    playerRef.current = null;
    micRef.current?.stream.getTracks().forEach((track) => track.stop());
    void micRef.current?.context.close();
    micRef.current = null;
    stopCamera();
    setThinking(false);
    setSpeaking(false);
    setHearing(false);
    setPhase("setup");
  }, [stopCamera]);

  useEffect(() => teardown, [teardown]);

  const send = (message: object) => {
    const socket = socketRef.current;
    if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(message));
  };

  const startMicrophone = async (socket: WebSocket) => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });
      const context = new AudioContext();
      await context.audioWorklet.addModule("/mic-worklet.js");
      const node = new AudioWorkletNode(context, "mic-processor");
      node.port.onmessage = (event: MessageEvent<ArrayBuffer>) => {
        if (socket.readyState === WebSocket.OPEN) socket.send(event.data);
      };
      context.createMediaStreamSource(stream).connect(node);
      micRef.current = { stream, context };
    } catch {
      setMicOn(false);
      setError("mic");
    }
  };

  const onServerMessage = (data: {
    type: string;
    text?: string;
    turn?: number;
    active?: boolean;
    code?: string;
  }) => {
    switch (data.type) {
      case "ready":
        setPhase("live");
        break;
      case "user":
        setLines((current) => [
          ...current,
          { id: ++lineIdRef.current, role: "user", text: data.text ?? "" },
        ]);
        break;
      case "assistant":
        setThinking(false);
        setLines((current) => {
          const last = current[current.length - 1];
          if (last?.role === "assistant" && last.turn === data.turn) {
            return [...current.slice(0, -1), { ...last, text: `${last.text} ${data.text}` }];
          }
          return [
            ...current,
            { id: ++lineIdRef.current, role: "assistant", text: data.text ?? "", turn: data.turn },
          ];
        });
        break;
      case "thinking":
        setThinking(true);
        break;
      case "turn_end":
        setThinking(false);
        break;
      case "listening":
        setHearing(Boolean(data.active));
        break;
      case "interrupt":
        playerRef.current?.interrupt();
        setThinking(false);
        break;
      case "error":
        setThinking(false);
        setError(data.code === "ollama" ? "ollama" : data.code === "setup" ? "setup" : "unknown");
        break;
    }
  };

  const start = async () => {
    const canvas = canvasRef.current;
    if (!avatar || voices.length === 0 || !config || !canvas) return;
    setError("");
    setLines([]);
    setElapsed(0);
    setMicOn(true);
    closingRef.current = false;
    setPhase("connecting");

    try {
      const res = await fetch(`${SERVER}/api/avatars/${encodeURIComponent(avatar.id)}/thumb`);
      if (!res.ok) throw new Error("portrait");

      canvas.width = avatar.width;
      canvas.height = avatar.height;
      const player = new AvatarPlayer(canvas, await res.blob(), config.fps, setSpeaking);
      playerRef.current = player;
      await player.start();

      const socket = new WebSocket(SOCKET);
      socket.binaryType = "arraybuffer";
      socketRef.current = socket;
      socket.onmessage = (event) => {
        if (typeof event.data === "string") onServerMessage(JSON.parse(event.data));
        else playerRef.current?.addChunk(event.data as ArrayBuffer);
      };
      socket.onclose = () => {
        if (closingRef.current || socketRef.current !== socket) return;
        teardown();
        setError("closed");
      };
      await new Promise<void>((resolve, reject) => {
        socket.onopen = () => resolve();
        socket.onerror = () => reject(new Error("socket"));
      });

      await startMicrophone(socket);
      // The avatar reacts to the user's face as well as their voice; it is optional.
      void startCamera(false);
      socket.send(
        JSON.stringify({
          type: "start",
          lang,
          avatar: avatar.id,
          bargeIn,
          index: player.index(),
        }),
      );
      tickRef.current = setInterval(
        () => send({ type: "tick", index: playerRef.current?.index() ?? 0 }),
        250,
      );
    } catch {
      teardown();
      setError("server");
    }
  };

  // ---- Controls ----------------------------------------------------------
  const sendText = (text: string) => {
    const clean = text.trim();
    if (!clean) return;
    playerRef.current?.interrupt();
    send({ type: "text", text: clean });
  };

  const toggleMic = () => {
    const next = !micOn;
    setMicOn(next);
    setHearing(false);
    send({ type: "mute", on: !next });
  };

  const startCamera = async (reportRefusal: boolean) => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" },
      });
      camRef.current = stream;
      setCamOn(true);
      const video = selfVideoRef.current;
      if (video) video.srcObject = stream;
      // A few small pictures a second are enough for the avatar to follow a face.
      const canvas = (faceCanvasRef.current ??= document.createElement("canvas"));
      canvas.width = 320;
      canvas.height = 240;
      clearInterval(faceTimerRef.current);
      faceTimerRef.current = setInterval(() => {
        if (!video || video.readyState < 2) return;
        canvas.getContext("2d")?.drawImage(video, 0, 0, canvas.width, canvas.height);
        send({ type: "face", image: canvas.toDataURL("image/jpeg", 0.7) });
      }, 130);
    } catch {
      if (reportRefusal) setError("camera");
    }
  };

  const toggleCamera = async () => {
    if (camOn) stopCamera();
    else await startCamera(true);
  };

  const interrupt = () => {
    playerRef.current?.interrupt();
    send({ type: "interrupt" });
  };

  // ---- Timer and transcript scroll ---------------------------------------
  const live = phase === "live";

  useEffect(() => {
    if (!live) return;
    const timer = setInterval(() => setElapsed((value) => value + 1), 1000);
    return () => clearInterval(timer);
  }, [live]);

  useEffect(() => {
    const el = transcriptRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [lines]);

  const status = speaking
    ? "speaking"
    : thinking
      ? "thinking"
      : !micOn
        ? "muted"
        : hearing
          ? "hearing"
          : "listening";

  // ---- Render ------------------------------------------------------------
  return (
    <main className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true" />
          Vis-à-vis
          <span className="badge">{t.badge}</span>
        </div>
        <div className="lang" role="group" aria-label="Langue / Language">
          {(["fr", "en"] as const).map((code) => (
            <button
              key={code}
              type="button"
              className={lang === code ? "on" : ""}
              disabled={phase !== "setup"}
              onClick={() => setLang(code)}
            >
              {code.toUpperCase()}
            </button>
          ))}
        </div>
      </header>

      {error && (
        <p className="error" role="alert">
          {t.errors[error]}
          {error === "server" && phase === "setup" && (
            <button type="button" className="ghost" onClick={loadConfig}>
              {t.retry}
            </button>
          )}
        </p>
      )}

      <section className="setup" hidden={phase !== "setup"}>
        <h1>{t.tagline}</h1>

        {configState === "loading" && <p className="muted">{t.loading}</p>}

        {configState === "ready" && config && (
          <>
            <h2>{t.chooseAvatar}</h2>
            {config.avatars.length === 0 && <p className="muted">{t.errors.noAvatars}</p>}
            <div className="avatars">
              {config.avatars.map((a) => (
                <button
                  key={a.id}
                  type="button"
                  className={`avatar-card${a.id === avatarId ? " on" : ""}`}
                  aria-pressed={a.id === avatarId}
                  onClick={() => setAvatarId(a.id)}
                >
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={`${SERVER}/api/avatars/${encodeURIComponent(a.id)}/thumb`} alt="" />
                  <span className="avatar-name">{a.id}</span>
                </button>
              ))}
            </div>

            <div className="options">
              {voices.length === 0 && <small className="muted">{t.errors.noVoices}</small>}

              <label className="field check">
                <span className="row">
                  <input
                    type="checkbox"
                    checked={bargeIn}
                    onChange={(e) => setBargeIn(e.target.checked)}
                  />
                  {t.bargeIn}
                </span>
                <small className="muted">{t.bargeInHint}</small>
              </label>
            </div>

            <button
              type="button"
              className="primary"
              onClick={() => void start()}
              disabled={!avatar || voices.length === 0}
            >
              {t.start}
            </button>
            <p className="muted small">Ollama · {config.model}</p>
          </>
        )}
      </section>

      <section className="call" hidden={phase === "setup"}>
        <div className="stage">
          <canvas ref={canvasRef} />
          {phase === "connecting" && <div className="veil">{t.connecting}</div>}
          {live && (
            <>
              <div className={`status ${status}`}>
                <span className="dot" />
                {t[status]}
              </div>
              <div className="timer">{formatTime(elapsed)}</div>
            </>
          )}
          <video ref={selfVideoRef} className="self" autoPlay playsInline muted hidden={!camOn} />
        </div>

        <div className="controls">
          <button type="button" className={micOn ? "" : "off"} onClick={toggleMic}>
            {t.mic}
          </button>
          <button type="button" className={camOn ? "" : "off"} onClick={() => void toggleCamera()}>
            {t.camera}
          </button>
          <button type="button" onClick={interrupt} disabled={!speaking && !thinking}>
            {t.interrupt}
          </button>
          <button type="button" className="danger" onClick={teardown}>
            {t.end}
          </button>
        </div>

        <aside className="transcript">
          <h2>{t.transcript}</h2>
          <div className="lines" ref={transcriptRef}>
            {lines.length === 0 && <p className="muted">{t.emptyTranscript}</p>}
            {lines.map((line) => (
              <p key={line.id} className={line.role}>
                <strong>{line.role === "user" ? t.you : (avatar?.id ?? t.avatar)}</strong>
                {line.text}
              </p>
            ))}
          </div>
          <form
            onSubmit={(e) => {
              e.preventDefault();
              if (!live) return;
              sendText(draft);
              setDraft("");
            }}
          >
            <input
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              placeholder={t.typeHere}
              disabled={!live}
            />
            <button type="submit" disabled={!live || !draft.trim()}>
              {t.send}
            </button>
          </form>
        </aside>
      </section>
    </main>
  );
}
