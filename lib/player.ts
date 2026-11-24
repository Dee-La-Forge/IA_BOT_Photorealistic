// Draws the avatar on a canvas and plays its voice, both driven by one clock
// (the audio clock), so lips and sound stay in step.
//
// The server generates the avatar's video for the whole call and sends it in
// short blocks: for given frame numbers, the pictures to show and, when the
// avatar is speaking, the voice that goes with them. Until the first block
// arrives the avatar's portrait is shown; if a block is late the last picture
// stays on screen.

interface Frame {
  bitmap: ImageBitmap;
  x: number;
  y: number;
}

interface ChunkHeader {
  start: number;
  rate: number;
  samples: number;
  voice: boolean;
  patches: { x: number; y: number; size: number }[];
}

export class AvatarPlayer {
  private context: AudioContext;
  private origin = 0;
  private raf = 0;
  private shown = -1;
  private portrait: ImageBitmap | null = null;
  private frames = new Map<number, Frame>();
  private current: Frame | null = null; // the picture on screen
  private sources = new Set<AudioBufferSourceNode>();
  private speakStart = 0;
  private speakEnd = 0;
  private speaking = false;
  private stopped = false;
  private draw: CanvasRenderingContext2D;

  constructor(
    canvas: HTMLCanvasElement,
    portrait: Blob,
    private fps: number,
    private onSpeaking: (speaking: boolean) => void,
  ) {
    this.draw = canvas.getContext("2d")!;
    this.context = new AudioContext({ latencyHint: "interactive" });
    createImageBitmap(portrait).then(
      (bitmap) => {
        if (this.stopped) bitmap.close();
        else this.portrait = bitmap;
      },
      () => {},
    );
  }

  async start() {
    await this.context.resume();
    this.origin = this.context.currentTime;
    const loop = () => {
      if (this.stopped) return;
      this.render();
      this.raf = requestAnimationFrame(loop);
    };
    loop();
  }

  /** Current position on the frame clock (fractional). */
  index() {
    return (this.context.currentTime - this.origin) * this.fps;
  }

  private render() {
    const index = Math.floor(this.index());
    if (index === this.shown) return;
    this.shown = index;

    const frame = this.frames.get(index);
    if (frame) {
      this.frames.delete(index);
      this.current?.bitmap.close();
      this.current = frame;
    }
    // The live video covers the face and shoulders; the rest of the picture stays as it is.
    if (this.portrait) this.draw.drawImage(this.portrait, 0, 0);
    if (this.current) this.draw.drawImage(this.current.bitmap, this.current.x, this.current.y);

    for (const [i, old] of this.frames) {
      if (i >= index) continue;
      old.bitmap.close();
      this.frames.delete(i);
    }

    const speaking = index >= this.speakStart && index < this.speakEnd;
    if (speaking !== this.speaking) {
      this.speaking = speaking;
      this.onSpeaking(speaking);
    }
  }

  /** Handles one binary message from the server: pictures, and voice if the avatar speaks. */
  addChunk(buffer: ArrayBuffer) {
    const view = new DataView(buffer);
    const headerLength = view.getUint32(0, true);
    const header = JSON.parse(
      new TextDecoder().decode(new Uint8Array(buffer, 4, headerLength)),
    ) as ChunkHeader;
    let offset = 4 + headerLength;

    if (header.samples > 0) {
      const pcm = new Int16Array(buffer.slice(offset, offset + header.samples * 2));
      offset += header.samples * 2;
      const audio = this.context.createBuffer(1, pcm.length, header.rate);
      const channel = audio.getChannelData(0);
      for (let i = 0; i < pcm.length; i++) channel[i] = pcm[i] / 32768;
      const source = this.context.createBufferSource();
      source.buffer = audio;
      source.connect(this.context.destination);
      const when = this.origin + header.start / this.fps;
      const now = this.context.currentTime;
      if (when >= now) source.start(when);
      else if (now - when < audio.duration) source.start(now, now - when);
      this.sources.add(source);
      source.onended = () => this.sources.delete(source);
    }

    header.patches.forEach((info, k) => {
      const blob = new Blob([buffer.slice(offset, offset + info.size)], { type: "image/jpeg" });
      offset += info.size;
      const index = header.start + k;
      createImageBitmap(blob).then(
        (bitmap) => {
          if (this.stopped || index < this.shown) bitmap.close();
          else this.frames.set(index, { bitmap, x: info.x, y: info.y });
        },
        () => {},
      );
    });

    if (header.voice) {
      if (this.shown >= this.speakEnd) this.speakStart = header.start;
      this.speakEnd = Math.max(this.speakEnd, header.start + header.patches.length);
    }
  }

  /** Silences the avatar right away; its video simply carries on. */
  interrupt() {
    for (const source of this.sources) {
      source.onended = null;
      try {
        source.stop();
      } catch {
        /* not started */
      }
    }
    this.sources.clear();
    this.speakEnd = 0;
  }

  stop() {
    this.stopped = true;
    cancelAnimationFrame(this.raf);
    this.interrupt();
    for (const frame of this.frames.values()) frame.bitmap.close();
    this.frames.clear();
    this.current?.bitmap.close();
    this.current = null;
    this.portrait?.close();
    this.portrait = null;
    void this.context.close();
  }
}
