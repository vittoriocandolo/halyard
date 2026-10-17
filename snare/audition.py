"""Audition: play the exact render buffers in context, and re-render on every edit (plan §7.5).

Hits are unchoked one-shots: each plays its buffer to the end, so overlapping
tails sum as in a DAW. With a tempo, hits land on chosen beats of 4/4 bars,
the backbeat by default; without one, they repeat every rendered length plus
0.5 s. Several patches cycle hit by hit, for velocity layers or for A/B, and
each patch's takes cycle over its own hits. A new buffer takes over at the
next hit that plays its patch.

The watcher tags each file change with an increasing generation and renders
it in the background. A finished render is used only if its generation is
still the newest, so stale results are dropped; an invalid edit, or a
gain-mode render over full scale, leaves the last valid buffer playing.

A bed (`--with`), such as the track's kick-and-hat loop, plays under the hits
and restarts on the tempo grid, so it never drifts from them. A sample
(`--against`) alternates with the render hit by hit, matched to the render's
loudness and rematched on every re-render: the sample is scaled, never the
render. With `--blind` each pair plays in random order, revealed when
audition stops. Nothing is ever resampled.
"""

from __future__ import annotations

import math
import os
import random
import sys
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np

from . import wav
from .loudness import momentary_max
from .patch import PatchError, loads
from .render import FORMATS, RenderError, db, quantize, render_takes

GAP_S = 0.5
BAR_TOLERANCE_S = 0.001  # how far a bed may be from a whole number of bars


class AuditionError(Exception):
    pass


def load_audio(path) -> tuple[int, np.ndarray]:
    """A float or PCM WAV as float64 frames × channels, never resampled."""
    try:
        w = wav.read(path)
    except (OSError, ValueError) as e:
        raise AuditionError(f"{path}: {e}") from None
    x = w.as_float()
    return w.rate, x if x.ndim == 2 else x[:, None]


def playable(audio: list[np.ndarray], report) -> list[np.ndarray]:
    """What `render` writes, as float32: PCM formats add only their quantization."""
    bits = FORMATS[report.format]
    if bits is None:
        return audio
    return [
        (quantize(a, bits, t.seed) / float(2 ** (bits - 1))).astype(np.float32) for a, t in zip(audio, report.takes)
    ]


class Bed:
    """A loop under the hits, restarted every `bars` bars on the tempo grid.

    Loop k starts at round(k · bars · 4 · 60 / bpm · rate) after the first hit's
    bar, the same rounding as the hits, and plays as much of the bed as fits
    before the next start.
    """

    def __init__(self, frames: np.ndarray, rate: int, bpm: float, bars: int):
        self.frames = np.ascontiguousarray(frames, dtype=np.float32)
        self.channels = self.frames.shape[1]
        self.rate, self.bpm, self.bars = rate, bpm, bars

    def start(self, k: int) -> int:
        return round(k * self.bars * 4 * self.rate * 60 / self.bpm)

    def fill(self, clock: int, frames: int) -> np.ndarray:
        """The bed over [clock, clock + frames), with the clock counted from the first bar."""
        out = np.zeros((frames, self.channels), dtype=np.float32)
        k = max(0, int(clock // (self.bars * 4 * self.rate * 60 / self.bpm)) - 1)
        while self.start(k) < clock + frames:
            s, e = self.start(k), min(self.start(k + 1), self.start(k) + len(self.frames))
            a, b = max(clock, s), min(clock + frames, e)
            if a < b:
                out[a - clock : b - clock] += self.frames[a - s : b - s]
            k += 1
        return out


class Hits:
    """Mixes hits into output blocks. `offer` may be called from any thread.

    Slot i holds patch i's takes. Hits cycle through the slots that have a
    buffer, and each slot's takes cycle over its own hits. With a sample
    (slot 0 only), hits alternate between the render and the sample in pairs;
    `rng` makes each pair's order random, and `order` records it (True when
    the render played first).
    """

    def __init__(
        self, rate: int, slots: int = 1, bpm: float | None = None, beats=(2, 4), *, bed: Bed | None = None, rng=None
    ):
        self.rate, self.bpm, self.beats, self.bed, self.rng = rate, bpm, tuple(beats), bed, rng
        self.channels = bed.channels if bed is not None else 1
        self._offers: list[tuple | None] = [None] * slots
        self._lock = threading.Lock()
        self.generations = [-1] * slots
        self.buffers: list[list[np.ndarray] | None] = [None] * slots
        self.sample: np.ndarray | None = None
        self.played = [0] * slots  # hits per slot, which pick its take
        self.last_slot = -1
        self.voices: list[tuple[int, np.ndarray]] = []  # (start sample, buffer)
        self.clock = 0  # index of the next output sample, counted from the first hit's bar
        self.started = False
        self.next_hit: int | None = None
        self.count = 0  # hits started
        self.render_hits = 0
        self.order: list[bool] = []
        self.log: list[tuple[int, int]] = []  # (slot, take) of each hit, the sample being slot −1

    @property
    def generation(self) -> int:
        return self.generations[0]

    def offer(self, slot: int, generation: int, buffers: list[np.ndarray], sample: np.ndarray | None = None) -> None:
        """Queue a slot's take buffers, and the sample matched to them, for that slot's next hit.

        An older generation never replaces a newer one.
        """
        with self._lock:
            current = self._offers[slot]
            if current is None or generation > current[0]:
                self._offers[slot] = (generation, list(buffers), sample)

    def wait(self) -> float:
        """Seconds until the next hit."""
        return 0.0 if self.next_hit is None else max(0, self.next_hit - self.clock) / self.rate

    def _take_offers(self) -> None:
        with self._lock:
            for i, offered in enumerate(self._offers):
                if offered is not None and offered[0] > self.generations[i]:
                    self.generations[i], self.buffers[i] = offered[0], offered[1]
                    if i == 0:
                        self.sample = offered[2]

    def _hit_time(self, n: int) -> int:
        """Where hit n starts: on the beat grid, counted from the first bar."""
        per_bar = len(self.beats)
        beat = (n // per_bar) * 4 + self.beats[n % per_bar] - 1
        return round(beat * self.rate * 60 / self.bpm)

    def _next_buffer(self) -> np.ndarray:
        if self.sample is not None:
            if self.count % 2 == 0:
                self.order.append(True if self.rng is None else self.rng.random() < 0.5)
            if (self.count % 2 == 0) != self.order[-1]:
                self.log.append((-1, 0))
                return self.sample
        ready = [i for i, b in enumerate(self.buffers) if b]
        slot = next((i for i in ready if i > self.last_slot), ready[0])
        self.last_slot = slot
        takes = self.buffers[slot]
        take = self.played[slot] % len(takes)
        self.played[slot] += 1
        self.log.append((slot, take))
        return takes[take]

    def fill(self, frames: int) -> np.ndarray:
        if not self.started:
            self._take_offers()
            if not any(self.buffers):
                return np.zeros(frames if self.channels == 1 else (frames, self.channels), dtype=np.float32)
            self.started = True
            self.next_hit = self._hit_time(0) if self.bpm else 0
        out = np.zeros(frames, dtype=np.float32)
        end = self.clock + frames
        while self.next_hit is not None and self.next_hit < end:
            self._take_offers()
            buf = self._next_buffer()
            self.voices.append((self.next_hit, buf))
            self.count += 1
            if self.bpm:
                self.next_hit = self._hit_time(self.count)
            else:
                self.next_hit += buf.size + round(GAP_S * self.rate)
        playing = []
        for start, buf in self.voices:
            a, b = max(self.clock, start), min(end, start + buf.size)
            if a < b:
                out[a - self.clock : b - self.clock] += buf[a - start : b - start]
            if start + buf.size > end:
                playing.append((start, buf))
        self.voices = playing
        if self.bed is not None:
            mix = self.bed.fill(self.clock, frames)
            mix += out[:, None]  # the snare sits in the centre
            out = mix[:, 0] if self.channels == 1 else mix
        self.clock = end
        return out


class Watcher:
    """Renders the newest version of a file in the background."""

    def __init__(
        self,
        path,
        work: Callable[[str], object],
        done: Callable[[int, object, float, bool], None],
        text: str | None = None,
    ):
        self.path, self.work, self.done = path, work, done
        self.generation = 0
        self._stamp, self._text = None, text  # `text`: the version already rendered, if any
        self._pending: tuple[int, str] | None = None
        self._cv = threading.Condition()
        threading.Thread(target=self._run, daemon=True).start()

    def poll(self) -> None:
        """Start a new generation if the file's content changed."""
        try:
            st = os.stat(self.path)
            stamp = (st.st_mtime_ns, st.st_size)
            if stamp == self._stamp:
                return
            with open(self.path, encoding="utf-8") as f:
                text = f.read()
        except OSError:
            return
        self._stamp = stamp
        if text == self._text:
            return
        self._text = text
        self.request(text)

    def request(self, text: str) -> None:
        with self._cv:
            self.generation += 1
            self._pending = (self.generation, text)  # an older request not yet started is superseded
            self._cv.notify()

    def _run(self) -> None:
        while True:
            with self._cv:
                while self._pending is None:
                    self._cv.wait()
                generation, text = self._pending
                self._pending = None
            start = time.perf_counter()
            try:
                result = self.work(text)
            except Exception as e:  # reported by `done`; the last valid buffer keeps playing
                result = e
            elapsed = time.perf_counter() - start
            with self._cv:
                newest = generation == self.generation
            self.done(generation, result, elapsed, newest)


class OverError(Exception):
    def __init__(self, path, report):
        super().__init__(f"{path}: over full scale by {db(report.peak):.2f} dB: lower output.db to audition it")


class Session:
    """One audition: the hits, the renders behind them, and the watchers. `poll` is called about every 0.1 s."""

    def __init__(
        self,
        paths,
        *,
        watch: bool = False,
        bpm: float | None = None,
        beats=(2, 4),
        bed=None,
        against=None,
        blind: bool = False,
        takes: int = 1,
        rng=None,
        out=None,
    ):
        out = out or sys.stdout
        self.paths = [Path(p) for p in ([paths] if isinstance(paths, (str, Path)) else paths)]
        self.out, self.takes, self.bpm = out, takes, bpm
        if against is not None and len(self.paths) > 1:
            raise AuditionError("--against compares one patch with a sample; give one patch")
        if blind and against is None:
            raise AuditionError("--blind needs --against: a blind comparison needs a sample")
        if bed is not None and bpm is None:
            raise AuditionError("--with needs --bpm, so the hits land on the bed's beats")
        texts = [self._text(p) for p in self.paths]
        first = self._render(texts[0])
        audio, report = first
        self._say(0, None, str(report))
        if report.over:
            raise OverError(self.paths[0], report)
        self.rate = report.sample_rate
        self.bed = self._bed(bed, bpm) if bed is not None else None
        self.sample, self.sample_lufs = self._sample(against) if against is not None else (None, None)
        self.blind = blind
        self.hits = Hits(
            self.rate, len(self.paths), bpm, beats, bed=self.bed, rng=(rng or random.SystemRandom()) if blind else None
        )
        self.buffers: list[list[np.ndarray] | None] = [None] * len(self.paths)
        for note in self._accept(0, 0, audio, report):
            print(note, file=self.out)
        self._pairs_said = 0
        self.watchers = [
            Watcher(p, self._render, self._done_for(i), t) for i, (p, t) in enumerate(zip(self.paths, texts))
        ]
        # The first patch plays as soon as it is ready; the others render behind it.
        for i, text in enumerate(texts[1:], 1):
            self.watchers[i].request(text)
        self.watch = watch

    def _text(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except OSError as e:
            raise AuditionError(f"{path}: {e.strerror}") from None

    def _render(self, text: str):
        audio, report = render_takes(loads(text), self.takes)
        return playable(audio, report), report

    def _say(self, slot: int, generation: int | None, message: str) -> None:
        if generation is None:
            tag = "" if len(self.paths) == 1 else f"{self.paths[slot].name}\n"
            print(tag + message, file=self.out)
            return
        tag = f"[{generation}]" if len(self.paths) == 1 else f"[{self.paths[slot].name} {generation}]"
        print(f"{tag} {message}", file=self.out)

    def _bed(self, path, bpm) -> Bed:
        rate, frames = load_audio(path)
        if rate != self.rate:
            raise AuditionError(f"{path}: {rate} Hz, but the patch renders at {self.rate} Hz; audition never resamples")
        if frames.shape[1] > 2:
            raise AuditionError(f"{path}: {frames.shape[1]} channels; a bed must be mono or stereo")
        bar = self.rate * 4 * 60 / bpm
        bars = round(frames.shape[0] / bar)
        if bars < 1 or abs(frames.shape[0] - bars * bar) > BAR_TOLERANCE_S * self.rate:
            raise AuditionError(
                f"{path}: {frames.shape[0] / bar:.3f} bars at {bpm:g} bpm; "
                "a bed must be within 1 ms of a whole number of bars"
            )
        print(
            f"{'bed':<9}{path}: {bars} bar{'s' if bars > 1 else ''} at {bpm:g} bpm, "
            f"{'stereo' if frames.shape[1] == 2 else 'mono'}",
            file=self.out,
        )
        return Bed(frames, self.rate, bpm, bars)

    def _sample(self, path) -> tuple[np.ndarray, float]:
        rate, frames = load_audio(path)
        if rate != self.rate:
            raise AuditionError(f"{path}: {rate} Hz, but the patch renders at {self.rate} Hz; audition never resamples")
        mid = frames.mean(axis=1)  # a stereo sample plays as its mid channel
        level = momentary_max(mid, rate)
        if level == -math.inf:
            raise AuditionError(f"{path}: the sample is silent")
        return mid, level

    def _matched(self, report) -> tuple[np.ndarray | None, list[str]]:
        """The sample matched to this render's loudness (its first take's), and what to say about it."""
        if self.sample is None:
            return None, []
        loud = report.takes[0].loudness
        gain = loud - self.sample_lufs if loud > -math.inf else -math.inf
        matched = (self.sample * 10.0 ** (gain / 20.0)).astype(np.float32)
        peak = float(np.max(np.abs(matched)))
        clip = (
            f"; its peak is then {db(peak):+.1f} dBFS, which the device will clip, so lower output.db"
            if peak > 1.0
            else ""
        )
        return matched, [
            f"{'sample':<9}plays at {gain:+.1f} dB, matched to the render's loudness "
            f"(the largest 400 ms momentary loudness, BS.1770){clip}"
        ]

    def _mix_warning(self) -> str | None:
        """Whether the hits, with the bed under them, would clip the device over one pattern."""
        if not self.bpm or not any(self.buffers):
            return None
        probe = Hits(self.rate, len(self.buffers), self.bpm, self.hits.beats, bed=self.bed)
        for i, b in enumerate(self.buffers):
            if b:
                probe.offer(i, 0, b)
        longest = max(x.size for b in self.buffers if b for x in b)
        span = (self.bed.start(1) if self.bed is not None else round(2 * 4 * self.rate * 60 / self.bpm)) + longest
        total = sum(len(b) for b in self.buffers if b)
        peak = float(np.max(np.abs(probe.fill(span * max(1, total)))))
        if peak <= 1.0:
            return None
        under = "over the bed " if self.bed is not None else ""
        return f"warning: the hits {under}peak at {db(peak):+.1f} dBFS, which the device will clip; lower " + (
            "the bed or output.db" if self.bed is not None else "output.db"
        )

    def _accept(self, slot: int, generation: int, audio, report) -> list[str]:
        """Offer a render to its slot; returns what to say about the sample and the mix."""
        matched, notes = self._matched(report) if slot == 0 else (None, [])
        self.buffers[slot] = audio
        self.hits.offer(slot, generation, audio, matched)
        warning = self._mix_warning()
        return notes + ([warning] if warning else [])

    def poll(self) -> None:
        if self.watch:
            for w in self.watchers:
                w.poll()
        if self.blind:
            started = (self.hits.count + 1) // 2
            while self._pairs_said < started:
                self._pairs_said += 1
                print(f"pair {self._pairs_said}", file=self.out)

    def reveal(self) -> str | None:
        """The blind order, once audition stops: R is the render, S the sample."""
        if not self.blind:
            return None
        pairs = " ".join(f"{i + 1}:{'RS' if first else 'SR'}" for i, first in enumerate(self.hits.order))
        return f"blind order (R = render, S = sample): {pairs or 'no pair played'}"

    def _done_for(self, slot: int):
        def done(generation, result, elapsed, newest) -> None:
            say = lambda msg: self._say(slot, generation, msg)  # noqa: E731
            if not newest:
                say("dropped: a newer edit is rendering")
            elif isinstance(result, PatchError):
                say(f"invalid patch, still playing the last valid one:\n{result}")
            elif isinstance(result, (RenderError, Exception)):
                say(f"render failed, still playing the last valid one: {result}")
            else:
                audio, report = result
                if report.over:
                    say(
                        f"over full scale by {db(report.peak):.2f} dB; lower output.db. "
                        "Still playing the last valid buffer."
                    )
                elif report.sample_rate != self.rate:
                    say(
                        f"sample rate {report.sample_rate} Hz, but audition plays at {self.rate} Hz; restart audition "
                        "to hear it. Still playing the last valid buffer."
                    )
                else:
                    notes = self._accept(slot, generation, audio, report)
                    say("\n".join([f"rendered in {1000 * elapsed:.0f} ms; plays at its next hit", str(report), *notes]))

        return done


def audition(
    paths,
    *,
    watch: bool = False,
    bpm: float | None = None,
    beats=(2, 4),
    bed=None,
    against=None,
    blind: bool = False,
    takes: int = 1,
    out=None,
    stream_factory=None,
    rng=None,
) -> int:
    """Play the patches until interrupted; with `watch`, re-render on every edit."""
    out = out or sys.stdout
    try:
        session = Session(
            paths,
            watch=watch,
            bpm=bpm,
            beats=beats,
            bed=bed,
            against=against,
            blind=blind,
            takes=takes,
            rng=rng,
            out=out,
        )
    except PatchError as e:
        print("invalid patch\n" + "\n".join(f"  {m}" for m in e.problems), file=sys.stderr)
        return 2
    except (AuditionError, RenderError) as e:
        print(e, file=sys.stderr)
        return 2
    except OverError as e:
        print(e, file=sys.stderr)
        return 1
    try:
        stream = (stream_factory or _sounddevice_stream)(session.rate, session.hits.channels, session.hits.fill)
    except RuntimeError as e:
        print(e, file=sys.stderr)
        return 1
    try:
        with stream:
            while True:
                time.sleep(0.1)
                session.poll()
    except KeyboardInterrupt:
        return 0
    finally:
        revealed = session.reveal()
        if revealed:
            print(revealed, file=out)


def _sounddevice_stream(rate: int, channels: int, fill):
    try:
        import sounddevice as sd
    except (ImportError, OSError) as e:
        raise RuntimeError(f"audition needs the sounddevice package and PortAudio ({e})") from None
    try:
        sd.check_output_settings(samplerate=rate, channels=channels, dtype="float32")
    except Exception as e:
        raise RuntimeError(f"the output device does not support {rate} Hz ({e}); audition never resamples") from None

    def callback(outdata, frames, _time, _status):
        outdata[:] = fill(frames).reshape(frames, channels)

    return sd.OutputStream(samplerate=rate, channels=channels, dtype="float32", callback=callback)
