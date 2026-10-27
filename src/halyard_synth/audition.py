# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

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

import importlib
import math
import os
import random
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, Protocol, TextIO, cast

import numpy as np

from . import wav
from .loudness import momentary_max
from .patch import PatchError, loads, read_text
from .render import FORMATS, RenderError, Report, db, quantize, render_takes

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from contextlib import AbstractContextManager

    from numpy.typing import ArrayLike, NDArray

    type Floats = NDArray[np.float64]
    type Audio = NDArray[np.float32]
    type Fill = Callable[[int], Audio]
    type Rendered = tuple[list[Audio], Report]
    type Path_ = str | os.PathLike[str]

    class Done(Protocol):
        """What reports a finished render: its generation, its result or failure, its time, and if it is newest."""

        def __call__(self, generation: int, result: object, elapsed: float, /, *, newest: bool) -> None:
            """Report the render."""

    class Chance(Protocol):
        """What draws a blind pair's order: random.SystemRandom, or a seeded generator in a test."""

        def random(self) -> float:
            """Draw a number in [0, 1)."""


GAP_S = 0.5
BAR_TOLERANCE_S = 0.001  # how far a bed may be from a whole number of bars
BEATS_PER_BAR = 4
STEREO = 2
FAIR = 0.5  # a blind pair's chance of playing the render first
POLL_S = 0.1  # how often the session looks at its files and its pairs
# What a render can fail with: its own errors, and the lookup and type errors a bug would raise, so that a failed
# render is reported and the watcher keeps watching.
FAILURES = (ValueError, RuntimeError, OSError, ArithmeticError, MemoryError, LookupError, TypeError)


class Offer(NamedTuple):
    """A slot's newest render: its generation, its take buffers, and the sample matched to them."""

    generation: int
    buffers: list[Audio]
    sample: Audio | None


class AuditionError(Exception):
    """An audition that cannot start, with what to change."""


class Tempo(NamedTuple):
    """The beat grid: the tempo, and the beats of each 4/4 bar that hits land on."""

    bpm: float
    beats: tuple[int, ...] = (2, 4)


@dataclass(frozen=True)
class Settings:
    """How an audition plays: watching, its tempo and beats, a bed, a sample to compare, blind, and takes."""

    watch: bool = False
    bpm: float | None = None
    beats: tuple[int, ...] = (2, 4)
    bed: Path_ | None = None
    against: Path_ | None = None
    blind: bool = False
    takes: int = 1

    @property
    def tempo(self) -> Tempo | None:
        """The beat grid, or None to repeat hits freely."""
        return None if self.bpm is None else Tempo(self.bpm, self.beats)


DEFAULTS = Settings()


def load_audio(path: Path_) -> tuple[int, Floats]:
    """Read a float or PCM WAV as float64 frames × channels, never resampled."""
    try:
        w = wav.read(path)
    except (OSError, ValueError) as e:
        msg = f"{path}: {e}"
        raise AuditionError(msg) from None
    x = w.as_float()
    return w.rate, x if x.ndim == STEREO else x[:, None]


def playable(audio: list[Audio], report: Report) -> list[Audio]:
    """Return what `render` writes, as float32: PCM formats add only their quantization."""
    bits = FORMATS[report.format]
    if bits is None:
        return audio
    return [
        (quantize(a, bits, t.seed) / float(2 ** (bits - 1))).astype(np.float32)
        for a, t in zip(audio, report.takes, strict=True)
    ]


class Bed:
    """A loop under the hits, restarted every `bars` bars on the tempo grid.

    Loop k starts at round(k · bars · 4 · 60 / bpm · rate) after the first hit's
    bar, the same rounding as the hits, and plays as much of the bed as fits
    before the next start.
    """

    def __init__(self, frames: ArrayLike, rate: int, bpm: float, bars: int) -> None:
        """Keep the bed's frames as float32, and its tempo grid."""
        self.frames = np.ascontiguousarray(frames, dtype=np.float32)
        self.channels = self.frames.shape[1]
        self.rate, self.bpm, self.bars = rate, bpm, bars

    def start(self, k: int) -> int:
        """Return where loop k starts."""
        return round(k * self.bars * BEATS_PER_BAR * self.rate * 60 / self.bpm)

    def fill(self, clock: int, frames: int) -> Audio:
        """Return the bed over [clock, clock + frames), with the clock counted from the first bar."""
        out = np.zeros((frames, self.channels), dtype=np.float32)
        k = max(0, int(clock // (self.bars * BEATS_PER_BAR * self.rate * 60 / self.bpm)) - 1)
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
        self,
        rate: int,
        slots: int = 1,
        tempo: Tempo | None = None,
        *,
        bed: Bed | None = None,
        rng: Chance | None = None,
    ) -> None:
        """Start with no buffers: nothing plays until a slot is offered one."""
        self.rate, self.bed, self.rng = rate, bed, rng
        self.bpm: float | None = None if tempo is None else tempo.bpm
        self.beats: tuple[int, ...] = (2, 4) if tempo is None else tuple(tempo.beats)
        self.channels = bed.channels if bed is not None else 1
        self._offers: list[Offer | None] = [None] * slots
        self._lock = threading.Lock()
        self.generations = [-1] * slots
        self.buffers: list[list[Audio] | None] = [None] * slots
        self.sample: Audio | None = None
        self.played = [0] * slots  # hits per slot, which pick its take
        self.last_slot = -1
        self.voices: list[tuple[int, Audio]] = []  # (start sample, buffer)
        self.clock = 0  # index of the next output sample, counted from the first hit's bar
        self.started = False
        self.next_hit: int | None = None
        self.count = 0  # hits started
        self.render_hits = 0
        self.order: list[bool] = []
        self.log: list[tuple[int, int]] = []  # (slot, take) of each hit, the sample being slot −1

    @property
    def generation(self) -> int:
        """The generation of the first slot's buffers."""
        return self.generations[0]

    def offer(self, slot: int, generation: int, buffers: Sequence[Audio], sample: Audio | None = None) -> None:
        """Queue a slot's take buffers, and the sample matched to them, for that slot's next hit.

        An older generation never replaces a newer one.
        """
        with self._lock:
            current = self._offers[slot]
            if current is None or generation > current.generation:
                self._offers[slot] = Offer(generation, list(buffers), sample)

    def offered(self, slot: int) -> Offer | None:
        """Return the newest render offered to a slot, whether or not a hit has taken it yet."""
        with self._lock:
            return self._offers[slot]

    def _take_offers(self) -> None:
        """Take every offer newer than its slot's buffers."""
        with self._lock:
            for i, offered in enumerate(self._offers):
                if offered is not None and offered.generation > self.generations[i]:
                    self.generations[i], self.buffers[i] = offered.generation, offered.buffers
                    if i == 0:
                        self.sample = offered.sample

    def _hit_time(self, n: int, bpm: float) -> int:
        """Return where hit n starts: on the beat grid, counted from the first bar."""
        per_bar = len(self.beats)
        beat = (n // per_bar) * BEATS_PER_BAR + self.beats[n % per_bar] - 1
        return round(beat * self.rate * 60 / bpm)

    def _next_buffer(self) -> Audio:
        """Return the buffer the next hit plays, and log it."""
        if self.sample is not None:
            if self.count % 2 == 0:
                self.order.append(True if self.rng is None else self.rng.random() < FAIR)
            if (self.count % 2 == 0) != self.order[-1]:
                self.log.append((-1, 0))
                return self.sample
        ready = [i for i, b in enumerate(self.buffers) if b]
        slot = next((i for i in ready if i > self.last_slot), ready[0])
        self.last_slot = slot
        takes = self.buffers[slot] or []  # a ready slot has takes
        take = self.played[slot] % len(takes)
        self.played[slot] += 1
        self.log.append((slot, take))
        return takes[take]

    def fill(self, frames: int) -> Audio:
        """Mix the next `frames` output samples: the hits that start or still ring in them, and the bed."""
        if not self.started:
            self._take_offers()
            if not any(self.buffers):
                return np.zeros(frames if self.channels == 1 else (frames, self.channels), dtype=np.float32)
            self.started = True
            self.next_hit = self._hit_time(0, self.bpm) if self.bpm else 0
        out = np.zeros(frames, dtype=np.float32)
        end = self.clock + frames
        while self.next_hit is not None and self.next_hit < end:
            self._take_offers()
            buf = self._next_buffer()
            self.voices.append((self.next_hit, buf))
            self.count += 1
            if self.bpm:
                self.next_hit = self._hit_time(self.count, self.bpm)
            else:
                self.next_hit += buf.size + round(GAP_S * self.rate)
        playing = []
        for start, buf in self.voices:  # each still rings in this block, or starts in it
            a, b = max(self.clock, start), min(end, start + buf.size)
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
        path: Path_,
        work: Callable[[str], object],
        done: Done,
        text: str | None = None,
    ) -> None:
        """Watch a file, rendering with `work` and reporting to `done`; `text` is the version already rendered."""
        self.path, self.work, self.done = Path(path), work, done
        self.generation = 0
        self._stamp: tuple[int, int] | None = None
        self._text = text
        self._pending: tuple[int, str] | None = None
        self._cv = threading.Condition()
        threading.Thread(target=self._run, daemon=True).start()

    def poll(self) -> None:
        """Start a new generation if the file's content changed."""
        try:
            st = self.path.stat()
            stamp = (st.st_mtime_ns, st.st_size)
            if stamp == self._stamp:
                return
            text = read_text(self.path)
        except (OSError, UnicodeDecodeError):  # gone, or caught mid-save: the next look reads it whole
            return
        self._stamp = stamp
        if text == self._text:
            return
        self._text = text
        self.request(text)

    def request(self, text: str) -> None:
        """Render this text as a new generation."""
        with self._cv:
            self.generation += 1
            self._pending = (self.generation, text)  # an older request not yet started is superseded
            self._cv.notify()

    def _run(self) -> None:
        """Render each pending request, and report it."""
        while True:
            with self._cv:
                while self._pending is None:
                    self._cv.wait()
                generation, text = self._pending
                self._pending = None
            start = time.perf_counter()
            try:
                result = self.work(text)
            except FAILURES as e:  # reported by `done`; the last valid buffer keeps playing
                result = e
            elapsed = time.perf_counter() - start
            with self._cv:
                newest = generation == self.generation
            self.done(generation, result, elapsed, newest=newest)


class OverError(Exception):
    """A first render over full scale, which audition cannot play."""

    def __init__(self, path: Path, report: Report) -> None:
        """Say how far over the render is, and what to lower."""
        super().__init__(f"{path}: over full scale by {db(report.peak):.2f} dB: lower output.db to audition it")


class Session:
    """One audition: the hits, the renders behind them, and the watchers. `poll` is called about every 0.1 s."""

    def __init__(
        self,
        paths: Path_ | Sequence[Path_],
        settings: Settings = DEFAULTS,
        *,
        rng: Chance | None = None,
        out: TextIO | None = None,
    ) -> None:
        """Render every patch, load the bed and the sample, and start a watcher per patch."""
        self.paths = [Path(p) for p in ([paths] if isinstance(paths, (str, os.PathLike)) else paths)]
        self.out, self.takes, self.bpm = out or sys.stdout, settings.takes, settings.bpm
        if settings.against is not None and len(self.paths) > 1:
            msg = "--against compares one patch with a sample; give one patch"
            raise AuditionError(msg)
        if settings.blind and settings.against is None:
            msg = "--blind needs --against: a blind comparison needs a sample"
            raise AuditionError(msg)
        if settings.bed is not None and settings.bpm is None:
            msg = "--with needs --bpm, so the hits land on the bed's beats"
            raise AuditionError(msg)
        texts = [self._text(p) for p in self.paths]
        audio, report = self._render(texts[0])
        self._say(0, None, str(report))
        if report.over:
            raise OverError(self.paths[0], report)
        self.rate = report.sample_rate
        self.bed = (
            self._bed(settings.bed, settings.bpm) if settings.bed is not None and settings.bpm is not None else None
        )
        self.sample: Floats | None = None
        self.sample_lufs = -math.inf
        if settings.against is not None:
            self.sample, self.sample_lufs = self._sample(settings.against)
        self.blind = settings.blind
        self.hits = Hits(
            self.rate,
            len(self.paths),
            settings.tempo,
            bed=self.bed,
            rng=(rng or random.SystemRandom()) if settings.blind else None,
        )
        self.buffers: list[list[Audio] | None] = [None] * len(self.paths)
        self.out.write("".join(f"{note}\n" for note in self._accept(0, 0, audio, report)))
        self._pairs_said = 0
        self.watchers = [
            Watcher(p, self._render, self._done_for(i), t)
            for i, (p, t) in enumerate(zip(self.paths, texts, strict=True))
        ]
        # The first patch plays as soon as it is ready; the others render behind it.
        for i, text in enumerate(texts[1:], 1):
            self.watchers[i].request(text)
        self.watch = settings.watch

    def _text(self, path: Path) -> str:
        """Read a patch file's text, in UTF-8 or UTF-16."""
        try:
            return read_text(path)
        except OSError as e:
            msg = f"{path}: {e.strerror}"
            raise AuditionError(msg) from None
        except UnicodeDecodeError:
            msg = f"{path}: not UTF-8 or UTF-16 text"
            raise AuditionError(msg) from None

    def _render(self, text: str) -> Rendered:
        """Render a patch's takes as they would be written; an over render, never written, is left as it is."""
        audio, report = render_takes(loads(text), self.takes)
        return (audio if report.over else playable(audio, report)), report

    def _say(self, slot: int, generation: int | None, message: str) -> None:
        """Report about a slot, tagged by its file and its generation."""
        if generation is None:
            tag = "" if len(self.paths) == 1 else f"{self.paths[slot].name}\n"
            self.out.write(f"{tag}{message}\n")
            return
        tag = f"[{generation}]" if len(self.paths) == 1 else f"[{self.paths[slot].name} {generation}]"
        self.out.write(f"{tag} {message}\n")

    def _bed(self, path: Path_, bpm: float) -> Bed:
        """Load a bed: at the render's rate, mono or stereo, a whole number of bars at the tempo."""
        rate, frames = load_audio(path)
        if rate != self.rate:
            msg = f"{path}: {rate} Hz, but the patch renders at {self.rate} Hz; audition never resamples"
            raise AuditionError(msg)
        if frames.shape[1] > STEREO:
            msg = f"{path}: {frames.shape[1]} channels; a bed must be mono or stereo"
            raise AuditionError(msg)
        bar = self.rate * BEATS_PER_BAR * 60 / bpm
        bars = round(frames.shape[0] / bar)
        if bars < 1 or abs(frames.shape[0] - bars * bar) > BAR_TOLERANCE_S * self.rate:
            msg = (
                f"{path}: {frames.shape[0] / bar:.3f} bars at {bpm:g} bpm; "
                "a bed must be within 1 ms of a whole number of bars"
            )
            raise AuditionError(msg)
        self.out.write(
            f"{'bed':<9}{path}: {bars} bar{'s' if bars > 1 else ''} at {bpm:g} bpm, "
            f"{'stereo' if frames.shape[1] == STEREO else 'mono'}\n",
        )
        return Bed(frames, self.rate, bpm, bars)

    def _sample(self, path: Path_) -> tuple[Floats, float]:
        """Load the sample to compare against, at the render's rate, as its mid channel, and its loudness."""
        rate, frames = load_audio(path)
        if rate != self.rate:
            msg = f"{path}: {rate} Hz, but the patch renders at {self.rate} Hz; audition never resamples"
            raise AuditionError(msg)
        mid: Floats = frames.mean(axis=1)  # a stereo sample plays as its mid channel
        level = momentary_max(mid, rate)
        if level == -math.inf:
            msg = f"{path}: the sample is silent"
            raise AuditionError(msg)
        return mid, level

    def _matched(self, report: Report) -> tuple[Audio | None, list[str]]:
        """Return the sample matched to this render's loudness (its first take's), and what to say about it."""
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
        note = (
            f"{'sample':<9}plays at {gain:+.1f} dB, matched to the render's loudness "
            f"(the largest 400 ms momentary loudness, BS.1770){clip}"
        )
        return matched, [note]

    def _mix_warning(self) -> str | None:
        """Warn if the hits, with the bed under them, would clip the device over one pattern."""
        if not self.bpm or not any(self.buffers):
            return None
        probe = Hits(self.rate, len(self.buffers), Tempo(self.bpm, self.hits.beats), bed=self.bed)
        for i, b in enumerate(self.buffers):
            if b:
                probe.offer(i, 0, b)
        longest = max(x.size for b in self.buffers if b for x in b)
        bars = round(2 * BEATS_PER_BAR * self.rate * 60 / self.bpm)
        span = (self.bed.start(1) if self.bed is not None else bars) + longest
        total = sum(len(b) for b in self.buffers if b)
        peak = float(np.max(np.abs(probe.fill(span * max(1, total)))))
        if peak <= 1.0:
            return None
        under = "over the bed " if self.bed is not None else ""
        lower = "the bed or output.db" if self.bed is not None else "output.db"
        return f"warning: the hits {under}peak at {db(peak):+.1f} dBFS, which the device will clip; lower {lower}"

    def _accept(self, slot: int, generation: int, audio: list[Audio], report: Report) -> list[str]:
        """Offer a render to its slot, and return what to say about the sample and the mix."""
        matched, notes = self._matched(report) if slot == 0 else (None, [])
        self.buffers[slot] = audio
        self.hits.offer(slot, generation, audio, matched)
        warning = self._mix_warning()
        return notes + ([warning] if warning else [])

    def poll(self) -> None:
        """Look at the files when watching, and name each blind pair as it starts."""
        if self.watch:
            for w in self.watchers:
                w.poll()
        if self.blind:
            started = (self.hits.count + 1) // 2
            while self._pairs_said < started:
                self._pairs_said += 1
                self.out.write(f"pair {self._pairs_said}\n")

    def reveal(self) -> str | None:
        """Return the blind order, once audition stops: R is the render, S the sample."""
        if not self.blind:
            return None
        pairs = " ".join(f"{i + 1}:{'RS' if first else 'SR'}" for i, first in enumerate(self.hits.order))
        return f"blind order (R = render, S = sample): {pairs or 'no pair played'}"

    def _done_for(self, slot: int) -> Done:
        """Return what reports a finished render of a slot, and offers it when it is the newest and playable."""

        def done(generation: int, result: object, elapsed: float, *, newest: bool) -> None:
            self._say(slot, generation, self._outcome(slot, generation, result, elapsed, newest=newest))

        return done

    def _outcome(self, slot: int, generation: int, result: object, elapsed: float, *, newest: bool) -> str:
        """Say what became of a finished render, offering it to its slot when it can play."""
        if not newest:
            return "dropped: a newer edit is rendering"
        if isinstance(result, PatchError):
            return f"invalid patch, still playing the last valid one:\n{result}"
        if isinstance(result, BaseException):
            return f"render failed, still playing the last valid one: {result}"
        audio, report = cast("Rendered", result)
        if report.over:
            return f"over full scale by {db(report.peak):.2f} dB; lower output.db. Still playing the last valid buffer."
        if report.sample_rate != self.rate:
            return (
                f"sample rate {report.sample_rate} Hz, but audition plays at {self.rate} Hz; restart audition "
                "to hear it. Still playing the last valid buffer."
            )
        notes = self._accept(slot, generation, audio, report)
        return "\n".join([f"rendered in {1000 * elapsed:.0f} ms; plays at its next hit", str(report), *notes])


def audition(
    paths: Path_ | Sequence[Path_],
    settings: Settings = DEFAULTS,
    *,
    out: TextIO | None = None,
    stream_factory: Callable[[int, int, Fill], AbstractContextManager[object]] | None = None,
    rng: Chance | None = None,
) -> int:
    """Play the patches until interrupted; with `settings.watch`, re-render on every edit."""
    report = out or sys.stdout
    try:
        session = Session(paths, settings, rng=rng, out=report)
    except PatchError as e:
        sys.stderr.write("invalid patch\n" + "".join(f"  {m}\n" for m in e.problems))
        return 2
    except (AuditionError, RenderError) as e:
        sys.stderr.write(f"{e}\n")
        return 2
    except OverError as e:
        sys.stderr.write(f"{e}\n")
        return 1
    try:
        stream = (stream_factory or device_stream)(session.rate, session.hits.channels, session.hits.fill)
    except RuntimeError as e:
        sys.stderr.write(f"{e}\n")
        return 1
    try:
        with stream:
            while True:
                time.sleep(POLL_S)
                session.poll()
    except KeyboardInterrupt:
        return 0
    finally:
        revealed = session.reveal()
        if revealed:
            report.write(f"{revealed}\n")


def device_stream(rate: int, channels: int, fill: Fill) -> AbstractContextManager[object]:
    """Open an output stream at exactly `rate` that plays what `fill` mixes, or say why the device cannot."""
    try:
        # Imported here: importing sounddevice starts PortAudio, which only audition needs.
        sd = importlib.import_module("sounddevice")
    except (ImportError, OSError) as e:
        msg = f"audition needs the sounddevice package and PortAudio ({e})"
        raise RuntimeError(msg) from None
    try:
        sd.check_output_settings(samplerate=rate, channels=channels, dtype="float32")
    except (ValueError, sd.PortAudioError) as e:
        msg = f"the output device does not support {rate} Hz ({e}); audition never resamples"
        raise RuntimeError(msg) from None

    def callback(outdata: Audio, frames: int, _time: object, _status: object) -> None:
        """Fill the device's buffer with the next frames."""
        outdata[:] = fill(frames).reshape(frames, channels)

    stream: AbstractContextManager[object] = sd.OutputStream(
        samplerate=rate,
        channels=channels,
        dtype="float32",
        callback=callback,
    )
    return stream
