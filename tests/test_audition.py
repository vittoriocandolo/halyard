# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Audition: the hit scheduler, beds, samples and blind pairs, sessions, watching and the device (plan §7.5)."""

import io
import os
import sys
import threading
import time
import types
from collections.abc import Callable
from pathlib import Path
from typing import override

import numpy as np
import pytest
import soundfile
from numpy.typing import ArrayLike, NDArray

from halyard_synth import audition
from halyard_synth.loudness import momentary_max
from halyard_synth.patch import loads
from halyard_synth.render import quantize, render, render_takes

from .conftest import EXAMPLE

type Audio = NDArray[np.float32]

RATE = 1000  # a toy rate keeps the hit arithmetic readable
DEVICE = 48000  # the example's rate
GAP = 500  # a free-running hit repeats after its length and this many toy samples
BEAT = 500  # a beat at 120 bpm, in toy samples
STEREO = 2
MATCHED_DB = 0.01  # the sample is matched to the render's loudness within this
FLOAT = 1e-7  # float32 sums of a bed and a hit, to this
REFUSED, BAD_INPUT = 1, 2
NEW = 2.0  # the value of a second buffer, told apart from the first's 1


def play(hits: audition.Hits, frames: int, block: int) -> Audio:
    """Mix `frames` samples of hits, in blocks of `block`."""
    out = []
    while frames > 0:
        n = min(block, frames)
        out.append(hits.fill(n))
        frames -= n
    return np.concatenate(out)


def ones(n: int, value: float = 1.0) -> Audio:
    """Return a buffer of n equal samples."""
    return np.full(n, value, dtype=np.float32)


class Collect(io.StringIO):
    """A text stream that keeps every write, as audition reports one message a write."""

    def __init__(self, sink: list[str]) -> None:
        """Keep the writes in sink."""
        super().__init__()
        self.sink = sink

    @override
    def write(self, s: str) -> int:
        """Keep one write."""
        self.sink.append(s)
        return len(s)


# The hit scheduler.


def test_hits_repeat_after_each_hit_plus_half_a_second() -> None:
    """Repeat a hit, without a tempo, after its own length and half a second."""
    buf = np.arange(1, 301, dtype=np.float32)
    hits = audition.Hits(RATE)
    hits.offer(0, 0, [buf])
    out = play(hits, 2000, 64)
    period = 300 + GAP
    for start in (0, period, 2 * period):
        assert np.array_equal(out[start : start + 300], buf)
        assert np.all(out[start + 300 : start + period] == 0)


def test_nothing_plays_until_a_buffer_arrives() -> None:
    """Play silence until the first buffer is offered."""
    hits = audition.Hits(RATE)
    assert np.all(play(hits, 500, 100) == 0)
    hits.offer(0, 0, [ones(10)])
    out = play(hits, 100, 100)
    assert np.all(out[:10] == 1)
    assert np.all(out[10:] == 0)


def test_block_size_never_moves_a_hit() -> None:
    """Mix the same output whatever the device asks for at a time."""
    buf = np.random.default_rng(0).normal(size=333).astype(np.float32)
    outs = []
    for block in (1, 7, 64, 1000, 4096):
        hits = audition.Hits(RATE, tempo=audition.Tempo(97.0))
        hits.offer(0, 0, [buf])
        outs.append(play(hits, 5000, block))
    for o in outs[1:]:
        assert np.array_equal(o, outs[0])


def test_new_buffer_takes_over_at_the_next_hit() -> None:
    """Finish a playing hit with its own buffer, and play a new one from the next hit."""
    hits = audition.Hits(RATE)
    hits.offer(0, 0, [ones(300)])
    first = play(hits, 150, 50)
    hits.offer(0, 1, [ones(100, NEW)])  # arrives mid-hit
    out = np.concatenate((first, play(hits, 2000, 50)))
    assert np.all(out[:300] == 1.0)  # the playing hit finishes with its own buffer
    assert np.all(out[300:800] == 0.0)
    assert np.all(out[800:900] == NEW)  # the next hit plays the new buffer...
    assert np.all(out[900:1400] == 0.0)
    assert np.all(out[1400:1500] == NEW)  # ...and the period follows its length


def test_older_generations_never_replace_newer_ones() -> None:
    """Keep the newest generation offered, whatever the order the offers arrive in."""
    hits = audition.Hits(RATE)
    hits.offer(0, 2, [ones(10, NEW)])
    hits.offer(0, 1, [ones(10, 1.0)])
    assert np.all(play(hits, 10, 10) == NEW)


def test_the_backbeat_by_default_and_other_beats_on_request() -> None:
    """Hit on beats 2 and 4 by default, or on the beats asked for."""
    # At 120 bpm a beat is 500 samples and a bar 2000: the snare lands on beats 2 and 4.
    hits = audition.Hits(RATE, tempo=audition.Tempo(120.0))
    hits.offer(0, 0, [ones(1)])
    assert np.array_equal(np.flatnonzero(play(hits, 8000, 333)), [500, 1500, 2500, 3500, 4500, 5500, 6500, 7500])
    hits = audition.Hits(RATE, tempo=audition.Tempo(120.0, (1, 2, 3, 4)))
    hits.offer(0, 0, [ones(1)])
    assert np.array_equal(np.flatnonzero(play(hits, 3000, 256)), [0, 500, 1000, 1500, 2000, 2500])
    hits = audition.Hits(RATE, tempo=audition.Tempo(120.0, (3,)))  # half time
    hits.offer(0, 0, [ones(1)])
    assert np.array_equal(np.flatnonzero(play(hits, 6000, 999)), [1000, 3000, 5000])


def test_overlapping_tails_sum_and_the_grid_never_drifts() -> None:
    """Sum overlapping tails, and place every hit on the beat grid however long a beat is."""
    hits = audition.Hits(RATE, tempo=audition.Tempo(120.0, (1, 2, 3, 4)))
    hits.offer(0, 0, [ones(900)])
    out = play(hits, 2000, 128)
    assert np.all(out[:500] == 1)
    assert np.all(out[500:900] == 1 + 1)  # two hits ring together
    assert np.all(out[900:1000] == 1)
    # Fractional beat lengths never drift: beat b starts at round(b · rate · 60 / bpm).
    hits = audition.Hits(RATE, tempo=audition.Tempo(140.0, (1, 2, 3, 4)))
    hits.offer(0, 0, [ones(1)])
    out = play(hits, 20000, 333)
    assert np.array_equal(np.flatnonzero(out), [round(i * RATE * 60 / 140) for i in range(47)])


def test_patches_cycle_hit_by_hit_and_takes_cycle_within_each() -> None:
    """Cycle the patches hit by hit, and each patch's takes over its own hits."""
    hits = audition.Hits(RATE, 2, audition.Tempo(120.0, (1, 2, 3, 4)))
    hits.offer(0, 0, [ones(1, 1.0), ones(1, 2.0)])
    hits.offer(1, 0, [ones(1, 10.0), ones(1, 20.0), ones(1, 30.0)])
    out = play(hits, 6000, 250)
    played = [out[BEAT * i] for i in range(12)]
    assert played == [1, 10, 2, 20, 1, 30, 2, 10, 1, 20, 2, 30]
    assert hits.log[:4] == [(0, 0), (1, 0), (0, 1), (1, 1)]


def test_the_first_patch_plays_while_the_others_render() -> None:
    """Play the first patch alone until the others are ready."""
    hits = audition.Hits(RATE, 2, audition.Tempo(120.0, (1, 2, 3, 4)))
    hits.offer(0, 0, [ones(1, 1.0)])
    early = play(hits, 1500, 100)
    hits.offer(1, 0, [ones(1, 5.0)])
    late = play(hits, 2000, 100)
    assert [early[0], early[500], early[1000]] == [1, 1, 1]
    assert [late[0], late[500], late[1000], late[1500]] == [5, 1, 5, 1]


# Beds and samples.


def test_a_bed_loops_under_the_hits_on_the_tempo_grid() -> None:
    """Loop a bed under the hits, restarted on the grid, the same whatever the block size."""
    bpm = 120.0
    bed = (np.arange(1, 4001, dtype=np.float32) / 10000)[:, None]  # two bars, mono
    snare = ones(10, 0.5)
    outs = []
    for block in (1, 64, 333, 4096):
        hits = audition.Hits(RATE, tempo=audition.Tempo(bpm), bed=audition.Bed(bed, RATE, bpm, 2))
        hits.offer(0, 0, [snare])
        outs.append(play(hits, 12000, block))
    for o in outs[1:]:
        assert np.array_equal(o, outs[0])
    expected = np.tile(bed[:, 0], 3)
    for beat in (1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23):  # beats 2 and 4 of each bar
        expected[BEAT * beat : BEAT * beat + 10] += 0.5
    assert outs[0].ndim == 1
    assert np.allclose(outs[0], expected, atol=FLOAT)


def test_a_bed_a_little_short_of_whole_bars_never_drifts() -> None:
    """Restart a bed one sample short of two bars on the grid, leaving the gap at its end."""
    bed = np.ones((3999, 1), dtype=np.float32)  # one sample short of two bars at 120 bpm
    hits = audition.Hits(RATE, tempo=audition.Tempo(120.0), bed=audition.Bed(bed, RATE, 120.0, 2))
    hits.offer(0, 0, [np.zeros(1, dtype=np.float32)])
    out = play(hits, 12000, 256)
    # Each loop restarts on the grid: the missing sample is a gap at the end of every period.
    assert np.array_equal(np.flatnonzero(out == 0), [3999, 7999, 11999])


def test_a_stereo_bed_opens_a_stereo_stream_with_the_snare_centred() -> None:
    """Open a stereo stream for a stereo bed, the hits in its centre."""
    bed = np.stack([np.full(2000, 0.1), np.full(2000, -0.1)], axis=1).astype(np.float32)
    hits = audition.Hits(RATE, tempo=audition.Tempo(120.0), bed=audition.Bed(bed, RATE, 120.0, 1))
    snare = np.linspace(0, 1, 50, dtype=np.float32)
    hits.offer(0, 0, [snare])
    out = play(hits, 4000, 100)
    assert hits.channels == STEREO
    assert out.shape == (4000, STEREO)
    centre = out - np.tile(bed, (2, 1))
    assert np.allclose(centre[:, 0], centre[:, 1])
    assert np.allclose(centre[500:550, 0], snare)


def test_hits_alternate_between_the_render_and_the_sample() -> None:
    """Alternate the render and the sample hit by hit, the render first."""
    hits = audition.Hits(RATE)
    hits.offer(0, 0, [ones(100, 1.0)], ones(200, NEW))
    out = play(hits, 3000, 64)
    # Without a tempo each hit repeats after its own length plus 0.5 s: render, sample, render, …
    assert np.all(out[:100] == 1.0)
    assert np.all(out[600:800] == NEW)
    assert np.all(out[1300:1400] == 1.0)
    assert np.all(out[100:600] == 0.0)
    assert np.all(out[800:1300] == 0.0)
    assert hits.order == [True, True, True]


def test_blind_pairs_play_in_the_recorded_random_order() -> None:
    """Play each blind pair in a random order, which is recorded."""
    hits = audition.Hits(RATE, tempo=audition.Tempo(120.0, (1, 2, 3, 4)), rng=np.random.default_rng(7))
    hits.offer(0, 0, [ones(10, 1.0)], ones(10, 2.0))
    out = play(hits, BEAT * 40, 128)
    played = [out[BEAT * i] for i in range(40)]
    for pair, render_first in enumerate(hits.order):
        assert played[2 * pair : 2 * pair + 2] == ([1.0, 2.0] if render_first else [2.0, 1.0])
    assert 0 < sum(hits.order) < len(hits.order)  # both orders occur


# Sessions.


def write_wav(path: Path, data: ArrayLike, rate: int, subtype: str = "FLOAT") -> Path:
    """Write a WAV file with soundfile, and return its path."""
    soundfile.write(path, data, rate, subtype=subtype)
    return path


@pytest.fixture
def example(tmp_path: Path) -> Path:
    """Write the example patch to a file."""
    path = tmp_path / "snare.toml"
    path.write_text(EXAMPLE, encoding="utf-8")
    return path


def test_audition_plays_the_exact_render_buffer(example: Path) -> None:
    """Play the exact float32 render."""
    expected, _ = render(loads(EXAMPLE))
    session = audition.Session(example, out=Collect([]))
    assert session.rate == DEVICE
    heard = np.concatenate([session.hits.fill(512) for _ in range(40)])
    assert np.array_equal(heard[: expected.size], expected)


def test_pcm_patches_play_their_quantized_samples(tmp_path: Path) -> None:
    """Play a PCM patch as its file would hold it: dithered and quantized."""
    path = tmp_path / "pcm.toml"
    text = EXAMPLE.replace('"float32"     #', '"pcm16"       #')
    path.write_text(text, encoding="utf-8")
    audio, _ = render(loads(text))
    expected = quantize(audio, 16, 1) / 2.0**15
    heard = np.concatenate([audition.Session(path, out=Collect([])).hits.fill(20000)])
    np.testing.assert_array_equal(heard[: expected.size], expected.astype(np.float32))


def test_takes_cycle_in_a_session(example: Path) -> None:
    """Cycle a session's takes hit by hit."""
    audio, _ = render_takes(loads(EXAMPLE), 3)
    session = audition.Session(example, audition.Settings(takes=3), out=Collect([]))
    period = audio[0].size + 24000
    heard = np.concatenate([session.hits.fill(4800) for _ in range(40)])
    for j in range(3):
        start = j * period
        assert np.array_equal(heard[start : start + audio[j].size], audio[j])


def test_several_patches_render_behind_the_first(tmp_path: Path, example: Path) -> None:
    """Render the other patches behind the first, and play them in turn on the grid."""
    other = tmp_path / "other.toml"
    other.write_text(EXAMPLE.replace("tune_hz  = 185.0", "tune_hz  = 220.0"), encoding="utf-8")
    first, _ = render(loads(EXAMPLE))
    second, _ = render(loads(other.read_text(encoding="utf-8")))
    session = audition.Session([example, other], audition.Settings(bpm=120.0, beats=(1, 2, 3, 4)), out=Collect([]))
    wait_for(lambda: session.buffers[1] is not None)
    heard = np.concatenate([session.hits.fill(4800) for _ in range(20)])
    assert np.array_equal(heard[: first.size], first)  # beat 1: the first patch
    beat = 24000
    np.testing.assert_allclose(
        heard[beat : beat + second.size],
        second
        + np.concatenate((first[beat:], np.zeros(max(0, second.size - first.size + beat), np.float32)))[: second.size],
        atol=1e-6,
    )


def test_audition_runs_until_interrupted_and_reports_what_it_cannot_play(example: Path) -> None:
    """Run until interrupted, refuse an over and an invalid patch, and say when the device cannot play."""
    seen: dict[str, object] = {}

    class Stream:
        def __init__(self, rate: int, channels: int, fill: Callable[[int], Audio]) -> None:
            seen["rate"], seen["channels"], seen["first"] = rate, channels, fill(64)

        def __enter__(self) -> None:
            raise KeyboardInterrupt

        def __exit__(self, *exc: object) -> None:
            return None

    assert audition.audition([example], stream_factory=Stream, out=Collect([])) == 0
    assert seen["rate"] == DEVICE
    assert seen["channels"] == 1
    example.write_text(EXAMPLE.replace("db          = -6.0", "db          = 12.0"), encoding="utf-8")
    assert audition.audition([example], stream_factory=Stream, out=Collect([])) == REFUSED  # over full scale
    example.write_text(EXAMPLE.replace("damping  = 1.0", "damping  = 9.0"), encoding="utf-8")
    assert audition.audition([example], stream_factory=Stream, out=Collect([])) == BAD_INPUT  # invalid

    def unsupported(rate: int, _channels: int, _fill: Callable[[int], Audio]) -> Stream:
        msg = f"the output device does not support {rate} Hz"
        raise RuntimeError(msg)

    example.write_text(EXAMPLE, encoding="utf-8")
    assert audition.audition([example], stream_factory=unsupported, out=Collect([])) == REFUSED


def wait_for(condition: Callable[[], object], timeout: float = 30.0) -> None:
    """Wait until the condition holds, or fail after the timeout."""
    deadline = time.time() + timeout
    while not condition():
        assert time.time() < deadline, "timed out"
        time.sleep(0.02)


def said(log: list[str], generation: int) -> bool:
    """Say whether audition has reported on a generation: its messages start with the tag."""
    return any(chunk.startswith(f"[{generation}] ") for chunk in log)


def edit(path: Path, text: str) -> None:
    """Rewrite a file, its modification time a second later, so a watcher sees it within one clock tick."""
    # Bump the modification time so the change is seen even within one clock tick.
    path.write_text(text, encoding="utf-8")
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))


def test_watcher_renders_only_the_newest_version(tmp_path: Path) -> None:
    """Render only the newest version of a file, dropping those superseded before they start."""
    path = tmp_path / "snare.toml"
    path.write_text("v0", encoding="utf-8")
    gate = threading.Event()
    rendered: list[str] = []
    results: list[tuple[int, object, bool]] = []

    def work(text: str) -> str:
        rendered.append(text)
        gate.wait(5)
        return text

    def done(generation: int, result: object, _elapsed: float, *, newest: bool) -> None:
        results.append((generation, result, newest))

    w = audition.Watcher(path, work, done, text="v0")
    w.poll()
    assert w.generation == 0  # unchanged content starts nothing
    for i in range(1, 5):
        path.write_text(f"v{i}", encoding="utf-8")
        w.poll()
        time.sleep(0.05)
    gate.set()
    newest = 4
    wait_for(lambda: results and results[-1][0] == newest, 5)
    # v1 was already rendering; v2 and v3 were superseded before they started.
    assert rendered == ["v1", "v4"]
    assert results == [(1, "v1", False), (4, "v4", True)]


def test_watching_swaps_buffers_and_keeps_the_last_valid_one(example: Path) -> None:
    """Keep playing the last valid buffer through an invalid edit, and swap in a valid one at the next hit."""
    log: list[str] = []
    session = audition.Session(example, audition.Settings(watch=True), out=Collect(log))
    original = session.hits.fill(DEVICE // 2)  # the first hit starts playing
    assert session.hits.generation == 0

    edit(example, EXAMPLE.replace("damping  = 1.0", "damping  = 9.0"))
    session.poll()
    wait_for(lambda: said(log, 1))
    assert "[1] invalid patch, still playing the last valid one" in "".join(log)

    quieter = EXAMPLE.replace("db          = -6.0", "db          = -12.0")
    edit(example, quieter)
    session.poll()
    wait_for(lambda: said(log, 2))
    assert "[2] rendered in" in "".join(log)
    assert "plays at its next hit" in "".join(log)
    assert session.hits.generation == 0  # nothing changes until the next hit
    expected, _ = render(loads(quieter))
    rest = np.concatenate([session.hits.fill(1000) for _ in range(60)])
    heard = np.concatenate((original, rest))
    old, _ = render(loads(EXAMPLE))
    assert np.array_equal(heard[: old.size], old)
    start = old.size + 24000  # the next hit: the rendered length plus 0.5 s
    assert np.all(heard[old.size : start] == 0)
    assert np.array_equal(heard[start : start + expected.size], expected)
    assert session.hits.generation == 1 + 1  # the valid edit, after the invalid one


def test_a_session_matches_the_sample_and_plays_over_the_bed(tmp_path: Path, example: Path) -> None:
    """Match the sample to the render, and play the exact render over a stereo bed."""
    audio, report = render(loads(EXAMPLE))
    other, _ = render(loads(EXAMPLE.replace("tune_hz  = 185.0", "tune_hz  = 230.0")))
    sample = write_wav(tmp_path / "sample.wav", np.stack([other, other * 0.5], axis=1) * 0.3, DEVICE, "PCM_24")
    bed = write_wav(tmp_path / "bed.wav", np.full((96000, 2), 0.01), DEVICE)  # one bar at 120 bpm, stereo
    log: list[str] = []
    session = audition.Session(example, audition.Settings(bpm=120.0, bed=bed, against=sample), out=Collect(log))
    text = "".join(log)
    assert "bed      " in text
    assert "1 bar at 120 bpm, stereo" in text
    assert "sample   plays at" in text
    assert session.hits.channels == STEREO
    offer = session.hits.offered(0)
    assert offer is not None
    assert offer.sample is not None
    # The sample is scaled to the render, never the render to the sample.
    assert momentary_max(offer.sample, DEVICE) == pytest.approx(report.takes[0].loudness, abs=MATCHED_DB)
    heard = session.hits.fill(DEVICE)  # beat 2: the render, exact, over the bed
    assert np.allclose(heard[24000 : 24000 + audio.size, 0] - 0.01, audio, atol=FLOAT)
    assert np.allclose(heard[:24000], 0.01)


def test_the_sample_is_rematched_on_every_re_render(tmp_path: Path, example: Path) -> None:
    """Rematch the sample to every new render."""
    other, _ = render(loads(EXAMPLE.replace("tune_hz  = 185.0", "tune_hz  = 230.0")))
    sample = write_wav(tmp_path / "sample.wav", other, DEVICE)
    log: list[str] = []
    session = audition.Session(example, audition.Settings(watch=True, against=sample), out=Collect(log))
    first = session.hits.offered(0)
    assert first is not None
    before = first.sample
    assert before is not None
    quieter = EXAMPLE.replace("db          = -6.0", "db          = -18.0")
    edit(example, quieter)
    session.poll()
    wait_for(lambda: said(log, 1))
    second = session.hits.offered(0)
    assert second is not None
    after = second.sample
    assert after is not None
    assert 20 * np.log10(np.max(np.abs(before)) / np.max(np.abs(after))) == pytest.approx(12.0, abs=MATCHED_DB)


def test_a_session_refuses_what_it_cannot_play_faithfully(tmp_path: Path, example: Path) -> None:
    """Refuse beds and samples at another rate, beds off whole bars, a silent sample and settings that cannot work."""
    wrong_rate = write_wav(tmp_path / "44k.wav", np.full((88200, 1), 0.1), 44100)
    uneven = write_wav(tmp_path / "uneven.wav", np.full((144000, 1), 0.1), 48000)  # 1.5 bars at 120 bpm
    near = write_wav(tmp_path / "near.wav", np.full((96000 + 96, 1), 0.1), 48000)  # 2 ms past one bar
    silent = write_wav(tmp_path / "silent.wav", np.zeros(4800), 48000)
    for kwargs, message in (
        ({"bpm": 120.0, "bed": wrong_rate}, "never resamples"),
        ({"bpm": 120.0, "bed": uneven}, "1.500 bars at 120 bpm; a bed must be within 1 ms of a whole number of bars"),
        ({"bpm": 120.0, "bed": near}, "within 1 ms"),
        ({"bed": uneven}, "--with needs --bpm"),
        ({"against": wrong_rate}, "never resamples"),
        ({"against": silent}, "the sample is silent"),
        ({"blind": True}, "--blind needs --against"),
        ({"against": tmp_path / "missing.wav"}, "missing.wav"),
        ({"against": write_wav(tmp_path / "nan.wav", np.full(4800, np.nan), 48000)}, "not numbers or are infinite"),
        ({"bpm": 120.0, "bed": write_wav(tmp_path / "inf.wav", np.full(96000, np.inf), 48000)}, "are infinite"),
    ):
        with pytest.raises(audition.AuditionError, match=message):
            audition.Session(example, audition.Settings(**kwargs), out=Collect([]))
    with pytest.raises(audition.AuditionError, match="give one patch"):
        audition.Session([example, example], audition.Settings(against=silent), out=Collect([]))
    # 0.5 ms off whole bars is within tolerance.
    close = write_wav(tmp_path / "close.wav", np.full((96000 + 24, 1), 0.01), 48000)
    audition.Session(example, audition.Settings(bpm=120.0, bed=close), out=Collect([]))


def test_audition_reveals_the_blind_order_when_it_stops(tmp_path: Path, example: Path) -> None:
    """Reveal the blind order when audition stops."""
    other, _ = render(loads(EXAMPLE.replace("tune_hz  = 185.0", "tune_hz  = 230.0")))
    sample = write_wav(tmp_path / "sample.wav", other, 48000)
    log: list[str] = []

    class Stream:
        def __init__(self, _rate: int, _channels: int, fill: Callable[[int], Audio]) -> None:
            for _ in range(24):  # six seconds: twelve hits on 2 and 4 at 120 bpm
                fill(12000)

        def __enter__(self) -> None:
            raise KeyboardInterrupt

        def __exit__(self, *exc: object) -> None:
            return None

    code = audition.audition(
        [example],
        audition.Settings(bpm=120.0, against=sample, blind=True),
        stream_factory=Stream,
        out=Collect(log),
        rng=np.random.default_rng(3),
    )
    text = "".join(log)
    assert code == 0
    assert "blind order (R = render, S = sample): 1:" in text
    pairs = 3  # six hits on the backbeat in six seconds
    assert text.count(":RS") + text.count(":SR") == pairs


def test_clipping_is_flagged(tmp_path: Path, example: Path) -> None:
    """Warn when the hits over a bed, or a matched sample, would clip the device."""
    loud = write_wav(tmp_path / "loud.wav", np.full((96000, 1), 0.9), 48000)
    quiet = write_wav(tmp_path / "quiet.wav", np.full((96000, 1), 0.01), 48000)
    log: list[str] = []
    audition.Session(example, audition.Settings(bpm=120.0, bed=loud), out=Collect(log))
    assert "warning: the hits over the bed peak at" in "".join(log)
    log.clear()
    audition.Session(example, audition.Settings(bpm=120.0, bed=quiet), out=Collect(log))
    assert "warning" not in "".join(log)
    # A quiet sample matched to a loud render would clip the device.
    other, _ = render(loads(EXAMPLE.replace("tune_hz  = 185.0", "tune_hz  = 230.0")))
    faint = write_wav(tmp_path / "faint.wav", other * 0.01 + np.pad([0.5], (0, other.size - 1)), 48000)
    log.clear()
    audition.Session(example, audition.Settings(against=faint), out=Collect(log))
    assert "which the device will clip" in "".join(log)


def test_a_watcher_ignores_an_unchanged_or_unreadable_file(tmp_path: Path) -> None:
    """Start nothing when a file is unchanged since the last look, or cannot be read."""
    path = tmp_path / "snare.toml"
    path.write_text("v0", encoding="utf-8")
    requests: list[str] = []

    def done(generation: int, result: object, elapsed: float, *, newest: bool) -> None:
        requests.append(f"{generation} {result} {elapsed > 0} {newest}")

    w = audition.Watcher(path, lambda text: text, done)
    w.poll()  # the first look reads it
    wait_for(lambda: requests)
    w.poll()  # unchanged: not even read
    path.unlink()
    w.poll()  # gone: nothing to read
    assert w.generation == 1


def test_a_finished_render_is_reported_for_what_it_is(example: Path) -> None:
    """Report a dropped render, a failed one, one over full scale and one at another rate, playing on."""
    log: list[str] = []
    session = audition.Session(example, out=Collect(log))
    done = session.watchers[0].done
    done(1, None, 0.1, newest=False)
    done(2, ValueError("a bug"), 0.1, newest=True)
    done(3, render_takes(loads(EXAMPLE.replace("db          = -6.0", "db          = 12.0"))), 0.1, newest=True)
    done(4, render_takes(loads(EXAMPLE.replace("sample_rate = 48000", "sample_rate = 44100"))), 0.1, newest=True)
    text = "".join(log)
    assert "[1] dropped: a newer edit is rendering" in text
    assert "[2] render failed, still playing the last valid one: a bug" in text
    assert "[3] over full scale by" in text
    assert "[4] sample rate 44100 Hz, but audition plays at 48000 Hz" in text
    offer = session.hits.offered(0)
    assert offer is not None
    assert offer.generation == 0  # none of them is offered: the first render plays on


def test_an_over_pcm_patch_is_refused_as_over(tmp_path: Path) -> None:
    """Refuse a PCM patch over full scale as over, as a float one is, before any device opens."""
    path = tmp_path / "loud.toml"
    loud = EXAMPLE.replace('"float32"     #', '"pcm16"       #').replace("db          = -6.0", "db          = 12.0")
    path.write_text(loud, encoding="utf-8")

    def no_device(rate: int, channels: int, fill: Callable[[int], Audio]) -> io.StringIO:
        raise AssertionError((rate, channels, fill))

    assert audition.audition([path], stream_factory=no_device, out=Collect([])) == REFUSED


def test_a_session_refuses_a_missing_patch_and_a_bed_of_three_channels(tmp_path: Path, example: Path) -> None:
    """Refuse a patch file that is not there, and a bed of more than two channels."""
    assert audition.audition([tmp_path / "missing.toml"], out=Collect([])) == BAD_INPUT
    surround = write_wav(tmp_path / "surround.wav", np.full((96000, 3), 0.01), DEVICE)
    with pytest.raises(audition.AuditionError, match="3 channels; a bed must be mono or stereo"):
        audition.Session(example, audition.Settings(bpm=120.0, bed=surround), out=Collect([]))


def test_blind_pairs_are_named_as_they_start(tmp_path: Path, example: Path) -> None:
    """Name each blind pair when it starts, and poll no file when not watching."""
    other, _ = render(loads(EXAMPLE.replace("tune_hz  = 185.0", "tune_hz  = 230.0")))
    sample = write_wav(tmp_path / "sample.wav", other, DEVICE)
    log: list[str] = []
    settings = audition.Settings(bpm=120.0, against=sample, blind=True)
    session = audition.Session(example, settings, rng=np.random.default_rng(1), out=Collect(log))
    session.hits.fill(4 * DEVICE)  # two bars: four hits, two pairs
    session.poll()
    assert "".join(log).count("pair ") == len(session.hits.order)


def test_audition_polls_until_interrupted(monkeypatch: pytest.MonkeyPatch, example: Path) -> None:
    """Poll the session between sleeps, until a keyboard interrupt stops it."""
    naps: list[float] = []
    polls = 3

    def nap(seconds: float) -> None:
        naps.append(seconds)
        if len(naps) == polls:
            raise KeyboardInterrupt

    class Stream:
        def __init__(self, _rate: int, _channels: int, _fill: Callable[[int], Audio]) -> None:
            pass

        def __enter__(self) -> None:
            return None

        def __exit__(self, *exc: object) -> None:
            return None

    monkeypatch.setattr(time, "sleep", nap)
    assert audition.audition([example], audition.Settings(watch=True), stream_factory=Stream, out=Collect([])) == 0
    assert naps == [audition.POLL_S] * polls


class RefusedError(Exception):
    """The fake device's PortAudio error."""


def fake_sounddevice(rates: set[int], opened: list[dict[str, object]]) -> types.SimpleNamespace:
    """Make a stand-in for sounddevice that accepts these rates and records the streams it opens."""

    def check_output_settings(*, samplerate: int, channels: int, dtype: str) -> None:
        if samplerate not in rates:
            msg = f"invalid sample rate {samplerate} for {channels} channels of {dtype}"
            raise RefusedError(msg)

    def output_stream(**settings: object) -> dict[str, object]:
        opened.append(settings)
        return settings

    return types.SimpleNamespace(
        check_output_settings=check_output_settings,
        PortAudioError=RefusedError,
        OutputStream=output_stream,
    )


def test_the_device_stream_plays_the_mix_or_says_why_it_cannot(monkeypatch: pytest.MonkeyPatch) -> None:
    """Open a stream at exactly the rate whose callback plays the mix; say so when PortAudio or the rate is missing."""
    opened: list[dict[str, object]] = []
    device = fake_sounddevice({DEVICE}, opened)

    monkeypatch.setitem(sys.modules, "sounddevice", device)
    audition.device_stream(DEVICE, STEREO, lambda frames: np.full((frames, STEREO), 0.25, dtype=np.float32))
    (settings,) = opened
    assert (settings["samplerate"], settings["channels"], settings["dtype"]) == (DEVICE, STEREO, "float32")
    outdata = np.zeros((8, STEREO), dtype=np.float32)
    callback = settings["callback"]
    assert callable(callback)
    callback(outdata, 8, None, None)
    assert np.all(outdata == np.float32(0.25))
    with pytest.raises(RuntimeError, match="does not support 44100 Hz"):
        audition.device_stream(44100, 1, lambda frames: np.zeros(frames, dtype=np.float32))

    monkeypatch.setitem(sys.modules, "sounddevice", None)  # as if it were not installed
    with pytest.raises(RuntimeError, match="needs the sounddevice package and PortAudio"):
        audition.device_stream(DEVICE, 1, lambda frames: np.zeros(frames, dtype=np.float32))


def test_a_patch_that_is_not_text_is_refused_and_a_half_saved_one_waited_for(tmp_path: Path) -> None:
    """Refuse a patch that is not UTF-8 or UTF-16 at the start, and wait while a watched one cannot be read."""
    path = tmp_path / "binary.toml"
    path.write_bytes(b"instrument = \xff\n")
    with pytest.raises(audition.AuditionError, match="not UTF-8 or UTF-16 text"):
        audition.Session(path, out=Collect([]))
    path.write_text(EXAMPLE, encoding="utf-16")
    session = audition.Session(path, audition.Settings(watch=True), out=Collect([]))
    edit_bytes = b"instrument = \xff\n"
    path.write_bytes(edit_bytes)
    session.poll()  # caught mid-save: nothing starts
    assert session.watchers[0].generation == 0


def test_a_stopped_watcher_ends_and_reports_nothing_more(tmp_path: Path) -> None:
    """Stop a watcher mid-render: its thread ends, the render is not reported, and nothing new starts."""
    path = tmp_path / "snare.toml"
    path.write_text("v0", encoding="utf-8")
    rendering, release = threading.Event(), threading.Event()
    reported: list[int] = []

    def work(text: str) -> str:
        rendering.set()
        assert release.wait(5)
        return text

    def done(generation: int, _result: object, _elapsed: float, *, newest: bool) -> None:
        reported.append(generation if newest else -generation)

    w = audition.Watcher(path, work, done)
    w.request("v1")
    assert rendering.wait(5)
    w.stop()
    w.request("v2")  # ignored once stopped
    release.set()
    w.thread.join(5)
    assert not w.thread.is_alive()
    assert reported == []
    idle = audition.Watcher(path, work, done)
    idle.stop()  # with nothing pending
    idle.thread.join(5)
    assert not idle.thread.is_alive()


def test_audition_stops_its_watchers_when_it_ends(example: Path) -> None:
    """End every watcher's thread when audition stops."""

    def watchers() -> set[threading.Thread]:
        return {t for t in threading.enumerate() if t.name.endswith("(_run)")}

    before = watchers()

    class Stream:
        def __init__(self, _rate: int, _channels: int, _fill: Callable[[int], Audio]) -> None:
            pass

        def __enter__(self) -> None:
            raise KeyboardInterrupt

        def __exit__(self, *exc: object) -> None:
            return None

    assert audition.audition([example, example], stream_factory=Stream, out=Collect([])) == 0
    wait_for(lambda: watchers() <= before, 5)
