# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""The WAV files: read by independent readers, their chunks, root key, provenance, PCM and dither (docs/design.md)."""

import json
import math
import os
import re
import struct
import warnings
import wave
from pathlib import Path
from typing import NoReturn

import numpy as np
import pytest
import soundfile
from numpy.typing import NDArray
from scipy.io import wavfile

from halyard_synth import __version__, cli, noise, render, wav
from halyard_synth.patch import SAMPLE_RATES, Patch, dumps, loads

from .conftest import EXAMPLE, make

type Audio = NDArray[np.float32]
type Chunk = tuple[bytes, int, bytes, int]  # (id, size, payload, offset)

RATE = 48000  # the example's rate
FMT_FLOAT = 18  # a float fmt chunk's size, with its cbSize
SMPL = 36  # a smpl chunk's size, without loops
PERIOD_NS = 20833  # the sample period at 48 kHz, in ns
F_SHARP_3 = 54  # the example's tuning, 185 Hz, as a MIDI note
INTERNAL, FACTOR = 1536000, 32  # the 48 kHz family's internal rate and factor
CHANNELS = 2  # a stereo bed
QUANTIZATION = 1.5  # a PCM error is at most the dither's 1 LSB and half an LSB of rounding
MOMENT = 0.006  # the dithered error's mean and variance agree with TPDF's to this, over 200,000 samples
CENTS = 0.06  # the smpl fraction is the pitch above the note within this


def walk(path: Path) -> list[Chunk]:
    """Walk the chunks independently: (id, size, payload, offset) in file order."""
    data = path.read_bytes()
    assert data[:4] == b"RIFF"
    assert data[8:12] == b"WAVE"
    assert struct.unpack_from("<I", data, 4)[0] == len(data) - 8
    pos, out = 12, []
    while pos < len(data):
        cid, size = data[pos : pos + 4], struct.unpack_from("<I", data, pos + 4)[0]
        out.append((cid, size, data[pos + 8 : pos + 8 + size], pos))
        pos += 8 + size + (size & 1)
    assert pos == len(data)
    return out


def export(path: Path, p: Patch) -> tuple[Audio, render.Report]:
    """Render a patch's take and write it."""
    audio, report = render.render(p)
    render.export(path, audio, p, report, report.takes[0])
    return audio, report


@pytest.fixture
def exported(tmp_path: Path) -> tuple[Path, Audio, render.Report]:
    """Write the example's take, and return its file, its audio and its report."""
    p = loads(EXAMPLE)
    path = tmp_path / "snare.wav"
    audio, report = export(path, p)
    return path, audio, report


def test_independent_readers_get_the_exact_samples(exported: tuple[Path, Audio, render.Report]) -> None:
    """Read the example's file with soundfile, scipy.io.wavfile and our reader, and find the exact samples."""
    path, audio, _ = exported
    data, rate = soundfile.read(path, dtype="float32")
    assert rate == RATE
    assert np.array_equal(data, audio)
    info = soundfile.info(path)
    assert info.subtype == "FLOAT"
    assert info.channels == 1
    assert info.frames == audio.size
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", wavfile.WavFileWarning)  # it does not know smpl or snar
        rate, data = wavfile.read(path)
    assert rate == RATE
    assert data.dtype == np.float32
    assert np.array_equal(data, audio)
    ours = wav.read(path)
    assert ours.rate == RATE
    assert np.array_equal(ours.samples, audio)
    assert ours.channels == 1


def test_headers(exported: tuple[Path, Audio, render.Report]) -> None:
    """Walk the chunks: fmt, fact, smpl, the private chunk and data, with float32 headers."""
    path, audio, _ = exported
    chunks = walk(path)
    assert [c[0] for c in chunks] == [b"fmt ", b"fact", b"smpl", b"snar", b"data"]
    fmt = chunks[0]
    assert fmt[1] == FMT_FLOAT
    assert struct.unpack("<HHIIHHH", fmt[2]) == (3, 1, 48000, 192000, 4, 32, 0)
    assert struct.unpack("<I", chunks[1][2]) == (audio.size,)
    assert chunks[4][1] == 4 * audio.size


def test_smpl_chunk_maps_the_root_key(exported: tuple[Path, Audio, render.Report]) -> None:
    """Find the smpl chunk's period, unity note F#3 and pitch fraction, and no loops."""
    path, _, _ = exported
    smpl = {c[0]: c[2] for c in walk(path)}[b"smpl"]
    fields = struct.unpack("<9I", smpl)
    assert len(smpl) == SMPL
    assert fields[2] == PERIOD_NS
    note, fraction = fields[3], fields[4]
    assert note == F_SHARP_3
    assert note + fraction / 2**32 == pytest.approx(69 + 12 * np.log2(185 / 440), abs=1e-9)
    assert fields[7] == 0
    assert fields[8] == 0


def test_a_silent_body_has_no_root_key(tmp_path: Path) -> None:
    """Write a clap, whose body is silent, without a smpl chunk."""
    path = tmp_path / "clap.wav"
    export(path, loads(cli.preset_text("clap")))
    assert [c[0] for c in walk(path)] == [b"fmt ", b"fact", b"snar", b"data"]
    assert wav.read(path).root is None


@pytest.mark.parametrize(
    ("hz", "note", "cents"),
    [
        (440.0, 69, 0.0),
        (146.832, 50, 0.0),
        (185.0, 54, 0.03),
        (1800.0, 93, 38.9),
        (41.19, 27, 99.4),
        (4000.0, 107, 21.3),
    ],
)
def test_root_note(hz: float, note: int, cents: float) -> None:
    """Find the unity note at or below the pitch, half a cent flat counting as the note, and the fraction above it."""
    n, fraction = wav.root_note(hz)
    assert n == note
    assert 100 * fraction / 2**32 == pytest.approx(cents, abs=CENTS)
    assert 0 <= fraction < 2**32


def test_provenance(exported: tuple[Path, Audio, render.Report]) -> None:
    """Find the patch, the renderer, the take, the gains, the decimator and the libraries recorded in the file."""
    path, _, _report = exported
    meta = wav.read(path).meta
    assert meta is not None
    assert meta["patch"] == dumps(loads(EXAMPLE))
    assert loads(meta["patch"]) == loads(EXAMPLE)
    assert meta["renderer"] == f"halyard {__version__}"
    assert (meta["seed"], meta["take"], meta["takes"]) == (1, 1, 1)
    assert meta["set_gain_db"] == pytest.approx(-6.0)
    assert meta["take_gain_db"] == 0.0
    assert meta["decimator"]["sha256"] == render.decimator(48000).sha256
    assert meta["decimator"]["internal_rate"] == INTERNAL
    assert meta["decimator"]["factor"] == FACTOR
    assert {"python", "numpy", "scipy"} <= set(meta)
    raw = {c[0]: c[2] for c in walk(path)}[b"snar"]
    assert json.loads(raw.decode("utf-8")) == meta


def test_each_take_records_its_own_patch_and_gains(tmp_path: Path) -> None:
    """Find each take's file holding its own seed and gains, which render its samples again."""
    p = make(**{"output.mode": "peak", "output.db": -1.0})
    audio, report = render.render_takes(p, 3, level_match=True)
    for a, take in zip(audio, report.takes, strict=True):
        path = tmp_path / f"t{take.number}.wav"
        render.export(path, a, p, report, take)
        meta = wav.read(path).meta
        assert meta is not None
        assert loads(meta["patch"]).seed == take.seed == p.seed + take.number - 1
        assert (meta["take"], meta["takes"]) == (take.number, len(audio))
        assert meta["set_gain_db"] == report.set_gain_db
        assert meta["take_gain_db"] == take.gain_db
        # The file holds the take's gains: its patch rendered alone and scaled by them gives the same samples.
        s = render.synthesize(loads(meta["patch"]))
        gain = 10 ** ((meta["set_gain_db"] + meta["take_gain_db"]) / 20)
        np.testing.assert_allclose(wav.read(path).samples, (s.out * gain).astype(np.float32), rtol=1e-6, atol=1e-9)


def test_riff_padding(tmp_path: Path) -> None:
    """Pad an odd-length private chunk, so the data chunk stays word-aligned and every reader reads it."""
    # An odd-length private chunk is padded so the data chunk stays word-aligned.
    for text in ("x", "xy", "π"):
        path = tmp_path / "pad.wav"
        samples = np.linspace(-1, 1, 7, dtype=np.float32)
        wav.write(path, samples, wav.Format(44100), meta={"patch": text})
        chunks = walk(path)
        private = next(c for c in chunks if c[0] == b"snar")
        data = next(c for c in chunks if c[0] == b"data")
        assert data[3] % 2 == 0
        if private[1] % 2:
            raw = path.read_bytes()
            assert raw[private[3] + 8 + private[1]] == 0
        assert np.array_equal(soundfile.read(path, dtype="float32")[0], samples)
        assert wav.read(path).meta == {"patch": text}


def test_reads_beds_and_samples_but_rejects_other_formats(tmp_path: Path) -> None:
    """Read stereo beds and samples as columns, plain or extensible, and refuse formats audition cannot play."""
    stereo = np.stack([np.linspace(-0.5, 0.5, 10), np.linspace(0.5, -0.5, 10)], axis=1)
    for subtype, bits in (("PCM_16", 16), ("PCM_24", 24), ("FLOAT", None)):
        for fmt in ("WAV", "WAVEX"):
            path = tmp_path / f"stereo-{subtype}-{fmt}.wav"
            soundfile.write(path, stereo, 44100, subtype=subtype, format=fmt)
            w = wav.read(path)
            assert w.channels == CHANNELS
            assert w.samples.shape == (10, CHANNELS)
            assert w.bits == bits
            np.testing.assert_allclose(w.as_float(), soundfile.read(path)[0], atol=1e-9)
    for name, data, subtype in (
        ("pcm32", np.zeros(10), "PCM_32"),
        ("double", np.zeros(10), "DOUBLE"),
        ("u8", np.zeros(10), "PCM_U8"),
    ):
        path = tmp_path / f"{name}.wav"
        soundfile.write(path, data, 48000, subtype=subtype)
        with pytest.raises(ValueError, match="expected 32-bit float or 16/24-bit PCM"):
            wav.read(path)
        assert b"fmt " in wav.chunks(path)
    with pytest.raises(ValueError, match="float32"):
        wav.write(tmp_path / "x.wav", np.zeros(4), wav.Format(48000))  # float64 is not a format
    with pytest.raises(ValueError, match="16 or 24 bits"):
        wav.write(tmp_path / "x.wav", np.zeros(4, dtype=np.int32), wav.Format(48000, 20))
    with pytest.raises(ValueError, match="out of range"):
        wav.write(tmp_path / "x.wav", np.array([2**15], dtype=np.int32), wav.Format(48000, 16))  # out of range


def test_every_rate_writes_a_valid_file(tmp_path: Path) -> None:
    """Write a file at every rate that soundfile reads back exactly."""
    for rate in SAMPLE_RATES:
        path = tmp_path / f"{rate}.wav"
        audio, _ = export(path, make(**{"output.sample_rate": rate}))
        data, sr = soundfile.read(path, dtype="float32")
        assert sr == rate
        assert np.array_equal(data, audio)


@pytest.mark.parametrize(("fmt", "bits", "subtype"), [("pcm24", 24, "PCM_24"), ("pcm16", 16, "PCM_16")])
def test_pcm_exports_read_back_exactly(tmp_path: Path, fmt: str, bits: int, subtype: str) -> None:
    """Read 24- and 16-bit PCM back exactly with three readers: the dithered samples, chunks and dither's record."""
    p = make(**{"output.format": fmt})
    path = tmp_path / f"{fmt}.wav"
    audio, _report = export(path, p)
    expected = render.quantize(audio, bits, p.seed)
    info = soundfile.info(path)
    assert info.subtype == subtype
    assert info.channels == 1
    assert info.frames == audio.size
    data, rate = soundfile.read(path, dtype="int32")  # left-justified in 32 bits
    assert rate == RATE
    assert np.array_equal(data >> (32 - bits), expected)
    with wave.open(str(path)) as w:
        assert (w.getsampwidth(), w.getnchannels(), w.getframerate()) == (bits // 8, 1, RATE)
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.uint8).reshape(-1, bits // 8).astype(np.int32)
    ints = sum(raw[:, i] << (8 * i) for i in range(bits // 8))
    assert np.array_equal((ints << (32 - bits)) >> (32 - bits), expected)
    ours = wav.read(path)
    assert ours.bits == bits
    assert np.array_equal(ours.samples, expected)
    # The error is the dither plus the rounding, and the silent first sample stays zero.
    x = audio.astype(float) * 2 ** (bits - 1)
    assert np.max(np.abs(expected - x)) < QUANTIZATION
    assert expected[0] == 0
    assert [c[0] for c in walk(path)] == [b"fmt ", b"smpl", b"snar", b"data"]  # no fact chunk for PCM
    meta = ours.meta
    assert meta is not None
    assert loads(meta["patch"]) == p
    assert meta["pcm"] == {
        "bits": bits,
        "dither": "TPDF, ±1 LSB; exact zeros kept",
        "dither_key": "snare/dither/v1",
        "seed": 1,
    }
    # The same patch writes the same bytes again.
    export(tmp_path / "again.wav", p)
    assert path.read_bytes() == (tmp_path / "again.wav").read_bytes()


def test_exact_zeros_stay_zero() -> None:
    """Keep exact zeros zero when dithering, and dither the rest by at most 1 LSB."""
    x = np.zeros(1000, dtype=np.float32)
    x[500:510] = 0.25
    q = render.quantize(x, 16, 1)
    assert np.all(q[:500] == 0)
    assert np.all(q[510:] == 0)
    assert np.all(np.abs(q[500:510] - 8192) <= 1)


def test_24_bit_packing_round_trips(tmp_path: Path) -> None:
    """Pack and read 24-bit samples across their whole range."""
    values = np.array([-(2**23), 2**23 - 1, -1, 0, 1, 12345, -54321, 2**22], dtype=np.int32)
    path = tmp_path / "pack.wav"
    wav.write(path, values, wav.Format(48000, 24))
    assert np.array_equal(wav.read(path).samples, values)
    assert np.array_equal(soundfile.read(path, dtype="int32")[0] >> 8, values)


@pytest.mark.parametrize("fraction", [0.0, 0.25, 0.5, 0.75])
def test_dither_leaves_no_noise_modulation(fraction: float) -> None:
    """Find the dithered error's mean 0 and power ¼ LSB² wherever the signal lies between two steps."""
    # With TPDF dither the total error has mean 0 and power 1/4 LSB² wherever the signal lies between
    # two steps: the first two moments of the error do not depend on the signal.
    x = np.full(200_000, (1000 + fraction) / 2**15, dtype=np.float32)
    e = render.quantize(x, 16, seed=3) - x.astype(float) * 2**15
    assert abs(e.mean()) < MOMENT
    assert e.var() == pytest.approx(0.25, abs=MOMENT)


def test_dither_is_bit_identical_per_seed() -> None:
    """Dither the same per seed and differently between seeds, under its own key."""
    x = (0.3 * np.sin(np.arange(10000) / 7)).astype(np.float32)
    assert np.array_equal(render.quantize(x, 16, 5), render.quantize(x, 16, 5))
    assert not np.array_equal(render.quantize(x, 16, 5), render.quantize(x, 16, 6))
    assert noise.DITHER_KEY == b"snare/dither/v1"


def test_pcm_keeps_room_for_dither() -> None:
    """Keep PCM 2 LSB below full scale in peak mode, and call a gain-mode render over the PCM ceiling over."""
    for fmt, bits in (("pcm16", 16), ("pcm24", 24)):
        p = make(**{"output.mode": "peak", "output.db": 0.0, "drive.db": 12.0, "output.format": fmt})
        audio, report = render.render(p)
        assert np.max(np.abs(audio)) <= render.ceiling(fmt) * (1 + 2**-24)
        assert not report.over
        q = render.quantize(audio, bits, p.seed)
        assert q.max() <= 2 ** (bits - 1) - 1
        assert q.min() >= -(2 ** (bits - 1))
    # Float keeps its full-scale target; a gain-mode render between the ceiling and full scale is over for PCM.
    assert np.max(
        np.abs(render.render(make(**{"output.mode": "peak", "output.db": 0.0, "drive.db": 12.0}))[0]),
    ) == pytest.approx(1.0)
    unity = np.max(np.abs(render.synthesize(make()).out))
    gain_db = 20 * math.log10((1 - 2**-15) / unity)  # half an LSB below full scale: fine as float, over as pcm16
    assert not render.render(make(**{"output.db": round(gain_db, 6)}))[1].over
    over = render.render(make(**{"output.db": round(gain_db, 6), "output.format": "pcm16"}))[1]
    assert over.over
    assert "16-bit ceiling" in str(over)


def riff(*chunks: tuple[bytes, bytes]) -> bytes:
    """Build a RIFF WAVE file from its chunks."""
    body = b"".join(cid + struct.pack("<I", len(data)) + data + b"\0" * (len(data) % 2) for cid, data in chunks)
    return b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body


def test_every_file_the_reader_refuses_is_named(tmp_path: Path) -> None:
    """Refuse what is not RIFF WAVE, a file without fmt or data, a frame not of whole bytes, and a file cut short."""
    fmt = struct.pack("<HHIIHH", wav.PCM, 1, RATE, 2 * RATE, 2, 16)
    odd_bits = struct.pack("<HHIIHH", wav.PCM, 1, RATE, 2 * RATE, 2, 12)
    no_channels = struct.pack("<HHIIHH", wav.PCM, 0, RATE, 0, 0, 16)
    no_rate = struct.pack("<HHIIHH", wav.PCM, 1, 0, 0, 2, 16)
    for name, data, message in (
        ("text.wav", b"not a wave file", "not a RIFF WAVE file"),
        ("short.wav", b"RIFF", "not a RIFF WAVE file"),
        ("aiff.wav", b"FORM\0\0\0\4AIFF", "not a RIFF WAVE file"),
        ("nofmt.wav", riff((b"data", b"\0\0")), "missing fmt or data chunk"),
        ("nodata.wav", riff((b"fmt ", fmt)), "missing fmt or data chunk"),
        ("12bit.wav", riff((b"fmt ", odd_bits), (b"data", b"\0\0")), "unsupported format, 12-bit samples in 1 channel"),
        ("empty.wav", riff((b"fmt ", no_channels), (b"data", b"")), "unsupported format, 16-bit samples in 0 channels"),
        ("0hz.wav", riff((b"fmt ", no_rate), (b"data", b"\0\0")), "16-bit samples in 1 channel at 0 Hz"),
        ("cutfmt.wav", riff((b"fmt ", fmt[:2]), (b"data", b"\0\0")), "a fmt or smpl chunk is cut short"),
        (
            "cutframe.wav",
            riff((b"fmt ", fmt), (b"data", b"\0\0\0")),
            "the data chunk ends inside a frame: 3 bytes, in frames of 2",
        ),
        (
            "cutfile.wav",
            riff((b"fmt ", fmt))[:-4],
            "the 'fmt ' chunk is cut short: it declares 16 bytes, and 12 remain",
        ),
        ("cutsmpl.wav", riff((b"fmt ", fmt), (b"data", b"\0\0"), (b"smpl", b"\0")), "a fmt or smpl chunk is cut short"),
        ("garbled.wav", riff((b"fmt ", fmt), (b"data", b"\0\0"), (wav.PRIVATE, b"{")), "the embedded patch is damaged"),
        ("deep.wav", riff((b"fmt ", fmt), (b"data", b"\0\0"), (wav.PRIVATE, b"[" * 100_000)), "patch is damaged"),
        ("list.wav", riff((b"fmt ", fmt), (b"data", b"\0\0"), (wav.PRIVATE, b"[1]")), "the embedded patch is damaged"),
    ):
        path = tmp_path / name
        path.write_bytes(data)
        with pytest.raises(ValueError, match=re.escape(message)):
            wav.read(path)
    with pytest.raises(ValueError, match="expected a mono buffer"):
        wav.write(tmp_path / "x.wav", np.zeros((4, 2), dtype=np.float32), wav.Format(RATE))


def test_a_last_chunk_may_lack_its_pad_byte(tmp_path: Path) -> None:
    """Read a file whose last chunk, of odd size, ends without the pad byte that would hold nothing."""
    path = tmp_path / "odd.wav"
    path.write_bytes(riff((b"abcd", b"xyz"))[:-1])
    assert wav.chunks(path) == {b"abcd": b"xyz"}


def test_save_replaces_a_file_whole_or_not_at_all(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace a file, then interrupt the next replacement, which leaves the file and no temporary behind."""
    path = tmp_path / "hit.wav"
    path.write_bytes(b"the old file")
    wav.save(path, b"the new file")
    assert path.read_bytes() == b"the new file"

    def interrupted(_src: object, _dst: object) -> NoReturn:
        raise KeyboardInterrupt

    monkeypatch.setattr(os, "replace", interrupted)
    with pytest.raises(KeyboardInterrupt):
        wav.save(path, b"a file that never lands")
    assert path.read_bytes() == b"the new file"
    assert [p.name for p in tmp_path.iterdir()] == ["hit.wav"]


def test_save_names_the_file_asked_for_when_it_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Report a failed replacement under the file's own name, not its temporary's."""
    path = tmp_path / "out.wav"

    def refused(self: Path, _target: object) -> NoReturn:
        raise PermissionError(13, "Permission denied", str(self))

    monkeypatch.setattr(Path, "replace", refused)
    with pytest.raises(PermissionError) as caught:
        wav.save(path, b"data")
    assert caught.value.filename == os.path.realpath(path)
