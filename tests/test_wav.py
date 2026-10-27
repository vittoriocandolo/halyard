# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

import json
import math
import struct
import warnings

import numpy as np
import pytest
import soundfile
from conftest import EXAMPLE, make
from scipy.io import wavfile

from halyard_synth import __version__, cli, noise, wav
from halyard_synth import render as R
from halyard_synth.patch import dumps, loads


def walk(path):
    """An independent chunk walker: (id, size, payload, offset) in file order."""
    data = open(path, "rb").read()
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE"
    assert struct.unpack_from("<I", data, 4)[0] == len(data) - 8
    pos, out = 12, []
    while pos < len(data):
        cid, size = data[pos : pos + 4], struct.unpack_from("<I", data, pos + 4)[0]
        out.append((cid, size, data[pos + 8 : pos + 8 + size], pos))
        pos += 8 + size + (size & 1)
    assert pos == len(data)
    return out


def export(path, p):
    audio, report = R.render(p)
    R.export(path, audio, p, report, report.takes[0])
    return audio, report


@pytest.fixture
def exported(tmp_path):
    p = loads(EXAMPLE)
    path = tmp_path / "snare.wav"
    audio, report = export(path, p)
    return path, audio, report


def test_independent_readers_get_the_exact_samples(exported):
    path, audio, _ = exported
    data, rate = soundfile.read(path, dtype="float32")
    assert rate == 48000 and np.array_equal(data, audio)
    info = soundfile.info(path)
    assert info.subtype == "FLOAT" and info.channels == 1 and info.frames == audio.size
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", wavfile.WavFileWarning)  # it does not know smpl or snar
        rate, data = wavfile.read(path)
    assert rate == 48000 and data.dtype == np.float32 and np.array_equal(data, audio)
    ours = wav.read(path)
    assert ours.rate == 48000 and np.array_equal(ours.samples, audio) and ours.channels == 1


def test_headers(exported):
    path, audio, _ = exported
    chunks = walk(path)
    assert [c[0] for c in chunks] == [b"fmt ", b"fact", b"smpl", b"snar", b"data"]
    fmt = chunks[0]
    assert fmt[1] == 18
    assert struct.unpack("<HHIIHHH", fmt[2]) == (3, 1, 48000, 192000, 4, 32, 0)
    assert struct.unpack("<I", chunks[1][2]) == (audio.size,)
    assert chunks[4][1] == 4 * audio.size


def test_smpl_chunk_maps_the_root_key(exported):
    path, _, _ = exported
    smpl = dict((c[0], c[2]) for c in walk(path))[b"smpl"]
    fields = struct.unpack("<9I", smpl)
    assert len(smpl) == 36
    assert fields[2] == 20833  # the sample period in ns at 48 kHz
    note, fraction = fields[3], fields[4]
    assert note == 54  # F#3
    assert note + fraction / 2**32 == pytest.approx(69 + 12 * np.log2(185 / 440), abs=1e-9)
    assert fields[7] == 0 and fields[8] == 0  # no loops, no sampler data


def test_a_silent_body_has_no_root_key(tmp_path):
    path = tmp_path / "clap.wav"
    export(path, loads(cli.reference_text("clap")))
    assert [c[0] for c in walk(path)] == [b"fmt ", b"fact", b"snar", b"data"]
    assert wav.read(path).root is None


@pytest.mark.parametrize(
    "hz, note, cents",
    [
        (440.0, 69, 0.0),
        (146.832, 50, 0.0),
        (185.0, 54, 0.03),
        (1800.0, 93, 38.9),
        (41.19, 27, 99.4),
        (4000.0, 107, 21.3),
    ],
)
def test_root_note(hz, note, cents):
    n, fraction = wav.root_note(hz)
    assert n == note
    assert 100 * fraction / 2**32 == pytest.approx(cents, abs=0.06)
    assert 0 <= fraction < 2**32


def test_provenance(exported):
    path, _, report = exported
    meta = wav.read(path).meta
    assert meta["patch"] == dumps(loads(EXAMPLE))
    assert loads(meta["patch"]) == loads(EXAMPLE)
    assert meta["renderer"] == f"halyard {__version__}"
    assert (meta["seed"], meta["take"], meta["takes"]) == (1, 1, 1)
    assert meta["set_gain_db"] == pytest.approx(-6.0) and meta["take_gain_db"] == 0.0
    assert meta["decimator"]["sha256"] == R.decimator(48000).sha256
    assert meta["decimator"]["internal_rate"] == 1536000 and meta["decimator"]["factor"] == 32
    assert {"python", "numpy", "scipy"} <= set(meta)
    raw = dict((c[0], c[2]) for c in walk(path))[b"snar"]
    assert json.loads(raw.decode("utf-8")) == meta


def test_each_take_records_its_own_patch_and_gains(tmp_path):
    p = make(**{"output.mode": "peak", "output.db": -1.0})
    audio, report = R.render_takes(p, 3, level_match=True)
    for a, take in zip(audio, report.takes):
        path = tmp_path / f"t{take.number}.wav"
        R.export(path, a, p, report, take)
        meta = wav.read(path).meta
        assert loads(meta["patch"]).seed == take.seed == p.seed + take.number - 1
        assert (meta["take"], meta["takes"]) == (take.number, 3)
        assert meta["set_gain_db"] == report.set_gain_db and meta["take_gain_db"] == take.gain_db
        # The file holds the take's gains: its patch rendered alone and scaled by them gives the same samples.
        s = R.synthesize(loads(meta["patch"]))
        gain = 10 ** ((meta["set_gain_db"] + meta["take_gain_db"]) / 20)
        np.testing.assert_allclose(wav.read(path).samples, (s.out * gain).astype(np.float32), rtol=1e-6, atol=1e-9)


def test_riff_padding(tmp_path):
    # An odd-length private chunk is padded so the data chunk stays word-aligned.
    for text in ("x", "xy", "π"):
        path = tmp_path / "pad.wav"
        samples = np.linspace(-1, 1, 7, dtype=np.float32)
        wav.write(path, samples, 44100, meta={"patch": text})
        chunks = walk(path)
        private = next(c for c in chunks if c[0] == b"snar")
        data = next(c for c in chunks if c[0] == b"data")
        assert data[3] % 2 == 0
        if private[1] % 2:
            raw = open(path, "rb").read()
            assert raw[private[3] + 8 + private[1]] == 0
        assert np.array_equal(soundfile.read(path, dtype="float32")[0], samples)
        assert wav.read(path).meta == {"patch": text}


def test_reads_beds_and_samples_but_rejects_other_formats(tmp_path):
    stereo = np.stack([np.linspace(-0.5, 0.5, 10), np.linspace(0.5, -0.5, 10)], axis=1)
    for subtype, bits in (("PCM_16", 16), ("PCM_24", 24), ("FLOAT", None)):
        for fmt in ("WAV", "WAVEX"):
            path = tmp_path / f"stereo-{subtype}-{fmt}.wav"
            soundfile.write(path, stereo, 44100, subtype=subtype, format=fmt)
            w = wav.read(path)
            assert w.channels == 2 and w.samples.shape == (10, 2) and w.bits == bits
            np.testing.assert_allclose(w.as_float(), soundfile.read(path)[0], atol=1e-9)
    for name, data, subtype in (
        ("pcm32", np.zeros(10), "PCM_32"),
        ("double", np.zeros(10), "DOUBLE"),
        ("u8", np.zeros(10), "PCM_U8"),
    ):
        path = tmp_path / f"{name}.wav"
        soundfile.write(path, data, 48000, subtype=subtype)
        with pytest.raises(ValueError):
            wav.read(path)
        assert b"fmt " in wav.chunks(path)
    with pytest.raises(ValueError):
        wav.write(tmp_path / "x.wav", np.zeros(4), 48000)  # float64 is not a format
    with pytest.raises(ValueError):
        wav.write(tmp_path / "x.wav", np.zeros(4, dtype=np.int32), 48000, bits=20)
    with pytest.raises(ValueError):
        wav.write(tmp_path / "x.wav", np.array([2**15], dtype=np.int32), 48000, bits=16)  # out of range


def test_every_rate_writes_a_valid_file(tmp_path):
    for rate in (44100, 48000, 88200, 96000):
        path = tmp_path / f"{rate}.wav"
        audio, _ = export(path, make(**{"output.sample_rate": rate}))
        data, sr = soundfile.read(path, dtype="float32")
        assert sr == rate and np.array_equal(data, audio)


@pytest.mark.parametrize("fmt, bits, subtype", [("pcm24", 24, "PCM_24"), ("pcm16", 16, "PCM_16")])
def test_pcm_exports_read_back_exactly(tmp_path, fmt, bits, subtype):
    import wave

    p = make(**{"output.format": fmt})
    path = tmp_path / f"{fmt}.wav"
    audio, report = export(path, p)
    expected = R.quantize(audio, bits, p.seed)
    info = soundfile.info(path)
    assert info.subtype == subtype and info.channels == 1 and info.frames == audio.size
    data, rate = soundfile.read(path, dtype="int32")  # left-justified in 32 bits
    assert rate == 48000 and np.array_equal(data >> (32 - bits), expected)
    with wave.open(str(path)) as w:
        assert (w.getsampwidth(), w.getnchannels(), w.getframerate()) == (bits // 8, 1, 48000)
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.uint8).reshape(-1, bits // 8).astype(np.int32)
    ints = sum(raw[:, i] << (8 * i) for i in range(bits // 8))
    assert np.array_equal((ints << (32 - bits)) >> (32 - bits), expected)
    ours = wav.read(path)
    assert ours.bits == bits and np.array_equal(ours.samples, expected)
    # The error is the dither plus the rounding, and the silent first sample stays zero.
    x = audio.astype(float) * 2 ** (bits - 1)
    assert np.max(np.abs(expected - x)) < 1.5 and expected[0] == 0
    assert [c[0] for c in walk(path)] == [b"fmt ", b"smpl", b"snar", b"data"]  # no fact chunk for PCM
    meta = ours.meta
    assert loads(meta["patch"]) == p
    assert meta["pcm"] == {
        "bits": bits,
        "dither": "TPDF, ±1 LSB; exact zeros kept",
        "dither_key": "snare/dither/v1",
        "seed": 1,
    }
    # The same patch writes the same bytes again.
    export(tmp_path / "again.wav", p)
    assert open(path, "rb").read() == open(tmp_path / "again.wav", "rb").read()


def test_exact_zeros_stay_zero():
    x = np.zeros(1000, dtype=np.float32)
    x[500:510] = 0.25
    q = R.quantize(x, 16, 1)
    assert np.all(q[:500] == 0) and np.all(q[510:] == 0) and np.all(np.abs(q[500:510] - 8192) <= 1)


def test_24_bit_packing_round_trips(tmp_path):
    values = np.array([-(2**23), 2**23 - 1, -1, 0, 1, 12345, -54321, 2**22], dtype=np.int32)
    path = tmp_path / "pack.wav"
    wav.write(path, values, 48000, bits=24)
    assert np.array_equal(wav.read(path).samples, values)
    assert np.array_equal(soundfile.read(path, dtype="int32")[0] >> 8, values)


@pytest.mark.parametrize("fraction", [0.0, 0.25, 0.5, 0.75])
def test_dither_leaves_no_noise_modulation(fraction):
    # With TPDF dither the total error has mean 0 and power 1/4 LSB² wherever the signal lies between
    # two steps: the first two moments of the error do not depend on the signal.
    x = np.full(200_000, (1000 + fraction) / 2**15, dtype=np.float32)
    e = R.quantize(x, 16, seed=3) - x.astype(float) * 2**15
    assert abs(e.mean()) < 0.006 and e.var() == pytest.approx(0.25, abs=0.006)


def test_dither_is_bit_identical_per_seed():
    x = (0.3 * np.sin(np.arange(10000) / 7)).astype(np.float32)
    assert np.array_equal(R.quantize(x, 16, 5), R.quantize(x, 16, 5))
    assert not np.array_equal(R.quantize(x, 16, 5), R.quantize(x, 16, 6))
    assert noise.DITHER_KEY == b"snare/dither/v1"


def test_pcm_keeps_room_for_dither():
    for fmt, bits in (("pcm16", 16), ("pcm24", 24)):
        p = make(**{"output.mode": "peak", "output.db": 0.0, "drive.db": 12.0, "output.format": fmt})
        audio, report = R.render(p)
        assert np.max(np.abs(audio)) <= R.ceiling(fmt) * (1 + 2**-24) and not report.over
        q = R.quantize(audio, bits, p.seed)
        assert q.max() <= 2 ** (bits - 1) - 1 and q.min() >= -(2 ** (bits - 1))
    # Float keeps its full-scale target; a gain-mode render between the ceiling and full scale is over for PCM.
    assert np.max(
        np.abs(R.render(make(**{"output.mode": "peak", "output.db": 0.0, "drive.db": 12.0}))[0])
    ) == pytest.approx(1.0)
    unity = np.max(np.abs(R.synthesize(make()).out))
    gain_db = 20 * math.log10((1 - 2**-15) / unity)  # half an LSB below full scale: fine as float, over as pcm16
    assert not R.render(make(**{"output.db": round(gain_db, 6)}))[1].over
    over = R.render(make(**{"output.db": round(gain_db, 6), "output.format": "pcm16"}))[1]
    assert over.over and "16-bit ceiling" in str(over)
