# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import soundfile

from halyard_synth import cli, wav
from halyard_synth.patch import dumps, loads
from halyard_synth.render import render, render_takes

from .conftest import EXAMPLE


def run(capsys, *args):
    code = cli.main([str(a) for a in args])
    out, err = capsys.readouterr()
    return code, out, err


@pytest.fixture
def example(tmp_path):
    path = tmp_path / "snare.toml"
    path.write_text(EXAMPLE)
    return path


@pytest.mark.parametrize("name", cli.REFERENCES)
def test_init_writes_complete_patches(capsys: pytest.CaptureFixture[str], name) -> None:
    code, out, _ = run(capsys, "init", name)
    assert code == 0
    assert out == dumps(loads(out))


def test_init_tunes_to_a_note_but_keeps_hertz(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(capsys, "init", "crack", "--tune", "D3")
    assert code == 0
    assert "tune_hz  = 146.832" in out
    assert loads(out).body.tune_hz == 146.832
    assert loads(run(capsys, "init", "tone", "--tune", "210.5")[1]).body.tune_hz == 210.5
    code, _, err = run(capsys, "init", "crack", "--tune", "H9")
    assert code == 2
    assert "not a note name" in err
    code, _, err = run(capsys, "init", "crack", "--tune", "C9")  # 8372 Hz is outside 40 … 4000 Hz
    assert code == 2
    assert "body.tune_hz" in err


def test_render_and_extract(capsys: pytest.CaptureFixture[str], tmp_path: Path, example) -> None:
    out = tmp_path / "snare.wav"
    code, text, _ = run(capsys, "render", example, "--out", out)
    assert code == 0
    assert "wrote" in text
    assert "F#3 +0.0 cents" in text
    data, rate = soundfile.read(out, dtype="float32")
    assert rate == 48000
    assert np.array_equal(data, render(loads(EXAMPLE))[0])
    code, text, _ = run(capsys, "extract", out)
    assert code == 0
    assert text == dumps(loads(EXAMPLE))


def test_render_writes_round_robin_takes(capsys: pytest.CaptureFixture[str], tmp_path: Path, example) -> None:
    out = tmp_path / "snare.wav"
    code, text, _ = run(capsys, "render", example, "--out", out, "--takes", 3, "--level-match")
    assert code == 0
    assert "3 takes sharing one gain" in text
    assert "level-matched" in text
    audio, _ = render_takes(loads(EXAMPLE), 3, level_match=True)
    for j in (1, 2, 3):
        path = tmp_path / f"snare_rr{j}.wav"
        assert np.array_equal(soundfile.read(path, dtype="float32")[0], audio[j - 1])
        # Each take's file holds its own patch: the patch with the take's seed.
        assert loads(run(capsys, "extract", path)[1]) == dataclasses.replace(loads(EXAMPLE), seed=j)
    assert not out.exists()
    code, _, err = run(capsys, "render", example, "--out", out, "--takes", 0)
    assert code == 2
    assert "--takes" in err


def test_render_refuses_overs_unless_allowed(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    patch = tmp_path / "loud.toml"
    patch.write_text(EXAMPLE.replace("db          = -6.0", "db          = 12.0"))
    out = tmp_path / "loud.wav"
    code, text, err = run(capsys, "render", patch, "--out", out)
    assert code == 1
    assert not out.exists()
    assert "not written" in err
    assert "Lower output.db by at least" in err
    code, _, _ = run(capsys, "render", patch, "--out", out, "--allow-over")
    assert code == 0
    assert np.max(np.abs(wav.read(out).samples)) > 1.0


def test_render_writes_pcm(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    for fmt, subtype in (("pcm24", "PCM_24"), ("pcm16", "PCM_16")):
        patch = tmp_path / f"{fmt}.toml"
        patch.write_text(EXAMPLE.replace('"float32"     #', f'"{fmt}"       #'))
        assert loads(patch.read_text()).output.format == fmt
        out = tmp_path / f"{fmt}.wav"
        code, text, _ = run(capsys, "render", patch, "--out", out)
        assert code == 0
        assert f"{fmt[3:]}-bit PCM" in text
        assert soundfile.info(out).subtype == subtype
        assert loads(run(capsys, "extract", out)[1]) == loads(patch.read_text())
        code, _, err = run(capsys, "render", patch, "--out", tmp_path / "x.wav", "--allow-over")
        assert code == 2
        assert "float32 only" in err
    loud = tmp_path / "loud.toml"
    loud.write_text(
        EXAMPLE.replace('"float32"     #', '"pcm24"       #').replace("db          = -6.0", "db          = 12.0"),
    )
    code, _, err = run(capsys, "render", loud, "--out", tmp_path / "loud.wav")
    assert code == 1
    assert "not written" in err
    assert "ceiling" in err
    assert not (tmp_path / "loud.wav").exists()


def test_errors_are_reported(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text(EXAMPLE.replace("damping  = 1.0", "damping  = 4.0").replace("seed = 1", "seed = -1"))
    code, _, err = run(capsys, "render", bad, "--out", tmp_path / "x.wav")
    assert code == 2
    assert "body.damping: 4 is outside 0 … 3" in err
    assert "seed:" in err
    code, _, err = run(capsys, "render", tmp_path / "missing.toml", "--out", "x.wav")
    assert code == 2
    assert "missing.toml" in err
    other = tmp_path / "plain.wav"
    soundfile.write(other, np.zeros(10), 48000)
    code, _, err = run(capsys, "extract", other)
    assert code == 2
    assert "no embedded snare patch" in err


def test_tune(capsys: pytest.CaptureFixture[str], example) -> None:
    code, out, _ = run(capsys, "tune", example, "D3")
    assert code == 0
    assert out == dumps(loads(out))
    assert loads(out).body.tune_hz == 146.832
    code, _, err = run(capsys, "tune", example, "H9")
    assert code == 2
    assert "cannot tune to H9" in err
    assert "not a note name" in err
    # A mode that the glide would take past 20 kHz is refused, never clamped.
    high = example.with_name("high.toml")
    high.write_text(EXAMPLE.replace("{ ratio = 2.136, level_db = -11.0 }", "{ ratio = 7.0,   level_db = -11.0 }"))
    assert run(capsys, "tune", high, "2400")[0] == 0
    code, _, err = run(capsys, "tune", high, "3000")
    assert code == 2
    assert "body.pitch[1].ratio" in err
    assert "for mode 3 is outside 10 Hz … 20 kHz" in err


def test_stretch(capsys: pytest.CaptureFixture[str], example) -> None:
    code, out, _ = run(capsys, "stretch", example, "all", "0.8")
    assert code == 0
    before, after = loads(EXAMPLE), loads(out)
    assert after.noise[1].amp[:2] == before.noise[1].amp[:2]  # the rise, up to the peak, is kept
    assert after.noise[1].amp[2][0] == pytest.approx(1.5 + 0.8 * 38.5)
    code, out, _ = run(capsys, "stretch", example, "body.pitch", "noise[1].amp", "2", "--after", "0.05")
    assert code == 0
    assert loads(out).body.pitch[-1][0] == 29.95
    assert loads(out).noise[0].amp[-1][0] == 15.95
    code, _, err = run(capsys, "stretch", example, "body.amp", "0.000001")
    assert code == 2
    assert "repeats the previous knot time" in err
    code, _, err = run(capsys, "stretch", example, "noise[3].amp", "2")
    assert code == 2
    assert "not a curve of this patch" in err


def test_bursts(capsys: pytest.CaptureFixture[str], example) -> None:
    code, out, _ = run(capsys, "bursts", example, "noise[1]", "--count", 4, "--spacing", 11)
    assert code == 0
    knots = loads(out).noise[0].amp
    assert len(knots) == 14
    assert knots[1] == (0.1, 1.0)
    assert knots[4] == (11.1, 1.0)
    code, _, err = run(capsys, "bursts", example, "noise[1]", "--count", 11, "--spacing", 11)
    assert code == 2
    assert "--count" in err
    code, _, err = run(capsys, "bursts", example, "noise[5]", "--count", 2, "--spacing", 11)
    assert code == 2
    assert "not a layer of this patch" in err


def test_softer(capsys: pytest.CaptureFixture[str], example, tmp_path: Path) -> None:
    code, out, _ = run(capsys, "softer", example, 12)
    assert code == 0
    q = loads(out)
    assert q.body.level_db == -16.0
    assert q.body.pitch[0] == (0.0, 1.005048)
    assert q.noise[0].band_hz[1] == 12137.2
    code, out, _ = run(capsys, "softer", example, 12, "--glide", 0, "--contact", 0)
    assert code == 0
    assert loads(out).body.pitch == loads(EXAMPLE).body.pitch
    code, _, err = run(capsys, "softer", example, 90)
    assert code == 2
    assert "cannot make it 90 dB softer" in err
    assert "-94 is outside" in err
    # Softer layers compose: 6 then 6 is 12, up to the decimal rounding it writes.
    six = tmp_path / "six.toml"
    six.write_text(run(capsys, "softer", example, 6)[1])
    twice = loads(run(capsys, "softer", six, 6)[1])
    assert twice.body.level_db == q.body.level_db
    assert abs(twice.noise[0].band_hz[1] - q.noise[0].band_hz[1]) <= 0.2


def test_the_report_explains_the_domain(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    patch = tmp_path / "driven.toml"
    patch.write_text(cli.reference_text("driven"))
    code, text, _ = run(capsys, "render", patch, "--out", tmp_path / "d.wav")
    assert code == 0
    assert "LUFS" in text
    assert "mostly kinks" in text
    assert "drive headroom +" in text


def test_verify(capsys: pytest.CaptureFixture[str], example) -> None:
    code, out, _ = run(capsys, "verify", example)
    assert code == 0
    assert "PASS" in out
    assert "E_rel" in out
    assert "6.144 and 12.288 MHz" in out
    code, out, _ = run(capsys, "verify", example, "--takes", 2)
    assert code == 0
    assert "take 2, seed 2" in out
    assert "2 of 2 takes pass" in out
