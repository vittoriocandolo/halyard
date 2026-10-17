import dataclasses

import numpy as np
import pytest
import soundfile
from conftest import EXAMPLE

from snare import cli, wav
from snare.patch import dumps, loads
from snare.render import render, render_takes


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
def test_init_writes_complete_patches(capsys, name):
    code, out, _ = run(capsys, "init", name)
    assert code == 0
    assert out == dumps(loads(out))


def test_init_tunes_to_a_note_but_keeps_hertz(capsys):
    code, out, _ = run(capsys, "init", "crack", "--tune", "D3")
    assert code == 0 and "tune_hz  = 146.832" in out
    assert loads(out).body.tune_hz == 146.832
    assert loads(run(capsys, "init", "tone", "--tune", "210.5")[1]).body.tune_hz == 210.5
    code, _, err = run(capsys, "init", "crack", "--tune", "H9")
    assert code == 2 and "not a note name" in err
    code, _, err = run(capsys, "init", "crack", "--tune", "C9")  # 8372 Hz is outside 40 … 4000 Hz
    assert code == 2 and "body.tune_hz" in err


def test_render_and_extract(capsys, tmp_path, example):
    out = tmp_path / "snare.wav"
    code, text, _ = run(capsys, "render", example, "--out", out)
    assert code == 0 and "wrote" in text and "F#3 +0.0 cents" in text
    data, rate = soundfile.read(out, dtype="float32")
    assert rate == 48000 and np.array_equal(data, render(loads(EXAMPLE))[0])
    code, text, _ = run(capsys, "extract", out)
    assert code == 0 and text == dumps(loads(EXAMPLE))


def test_render_writes_round_robin_takes(capsys, tmp_path, example):
    out = tmp_path / "snare.wav"
    code, text, _ = run(capsys, "render", example, "--out", out, "--takes", 3, "--level-match")
    assert code == 0 and "3 takes sharing one gain" in text and "level-matched" in text
    audio, _ = render_takes(loads(EXAMPLE), 3, level_match=True)
    for j in (1, 2, 3):
        path = tmp_path / f"snare_rr{j}.wav"
        assert np.array_equal(soundfile.read(path, dtype="float32")[0], audio[j - 1])
        # Each take's file holds its own patch: the patch with the take's seed.
        assert loads(run(capsys, "extract", path)[1]) == dataclasses.replace(loads(EXAMPLE), seed=j)
    assert not out.exists()
    code, _, err = run(capsys, "render", example, "--out", out, "--takes", 0)
    assert code == 2 and "--takes" in err


def test_render_refuses_overs_unless_allowed(capsys, tmp_path):
    patch = tmp_path / "loud.toml"
    patch.write_text(EXAMPLE.replace("db          = -6.0", "db          = 12.0"))
    out = tmp_path / "loud.wav"
    code, text, err = run(capsys, "render", patch, "--out", out)
    assert code == 1 and not out.exists()
    assert "not written" in err and "Lower output.db by at least" in err
    code, _, _ = run(capsys, "render", patch, "--out", out, "--allow-over")
    assert code == 0 and np.max(np.abs(wav.read(out).samples)) > 1.0


def test_render_writes_pcm(capsys, tmp_path):
    for fmt, subtype in (("pcm24", "PCM_24"), ("pcm16", "PCM_16")):
        patch = tmp_path / f"{fmt}.toml"
        patch.write_text(EXAMPLE.replace('"float32"     #', f'"{fmt}"       #'))
        assert loads(patch.read_text()).output.format == fmt
        out = tmp_path / f"{fmt}.wav"
        code, text, _ = run(capsys, "render", patch, "--out", out)
        assert code == 0 and f"{fmt[3:]}-bit PCM" in text and soundfile.info(out).subtype == subtype
        assert loads(run(capsys, "extract", out)[1]) == loads(patch.read_text())
        code, _, err = run(capsys, "render", patch, "--out", tmp_path / "x.wav", "--allow-over")
        assert code == 2 and "float32 only" in err
    loud = tmp_path / "loud.toml"
    loud.write_text(
        EXAMPLE.replace('"float32"     #', '"pcm24"       #').replace("db          = -6.0", "db          = 12.0")
    )
    code, _, err = run(capsys, "render", loud, "--out", tmp_path / "loud.wav")
    assert code == 1 and "not written" in err and "ceiling" in err and not (tmp_path / "loud.wav").exists()


def test_errors_are_reported(capsys, tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text(EXAMPLE.replace("damping  = 1.0", "damping  = 4.0").replace("seed = 1", "seed = -1"))
    code, _, err = run(capsys, "render", bad, "--out", tmp_path / "x.wav")
    assert code == 2 and "body.damping: 4 is outside 0 … 3" in err and "seed:" in err
    code, _, err = run(capsys, "render", tmp_path / "missing.toml", "--out", "x.wav")
    assert code == 2 and "missing.toml" in err
    other = tmp_path / "plain.wav"
    soundfile.write(other, np.zeros(10), 48000)
    code, _, err = run(capsys, "extract", other)
    assert code == 2 and "no embedded snare patch" in err


def test_tune(capsys, example):
    code, out, _ = run(capsys, "tune", example, "D3")
    assert code == 0 and out == dumps(loads(out)) and loads(out).body.tune_hz == 146.832
    code, _, err = run(capsys, "tune", example, "H9")
    assert code == 2 and "cannot tune to H9" in err and "not a note name" in err
    # A mode that the glide would take past 20 kHz is refused, never clamped.
    high = example.with_name("high.toml")
    high.write_text(EXAMPLE.replace("{ ratio = 2.136, level_db = -11.0 }", "{ ratio = 7.0,   level_db = -11.0 }"))
    assert run(capsys, "tune", high, "2400")[0] == 0
    code, _, err = run(capsys, "tune", high, "3000")
    assert code == 2 and "body.pitch[1].ratio" in err and "for mode 3 is outside 10 Hz … 20 kHz" in err


def test_stretch(capsys, example):
    code, out, _ = run(capsys, "stretch", example, "all", "0.8")
    assert code == 0
    before, after = loads(EXAMPLE), loads(out)
    assert after.noise[1].amp[:2] == before.noise[1].amp[:2]  # the rise, up to the peak, is kept
    assert after.noise[1].amp[2][0] == pytest.approx(1.5 + 0.8 * 38.5)
    code, out, _ = run(capsys, "stretch", example, "body.pitch", "noise[1].amp", "2", "--after", "0.05")
    assert code == 0 and loads(out).body.pitch[-1][0] == 29.95 and loads(out).noise[0].amp[-1][0] == 15.95
    code, _, err = run(capsys, "stretch", example, "body.amp", "0.000001")
    assert code == 2 and "repeats the previous knot time" in err
    code, _, err = run(capsys, "stretch", example, "noise[3].amp", "2")
    assert code == 2 and "not a curve of this patch" in err


def test_bursts(capsys, example):
    code, out, _ = run(capsys, "bursts", example, "noise[1]", "--count", 4, "--spacing", 11)
    assert code == 0
    knots = loads(out).noise[0].amp
    assert len(knots) == 14 and knots[1] == (0.1, 1.0) and knots[4] == (11.1, 1.0)
    code, _, err = run(capsys, "bursts", example, "noise[1]", "--count", 11, "--spacing", 11)
    assert code == 2 and "--count" in err
    code, _, err = run(capsys, "bursts", example, "noise[5]", "--count", 2, "--spacing", 11)
    assert code == 2 and "not a layer of this patch" in err


def test_softer(capsys, example, tmp_path):
    code, out, _ = run(capsys, "softer", example, 12)
    assert code == 0
    q = loads(out)
    assert q.body.level_db == -16.0 and q.body.pitch[0] == (0.0, 1.005048) and q.noise[0].band_hz[1] == 12137.2
    code, out, _ = run(capsys, "softer", example, 12, "--glide", 0, "--contact", 0)
    assert code == 0 and loads(out).body.pitch == loads(EXAMPLE).body.pitch
    code, _, err = run(capsys, "softer", example, 90)
    assert code == 2 and "cannot make it 90 dB softer" in err and "-94 is outside" in err
    # Softer layers compose: 6 then 6 is 12, up to the decimal rounding it writes.
    six = tmp_path / "six.toml"
    six.write_text(run(capsys, "softer", example, 6)[1])
    twice = loads(run(capsys, "softer", six, 6)[1])
    assert twice.body.level_db == q.body.level_db
    assert abs(twice.noise[0].band_hz[1] - q.noise[0].band_hz[1]) <= 0.2


def test_the_report_explains_the_domain(capsys, tmp_path):
    patch = tmp_path / "driven.toml"
    patch.write_text(cli.reference_text("driven"))
    code, text, _ = run(capsys, "render", patch, "--out", tmp_path / "d.wav")
    assert code == 0 and "LUFS" in text and "mostly kinks" in text and "drive headroom +" in text


def test_verify(capsys, example):
    code, out, _ = run(capsys, "verify", example)
    assert code == 0 and "PASS" in out and "E_rel" in out and "6.144 and 12.288 MHz" in out
    code, out, _ = run(capsys, "verify", example, "--takes", 2)
    assert code == 0 and "take 2, seed 2" in out and "2 of 2 takes pass" in out
