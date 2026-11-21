# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""The command line: every command, its output, and what it refuses (plan §7.6)."""

import dataclasses
import io
import runpy
import sys
from importlib.metadata import version
from pathlib import Path
from typing import override

import numpy as np
import pytest
import soundfile

from halyard_synth import cli, wav
from halyard_synth.patch import dumps, loads
from halyard_synth.render import render, render_takes

from .conftest import EXAMPLE

BAD_INPUT, REFUSED = 2, 1  # the exit statuses of bad input, and of a refusal
RATE = 48000
D3 = 146.832  # D3 in Hz, as `--tune D3` writes it
TAKES = 3
BURST_KNOTS = 14  # four bursts: 3N + 2 knots
PITCH_END_MS = 29.95  # the pitch's last knot, 15 ms, stretched twice after 0.05 ms, as decimals
STICK_END_MS = 15.95  # and the stick's, 8 ms


def run(capsys: pytest.CaptureFixture[str], *args: object) -> tuple[int, str, str]:
    """Run a command, and return its status, its output and its errors."""
    code = cli.main([str(a) for a in args])
    out, err = capsys.readouterr()
    return code, out, err


@pytest.fixture
def example(tmp_path: Path) -> Path:
    """Write the example patch to a file."""
    path = tmp_path / "snare.toml"
    path.write_text(EXAMPLE, encoding="utf-8")
    return path


@pytest.mark.parametrize("name", cli.REFERENCES)
def test_init_writes_complete_patches(capsys: pytest.CaptureFixture[str], name: str) -> None:
    """Print every reference as a complete, canonical patch."""
    code, out, _ = run(capsys, "init", name)
    assert code == 0
    assert out == dumps(loads(out))


def test_init_tunes_to_a_note_but_keeps_hertz(capsys: pytest.CaptureFixture[str]) -> None:
    """Tune a reference to a note or a frequency, writing Hz, and refuse what is not a note or out of range."""
    code, out, _ = run(capsys, "init", "crack", "--tune", "D3")
    assert code == 0
    assert "tune_hz  = 146.832" in out
    assert loads(out).body.tune_hz == D3
    hz = 210.5
    assert loads(run(capsys, "init", "tone", "--tune", str(hz))[1]).body.tune_hz == hz
    code, _, err = run(capsys, "init", "crack", "--tune", "H9")
    assert code == BAD_INPUT
    assert "not a note name" in err
    code, _, err = run(capsys, "init", "crack", "--tune", "C9")  # 8372 Hz is outside 40 … 4000 Hz
    assert code == BAD_INPUT
    assert "body.tune_hz" in err


def test_render_and_extract(capsys: pytest.CaptureFixture[str], tmp_path: Path, example: Path) -> None:
    """Render a patch to a file whose samples are the render, and extract the patch from it again."""
    out = tmp_path / "snare.wav"
    code, text, _ = run(capsys, "render", example, "--out", out)
    assert code == 0
    assert "wrote" in text
    assert "F#3 +0.0 cents" in text
    data, rate = soundfile.read(out, dtype="float32")
    assert rate == RATE
    assert np.array_equal(data, render(loads(EXAMPLE))[0])
    code, text, _ = run(capsys, "extract", out)
    assert code == 0
    assert text == dumps(loads(EXAMPLE))


def test_render_writes_round_robin_takes(capsys: pytest.CaptureFixture[str], tmp_path: Path, example: Path) -> None:
    """Write a set's takes as _rr files, each holding its own seed's patch, and refuse no takes."""
    out = tmp_path / "snare.wav"
    code, text, _ = run(capsys, "render", example, "--out", out, "--takes", TAKES, "--level-match")
    assert code == 0
    assert "3 takes sharing one gain" in text
    assert "level-matched" in text
    audio, _ = render_takes(loads(EXAMPLE), TAKES, level_match=True)
    for j in (1, 2, 3):
        path = tmp_path / f"snare_rr{j}.wav"
        assert np.array_equal(soundfile.read(path, dtype="float32")[0], audio[j - 1])
        # Each take's file holds its own patch: the patch with the take's seed.
        assert loads(run(capsys, "extract", path)[1]) == dataclasses.replace(loads(EXAMPLE), seed=j)
    assert not out.exists()
    code, _, err = run(capsys, "render", example, "--out", out, "--takes", 0)
    assert code == BAD_INPUT
    assert "--takes" in err


def test_render_refuses_overs_unless_allowed(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """Refuse a gain-mode render over full scale, saying what to lower, unless overs are allowed in float."""
    patch = tmp_path / "loud.toml"
    patch.write_text(EXAMPLE.replace("db          = -6.0", "db          = 12.0"), encoding="utf-8")
    out = tmp_path / "loud.wav"
    code, _text, err = run(capsys, "render", patch, "--out", out)
    assert code == REFUSED
    assert not out.exists()
    assert "not written" in err
    assert "Lower output.db by at least" in err
    code, _, _ = run(capsys, "render", patch, "--out", out, "--allow-over")
    assert code == 0
    assert np.max(np.abs(wav.read(out).samples)) > 1


def test_render_writes_pcm(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """Write 24- and 16-bit PCM, never with overs, and refuse a PCM render over its ceiling."""
    for fmt, subtype in (("pcm24", "PCM_24"), ("pcm16", "PCM_16")):
        patch = tmp_path / f"{fmt}.toml"
        patch.write_text(EXAMPLE.replace('"float32"     #', f'"{fmt}"       #'), encoding="utf-8")
        assert loads(patch.read_text(encoding="utf-8")).output.format == fmt
        out = tmp_path / f"{fmt}.wav"
        code, text, _ = run(capsys, "render", patch, "--out", out)
        assert code == 0
        assert f"{fmt[3:]}-bit PCM" in text
        assert soundfile.info(out).subtype == subtype
        assert loads(run(capsys, "extract", out)[1]) == loads(patch.read_text(encoding="utf-8"))
        code, _, err = run(capsys, "render", patch, "--out", tmp_path / "x.wav", "--allow-over")
        assert code == BAD_INPUT
        assert "float32 only" in err
    loud = tmp_path / "loud.toml"
    loud.write_text(
        EXAMPLE.replace('"float32"     #', '"pcm24"       #').replace("db          = -6.0", "db          = 12.0"),
        encoding="utf-8",
    )
    code, _, err = run(capsys, "render", loud, "--out", tmp_path / "loud.wav")
    assert code == REFUSED
    assert "not written" in err
    assert "ceiling" in err
    assert not (tmp_path / "loud.wav").exists()


def test_errors_are_reported(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """Report every problem of an invalid patch, a missing file, and a WAV with no patch in it."""
    bad = tmp_path / "bad.toml"
    invalid = EXAMPLE.replace("damping  = 1.0", "damping  = 4.0").replace("seed = 1", "seed = -1")
    bad.write_text(invalid, encoding="utf-8")
    code, _, err = run(capsys, "render", bad, "--out", tmp_path / "x.wav")
    assert code == BAD_INPUT
    assert "body.damping: 4 is outside 0 … 3" in err
    assert "seed:" in err
    code, _, err = run(capsys, "render", tmp_path / "missing.toml", "--out", "x.wav")
    assert code == BAD_INPUT
    assert "missing.toml" in err
    other = tmp_path / "plain.wav"
    soundfile.write(other, np.zeros(10), 48000)
    code, _, err = run(capsys, "extract", other)
    assert code == BAD_INPUT
    assert "no embedded snare patch" in err


def test_tune(capsys: pytest.CaptureFixture[str], example: Path) -> None:
    """Retune a patch, refuse what is not a note, and refuse a tuning that would take a mode past 20 kHz."""
    code, out, _ = run(capsys, "tune", example, "D3")
    assert code == 0
    assert out == dumps(loads(out))
    assert loads(out).body.tune_hz == D3
    code, _, err = run(capsys, "tune", example, "H9")
    assert code == BAD_INPUT
    assert "cannot tune to H9" in err
    assert "not a note name" in err
    # A mode that the glide would take past 20 kHz is refused, never clamped.
    high = example.with_name("high.toml")
    seventh = EXAMPLE.replace("{ ratio = 2.136, level_db = -11.0 }", "{ ratio = 7.0,   level_db = -11.0 }")
    high.write_text(seventh, encoding="utf-8")
    assert run(capsys, "tune", high, "2400")[0] == 0
    code, _, err = run(capsys, "tune", high, "3000")
    assert code == BAD_INPUT
    assert "body.pitch[1].ratio" in err
    assert "for mode 3 is outside 10 Hz … 20 kHz" in err


def test_stretch(capsys: pytest.CaptureFixture[str], example: Path) -> None:
    """Stretch every shape after its peak, or chosen curves after a time, and refuse what cannot be."""
    code, out, _ = run(capsys, "stretch", example, "all", "0.8")
    assert code == 0
    before, after = loads(EXAMPLE), loads(out)
    assert after.noise[1].amp[:2] == before.noise[1].amp[:2]  # the rise, up to the peak, is kept
    assert after.noise[1].amp[2][0] == pytest.approx(1.5 + 0.8 * 38.5)
    code, out, _ = run(capsys, "stretch", example, "body.pitch", "noise[1].amp", "2", "--after", "0.05")
    assert code == 0
    assert loads(out).body.pitch[-1][0] == PITCH_END_MS
    assert loads(out).noise[0].amp[-1][0] == STICK_END_MS
    code, _, err = run(capsys, "stretch", example, "body.amp", "0.000001")
    assert code == BAD_INPUT
    assert "repeats the previous knot time" in err
    code, _, err = run(capsys, "stretch", example, "noise[3].amp", "2")
    assert code == BAD_INPUT
    assert "not a curve of this patch" in err


def test_bursts(capsys: pytest.CaptureFixture[str], example: Path) -> None:
    """Rewrite a layer as a clap train, and refuse too many bursts or a layer the patch lacks."""
    code, out, _ = run(capsys, "bursts", example, "noise[1]", "--count", 4, "--spacing", 11)
    assert code == 0
    knots = loads(out).noise[0].amp
    assert len(knots) == BURST_KNOTS
    assert knots[1] == (0.1, 1.0)
    assert knots[4] == (11.1, 1.0)
    code, _, err = run(capsys, "bursts", example, "noise[1]", "--count", 11, "--spacing", 11)
    assert code == BAD_INPUT
    assert "--count" in err
    code, _, err = run(capsys, "bursts", example, "noise[5]", "--count", 2, "--spacing", 11)
    assert code == BAD_INPUT
    assert "not a layer of this patch" in err


def test_softer(capsys: pytest.CaptureFixture[str], example: Path, tmp_path: Path) -> None:
    """Print the same drum struck softer, by each rule or none, refusing too much, and composing."""
    code, out, _ = run(capsys, "softer", example, 12)
    assert code == 0
    q = loads(out)
    assert q.body.level_db == -4.0 - 12
    assert q.body.pitch[0] == (0.0, 1.005048)
    assert q.noise[0].band_hz[1] == round(16000 * 10 ** (-12 / 20 * 0.2), 1)
    code, out, _ = run(capsys, "softer", example, 12, "--glide", 0, "--contact", 0)
    assert code == 0
    assert loads(out).body.pitch == loads(EXAMPLE).body.pitch
    code, _, err = run(capsys, "softer", example, 90)
    assert code == BAD_INPUT
    assert "cannot make it 90 dB softer" in err
    assert "-94 is outside" in err
    # Softer layers compose: 6 then 6 is 12, up to the decimal rounding it writes.
    six = tmp_path / "six.toml"
    six.write_text(run(capsys, "softer", example, 6)[1], encoding="utf-8")
    twice = loads(run(capsys, "softer", six, 6)[1])
    assert twice.body.level_db == q.body.level_db
    assert abs(twice.noise[0].band_hz[1] - q.noise[0].band_hz[1]) <= 2 * 0.1  # two roundings to 0.1 Hz


def test_the_report_explains_the_domain(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """Find the driven reference's report naming its loudness, its cause and its drive headroom."""
    patch = tmp_path / "driven.toml"
    patch.write_text(cli.reference_text("driven"), encoding="utf-8")
    code, text, _ = run(capsys, "render", patch, "--out", tmp_path / "d.wav")
    assert code == 0
    assert "LUFS" in text
    assert "mostly kinks" in text
    assert "drive headroom +" in text


def test_verify(capsys: pytest.CaptureFixture[str], example: Path) -> None:
    """Verify a patch and its takes against the reference."""
    code, out, _ = run(capsys, "verify", example)
    assert code == 0
    assert "PASS" in out
    assert "E_rel" in out
    assert "6.144 and 12.288 MHz" in out
    code, out, _ = run(capsys, "verify", example, "--takes", 2)
    assert code == 0
    assert "take 2, seed 2" in out
    assert "2 of 2 takes pass" in out


@pytest.mark.parametrize(
    ("body", "message"),
    [
        pytest.param(b"{", "the embedded patch is damaged", id="unclosed"),
        pytest.param(b"[" * 100_000, "the embedded patch is damaged", id="nested"),
        pytest.param(b'{"patch": 5}', "the embedded patch is damaged", id="not-text"),
        pytest.param(
            b'{"patch": "\\u001b]52;c;cHduZWQ=\\u0007"}',
            "holds control characters (U+0007, U+001B)",
            id="osc-52",
        ),
        pytest.param(b'{"patch": "# snare\\u009b2J"}', "holds control characters (U+009B)", id="csi"),
    ],
)
def test_extract_prints_only_an_intact_patch(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    body: bytes,
    message: str,
) -> None:
    """Refuse a chunk that is garbled, nested to exhaust the parser or not text, or text a terminal would act on."""
    chunk = wav.PRIVATE + len(body).to_bytes(4, "little") + body + b"\0" * (len(body) % 2)
    path = tmp_path / "odd.wav"
    path.write_bytes(b"RIFF" + (4 + len(chunk)).to_bytes(4, "little") + b"WAVE" + chunk)
    code, out, err = run(capsys, "extract", path)
    assert (code, out) == (BAD_INPUT, "")
    assert message in err


def test_extract_refuses_what_is_not_a_wave_file(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """Refuse to extract from a file that is not RIFF WAVE, or is not there."""
    text = tmp_path / "notes.txt"
    text.write_text("not audio", encoding="utf-8")
    code, _, err = run(capsys, "extract", text)
    assert code == BAD_INPUT
    assert "not a RIFF WAVE file" in err
    code, _, err = run(capsys, "extract", tmp_path / "missing.wav")
    assert code == BAD_INPUT
    assert "missing.wav" in err


def test_audition_from_the_command_line(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """Pass audition's options through, and refuse beats, tempi and numbers that cannot be."""
    code, _, err = run(capsys, "audition", tmp_path / "missing.toml", "--bpm", 120, "--beats", "3,1")
    assert code == BAD_INPUT
    assert "missing.toml" in err
    for args, message in (
        (["--beats", "two"], "expected beats such as 2,4"),
        (["--beats", "0,2"], "beats of a 4/4 bar are 1 to 4"),
        (["--bpm", "0"], "--bpm must be positive"),
        (["--bpm", "fast"], "expected a number"),
        (["--bpm", "inf"], "expected a finite number"),
    ):
        with pytest.raises(SystemExit) as e:
            cli.main(["audition", "x.toml", *args])
        assert e.value.code == BAD_INPUT
        assert message in capsys.readouterr().err


def test_a_reader_that_goes_away_ends_the_command_quietly(monkeypatch: pytest.MonkeyPatch) -> None:
    """End with status 0, saying nothing, when the output closes, as `halyard init crack | head` does."""

    class Gone(io.StringIO):
        @override
        def write(self, s: str) -> int:
            raise BrokenPipeError

    errors = io.StringIO()
    monkeypatch.setattr(sys, "stdout", Gone())
    monkeypatch.setattr(sys, "stderr", errors)
    assert cli.main(["init", "crack"]) == 0
    assert errors.closed


def test_the_command_says_its_version(capsys: pytest.CaptureFixture[str]) -> None:
    """Say the version, as the package declares it."""
    with pytest.raises(SystemExit) as e:
        cli.main(["--version"])
    assert e.value.code == 0
    assert capsys.readouterr().out == f"halyard {version('halyard-synth')}\n"


def test_the_package_runs_as_the_command(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Run `python -m halyard_synth` as the halyard command, exiting with its status."""
    monkeypatch.setattr(sys, "argv", ["halyard", "init", "snap"])
    with pytest.raises(SystemExit) as e:
        runpy.run_module("halyard_synth", run_name="__main__")
    assert e.value.code == 0
    assert loads(capsys.readouterr().out) == loads(cli.reference_text("snap"))


def test_flam(capsys: pytest.CaptureFixture[str], example: Path) -> None:
    """Print the drum struck twice, and refuse a gap that is not positive."""
    code, out, _ = run(capsys, "flam", example, "--gap", 25)
    assert code == 0
    assert loads(out).body.amp[-1][0] == loads(EXAMPLE).body.amp[-1][0] + FLAM_GAP_MS
    code, out, _ = run(capsys, "flam", example, "--gap", 25, "--grace", 6, "--glide", 0, "--contact", 0)
    assert code == 0
    assert loads(out).noise[0].amp[1] == (0.05, round(10 ** (-6 / 20), 6))  # neither rise lengthened
    code, _, err = run(capsys, "flam", example, "--gap", 0)
    assert code == BAD_INPUT
    assert "cannot write a flam 0 ms wide" in err


FLAM_GAP_MS = 25.0


def test_roll(capsys: pytest.CaptureFixture[str], example: Path) -> None:
    """Print a roll, and refuse one a shape cannot hold, saying which."""
    code, out, _ = run(capsys, "roll", example, "--count", 3, "--spacing", 80, "--crescendo", 6)
    assert code == 0
    assert loads(out).body.amp[-1][0] == loads(EXAMPLE).body.amp[-1][0] + ROLL_LAST_MS
    code, _, err = run(capsys, "roll", example, "--count", 9, "--spacing", 60)
    assert code == BAD_INPUT
    assert "noise[1].amp" in err


ROLL_LAST_MS = 160.0  # the third stroke of 80 ms gaps


def test_stack(capsys: pytest.CaptureFixture[str], example: Path, tmp_path: Path) -> None:
    """Print a snare and a clap as one patch, and refuse a stack of two bodies or of a file that is no patch."""
    clap = tmp_path / "clap.toml"
    clap.write_text(cli.reference_text("clap"), encoding="utf-8")
    code, out, _ = run(capsys, "stack", example, clap, "--db", -4)
    assert code == 0
    assert len(loads(out).noise) == len(loads(EXAMPLE).noise) + len(loads(cli.reference_text("clap")).noise)
    code, _, err = run(capsys, "stack", example, example)
    assert code == BAD_INPUT
    assert "both patches have a sounding body" in err
    code, _, err = run(capsys, "stack", example, tmp_path / "missing.toml")
    assert code == BAD_INPUT
