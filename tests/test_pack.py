# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Packs: velocity layers with takes, one gain, and an SFZ mapping that a sampler reads (plan §7.3)."""

import dataclasses
import itertools
import math
import re
from pathlib import Path

import numpy as np
import pytest
import soundfile

from halyard_synth import cli, pack, patch, wav
from halyard_synth.loudness import momentary_max
from halyard_synth.patch import dumps, loads, softer

from .conftest import EXAMPLE

type Opcodes = dict[str, str]

TOP = 127  # the highest velocity
SNARE, CLAP = 38, 39  # the General MIDI keys
LAYERS, TAKES = 4, 3  # the ladder's layers, and the takes the built pack has of each
LAW_DB = 0.01  # the level follows the square law within this, measured from the files
STEP_DB = 0.02  # and never steps more than one velocity's step past it across a boundary
PEAK_DB = 1e-6  # in peak mode the loudest file is at its target
MATCHED_DB = 0.01  # level matching brings a layer's takes within this


def parse_sfz(text: str) -> list[tuple[str, Opcodes]]:
    """Read the mapping independently: its headers with their opcodes, in order."""
    text = re.sub(r"//[^\n]*", "", text)
    out = []
    for header, body in re.findall(r"<(\w+)>([^<]*)", text):
        opcodes = dict(re.findall(r"(\w+)=(\S+)", body))
        out.append((header, opcodes))
    return out


def regions(text: str) -> list[Opcodes]:
    """Return every region with the opcodes it inherits from its global and group."""
    glob: Opcodes = {}
    group: Opcodes = {}
    found: list[Opcodes] = []
    for header, ops in parse_sfz(text):
        if header == "global":
            glob = ops
        elif header == "group":
            group = ops
        elif header == "region":
            found.append({**glob, **group, **ops})
    return found


@pytest.fixture(scope="module")
def ladder(tmp_path_factory: pytest.TempPathFactory) -> list[Path]:
    """Write the crack reference and three softer layers as patch files."""
    d = tmp_path_factory.mktemp("ladder")
    base = loads(cli.reference_text("crack"))
    paths = []
    for db in (0, 8, 16, 24):
        path = d / ("snare.toml" if db == 0 else f"snare-{db}.toml")
        path.write_text(dumps(softer(base, db)), encoding="utf-8")
        paths.append(path)
    return paths


@pytest.fixture(scope="module")
def built(ladder: list[Path]) -> pack.Pack:
    """Build the ladder's pack, three takes a layer."""
    return pack.build([(p.stem, loads(p.read_text(encoding="utf-8"))) for p in ladder], pack.Options(TAKES))


def test_velocity_ranges_follow_the_square_law() -> None:
    """Map layers 0, 6, 12 and 24 dB down to contiguous velocities, each where the square law reaches its level."""
    ranges = pack.velocity_ranges([0.0, -6.0, -12.0, -24.0])
    assert ranges[0][1] == TOP
    assert ranges[-1][0] == 1
    for (lo, _), (_, hi) in itertools.pairwise(ranges):
        assert hi == lo - 1  # contiguous, no gaps and no overlaps
    # A layer 6 dB down runs up to where 40·log10(v/127) = −6, v = 89.9: never above its own loudness.
    assert ranges[1][1] == math.floor(127 * 10 ** (-6 / 40))
    assert ranges[0][0] == ranges[1][1] + 1
    with pytest.raises(pack.PackError, match="layers 2 and 3 are within a velocity step"):
        pack.velocity_ranges([0.0, -0.001, -0.002])
    with pytest.raises(pack.PackError, match=r"layer 2 is 90\.0 dB under the loudest"):
        pack.velocity_ranges([0.0, -90.0])


def test_the_curve_puts_each_layer_on_the_square_law() -> None:
    """Find each velocity's curve point putting the layer on 40·log₁₀(v/127), never above its own level."""
    for relative in (0.0, -5.0, -17.5):
        hi = math.floor(127 * 10 ** (relative / 40))
        curve = pack.curve_for(relative, 20, hi)
        assert max(curve) <= 1.0
        assert min(curve) > 0
        for v, c in zip(range(20, hi + 1), curve, strict=True):
            assert relative + 20 * math.log10(c) == pytest.approx(40 * math.log10(v / 127), abs=1e-9)


def test_every_velocity_and_take_maps_to_exactly_one_file(built: pack.Pack, tmp_path: Path) -> None:
    """Find every velocity and round robin mapped to exactly one existing file, and nothing else written."""
    written = pack.write(built, tmp_path)
    text = (tmp_path / "snare.sfz").read_text(encoding="utf-8")
    rs = regions(text)
    assert len(rs) == LAYERS * TAKES
    assert all(r["key"] == str(SNARE) and r["pitch_keytrack"] == "0" for r in rs)
    assert all(r["loop_mode"] == "one_shot" for r in rs)  # a short note never cuts the drum
    for v in range(1, 128):
        for position in (1, 2, 3):
            hits = [r for r in rs if int(r["lovel"]) <= v <= int(r["hivel"]) and int(r["seq_position"]) == position]
            assert len(hits) == 1
            assert hits[0]["seq_length"] == "3"
            assert (tmp_path / hits[0]["sample"]).exists()
    assert {p.name for p in written} == {r["sample"] for r in rs} | {"snare.sfz"}


def test_the_level_follows_the_square_law_across_layer_boundaries(built: pack.Pack, tmp_path: Path) -> None:
    """Measure each layer's file and its curve: the level follows 40·log₁₀(v/127) at every velocity."""
    pack.write(built, tmp_path)
    text = (tmp_path / "snare.sfz").read_text(encoding="utf-8")
    groups = [ops for header, ops in parse_sfz(text) if header == "group"]
    rate = built.layers[0].patch.output.sample_rate
    top = None
    levels = []
    assert len(groups) == len(built.layers)
    for group in groups:
        # The level of the layer: its takes together, measured from their files.
        names = [r["sample"] for r in regions(text) if r["lovel"] == group["lovel"]]
        loud = pack.mean_loudness(
            [momentary_max(soundfile.read(tmp_path / name, dtype="float64")[0], rate) for name in names],
        )
        top = loud if top is None else top
        for v in range(int(group["lovel"]), int(group["hivel"]) + 1):
            c = float(group[f"amp_velcurve_{v}"])
            levels.append((v, loud + 20 * math.log10(c) - top))
    levels.sort()
    assert [v for v, _ in levels] == list(range(1, TOP + 1))
    for v, level in levels:
        assert level == pytest.approx(40 * math.log10(v / TOP), abs=LAW_DB)
    # Within one velocity step across every boundary.
    step = {v: 40 * math.log10((v + 1) / v) for v in range(1, TOP)}
    for (v0, l0), (v1, l1) in itertools.pairwise(levels):
        assert v1 == v0 + 1
        assert 0 < l1 - l0 <= step[v0] + STEP_DB


def test_the_pack_shares_one_gain(ladder: list[Path]) -> None:
    """Build a pack in peak mode: its loudest file reaches the target, and every other keeps its natural level."""
    peak = dataclasses.replace(loads(ladder[0].read_text(encoding="utf-8")).output, mode="peak", db=-1.0)
    patches = [(p.stem, dataclasses.replace(loads(p.read_text(encoding="utf-8")), output=peak)) for p in ladder]
    takes = 2
    pk = pack.build(patches, pack.Options(takes))
    peaks = [float(np.max(np.abs(a))) for row in pk.audio for a in row]
    assert 20 * math.log10(max(peaks)) == pytest.approx(-1.0, abs=PEAK_DB)
    assert peaks.index(max(peaks)) < takes  # the loudest layer holds the loudest file
    assert len({round(p, 9) for p in peaks}) == len(peaks)  # the others keep their natural levels


def test_level_matching_within_each_layer(ladder: list[Path]) -> None:
    """Level-match a pack: every take of a layer as loud as its first."""
    patches = [(p.stem, loads(p.read_text(encoding="utf-8"))) for p in ladder[:2]]
    pk = pack.build(patches, pack.Options(4, level_match=True))
    for layer, files in zip(pk.layers, pk.audio, strict=True):
        rate = layer.patch.output.sample_rate
        first = momentary_max(files[0], rate)
        assert all(momentary_max(a, rate) == pytest.approx(first, abs=MATCHED_DB) for a in files)
        assert layer.spread < MATCHED_DB


def test_files_carry_their_take_and_layer(built: pack.Pack, tmp_path: Path) -> None:
    """Find each file named by its layer and take, and carrying its seed, its place in the pack and the gain."""
    pack.write(built, tmp_path)
    for i, layer in enumerate(built.layers, 1):
        for j, name in enumerate(layer.files("snare"), 1):
            assert name == f"snare_v{layer.hivel}_rr{j}.wav"
            meta = wav.read(tmp_path / name).meta
            assert meta is not None
            assert loads(meta["patch"]).seed == layer.patch.seed + j - 1
            where = {"layer": i, "layers": LAYERS, "lovel": layer.lovel, "hivel": layer.hivel, "key": SNARE}
            assert meta["pack"] == where
            assert meta["set_gain_db"] == built.set_gain_db


def test_a_pack_re_renders_bit_identically(ladder: list[Path], tmp_path: Path) -> None:
    """Build and write a pack twice, and find the same bytes in every file."""
    patches = [(p.stem, loads(p.read_text(encoding="utf-8"))) for p in ladder[:2]]
    for d in ("a", "b"):
        pack.write(pack.build(patches, pack.Options(2)), tmp_path / d)
    for f in sorted((tmp_path / "a").iterdir()):
        assert f.read_bytes() == (tmp_path / "b" / f.name).read_bytes()


def test_a_clap_takes_the_clap_key() -> None:
    """Map a pack without a body to the clap key, unless a key is given."""
    clap = loads(cli.reference_text("clap"))
    pk = pack.build([("clap", clap), ("clap-12", softer(clap, 12))], pack.Options(1))
    assert pk.key == CLAP
    assert f"key={CLAP}" in pk.sfz()
    chosen = 40
    assert pack.build([("clap", clap)], pack.Options(1, chosen)).key == chosen


def test_packs_refuse_what_they_cannot_map(ladder: list[Path]) -> None:
    """Refuse layers not loudest first, layers whose outputs differ, and a silent layer."""
    loud, soft = loads(ladder[0].read_text(encoding="utf-8")), loads(ladder[1].read_text(encoding="utf-8"))
    with pytest.raises(pack.PackError, match="not softer"):
        pack.build([("soft", soft), ("loud", loud)])
    other = loads(ladder[1].read_text(encoding="utf-8").replace("sample_rate = 48000", "sample_rate = 44100"))
    with pytest.raises(pack.PackError, match=r"output\.sample_rate"):
        pack.build([("loud", loud), ("other", other)])
    silent = dataclasses.replace(
        soft,
        body=dataclasses.replace(soft.body, level_db=-math.inf),
        noise=tuple(dataclasses.replace(n, level_db=-math.inf) for n in soft.noise),
    )
    with pytest.raises(pack.PackError, match="silent"):
        pack.build([("loud", loud), ("silent", silent)])


def test_the_command_writes_a_pack(ladder: list[Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Run `halyard pack`: it writes the files and their mapping, and refuses layers out of order."""
    out = tmp_path / "kit"
    code = cli.main(["pack", *map(str, ladder), "--takes", "2", "--out", str(out)])
    text = capsys.readouterr().out
    assert code == 0
    assert "4 velocity layers × 2 takes = 8 files" in text
    assert "velocities" in text
    assert (out / "snare.sfz").exists()
    assert len(list(out.glob("*.wav"))) == LAYERS * 2
    code = cli.main(["pack", str(ladder[1]), str(ladder[0]), "--out", str(tmp_path / "bad")])
    assert code == cli.CliError("").status
    assert "not softer" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("name", "holds"),
    [
        ("snare\n<region> sample=other.wav key=1", "U+000A < > ="),
        ("snare $KEY", "$"),
        ("snare // old", "//"),
        ("snare /* old", "/*"),
        ("snare\\..", "\\"),
        ("snare\u2028two", "U+2028"),
    ],
)
def test_packs_refuse_names_a_map_would_misread(ladder: list[Path], name: str, holds: str) -> None:
    """Refuse a layer or pack name that would add to, or break, the SFZ map it is written in."""
    loud = loads(ladder[0].read_text(encoding="utf-8"))
    with pytest.raises(pack.PackError, match=re.escape(f"it holds {holds};")):
        pack.build([(name, loud)])
    with pytest.raises(pack.PackError, match="a sampler trims the space around it"):
        pack.build([("snare", loud)], stem=" snare")


def test_the_command_refuses_a_patch_named_to_misread(
    ladder: list[Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Run `halyard pack` on a file named to add a region to the map, and write nothing."""
    evil = tmp_path / "snare\n<region> sample=other.wav key=1.toml"
    evil.write_text(ladder[0].read_text(encoding="utf-8"), encoding="utf-8")
    code = cli.main(["pack", str(evil), "--out", str(tmp_path / "kit")])
    assert code == cli.CliError("").status
    assert "cannot name a file in an SFZ map" in capsys.readouterr().err
    assert not (tmp_path / "kit").exists()


def test_no_velocity_plays_a_take_louder_than_it_was_rendered(ladder: list[Path]) -> None:
    """Find every curve at most 1 and no group volume, even for layers 18 dB apart: the map only lowers."""
    loud = loads(ladder[0].read_text(encoding="utf-8"))
    pk = pack.build([("loud", loud), ("far", softer(loud, 18))])
    assert all(max(layer.curve) <= 1.0 for layer in pk.layers)
    assert "volume=" not in pk.sfz()


def test_the_command_refuses_what_it_cannot_pack(
    ladder: list[Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Refuse no patch, a missing or invalid patch, a key outside MIDI's, and a pack over full scale: write nothing."""
    with pytest.raises(pack.PackError, match="at least one patch"):
        pack.build([])
    invalid = tmp_path / "invalid.toml"
    invalid.write_text("schema = 1\n", encoding="utf-8")
    over = tmp_path / "over.toml"
    over.write_text(EXAMPLE.replace("db          = -6.0", "db          = 12.0"), encoding="utf-8")  # 6 dB over
    out = tmp_path / "kit"
    for paths, options, status, message in (
        ([tmp_path / "missing.toml"], pack.Options(), 2, "missing.toml"),
        ([invalid], pack.Options(), 2, "invalid patch"),
        ([ladder[0]], pack.Options(key=128), 2, "--key: 128 is not a MIDI key"),
        ([over], pack.Options(), 1, "not written: the loudest file peaks at"),
    ):
        assert pack.pack(map(str, paths), out, options) == status
        assert message in capsys.readouterr().err
    assert not out.exists()


def test_a_layer_is_as_loud_as_its_takes_together_and_takes_that_vary_refuse_matching(ladder: list[Path]) -> None:
    """Place a layer by its takes' mean power, and refuse level matching that would undo [takes]."""
    assert pack.mean_loudness([-20.0, -20.0]) == pytest.approx(-20.0)
    assert pack.mean_loudness([-20.0, -math.inf]) == pytest.approx(-20.0 - 10 * math.log10(2))
    assert pack.mean_loudness([-math.inf, -math.inf]) == -math.inf
    varied = [
        (p.stem, dataclasses.replace(loads(p.read_text(encoding="utf-8")), takes=patch.Takes(2.0, 0.0)))
        for p in ladder[:2]
    ]
    with pytest.raises(pack.PackError, match="level matching would undo"):
        pack.build(varied, pack.Options(2, level_match=True))
    played = pack.build(varied, pack.Options(2))
    assert played.layers[0].loudness > played.layers[1].loudness
