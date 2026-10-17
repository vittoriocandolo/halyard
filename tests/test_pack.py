import dataclasses
import math
import re

import numpy as np
import pytest
import soundfile

from snare import cli, wav
from snare import pack as K
from snare.loudness import momentary_max
from snare.patch import dumps, loads, softer


def parse_sfz(text):
    """An independent reading of the mapping: headers with their opcodes, in order."""
    text = re.sub(r"//[^\n]*", "", text)
    out = []
    for header, body in re.findall(r"<(\w+)>([^<]*)", text):
        opcodes = dict(re.findall(r"(\w+)=(\S+)", body))
        out.append((header, opcodes))
    return out


def regions(text):
    """Every region with the opcodes it inherits from its global and group."""
    glob, group, found = {}, {}, []
    for header, ops in parse_sfz(text):
        if header == "global":
            glob = ops
        elif header == "group":
            group = ops
        elif header == "region":
            found.append({**glob, **group, **ops})
    return found


@pytest.fixture(scope="module")
def ladder(tmp_path_factory):
    """The crack reference and three softer layers, written as patch files."""
    d = tmp_path_factory.mktemp("ladder")
    base = loads(cli.reference_text("crack"))
    paths = []
    for db in (0, 8, 16, 24):
        path = d / ("snare.toml" if db == 0 else f"snare-{db}.toml")
        path.write_text(dumps(softer(base, db)))
        paths.append(path)
    return paths


@pytest.fixture(scope="module")
def built(ladder):
    return K.build([(p.stem, loads(p.read_text())) for p in ladder], takes=3)


def test_velocity_ranges_follow_the_square_law():
    ranges = K.velocity_ranges([0.0, -6.0, -12.0, -24.0])
    assert ranges[0][1] == 127 and ranges[-1][0] == 1
    for (lo, _), (_, hi) in zip(ranges, ranges[1:]):
        assert hi == lo - 1  # contiguous, no gaps and no overlaps
    # A layer 6 dB down sits where 40·log10(v/127) = −6: v ≈ 90; neighbours meet halfway in dB.
    natural = 127 * 10 ** (-6 / 40)
    assert ranges[1][0] <= natural <= ranges[1][1]
    assert ranges[0][0] == math.floor(math.sqrt(127 * natural)) + 1
    with pytest.raises(K.PackError):
        K.velocity_ranges([0.0, -0.001, -0.002])


def test_the_curve_puts_each_layer_on_the_square_law():
    for relative in (0.0, -5.0, -17.5):
        lo, hi = 40, 90
        volume, curve = K.curve_for(relative, lo, hi)
        assert max(curve) == 1.0 and min(curve) > 0
        for v, c in zip(range(lo, hi + 1), curve):
            assert relative + volume + 20 * math.log10(c) == pytest.approx(40 * math.log10(v / 127), abs=1e-9)


def test_every_velocity_and_take_maps_to_exactly_one_file(built, tmp_path):
    written = K.write(built, tmp_path)
    text = (tmp_path / "snare.sfz").read_text()
    rs = regions(text)
    assert len(rs) == 4 * 3 and all(r["key"] == "38" and r["pitch_keytrack"] == "0" for r in rs)
    for v in range(1, 128):
        for position in (1, 2, 3):
            hits = [r for r in rs if int(r["lovel"]) <= v <= int(r["hivel"]) and int(r["seq_position"]) == position]
            assert len(hits) == 1 and hits[0]["seq_length"] == "3"
            assert (tmp_path / hits[0]["sample"]).exists()
    assert {p.name for p in written} == {r["sample"] for r in rs} | {"snare.sfz"}


def test_the_level_follows_the_square_law_across_layer_boundaries(built, tmp_path):
    K.write(built, tmp_path)
    text = (tmp_path / "snare.sfz").read_text()
    groups = [ops for header, ops in parse_sfz(text) if header == "group"]
    rate = built.layers[0].patch.output.sample_rate
    top = None
    levels = []
    for group, layer in zip(groups, built.layers):
        # The level of the layer's first take, measured from its file.
        name = [r["sample"] for r in regions(text) if r["lovel"] == group["lovel"] and r["seq_position"] == "1"][0]
        data, _ = soundfile.read(tmp_path / name, dtype="float64")
        loud = momentary_max(data, rate)
        top = loud if top is None else top
        for v in range(int(group["lovel"]), int(group["hivel"]) + 1):
            c = float(group[f"amp_velcurve_{v}"])
            levels.append((v, loud + float(group["volume"]) + 20 * math.log10(c) - top))
    levels.sort()
    assert [v for v, _ in levels] == list(range(1, 128))
    for v, level in levels:
        assert level == pytest.approx(40 * math.log10(v / 127), abs=0.01)
    # Within one velocity step across every boundary.
    step = {v: 40 * math.log10((v + 1) / v) for v in range(1, 127)}
    for (v0, l0), (v1, l1) in zip(levels, levels[1:]):
        assert v1 == v0 + 1 and 0 < l1 - l0 <= step[v0] + 0.02


def test_the_pack_shares_one_gain(ladder):
    peak = dataclasses.replace(loads(ladder[0].read_text()).output, mode="peak", db=-1.0)
    patches = [(p.stem, dataclasses.replace(loads(p.read_text()), output=peak)) for p in ladder]
    pk = K.build(patches, takes=2)
    peaks = [float(np.max(np.abs(a))) for row in pk.audio for a in row]
    assert 20 * math.log10(max(peaks)) == pytest.approx(-1.0, abs=1e-6)
    assert peaks.index(max(peaks)) < 2  # the loudest layer holds the loudest file
    assert len({round(p, 9) for p in peaks}) == len(peaks)  # the others keep their natural levels


def test_level_matching_within_each_layer(ladder):
    patches = [(p.stem, loads(p.read_text())) for p in ladder[:2]]
    pk = K.build(patches, takes=4, level_match=True)
    for layer, files in zip(pk.layers, pk.audio):
        rate = layer.patch.output.sample_rate
        first = momentary_max(files[0], rate)
        assert all(momentary_max(a, rate) == pytest.approx(first, abs=0.01) for a in files)
        assert layer.spread < 0.01


def test_files_carry_their_take_and_layer(built, tmp_path):
    K.write(built, tmp_path)
    for i, layer in enumerate(built.layers, 1):
        for j, name in enumerate(layer.files("snare"), 1):
            assert name == f"snare_v{layer.hivel}_rr{j}.wav"
            meta = wav.read(tmp_path / name).meta
            assert loads(meta["patch"]).seed == layer.patch.seed + j - 1
            assert meta["pack"] == {"layer": i, "layers": 4, "lovel": layer.lovel, "hivel": layer.hivel, "key": 38}
            assert meta["set_gain_db"] == built.set_gain_db


def test_a_pack_re_renders_bit_identically(ladder, tmp_path):
    patches = [(p.stem, loads(p.read_text())) for p in ladder[:2]]
    for d in ("a", "b"):
        K.write(K.build(patches, takes=2), tmp_path / d)
    for f in sorted((tmp_path / "a").iterdir()):
        assert f.read_bytes() == (tmp_path / "b" / f.name).read_bytes()


def test_a_clap_takes_the_clap_key():
    clap = loads(cli.reference_text("clap"))
    pk = K.build([("clap", clap), ("clap-12", softer(clap, 12))], takes=1)
    assert pk.key == 39 and "key=39" in pk.sfz()
    assert K.build([("clap", clap)], takes=1, key=40).key == 40


def test_packs_refuse_what_they_cannot_map(ladder):
    loud, soft = loads(ladder[0].read_text()), loads(ladder[1].read_text())
    with pytest.raises(K.PackError, match="not softer"):
        K.build([("soft", soft), ("loud", loud)])
    other = loads(ladder[1].read_text().replace("sample_rate = 48000", "sample_rate = 44100"))
    with pytest.raises(K.PackError, match="output.sample_rate"):
        K.build([("loud", loud), ("other", other)])
    silent = dataclasses.replace(
        soft,
        body=dataclasses.replace(soft.body, level_db=-math.inf),
        noise=tuple(dataclasses.replace(n, level_db=-math.inf) for n in soft.noise),
    )
    with pytest.raises(K.PackError, match="silent"):
        K.build([("loud", loud), ("silent", silent)])


def test_the_command_writes_a_pack(ladder, tmp_path, capsys):
    out = tmp_path / "kit"
    code = cli.main(["pack", *map(str, ladder), "--takes", "2", "--out", str(out)])
    text = capsys.readouterr().out
    assert code == 0 and "4 velocity layers × 2 takes = 8 files" in text and "velocities" in text
    assert (out / "snare.sfz").exists() and len(list(out.glob("*.wav"))) == 8
    code = cli.main(["pack", str(ladder[1]), str(ladder[0]), "--out", str(tmp_path / "bad")])
    assert code == 2 and "not softer" in capsys.readouterr().err
