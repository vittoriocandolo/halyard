# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Fuzzing: a damaged or crafted file, chunk or patch is refused with an error that says why, never misread.

Each property runs on examples Hypothesis derives the same way every time (see conftest.py), from files
Halyard writes and references it ships, mutated, cut short or filled with anything.
"""

import contextlib
import io
import json
import math
import struct
import tomllib
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import event, given, settings
from hypothesis import strategies as st

from halyard_synth import cli, patch, render, wav
from halyard_synth.patch import PatchError

RATE = 48000
FORMATS = [(wav.PCM, 1, 16), (wav.PCM, 2, 24), (wav.FLOAT, 1, 32), (wav.EXTENSIBLE, 2, 16), (wav.FLOAT, 2, 64)]
LINE_ENDS = "\t\n\r"
EXIT_OK, BAD_INPUT = 0, 2
FRAMES = 2  # dimensions of the frames a file holds: one row per frame, one column per channel

type Json = bool | int | float | str | list["Json"] | dict[str, "Json"] | None


def riff(*chunks: tuple[bytes, bytes]) -> bytes:
    """Build a RIFF WAVE file from its chunks, each padded to an even length."""
    body = b"".join(cid + struct.pack("<I", len(d)) + d + b"\0" * (len(d) % 2) for cid, d in chunks)
    return b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body


def sound(tag: int, channels: int, bits: int) -> bytes:
    """Return a short, well-formed WAV of that format."""
    fmt = struct.pack("<HHIIHH", tag, channels, RATE, RATE * channels * bits // 8, channels * bits // 8, bits)
    return riff((b"fmt ", fmt), (b"data", bytes(range(96)) * (channels * bits // 8)))


@st.composite
def damaged(draw: st.DrawFn, base: bytes) -> bytes:
    """Return `base` with a few bytes overwritten, often in its headers, and perhaps cut short or extended."""
    data = bytearray(base)
    for _ in range(draw(st.integers(0, 8))):
        reach = len(data) if draw(st.booleans()) else min(len(data), 64)  # the headers are where sizes live
        data[draw(st.integers(0, reach - 1))] = draw(st.integers(0, 255))
    data = data[: draw(st.integers(0, len(data)))] if draw(st.booleans()) else data
    return bytes(data) + draw(st.binary(max_size=16))


WAVS = st.one_of(
    st.sampled_from(FORMATS).flatmap(lambda f: damaged(sound(*f))),
    st.binary(max_size=128).map(lambda b: b"RIFF" + b[:4] + b"WAVE" + b[4:]),
)
JSON: st.SearchStrategy[Json] = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats() | st.text(),
    lambda inner: st.lists(inner, max_size=4) | st.dictionaries(st.text(max_size=8), inner, max_size=4),
    max_leaves=12,
)
CHUNKS = st.one_of(
    st.binary(max_size=256),
    JSON.map(lambda v: json.dumps(v).encode()),
    st.text().map(lambda t: json.dumps({"patch": t}).encode()),
    st.integers(1, 3000).map(lambda depth: b"[" * depth),
)


@pytest.fixture(scope="module")
def scratch(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return a folder for the files the properties write, one at a time."""
    return tmp_path_factory.mktemp("fuzz")


@given(data=WAVS)
@settings(max_examples=400)
def test_any_wav_is_read_whole_or_refused(scratch: Path, data: bytes) -> None:
    """Read a damaged or crafted WAV as whole frames at a real rate, or refuse it with a ValueError."""
    path = scratch / "x.wav"
    path.write_bytes(data)
    try:
        w = wav.read(path)
    except ValueError:
        return
    assert w.rate > 0
    assert w.channels > 0
    frames = w.as_float().reshape(-1, w.channels)
    assert frames.dtype == np.float64
    assert frames.ndim == FRAMES


@given(body=CHUNKS)
@settings(max_examples=300)
def test_extract_writes_an_intact_patch_or_refuses(scratch: Path, body: bytes) -> None:
    """Write the embedded text exactly when it is text free of control characters, and refuse anything else."""
    path = scratch / "x.wav"
    path.write_bytes(riff((wav.PRIVATE, body)))
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        status = cli.main(["extract", str(path)])
    assert status in {EXIT_OK, BAD_INPUT}
    text = out.getvalue()
    assert not any(unicodedata.category(c) == "Cc" and c not in LINE_ENDS for c in text + err.getvalue())
    if status == EXIT_OK:
        assert text == json.loads(body)["patch"]


@st.composite
def edited_text(draw: st.DrawFn) -> str:
    """Return a shipped reference's text with a few spans replaced by anything."""
    text = cli.reference_text(draw(st.sampled_from(cli.REFERENCES)))
    for _ in range(draw(st.integers(1, 4))):
        start = draw(st.integers(0, len(text)))
        end = draw(st.integers(start, min(len(text), start + 12)))
        text = text[:start] + draw(st.text(max_size=8)) + text[end:]
    return text


def leaves(data: Json, path: tuple[str | int, ...] = ()) -> list[tuple[str | int, ...]]:
    """Return the path of every value in a parsed patch, tables and arrays included."""
    found: list[tuple[str | int, ...]] = [path] if path else []
    if isinstance(data, dict):
        for k, v in data.items():
            found += leaves(v, (*path, k))
    elif isinstance(data, list):
        for i, v in enumerate(data):
            found += leaves(v, (*path, i))
    return found


VALUES: st.SearchStrategy[Json] = st.one_of(
    st.floats(),  # NaN, infinities, subnormals, the largest and the smallest
    st.integers(),
    st.sampled_from([0.0, -0.0, 5e-324, 1e-320, 1e308, -1e308, 2**53 + 1, -1, 1e9]),
    JSON,
)


@st.composite
def edited_data(draw: st.DrawFn, *, near: bool = False) -> dict[str, Any]:
    """Return a shipped reference's data with a few values replaced by anything, or removed.

    With `near`, numbers are only scaled, by up to four times or down to zero, so that most patches stay valid
    and what they render is tried, at the edges of the domain included.
    """
    data: dict[str, Any] = tomllib.loads(cli.reference_text(draw(st.sampled_from(cli.REFERENCES))))
    for _ in range(draw(st.integers(1, 3))):
        *parents, last = draw(st.sampled_from(leaves(data)))
        holder: Any = data
        for key in parents:
            holder = holder[key]
        value = holder[last]
        if near:
            if isinstance(value, int | float) and not isinstance(value, bool):
                holder[last] = type(value)(value * draw(st.sampled_from([0.0, 0.5, 0.999, 1.001, 2.0, 4.0])))
        elif draw(st.booleans()) or isinstance(holder, list):
            holder[last] = draw(VALUES)
        else:
            del holder[last]
    return data


def accepted(p: patch.Patch) -> None:
    """Check what every accepted patch must be: written back exactly, and every number in it a number."""
    assert patch.loads(patch.dumps(p)) == p  # equality fails on NaN, so this also finds one accepted


@given(text=edited_text())
@settings(max_examples=300)
def test_any_patch_text_is_a_patch_or_a_patch_error(text: str) -> None:
    """Parse an edited reference into a patch that writes back as itself, or refuse it with a patch error."""
    try:
        p = patch.loads(text)
    except PatchError:
        event("refused")
        return
    event("accepted")
    accepted(p)


@given(data=edited_data())
@settings(max_examples=500)
def test_any_patch_data_is_a_patch_or_a_patch_error(data: dict[str, Any]) -> None:
    """Build a patch from edited data that writes back as itself, or refuse it with a patch error."""
    try:
        p = patch.from_dict(data)
    except PatchError:
        event("refused")
        return
    event("accepted")
    accepted(p)


@given(data=edited_data(near=True))
@settings(max_examples=40)
def test_any_accepted_patch_renders(data: dict[str, Any]) -> None:
    """Render every patch the validation accepts into finite audio, as its report says."""
    try:
        p = patch.from_dict(data)
    except PatchError:
        event("refused")
        return
    event("rendered")
    audio, report = render.render(p)
    assert np.all(np.isfinite(audio))
    assert all(math.isfinite(take.peak) for take in report.takes)
