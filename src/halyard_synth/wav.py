# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""A small RIFF writer and reader for mono WAV: IEEE float32, or PCM at 24 or 16 bits (plan §7.2).

Files carry fmt, fact (float only), smpl (so samplers map the root key; left
out when the body is silent), a private 'snar' chunk with the patch text and
provenance, and data. PCM uses the plain PCM format tag, which hardware
samplers read most widely. RIFF readers skip chunks they do not know.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from .patch import midi_pitch

if TYPE_CHECKING:
    import os

    from numpy.typing import ArrayLike, NDArray

    type Floats = NDArray[np.float64]

FLOAT = 3  # WAVE_FORMAT_IEEE_FLOAT
PCM = 1  # WAVE_FORMAT_PCM
EXTENSIBLE = 0xFFFE  # WAVE_FORMAT_EXTENSIBLE: the real tag opens the subformat GUID
EXTENSIBLE_FMT = 40  # the size of an extensible fmt chunk, which holds that GUID
PCM16, PCM24 = 16, 24
HEADER = 12  # "RIFF", the size and "WAVE"
PRIVATE = b"snar"
FMT = 16  # the bytes of a plain fmt chunk
SMPL = 36  # the bytes of a smpl chunk up to its first loop


class Format(NamedTuple):
    """The sample rate, and the PCM bit depth: 16 or 24, or None for float32."""

    rate: int
    bits: int | None = None


def root_note(hz: float) -> tuple[int, int]:
    """Smpl unity note and pitch fraction (units of 2⁻³² semitone, always upward).

    The unity note is the note at or below the pitch, except that a pitch less
    than half a cent flat of a note counts as that note, so samplers that
    ignore the fraction are never a semitone off.
    """
    p = midi_pitch(hz)
    note = math.floor(p + 0.005)
    return note, round(max(0.0, p - note) * 2**32)


def _chunk(cid: bytes, data: bytes) -> bytes:
    """Make a chunk: its ID, its size and its data, padded to an even length."""
    return cid + struct.pack("<I", len(data)) + data + b"\0" * (len(data) % 2)


def _pcm_bytes(samples: NDArray[np.integer[Any]], bits: int) -> bytes:
    """Pack integer samples as little-endian PCM of 16 or 24 bits."""
    if bits == PCM16:
        return samples.astype("<i2").tobytes()
    # 24 bits: the low three bytes of each little-endian 32-bit integer.
    return samples.astype("<i4").view(np.uint8).reshape(-1, 4)[:, :3].tobytes()


def write(
    path: str | os.PathLike[str],
    buffer: ArrayLike,
    fmt: Format,
    *,
    root: tuple[int, int] | None = None,
    meta: dict[str, Any] | None = None,
) -> None:
    """Write float32 samples, or with `fmt.bits` (16 or 24) integer PCM samples already quantized."""
    samples = np.asarray(buffer)
    rate, bits = fmt
    if samples.ndim != 1:
        msg = "expected a mono buffer"
        raise ValueError(msg)
    if bits is None:
        if samples.dtype != np.float32:
            msg = "expected a mono float32 buffer"
            raise ValueError(msg)
        chunks = _chunk(b"fmt ", struct.pack("<HHIIHHH", FLOAT, 1, rate, 4 * rate, 4, 32, 0))
        chunks += _chunk(b"fact", struct.pack("<I", samples.size))
        data = samples.astype("<f4").tobytes()
    else:
        if bits not in (PCM16, PCM24) or samples.dtype.kind != "i":
            msg = "expected integer samples and 16 or 24 bits"
            raise ValueError(msg)
        top = 2 ** (bits - 1)
        if samples.size and not (-top <= samples.min() and samples.max() < top):
            msg = f"samples out of range for {bits}-bit PCM"
            raise ValueError(msg)
        width = bits // 8
        chunks = _chunk(b"fmt ", struct.pack("<HHIIHH", PCM, 1, rate, width * rate, width, bits))
        data = _pcm_bytes(samples, bits)
    if root is not None:
        note, fraction = root
        chunks += _chunk(b"smpl", struct.pack("<9I", 0, 0, round(1e9 / rate), note, fraction, 0, 0, 0, 0))
    if meta is not None:
        chunks += _chunk(PRIVATE, json.dumps(meta, ensure_ascii=False, indent=1).encode("utf-8"))
    chunks += _chunk(b"data", data)
    with Path(path).open("wb") as f:
        f.write(b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks)


def chunks(path: str | os.PathLike[str]) -> dict[bytes, bytes]:
    """Return every top-level chunk of a RIFF WAVE file, by ID (first occurrence wins)."""
    data = Path(path).read_bytes()
    if len(data) < HEADER or data[:4] != b"RIFF" or data[8:HEADER] != b"WAVE":
        msg = f"{path}: not a RIFF WAVE file"
        raise ValueError(msg)
    end = min(8 + struct.unpack_from("<I", data, 4)[0], len(data))
    found: dict[bytes, bytes] = {}
    pos = HEADER
    while pos + 8 <= end:
        cid, size = data[pos : pos + 4], struct.unpack_from("<I", data, pos + 4)[0]
        found.setdefault(cid, data[pos + 8 : pos + 8 + size])
        pos += 8 + size + size % 2
    return found


@dataclass(frozen=True)
class Wav:
    """A file as read: its rate, its samples, its root note and its embedded patch and provenance."""

    rate: int
    samples: NDArray[Any]  # float32, or the integers of a PCM file; channels in columns when there are several
    root: tuple[int, int] | None
    meta: dict[str, Any] | None
    bits: int | None = None  # 16 or 24 for PCM, None for float
    channels: int = 1

    def as_float(self) -> Floats:
        """Return the samples as float64 in [−1, 1)."""
        out: Floats = self.samples.astype(float) if self.bits is None else self.samples / float(2 ** (self.bits - 1))
        return out


def read(path: str | os.PathLike[str]) -> Wav:
    """Read a WAV of 32-bit float or 16- or 24-bit PCM, mono or with several channels.

    This reads what `write` writes, and the beds and samples that audition plays
    against; several channels come back as columns.
    """
    found = chunks(path)
    if b"fmt " not in found or b"data" not in found:
        msg = f"{path}: missing fmt or data chunk"
        raise ValueError(msg)
    if len(found[b"fmt "]) < FMT or len(found.get(b"smpl", bytes(SMPL))) < SMPL:
        msg = f"{path}: a fmt or smpl chunk is cut short"
        raise ValueError(msg)
    tag, channels, rate, _, _, bits = struct.unpack_from("<HHIIHH", found[b"fmt "])
    if tag == EXTENSIBLE and len(found[b"fmt "]) >= EXTENSIBLE_FMT:
        tag = struct.unpack_from("<H", found[b"fmt "], 24)[0]
    raw = found[b"data"]
    frame = channels * bits // 8 if bits % 8 == 0 and channels > 0 else 0
    if not frame:
        msg = f"{path}: unsupported format, {bits}-bit samples in {channels} channel{'' if channels == 1 else 's'}"
        raise ValueError(msg)
    raw = raw[: len(raw) // frame * frame]
    samples: NDArray[Any]
    if (tag, bits) == (FLOAT, 32):
        samples, pcm = np.frombuffer(raw, dtype="<f4").astype(np.float32), None
    elif tag == PCM and bits == PCM16:
        samples, pcm = np.frombuffer(raw, dtype="<i2").astype(np.int32), PCM16
    elif tag == PCM and bits == PCM24:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        samples, pcm = (b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)) << 8 >> 8, PCM24  # sign-extend from bit 23
    else:
        msg = f"{path}: expected 32-bit float or 16/24-bit PCM, found format {tag} with {bits} bits"
        raise ValueError(msg)
    if channels > 1:
        samples = samples.reshape(-1, channels)
    root = struct.unpack_from("<9I", found[b"smpl"])[3:5] if b"smpl" in found else None
    meta = embedded(found[PRIVATE]) if PRIVATE in found else None
    if PRIVATE in found and meta is None:
        msg = f"{path}: the embedded patch is damaged"
        raise ValueError(msg)
    return Wav(rate, samples, root, meta, pcm, channels)


def embedded(chunk: bytes) -> dict[str, Any] | None:
    """Return the table a private chunk holds, or None when it is truncated, garbled or nested to exhaust the parser."""
    try:
        meta = json.loads(chunk.decode("utf-8"))
    except (ValueError, RecursionError):
        return None
    return meta if isinstance(meta, dict) else None
