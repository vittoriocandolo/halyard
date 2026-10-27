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

import numpy as np

from .patch import midi_pitch

FLOAT = 3  # WAVE_FORMAT_IEEE_FLOAT
PCM = 1  # WAVE_FORMAT_PCM
PRIVATE = b"snar"


def root_note(hz: float) -> tuple[int, int]:
    """smpl unity note and pitch fraction (units of 2⁻³² semitone, always upward).

    The unity note is the note at or below the pitch, except that a pitch less
    than half a cent flat of a note counts as that note, so samplers that
    ignore the fraction are never a semitone off.
    """
    p = midi_pitch(hz)
    note = math.floor(p + 0.005)
    return note, round(max(0.0, p - note) * 2**32)


def _chunk(cid: bytes, data: bytes) -> bytes:
    return cid + struct.pack("<I", len(data)) + data + b"\0" * (len(data) % 2)


def _pcm_bytes(samples: np.ndarray, bits: int) -> bytes:
    if bits == 16:
        return samples.astype("<i2").tobytes()
    # 24 bits: the low three bytes of each little-endian 32-bit integer.
    return samples.astype("<i4").view(np.uint8).reshape(-1, 4)[:, :3].tobytes()


def write(
    path,
    samples: np.ndarray,
    rate: int,
    *,
    bits: int | None = None,
    root: tuple[int, int] | None = None,
    meta: dict | None = None,
) -> None:
    """Write float32 samples, or with `bits` (16 or 24) integer PCM samples already quantized."""
    samples = np.asarray(samples)
    if samples.ndim != 1:
        raise ValueError("expected a mono buffer")
    if bits is None:
        if samples.dtype != np.float32:
            raise ValueError("expected a mono float32 buffer")
        chunks = _chunk(b"fmt ", struct.pack("<HHIIHHH", FLOAT, 1, rate, 4 * rate, 4, 32, 0))
        chunks += _chunk(b"fact", struct.pack("<I", samples.size))
        data = samples.astype("<f4").tobytes()
    else:
        if bits not in (16, 24) or samples.dtype.kind != "i":
            raise ValueError("expected integer samples and 16 or 24 bits")
        top = 2 ** (bits - 1)
        if samples.size and not (-top <= samples.min() and samples.max() < top):
            raise ValueError(f"samples out of range for {bits}-bit PCM")
        width = bits // 8
        chunks = _chunk(b"fmt ", struct.pack("<HHIIHH", PCM, 1, rate, width * rate, width, bits))
        data = _pcm_bytes(samples, bits)
    if root is not None:
        note, fraction = root
        chunks += _chunk(b"smpl", struct.pack("<9I", 0, 0, round(1e9 / rate), note, fraction, 0, 0, 0, 0))
    if meta is not None:
        chunks += _chunk(PRIVATE, json.dumps(meta, ensure_ascii=False, indent=1).encode("utf-8"))
    chunks += _chunk(b"data", data)
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks)


def chunks(path) -> dict[bytes, bytes]:
    """Every top-level chunk of a RIFF WAVE file, by ID (first occurrence wins)."""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"{path}: not a RIFF WAVE file")
    end = min(8 + struct.unpack_from("<I", data, 4)[0], len(data))
    found, pos = {}, 12
    while pos + 8 <= end:
        cid, size = data[pos : pos + 4], struct.unpack_from("<I", data, pos + 4)[0]
        found.setdefault(cid, data[pos + 8 : pos + 8 + size])
        pos += 8 + size + size % 2
    return found


@dataclass(frozen=True)
class Wav:
    rate: int
    samples: np.ndarray  # float32, or the integers of a PCM file; channels in columns when there are several
    root: tuple[int, int] | None
    meta: dict | None
    bits: int | None = None  # 16 or 24 for PCM, None for float
    channels: int = 1

    def as_float(self) -> np.ndarray:
        """The samples as float64 in [−1, 1)."""
        return self.samples.astype(float) if self.bits is None else self.samples / float(2 ** (self.bits - 1))


def read(path) -> Wav:
    """Read a WAV of 32-bit float or 16- or 24-bit PCM, mono or with several channels.

    This reads what `write` writes, and the beds and samples that audition plays
    against; several channels come back as columns.
    """
    found = chunks(path)
    if b"fmt " not in found or b"data" not in found:
        raise ValueError(f"{path}: missing fmt or data chunk")
    tag, channels, rate, _, _, bits = struct.unpack_from("<HHIIHH", found[b"fmt "])
    if tag == 0xFFFE and len(found[b"fmt "]) >= 40:  # WAVE_FORMAT_EXTENSIBLE: the real tag opens the subformat GUID
        tag = struct.unpack_from("<H", found[b"fmt "], 24)[0]
    raw = found[b"data"]
    frame = channels * bits // 8 if bits % 8 == 0 and channels > 0 else 0
    if not frame:
        raise ValueError(f"{path}: unsupported format ({channels} channels, {bits} bits)")
    raw = raw[: len(raw) // frame * frame]
    if (tag, bits) == (FLOAT, 32):
        samples, pcm = np.frombuffer(raw, dtype="<f4").astype(np.float32), None
    elif tag == PCM and bits == 16:
        samples, pcm = np.frombuffer(raw, dtype="<i2").astype(np.int32), 16
    elif tag == PCM and bits == 24:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        samples, pcm = (b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)) << 8 >> 8, 24  # sign-extend from bit 23
    else:
        raise ValueError(f"{path}: expected 32-bit float or 16/24-bit PCM, found format {tag} with {bits} bits")
    if channels > 1:
        samples = samples.reshape(-1, channels)
    root = struct.unpack_from("<9I", found[b"smpl"])[3:5] if b"smpl" in found else None
    meta = json.loads(found[PRIVATE].decode("utf-8")) if PRIVATE in found else None
    return Wav(rate, samples, root, meta, pcm, channels)
