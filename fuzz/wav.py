# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Fuzz the WAV reader: any bytes are read as a WAV file or refused with a ValueError.

Never misread, and never a crash.

    uv run --group fuzz python fuzz/wav.py [-max_total_time=SECONDS]

Linux on x86-64 only, where Atheris has wheels. A short file of every sample format seeds the corpus.
"""

import struct
import sys
import tempfile
from pathlib import Path

import atheris

with atheris.instrument_imports():
    from halyard_synth import wav

FILE = Path(tempfile.mkdtemp()) / "take.wav"
PCM, FLOAT, EXTENSIBLE = 1, 3, 0xFFFE
FORMATS = [(PCM, 8), (PCM, 16), (PCM, 24), (PCM, 32), (FLOAT, 32), (FLOAT, 64)]
GUID_TAIL = bytes.fromhex("000000001000800000aa00389b71")  # KSDATAFORMAT_SUBTYPE_*, after the format tag


def riff(tag: int, bits: int, channels: int, *, extensible: bool) -> bytes:
    """Return a WAV file of four frames of silence in the given format, plain or extensible."""
    align = bits // 8 * channels
    fmt = struct.pack("<HHIIHH", EXTENSIBLE if extensible else tag, channels, 48000, 48000 * align, align, bits)
    if extensible:
        fmt += struct.pack("<HHI", 22, bits, 0) + struct.pack("<H", tag) + GUID_TAIL
    data = bytes(4 * align)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(data)) + data
    return b"RIFF" + struct.pack("<I", len(body)) + body


def test_one(data: bytes) -> None:
    """Read one input as a WAV file."""
    FILE.write_bytes(data)
    try:
        wav.read(FILE)
    except ValueError:  # refused, with the reason
        return


def main() -> None:
    """Fuzz from a file of every format, with libFuzzer's own options from the command line."""
    corpus = Path(tempfile.mkdtemp())
    for tag, bits in FORMATS:
        for channels in (1, 2):
            for extensible in (False, True):
                name = f"{tag}-{bits}-{channels}{'-x' if extensible else ''}.wav"
                (corpus / name).write_bytes(riff(tag, bits, channels, extensible=extensible))
    atheris.Setup([*sys.argv, str(corpus)], test_one)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
