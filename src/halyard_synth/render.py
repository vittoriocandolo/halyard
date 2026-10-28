# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Rendering policy, the validated domain, takes and the output stage (plan §2.6, §3, §5, §7).

    out[m] = p · g · Σ_{k<L} h[k] y[mK − k],    y[n] = y(n / F) for 0 ≤ n < N, else 0

The internal rate is F = K × base, with base 44.1 or 48 kHz. The 88.2 and
96 kHz exports reuse their family's F and filter with factor K/2, so a 96 kHz
file is exactly the 48 kHz file with the in-between samples filled in.
Nothing trims, fades, high-passes or removes DC, and the whole filter tail
is kept.

Each take is judged on its own: the validated-domain status, what the
predicted aliasing comes from and how much drive would change the status,
all from the take's own render. A set of takes shares one gain.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import math
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from importlib import resources
from typing import TYPE_CHECKING, Any, NamedTuple, override

import numpy as np
import scipy
from scipy.signal import oaconvolve, upfirdn

from . import __version__, cores, noise, wav
from .loudness import momentary_max
from .patch import MAX_SEED, Patch, describe_tuning, dumps
from .voice import (
    Kinks,
    Slew,
    Voice,
    drive_alpha,
    internal,
    kink_bound,
    kink_bounds,
    kink_knots,
    kink_terms,
    sample_count,
    sources,
    tables,
    voice,
)

if TYPE_CHECKING:
    import os

    from numpy.typing import ArrayLike, NDArray

    from .voice import Tables

    type Floats = NDArray[np.float64]

K = 32  # frozen by the adversarial grid: at 16× the crack reference keeps only 1.7 dB of drive headroom (plan §3)
FAMILY = {44100: 44100, 88200: 44100, 48000: 48000, 96000: 48000}

# The validated domain (plan §5). The predicted aliasing error at unity gain is
#     SATURATION · exp(−π² / q) · P  +  the kink bound
# where q is the knee slew and P the output peak. A render is inside when that
# is at most LIMIT · P, or FLOOR when it is near-silent. SATURATION is refit on
# the adversarial grid at 32× (docs/grid/k32.csv): 6 dB of margin over every
# converged fixture needs 0.419, so it is frozen at 0.5 (the kick's grid froze 2).
SATURATION = 0.5
LIMIT = 1e-5  # −100 dB of the peak
FLOOR = 1e-7  # the absolute bound for near-silent renders
QUIET = 1e-2  # below this peak the absolute bound applies

DRIVE_MAX = 48.0  # the valid domain's largest drive
MAX_TAKES = 128

# Export formats: float32, or PCM with seeded TPDF dither (the value is the bit depth).
FORMATS = {"float32": None, "pcm24": 24, "pcm16": 16}


class RenderError(RuntimeError):
    """A render that cannot be made as asked."""


def ceiling(fmt: str) -> float:
    """Return the largest sample a format takes: full scale for float, 2 LSB below it for PCM, so dither never clips."""
    bits = FORMATS[fmt]
    return 1.0 if bits is None else 1.0 - 2.0 ** (2 - bits)


def db(x: float) -> float:
    """Return x in dB, or −inf for 0."""
    return 20 * math.log10(x) if x > 0 else -math.inf


# Frozen decimators.


@dataclass(frozen=True)
class Decimator:
    """A frozen decimator: its family's base rate, its factor K, its taps, their checksum and its stopband."""

    base: int
    k: int
    taps: Floats
    sha256: str
    stopband_db: float


def manifest() -> dict[str, Any]:
    """Return the shipped decimators' manifest: each one's file, size, checksum and stopband."""
    data: dict[str, Any] = json.loads(
        resources.files("halyard_synth").joinpath("data", "decimators.json").read_text(encoding="utf-8"),
    )
    return data


def load_taps(raw: bytes, entry: dict[str, Any]) -> Floats:
    """Return the coefficients from their shipped bytes, refused unless they match the checksum."""
    if hashlib.sha256(raw).hexdigest() != entry["sha256"] or len(raw) != 8 * entry["taps"]:
        msg = f"decimator {entry['base']}x{entry['K']} does not match its checksum"
        raise RenderError(msg)
    taps = np.frombuffer(raw, dtype="<f8").astype(float)
    taps.flags.writeable = False
    return taps


@functools.cache
def decimator(base: int, k: int = K) -> Decimator:
    """Return a frozen decimator from halyard_synth/data, verified against its checksum."""
    entry = manifest()["filters"][f"{base}x{k}"]
    raw = resources.files("halyard_synth").joinpath("data", entry["file"]).read_bytes()
    return Decimator(base, k, load_taps(raw, entry), entry["sha256"], entry["stopband_db"])


def decimate(taps: Floats, y: Floats, factor: int, workers: int | None = None) -> Floats:
    """Return upfirdn(taps, y, down=factor), computed in parallel chunks that match it bit for bit.

    Each output of upfirdn is one sequential sum over its own inputs. A chunk is
    given every input its outputs need, starting on the factor's grid, so each
    output is computed by exactly the same operations as in a single call.
    """
    length = taps.size
    n_out = (y.size + length - 2) // factor + 1
    workers = min(workers or cores.available(), max(1, n_out // 4096))
    if workers == 1:
        whole: Floats = upfirdn(taps, y, down=factor)
        return whole
    edges = [round(i * n_out / workers) for i in range(workers + 1)]

    def chunk(m0: int, m1: int) -> Floats:
        """Return outputs m0 … m1 − 1, from every input they need."""
        start = max(0, (m0 * factor - (length - 1)) // factor * factor)
        stop = y.size if m1 == n_out else (m1 - 1) * factor + 1
        out: Floats = upfirdn(taps, y[start:stop], down=factor)
        return out[m0 - start // factor : m1 - start // factor]

    with ThreadPoolExecutor(workers) as pool:
        return np.concatenate(list(pool.map(chunk, edges[:-1], edges[1:])))


def rates(sample_rate: int, k: int = K) -> tuple[int, int]:
    """Return (internal rate F, decimation factor) for an export rate."""
    base = FAMILY[sample_rate]
    return base * k, k * base // sample_rate


# Synthesis, and what the domain rule needs at every drive.

PROBES = 4  # output positions kept for every drive, in each of three places


@dataclass(frozen=True)
class Probes:
    """The exact output at a few chosen positions, at any drive.

    Every internal sample is a pure function of time, so the probes keep the
    saturator's input over their filter windows, and the output there follows
    exactly for any α. The largest of them never exceeds that drive's output
    peak, and at the render's own drive it is the peak. They sit at the
    loudest outputs of the render itself; at the loudest undriven outputs just
    after the crest of x, where a weak drive peaks; and at the loudest outputs
    at full drive just after the steepest zero crossing, where the overshoot
    of a saturated edge peaks. Nearby windows overlap almost entirely, so x is
    kept once over their union.
    """

    x: Floats  # the saturator input over the union of the windows
    windows: tuple[tuple[int, Floats], ...]  # per probe: (offset into x, the taps that weight its samples)

    def peak(self, alphas: ArrayLike) -> Floats:
        """Return the largest output among the probes at each α."""
        a = np.atleast_1d(np.asarray(alphas, dtype=float))
        out = np.empty(a.size)
        for i in range(0, a.size, 32):  # bounded temporaries
            c = a[i : i + 32, None]
            y = np.where(c > 0, np.tanh(c * self.x) / np.tanh(np.where(c > 0, c, 1.0)), self.x)
            out[i : i + 32] = np.max(np.abs(np.stack([y[:, o : o + h.size] @ h for o, h in self.windows])), axis=0)
        return out


@dataclass(frozen=True)
class Synthesis:
    """Everything before the output stage, at unity gain, for one take."""

    out: Floats
    rate: int
    factor: int
    source_samples: int
    slew: Slew
    kinks: Kinks = field(compare=False, repr=False)
    kink: float  # bound on the error from the curves' kinks, at unity gain and the patch's drive
    alpha: float
    band: float  # the decimator's passband edge, for the kink bound
    true_peak: float  # the largest value of the output's band-limited reconstruction, at unity gain
    probes: Probes | None = field(default=None, compare=False, repr=False)

    def peak(self, mode: str) -> float:
        """Return the peak a level mode targets, at unity gain: the true peak for "true-peak", else the sample peak."""
        return self.true_peak if mode == "true-peak" else float(np.max(np.abs(self.out), initial=0.0))


def reconstructed_peak(y: Floats, taps: Floats) -> float:
    """Return the true peak of the output that decimating y through `taps` gives, at unity gain.

    The decimator's output at every internal sample, not only at every K-th,
    is the output's band-limited reconstruction sampled at the internal rate:
    exact to the filter's stopband, 115 dB down. Its largest value, refined
    by the parabola through its two neighbours, is the true peak to within
    10⁻⁷ of it for anything below 20 kHz.
    """
    z = oaconvolve(y, taps)  # overlap-add: one long signal through a short filter
    i = int(np.argmax(np.abs(z), axis=None)) if z.size else 0
    if not 0 < i < z.size - 1:
        return float(np.abs(z[i])) if z.size else 0.0
    sign = math.copysign(1.0, z[i])
    a, b, c = sign * z[i - 1], sign * z[i], sign * z[i + 1]
    curve = a - 2 * b + c  # negative at a maximum
    return float(b - (a - c) ** 2 / (8 * curve)) if curve < 0 else float(b)


def loudest(mags: Floats, first: int, n: int, taken: list[int]) -> list[int]:
    """Return up to n indices of the largest nonzero values, apart from each other and from `taken`."""
    picked: list[int] = []
    for i in np.argsort(mags)[::-1][:256]:
        m = first + int(i)
        if mags[i] == 0 or len(picked) == n:
            break
        if all(abs(m - q) > 1 for q in taken + picked):
            picked.append(m)
    return picked


class Grid(NamedTuple):
    """A take's internal grid: its voice, the internal rate, the factor, the taps, its samples and its tables."""

    v: Voice
    rate: int
    factor: int
    taps: Floats
    samples: int
    tabs: Tables


def _probes(g: Grid, out: Floats, slew: Slew) -> Probes | None:
    """Choose the probes for a render, and keep the saturator's input under their windows."""
    v, rate, factor, taps, count, tabs = g
    picked = loudest(np.abs(out), 0, PROBES, [])
    # The output just after the crest without drive, and just after the steepest crossing at full drive,
    # computed exactly from x (the decimator's delay at low frequencies is a few outputs).
    a_max = drive_alpha(DRIVE_MAX)
    for n, alpha in ((slew.crest, 0.0), (slew.steep, a_max)):
        # The crest and the crossing are internal samples, so the output reaches past them.
        m0, m1 = max(0, n // factor - 8), min(out.size, n // factor + 24)
        n0 = max(0, m0 * factor - taps.size + 1) // factor * factor
        n1 = min(count, (m1 - 1) * factor + 1)
        x = sources(v, rate, n0, n1, tabs)
        y = np.tanh(alpha * x) / np.tanh(alpha) if alpha else x
        near = upfirdn(taps, y, down=factor)[m0 - n0 // factor : m1 - n0 // factor]
        picked += loudest(np.abs(near), m0, PROBES, picked)
    # out[m] = Σ_n h[mK − n] · y[n] over the internal samples 0 ≤ n < count that the filter reaches.
    # Every output reaches at least the last internal sample, so no span is empty.
    spans: list[tuple[int, int, int]] = []
    for m in picked:
        top = m * factor
        spans.append((m, max(0, top - taps.size + 1), min(top, count - 1) + 1))
    if not spans:
        return None
    merged: list[list[int]] = []
    for _, n0, n1 in sorted(spans, key=lambda s: s[1]):
        if merged and n0 <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], n1)
        else:
            merged.append([n0, n1])
    xs: list[Floats] = []
    where: list[tuple[int, int, int]] = []
    pos = 0
    for n0, n1 in merged:
        xs.append(sources(v, rate, n0, n1, tabs))
        where.append((n0, n1, pos))
        pos += n1 - n0
    windows = []
    for m, n0, n1 in spans:
        base = next(p0 + n0 - a for a, b, p0 in where if a <= n0 and n1 <= b)
        top = m * factor
        windows.append((base, np.ascontiguousarray(taps[top - np.arange(n0, n1)])))
    return Probes(np.concatenate(xs), tuple(windows))


def synthesize(p: Patch, *, k: int = K, taps: Floats | None = None) -> Synthesis:
    """Return the take that p's seed names, at unity gain, with everything the domain rule needs."""
    v = voice(p)
    rate, factor = rates(p.output.sample_rate, k)
    if taps is None:
        taps = decimator(FAMILY[p.output.sample_rate], k).taps
    tabs = tables(v, rate)
    n = sample_count(v.t_src, rate)
    y, slew = internal(v, rate, n, tabs)
    band = FAMILY[p.output.sample_rate] / 2
    kinks = kink_terms(v, rate, tabs)
    out = decimate(taps, y, factor)
    return Synthesis(
        out,
        rate,
        factor,
        n,
        slew,
        kinks,
        kink_bound(kinks, v.alpha, rate, band),
        v.alpha,
        band,
        reconstructed_peak(y, taps),
        _probes(Grid(v, rate, factor, taps, n, tabs), out, slew),
    )


# The validated domain.


@dataclass(frozen=True)
class Domain:
    """The validated-domain rule on one take: its predicted aliasing, what it allows and the output peak."""

    predicted: float  # predicted aliasing error, at unity gain
    allowed: float
    peak: float

    @property
    def inside(self) -> bool:
        """Whether the predicted error is within what the rule allows."""
        return self.predicted <= self.allowed

    @property
    def margin_db(self) -> float:
        """How far the prediction lies under what is allowed, in dB; negative outside."""
        return db(self.allowed / self.predicted) if self.predicted else math.inf

    @property
    def predicted_db(self) -> float:
        """The predicted error relative to the peak."""
        return db(self.predicted / self.peak) if self.peak else -math.inf


def allowed(peak: float) -> float:
    """Return the error the rule allows at an output peak: −100 dB of it, or the floor when near-silent."""
    return LIMIT * peak if peak >= QUIET else FLOOR


def domain(s: Synthesis) -> Domain:
    """Judge a take by the validated-domain rule."""
    peak = float(np.max(np.abs(s.out), initial=0.0))
    q = s.slew.knee(s.alpha)
    saturation = SATURATION * math.exp(-(math.pi**2) / q) * peak if q else 0.0
    return Domain(saturation + s.kink, allowed(peak), peak)


@dataclass(frozen=True)
class Cause:
    """Where the predicted aliasing comes from, at unity gain."""

    saturation: float  # the saturation term of the rule
    kinks: float  # the kink bound
    knot: str | None = None  # the knot that contributes most to the kink bound, e.g. "noise[1].amp[2]"
    knot_ms: float | None = None
    share: float = 0.0  # its share of the kink bound

    @property
    def dominant(self) -> str | None:
        """Which term is larger, saturation or kinks; None when neither predicts any."""
        if not (self.saturation or self.kinks):
            return None
        return "saturation" if self.saturation >= self.kinks else "kinks"

    @property
    def where(self) -> str:
        """Name the knot that contributes most to the kink bound, its time and its share."""
        return f"{100 * self.share:.0f}% at {self.knot} ({self.knot_ms:g} ms)"


def knot_ms(p: Patch, path: str) -> float:
    """Return the time of a knot named by its path, such as "noise[2].amp[3]", as the patch writes it."""
    curve, index = path[: path.rindex("[")], int(path[path.rindex("[") + 1 : -1])
    if curve == "body.amp":
        knots = p.body.amp
    elif curve == "body.pitch":
        knots = p.body.pitch
    else:
        knots = p.noise[int(curve[len("noise[") : curve.index("]")]) - 1].amp
    time: float = knots[index - 1][0]
    return time


def cause_of(p: Patch, s: Synthesis, d: Domain) -> Cause:
    """Find where a take's predicted aliasing comes from."""
    saturation = d.predicted - s.kink
    knots = kink_knots(s.kinks, s.alpha, s.rate, s.band)
    total = math.fsum(c for _, c in knots)
    if not total:
        return Cause(saturation, s.kink)
    path, c = knots[0]
    return Cause(saturation, s.kink, path, knot_ms(p, path), c / total)


HEADROOM_STEP_DB = 0.5  # the scan's grid; every jump of the knee slew is a candidate as well


@dataclass(frozen=True)
class Headroom:
    """How much the drive can change before the validated-domain status does, estimated from one render.

    `db` is positive inside and negative outside; +inf when the take stays
    inside at every drive up to 48 dB, −inf when it is outside even without
    drive. `edge_db` is the drive at the domain's edge.
    """

    db: float
    edge_db: float = math.nan  # nan where the status never changes


def alphas_of(drives: ArrayLike) -> Floats:
    """Return α with α / tanh α = 10^(drive/20), for many drives at once: bisection on [G − 1, G]."""
    g = 10.0 ** (np.asarray(drives, dtype=float) / 20.0)
    lo, hi = np.maximum(g - 1.0, 0.0), g.copy()
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        below = mid / np.tanh(np.where(mid > 0, mid, 1.0)) < g
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    out: Floats = np.where(g > 1.0, 0.5 * (lo + hi), 0.0)
    return out


def drives_of(alphas: ArrayLike) -> Floats:
    """Return the drive in dB of each α: 20·log₁₀(α / tanh α), and 0 for α = 0."""
    a = np.asarray(alphas, dtype=float)
    out: Floats = np.where(a > 0, 20.0 * np.log10(a / np.tanh(np.where(a > 0, a, 1.0))), 0.0)
    return out


def ratios(s: Synthesis, alphas: ArrayLike) -> Floats:
    """Return the rule's prediction over its limit at many α: above 1 is outside.

    The knee slew and the kink bound are exact at every α. The output peak is
    bounded from below by the probes, which can only raise the ratio, so the
    headroom errs on the safe side in both directions.
    """
    a = np.atleast_1d(np.asarray(alphas, dtype=float))
    p = s.probes.peak(a) if s.probes is not None else np.zeros(a.size)
    q = s.slew.knees.at(a)
    saturation = np.where(q > 0, SATURATION * np.exp(-(math.pi**2) / np.where(q > 0, q, 1.0)) * p, 0.0)
    out: Floats = (saturation + kink_bounds(s.kinks, a, s.rate, s.band)) / np.where(p >= QUIET, LIMIT * p, FLOOR)
    return out


def _edge(s: Synthesis, inside: float, outside: float) -> float:
    """Refine an inside/outside bracket in α, 16 points at a time, and return the inside end.

    Four rounds narrow a 0.5 dB bracket below 10⁻⁵ dB.
    """
    for _ in range(4):
        a = np.linspace(inside, outside, 17)[1:-1]
        out = np.flatnonzero(ratios(s, a) > 1.0)  # a runs from the inside end to the outside end
        if out.size:
            j = out[0]
            inside, outside = (a[j - 1] if j else inside), a[j]
        else:
            inside = a[-1]
    return float(inside)


def drive_headroom(s: Synthesis, drive_db: float, d: Domain) -> Headroom | None:
    """Find the drive at which the validated-domain status first changes, searched from the patch's own drive.

    The rule is not monotone in drive: a step near the knee counts only while
    α·reach ≤ 1, so just below the α where it stops counting the domain can
    have a narrow outside island. Candidates are every 0.5 dB of drive and
    both sides of every α at which the knee slew drops, so such an island is
    never skipped; the first change is then refined.
    """
    if d.peak == 0:
        return None
    a_max = drive_alpha(DRIVE_MAX)
    drops = s.slew.knees.drops()
    if d.inside:
        grid = np.append(np.arange(drive_db + HEADROOM_STEP_DB, DRIVE_MAX, HEADROOM_STEP_DB), DRIVE_MAX)
        below = drops * (1.0 - 1e-12)  # just before each drop, where the step still counts
        cand = np.concatenate((alphas_of(grid), below[(below > s.alpha) & (below <= a_max)]))
        cand = np.sort(cand[cand > s.alpha])
        out = np.flatnonzero(ratios(s, cand) > 1.0)
        if not out.size:
            return Headroom(math.inf)
        j = out[0]
        edge = _edge(s, cand[j - 1] if j else s.alpha, cand[j])
    else:
        grid = np.append(np.arange(drive_db - HEADROOM_STEP_DB, 0.0, -HEADROOM_STEP_DB), 0.0)
        above = drops * (1.0 + 1e-12)  # just past each drop, where the step no longer counts
        cand = np.concatenate((alphas_of(grid), above[above < s.alpha]))
        cand = np.sort(cand[cand < s.alpha])[::-1]
        inside = np.flatnonzero(ratios(s, cand) <= 1.0)
        if not inside.size:
            return Headroom(-math.inf)
        j = inside[0]
        edge = _edge(s, cand[j], cand[j - 1] if j else s.alpha)
    edge_db = float(drives_of(edge))
    return Headroom(edge_db - drive_db, edge_db)


def _floor1(x: float) -> float:
    """Round down to 0.1, so a suggested drive errs on the inside."""
    return math.floor(x * 10 + 1e-9) / 10


# Takes, the set gain and the report.


@dataclass(frozen=True)
class Take:
    """One take's own judgement and levels; the levels are those of the exported file."""

    number: int  # 1-based
    seed: int
    gain_db: float  # the take gain from level matching; 0 without
    peak: float
    true_peak: float
    loudness: float
    domain: Domain
    cause: Cause
    headroom: Headroom | None

    @property
    def inside(self) -> bool:
        """Whether the take is inside the validated domain."""
        return self.domain.inside

    def judgement(self) -> str:
        """Say whether the take is inside the validated domain, and by how much."""
        d = self.domain
        if not d.predicted:
            return "inside the validated domain (no predicted aliasing)"
        if d.inside:
            return (
                f"inside the validated domain: predicted aliasing {d.predicted_db:.0f} dB, margin {d.margin_db:.1f} dB"
            )
        return (
            f"OUTSIDE the validated domain by {-d.margin_db:.1f} dB (predicted aliasing {d.predicted_db:.0f} dB); "
            "check it with `halyard verify`"
        )

    def detail(self, drive_db: float) -> str | None:
        """Say what the prediction comes from, and what drive would change the status."""
        c, h, d = self.cause, self.headroom, self.domain
        if c.dominant is None or not d.predicted or h is None:  # no headroom is found only for silence
            return None
        text = "mostly saturation" if c.dominant == "saturation" else f"mostly kinks, {c.where}"
        if h.db == math.inf:
            return f"{text}; inside at every drive up to {DRIVE_MAX:g} dB"
        if h.db == -math.inf:
            knot = "" if c.dominant == "kinks" or c.knot is None else f", most at {c.knot} ({c.knot_ms:g} ms)"
            return f"{text}; outside even without drive: the kinks alone exceed the limit{knot}"
        if h.db >= 0:
            return f"{text}; drive headroom {_floor1(h.db):+.1f} dB"
        to = _floor1(h.edge_db)
        return f"{text}; drive headroom {to - drive_db:+.1f} dB: lower drive.db to {to:.1f} dB"


@dataclass(frozen=True)
class Report:
    """What a render says about itself: its tuning, its length, its levels and each take's judgement."""

    tune_hz: float
    modes_hz: tuple[float, ...]
    body_sounds: bool
    rims_hz: tuple[tuple[int, tuple[float, ...]], ...]  # each audible rim's number and its modes
    source_ms: float
    samples: int
    sample_rate: int
    format: str
    mode: str
    drive_db: float
    set_gain_db: float
    ceiling: float
    level_match: bool
    takes: tuple[Take, ...]
    internal_rate: int
    factor: int
    decimator: str
    render_s: float

    @property
    def peak(self) -> float:
        """The largest sample of any take."""
        return max(t.peak for t in self.takes)

    @property
    def over(self) -> bool:
        """A gain-mode render past the format's ceiling."""
        return self.mode == "gain" and self.peak > self.ceiling

    @property
    def inside(self) -> bool:
        """Whether every take is inside the validated domain."""
        return all(t.inside for t in self.takes)

    @property
    def exported_ms(self) -> float:
        """The length of the exported file."""
        return 1000 * self.samples / self.sample_rate

    @property
    def loudness_spread(self) -> float:
        """The range of the takes' loudness, in dB."""
        values = [t.loudness for t in self.takes if t.loudness > -math.inf]
        return max(values) - min(values) if values else 0.0

    @property
    def peak_spread(self) -> float:
        """The range of the takes' sample peaks, in dB."""
        values = [db(t.peak) for t in self.takes if t.peak > 0]
        return max(values) - min(values) if values else 0.0

    @property
    def uneven(self) -> bool:
        """The report flags a set whose loudness spreads over more than 1 dB (plan §7.1)."""
        return self.loudness_spread > 1.0

    def _over(self) -> str:
        """Say how far over the format's ceiling the render is, and what to do."""
        peak = self.peak
        if FORMATS[self.format] is None:
            return (
                f"{db(peak):+.2f} dBFS sample: over full scale; lower output.db by at least "
                f"{db(peak):.2f} dB, or pass --allow-over"
            )
        return (
            f"{db(peak):+.4f} dBFS sample: over the {FORMATS[self.format]}-bit ceiling of "
            f"{db(self.ceiling):.4f} dBFS, which keeps 2 LSB for dither; lower output.db by at least "
            f"{db(peak / self.ceiling):.4f} dB"
        )

    def _levels(self, t: Take) -> tuple[str, str]:
        """Describe a take's peak and loudness."""
        if t.peak == 0:
            peak = "silent"
        elif self.over and t.peak == self.peak:
            peak = self._over()
        else:
            peak = f"{db(t.peak):.2f} dBFS sample, {db(t.true_peak):.2f} dBTP true"
        loudness = (
            f"{t.loudness:.1f} LUFS, the largest 400 ms momentary loudness (BS.1770)"
            if t.loudness > -math.inf
            else "silent"
        )
        return peak, loudness

    def lines(self) -> list[str]:
        """Describe the render, a labelled line at a time."""
        modes = ", ".join(f"{f:.1f}" for f in self.modes_hz)
        tuning = f"{self.tune_hz:.3f} Hz, {describe_tuning(self.tune_hz)}"
        if self.body_sounds:
            tuning += f"; modes {modes} Hz"
        else:
            tuning += " (the body is silent)"
        for k, hz in self.rims_hz:
            tuning += f"; rim {k} " + ", ".join(f"{f:.1f}" for f in hz) + " Hz"
        pcm = self.format if FORMATS[self.format] is None else f"{FORMATS[self.format]}-bit PCM"
        first = self.takes[0]
        seeds = f"seed {first.seed}" if len(self.takes) == 1 else f"seeds {first.seed}–{self.takes[-1].seed}"
        rows = [
            ("tuning", tuning),
            (
                "length",
                (
                    f"{self.source_ms:.1f} ms source, {self.exported_ms:.1f} ms exported "
                    f"({self.samples} samples at {self.sample_rate} Hz, {pcm}), {seeds}"
                ),
            ),
        ]
        if len(self.takes) == 1:
            peak, loudness = self._levels(first)
            rows += [("peak", peak), ("loudness", loudness), ("domain", first.judgement())]
            detail = first.detail(self.drive_db)
            if detail:
                rows.append(("", detail))
        else:
            match = ", level-matched to take 1" if self.level_match else ""
            rows.append(("takes", f"{len(self.takes)} takes sharing one gain of {self.set_gain_db:+.2f} dB{match}"))
            spread = f"loudness spread {self.loudness_spread:.2f} dB, sample-peak spread {self.peak_spread:.2f} dB"
            if self.uneven:
                spread += "; the takes vary by more than 1 dB: consider --level-match"
            rows.append(("", spread))
            if self.over:
                rows.append(("peak", self._over()))
            for t in self.takes:
                gain = f", take gain {t.gain_db:+.2f} dB" if self.level_match else ""
                level = "silent" if t.peak == 0 else f"{db(t.peak):.2f} dBFS, {t.loudness:.1f} LUFS"
                rows.append((f"take {t.number}", f"seed {t.seed}: {level}{gain}"))
                rows.append(("", t.judgement()))
                detail = t.detail(self.drive_db)
                if detail:
                    rows.append(("", detail))
        rows.append(("render", f"{1000 * self.render_s:.0f} ms"))
        width = max(9, max(len(label) for label, _ in rows) + 1)
        return [f"{label:<{width}}{text}" for label, text in rows]

    @override
    def __str__(self) -> str:
        """Describe the render."""
        return "\n".join(self.lines())


def take_seeds(p: Patch, takes: object) -> list[int]:
    """Return the takes' seeds: take j plays seed + j − 1 (plan §6)."""
    if isinstance(takes, bool) or not isinstance(takes, int) or not 1 <= takes <= MAX_TAKES:
        msg = f"--takes: needs 1 to {MAX_TAKES} takes, found {takes!r}"
        raise RenderError(msg)
    if p.seed + takes - 1 > MAX_SEED:
        msg = f"--takes: seeds {p.seed} … {p.seed + takes - 1} pass 2**63 - 1; lower the patch's seed"
        raise RenderError(msg)
    return [p.seed + j for j in range(takes)]


def set_gains(synths: list[Synthesis], p: Patch, top: float, *, level_match: bool) -> tuple[float, list[float]]:
    """Return one gain for the whole set, and each take's gain from level matching (1 without)."""
    rate = p.output.sample_rate
    takes = [1.0] * len(synths)
    if level_match:
        loud = [momentary_max(s.out, rate) for s in synths]
        takes = [10.0 ** ((loud[0] - x) / 20.0) if loud[0] > -math.inf and x > -math.inf else 1.0 for x in loud]
    if p.output.mode == "gain":
        return 10.0 ** (p.output.db / 20.0), takes
    peak = max(s.peak(p.output.mode) * g for s, g in zip(synths, takes, strict=True))
    target = min(10.0 ** (p.output.db / 20.0), top)
    return (target / peak if peak > 0 else 1.0), takes


def output_stage(out: Floats, gain: float, polarity: int) -> NDArray[np.float32]:
    """Apply the gain, then the polarity, then one conversion to float32."""
    scaled = out * gain
    if polarity < 0:
        scaled = -scaled
    return scaled.astype(np.float32)


def render_takes(p: Patch, takes: int = 1, *, level_match: bool = False) -> tuple[list[NDArray[np.float32]], Report]:
    """Return the float32 audio of each take, before any PCM quantization, and the report of the set."""
    start = time.perf_counter()
    seeds = take_seeds(p, takes)
    top = ceiling(p.output.format)
    synths = [synthesize(dataclasses.replace(p, seed=s)) if s != p.seed else synthesize(p) for s in seeds]
    gain, take_gains = set_gains(synths, p, top, level_match=level_match)
    audio: list[NDArray[np.float32]] = []
    rows: list[Take] = []
    for j, (seed, s, g) in enumerate(zip(seeds, synths, take_gains, strict=True), 1):
        a = output_stage(s.out, gain * g, p.output.polarity)
        d = domain(s)
        audio.append(a)
        rows.append(
            Take(
                j,
                seed,
                db(g),
                float(np.max(np.abs(a), initial=0.0)),
                s.true_peak * gain * g,
                momentary_max(a, p.output.sample_rate),
                d,
                cause_of(p, s, d),
                drive_headroom(s, p.drive.db, d),
            ),
        )
    v = voice(p)
    report = Report(
        tune_hz=p.body.tune_hz,
        modes_hz=tuple(p.body.tune_hz * r for r, _ in p.body.modes),
        body_sounds=v.body_sounds,
        rims_hz=tuple((k, tuple(hz for hz, _ in r.modes)) for k, r in enumerate(p.rim, 1) if r.level_db > -math.inf),
        source_ms=p.length_ms,
        samples=audio[0].size,
        sample_rate=p.output.sample_rate,
        format=p.output.format,
        mode=p.output.mode,
        drive_db=p.drive.db,
        set_gain_db=db(gain),
        ceiling=top,
        level_match=level_match,
        takes=tuple(rows),
        internal_rate=synths[0].rate,
        factor=synths[0].factor,
        decimator=decimator(FAMILY[p.output.sample_rate]).sha256,
        render_s=time.perf_counter() - start,
    )
    return audio, report


def render(p: Patch) -> tuple[NDArray[np.float32], Report]:
    """Return the float32 audio of the patch's own take, and its report."""
    audio, report = render_takes(p, 1)
    return audio[0], report


# Export.


def quantize(audio: NDArray[np.float32], bits: int, seed: int) -> NDArray[np.int32]:
    """Round to `bits`-bit PCM with the seed's TPDF dither of ±1 LSB.

    Exact zeros stay zero, so the silent first sample and any digital silence
    stay silent. A render that is not over keeps its peak 2 LSB below full
    scale, so the dithered value always fits; an over one is never written.
    """
    top = 2 ** (bits - 1)
    x = audio.astype(float) * top
    q = np.round(np.where(x != 0.0, x + noise.dither(seed, x.size), 0.0))
    return q.astype(np.int32)


class Stamp(NamedTuple):
    """Which take a file holds, of how many, and the gains it was rendered with."""

    seed: int
    take: int
    takes: int
    set_gain_db: float
    take_gain_db: float


def provenance(p: Patch, stamp: Stamp, *, k: int = K) -> dict[str, Any]:
    """Return what a file records of how it was made: renderer, take, gains, filter and libraries."""
    rate, factor = rates(p.output.sample_rate, k)
    seed = stamp.seed
    meta: dict[str, Any] = {
        "renderer": f"halyard {__version__}",
        "seed": seed,
        "take": stamp.take,
        "takes": stamp.takes,
        "set_gain_db": stamp.set_gain_db,
        "take_gain_db": stamp.take_gain_db,
        "decimator": {
            "sha256": decimator(FAMILY[p.output.sample_rate], k).sha256,
            "internal_rate": rate,
            "factor": factor,
        },
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    bits = FORMATS[p.output.format]
    if bits is not None:
        meta["pcm"] = {
            "bits": bits,
            "dither": "TPDF, ±1 LSB; exact zeros kept",
            "dither_key": noise.DITHER_KEY.decode(),
            "seed": seed,
        }
    return meta


def write_take(
    path: str | os.PathLike[str],
    audio: NDArray[np.float32],
    p: Patch,
    stamp: Stamp,
    *,
    extra: dict[str, Any] | None = None,
) -> None:
    """Write one take's WAV in the patch's format, with its root note, its own patch text and provenance.

    The patch text is the take's: the patch with the take's seed, so `halyard
    extract` recovers exactly the patch that renders this file.
    """
    bits = FORMATS[p.output.format]
    samples = audio if bits is None else quantize(audio, bits, stamp.seed)
    root = wav.root_note(p.body.tune_hz) if voice(p).body_sounds else None
    meta = {
        "patch": dumps(dataclasses.replace(p, seed=stamp.seed)),
        **provenance(p, stamp),
        **(extra or {}),
    }
    wav.write(path, samples, wav.Format(p.output.sample_rate, bits), root=root, meta=meta)


def export(path: str | os.PathLike[str], audio: NDArray[np.float32], p: Patch, report: Report, take: Take) -> None:
    """Write one take of a render."""
    write_take(path, audio, p, Stamp(take.seed, take.number, len(report.takes), report.set_gain_db, take.gain_db))
