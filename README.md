# Halyard

A one-voice snare and clap synthesizer, rendered offline to mono WAV. It
implements the *Snare/Clap Synth design and implementation plan, revision 4*
([docs/PLAN.md](docs/PLAN.md)): a modal body and up to four layers of one
seeded noise are summed into one saturator and band-limited once by a frozen
minimum-phase decimator.

    x(t) = b(t) + Σ_j w_j(t),    y(t) = S_α(x(t)),    out[m] = p · g · Σ_k h[k] · y((mK − k) / F)

Everything before the decimator is a pure function of time, so a render
depends only on the patch, its seed, the renderer version and the frozen
filter. Each render states whether its aliasing is demonstrated below
−100 dB, by how much, what dominates it and how much more drive it takes to
leave that domain.

The user guide is [docs/GUIDE.md](docs/GUIDE.md): the workflow, every
control, recipes and how to read the report.

## Install

Halyard needs Python 3.12 or later.

```
uv tool install "halyard-synth[audition]"
```

or `pipx install "halyard-synth[audition]"`. The `audition` extra adds live
playback through sounddevice and PortAudio; without it every other command
works. From a checkout, `uv tool install ".[audition]"`; to work on Halyard,
see [CONTRIBUTING.md](CONTRIBUTING.md).

## Use

```
halyard init crack --tune D3 > snare.toml               # a complete starting patch (writes tune_hz = 146.832)
halyard audition snare.toml --watch --bpm 92 --with groove.wav   # on the backbeat, over the track's loop
halyard audition snare.toml --watch --against favourite.wav --blind
halyard render snare.toml --out snare.wav --takes 8      # round-robin takes snare_rr1.wav … snare_rr8.wav
halyard verify snare.toml --takes 8                      # each take against its reference render
halyard extract snare_rr3.wav > take3.toml               # the exact patch of any exported file
halyard tune snare.toml F3                               # retune: modes, glide and rattle follow
halyard stretch snare.toml all 0.8                       # fit every shape to a tempo
halyard bursts clap.toml 'noise[1]' --count 4 --spacing 11   # a clap train as ordinary knots
halyard softer snare.toml 12 > snare-12.toml             # the same drum struck 12 dB softer
halyard pack snare.toml snare-8.toml snare-16.toml snare-24.toml --takes 6 --out kit/
```

The six references are `tone`, `crack`, `fat`, `driven`, `clap` and `snap`.
They are matched in loudness (−27 LUFS), peak below −1 dBFS over seeds 1–16,
and every take of them is inside the validated domain with at least 16 dB of
drive headroom. They are starting points: their names never select hidden
algorithms.

The editing commands print a complete, canonical patch and never add a
parameter, so every change is visible and can be edited further. Knots,
layers and modes are numbered from 1, as the parser's messages and the report
number them: `noise[2].amp[3].ms` is the time of the third knot of the second
layer.

## The patch

A patch is a complete TOML description of one hit. Every field is required and
the renderer has no defaults, so a future default can never change an old
patch. The parser rejects anything outside the valid domain and never clamps
or repairs.

```toml
instrument = "snare"
schema = 1
seed = 1

[body]
tune_hz  = 185.0
level_db = -4.0
damping  = 1.0              # mode i decays ratio^damping times faster
modes = [                   # loudest mode exactly 0 dB
  { ratio = 1.0,   level_db = 0.0 },
  { ratio = 1.593, level_db = -5.0 },
  { ratio = 2.136, level_db = -11.0 },
]
pitch = [                   # ratio to tune_hz; starts at 0 ms, ends at 1
  { ms = 0.0,  ratio = 1.08 },
  { ms = 15.0, ratio = 1.0 },
]
amp = [                     # shape: 0 at both ends, highest point exactly 1
  { ms = 0.0,   gain = 0.0 },
  { ms = 0.4,   gain = 1.0 },
  { ms = 25.0,  gain = 0.45 },
  { ms = 90.0,  gain = 0.1 },
  { ms = 180.0, gain = 0.0 },
]

[[noise]]                   # noise[1]
level_db = -10.0
band_hz  = [1500.0, 16000.0]
rattle   = 0.0              # 0 … 1, gated at the body's fundamental
amp = [
  { ms = 0.0,  gain = 0.0 },
  { ms = 0.05, gain = 1.0 },
  { ms = 0.15, gain = 0.7 },
  { ms = 3.0,  gain = 0.1 },
  { ms = 8.0,  gain = 0.0 },
]

[[noise]]                   # noise[2]
level_db = -6.0
band_hz  = [2500.0, 9000.0]
rattle   = 0.5              # 0 … 1, gated at the body's fundamental
amp = [
  { ms = 0.0,   gain = 0.0 },
  { ms = 1.5,   gain = 1.0 },
  { ms = 40.0,  gain = 0.4 },
  { ms = 140.0, gain = 0.08 },
  { ms = 260.0, gain = 0.0 },
]

[drive]
db = 6.0                    # small-signal gain; 0 = bypass

[output]
sample_rate = 48000
format      = "float32"     # "float32" | "pcm24" | "pcm16"
mode        = "gain"        # "gain" | "peak"
db          = -6.0          # gain in dB, or peak target in dBFS
polarity    = 1
```

In brief:

- **Body.** One to four modes at fixed ratios of `tune_hz`, all following one
  pitch glide. With `damping` β, mode i plays the body shape's decay r_i^β
  times faster from the peak on.
- **Noise layers.** Every layer uses the same seeded noise, weighted by its
  own band. Identical bands give identical noise, a crossfade between two
  layers is a spectral morph, and editing a band reweights the same noise
  instead of rerolling it. `rattle` gates a layer at the body's fundamental,
  glide included, as snare wires slap the head.
- **Drive.** `drive.db` is the saturator's small-signal gain on the sum; 0 dB
  is exact bypass.
- **Seed.** Each seed is a take of the same sound. `--takes N` renders seeds
  seed, seed + 1, … as round robins that share one gain.

The valid domain:

| Quantity | Valid range |
|---|---|
| `tune_hz` | 40 – 4,000 |
| modes | 1 – 4; the first ratio exactly 1; strictly increasing, ≤ 8 |
| mode levels | −60 – 0 dB, with the loudest exactly 0 |
| body frequency f₀·r_i·r(t) at every pitch knot | 10 Hz – 20 kHz |
| `damping` | 0 – 3 |
| body and layer `level_db` | −inf (silence), or −90 – +24 |
| noise layers | 1 – 4 |
| `band_hz` | 20 ≤ low < high ≤ 20,000 |
| `rattle` | 0 – 1 |
| drive | 0 – 48 dB |
| knots per curve | pitch 2 – 32, amplitude 3 – 32 |
| knot spacing | ≥ 10 µs (compared on the decimal values as written) |
| body curves / noise curves | end by 2 s / by 1 s |
| output | 44.1, 48, 88.2 or 96 kHz; float32, pcm24 or pcm16; gain −60 – +24 dB; peak target −60 – 0 dBFS |
| `seed` | 0 – 2⁶³ − 1, takes included |

## What a render reports

```
tuning   185.000 Hz, F#3 +0.0 cents; modes 185.0, 294.7, 395.2 Hz
length   260.0 ms source, 261.9 ms exported (12571 samples at 48000 Hz, float32), seed 1
peak     -4.05 dBFS sample, -3.74 dBTP true (4× estimate, information only)
loudness -19.0 LUFS, the largest 400 ms momentary loudness (BS.1770)
domain   inside the validated domain: predicted aliasing -137 dB, margin 37.0 dB
         mostly kinks, 64% at noise[1].amp[2] (0.05 ms); drive headroom +20.6 dB
render   99 ms
```

The domain line is the renderer's own prediction of its aliasing error,
relative to the output peak; inside means at most −100 dB. The line under it
says what dominates the prediction (saturation, or the kinks with the knot
that contributes most) and how much more drive the take stands before its
status changes. Outside, the headroom is negative and the line says what to
lower `drive.db` to. For several takes the report gives each take's own
judgement, the loudness and peak spread, and flags a set whose loudness
spreads over more than 1 dB. GUIDE.md explains every line.

## How it renders

- **Rates.** The internal rate is 32× the base rate: 1.4112 MHz for the
  44.1 kHz family, 1.536 MHz for 48 kHz. The 88.2 and 96 kHz exports reuse
  their family's rate and filter with factor 16, so a 96 kHz file equals the
  48 kHz file with the in-between samples filled in, bit for bit.
- **Decimator.** One frozen minimum-phase FIR per family, passing 0–20 kHz
  within 0.001 dB and attenuating at least 110 dB from base/2 up. It ships
  as data with a SHA-256 checksum, so the sound cannot drift with SciPy
  versions.
- **Evaluation.** The internal signal is computed in blocks of 2¹⁶ samples
  aligned to the trigger, each source only over its own support, with curves
  evaluated segment by segment. A longer render never changes earlier
  samples. The noise tables of a render's layers are computed in parallel and
  cached; decimation runs in parallel chunks that are bit-identical to one
  `upfirdn` call.
- **Output.** One gain for the whole set (the output gain, or in peak mode
  the gain that brings the loudest take to the target), then polarity, then
  one conversion to float32, or one quantization to PCM with seeded TPDF
  dither that leaves exact zeros alone.
- **File.** `fmt`, `fact` (float), `smpl` with the root key (left out when
  the body is silent), a private `snar` chunk with the take's own patch text,
  its seed and gains, the renderer version, the decimator checksum and the
  dependency versions, and `data`.

## Development

```
uv run pytest                                    # the full suite (about a minute)
uv run python -m tools.design_decimator --check
uv run python -m tools.grid run --k 32           # the adversarial grid (about 10 minutes on 4 cores)
uv run python -m tools.grid analyze docs/grid/k16.csv docs/grid/k32.csv
uv run python -m tools.headroom run --dense      # the report's drive headroom against full renders (hours)
uv run python -m tools.headroom analyze docs/grid/headroom.csv
uv run python -m tools.references                # the six references over seeds 1–16
uv run python -m tools.bench                     # render time and memory on this machine
uv run python -m tools.golden                    # regenerate golden renders (a deliberate sound change only)
```

[docs/VALIDATION.md](docs/VALIDATION.md) records every numerical criterion of
the plan with its measured value, the adversarial grid, the choice of K, the
refit of the validated-domain rule and performance.
[docs/BENCHMARK.md](docs/BENCHMARK.md) is the protocol for the
sample-replacement benchmark.

| Module | Responsibility |
|---|---|
| `src/halyard_synth/patch.py` | schema, parsing, validation, canonical TOML; `tune`, `stretch`, `bursts`, `softer`; tuning helpers |
| `src/halyard_synth/curve.py` | Hermite curves, PCHIP slopes, exact integrals, knot jumps, decay compression |
| `src/halyard_synth/noise.py` | seeded noise, coherent layer tables (cached), values and slopes between samples, seeded dither |
| `src/halyard_synth/voice.py` | modal body, noise layers with rattle, drive; knee slew at every drive, kink bound |
| `src/halyard_synth/render.py` | rates, frozen decimators, parallel decimation, domain rule and drive headroom, takes, report, export |
| `src/halyard_synth/loudness.py` | BS.1770 K-weighting at every rate, momentary loudness, true peak |
| `src/halyard_synth/pack.py` | velocity layers, takes, one gain per pack, the SFZ mapping |
| `src/halyard_synth/reference.py` | the reference renderer and comparison behind `halyard verify` |
| `src/halyard_synth/wav.py` | RIFF writing and reading, float and PCM |
| `src/halyard_synth/audition.py` | hits on the beat grid, patches and takes cycling, beds, A/B and blind order, watching, playback |
| `src/halyard_synth/cli.py` | commands |
| `src/halyard_synth/data/` | frozen decimators and checksums, the six references |

## Status

| Milestone | State |
|---|---|
| 1. Complete path | Done. Every §8.1 criterion that does not need the grid passes (VALIDATION.md). |
| 2. Measure and freeze | Done. K = 32, the rule's constant (0.5) and the decimators are frozen from a 957-fixture grid at 16× and 32×; the drive headroom is checked on every fixture; golden renders begin. |
| 3. Audition and delivery | Done here with a simulated device: exact buffers, stale renders dropped, swaps at hit boundaries, patches and takes cycling, the bed on the grid. Packs parse back from their SFZ with the right velocity and take mapping. Playing through a real device and loading packs in two SFZ players needs you. |
| 4. Voicing | By measurement: the references are loudness-matched, inside with 16 dB or more of drive headroom for seeds 1–16, and every usability task moves its measurement the right way. Approval by ear, in context and across takes and strengths, needs you. |
| 5. Benchmark | Protocol written (BENCHMARK.md); it needs your samples and your ears. |

## License

Halyard is free software, under the [GNU General Public License](LICENSE),
version 3 or later.

The sounds you make with it are yours. An additional permission in
[NOTICE](NOTICE) lets you use the sounds Halyard renders, the patches and
SFZ maps it writes, and its reference patches under any terms you choose, in a
track or in a sample pack you sell.
