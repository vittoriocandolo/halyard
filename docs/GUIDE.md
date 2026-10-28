# User guide

This guide is for producers and sound designers who program snares and claps.
It covers the workflow, every control, recipes for common edits, and how to
read what a render reports. The design and its evidence are in
[PLAN.md](PLAN.md) and [VALIDATION.md](VALIDATION.md).

## The workflow

1. **Start from the nearest reference.** `halyard init crack --tune D3 > snare.toml`.
   The references are `tone`, `crack`, `fat`, `driven`, `clap` and `snap`
   (below). `--tune` takes a note name or a frequency; the patch stays in Hz.
2. **Voice it in context.** `halyard audition snare.toml --watch --bpm 92 --with groove.wav`
   plays the snare on beats 2 and 4 over your loop and re-renders every time
   you save the file. Edits take over at the next hit.
3. **Match it against the sample it should replace.** `halyard audition snare.toml --watch --against favourite.wav --blind`
   alternates your render with the sample, matched in loudness, in a random
   order that audition reveals when you stop it.
4. **Derive softer layers.** `halyard softer snare.toml 8 > snare-8.toml`, and so on.
5. **Pack them with takes for the sampler.** `halyard pack snare.toml snare-8.toml snare-16.toml snare-24.toml --takes 6 --out kit/`.

Every file is exact and reproducible: `halyard extract` gives back the exact
patch of any exported file, take included.

## The references

| Reference | What it is | Try |
|---|---|---|
| `tone` | Two tonal modes and a soft click over bright, snappy noise; round and clearly pitched | tuning, damping, the second mode's level |
| `crack` | A quick glide, a sharp click and bright wires with a short decay; tight | the stick's level and rise, `stretch` |
| `fat` | A lower tuning, a longer body, buzzing wires and a tail that darkens | rattle, the dark layer's band and length |
| `driven` | Body and noise saturated together; dense, still pitched | drive, the wires' level |
| `clap` | Four bursts and a diffuse tail, with no body | `bursts`, the bands, the tail layer |
| `snap` | A short, bright burst over a high ping that dies fast; a finger snap or rim click | the ping's tuning, the burst's band |

They are matched in loudness (−27 LUFS) so you can compare them fairly, and
they peak below −1 dBFS. Raise `output.db` as you need.

## Every control

A patch is complete: every field is required, and the renderer has no
defaults. Paths below are the ones the parser and the report use. They count
from 1: `noise[2].amp[3]` is the third knot of the second layer.

### Times and curves

Knot times are milliseconds from the hit. Knots must be at least 10 µs apart.
Every curve is a smooth monotone cubic through its knots (PCHIP): it never
overshoots between knots, so an amplitude shape stays within 0 … 1 and the
pitch is extreme only at its knots.

- **Amplitude shapes** (`body.amp`, `noise[N].amp`) start and end at gain 0,
  and their highest point is exactly 1. Levels are set separately, in dB.
- **Knots every 10 dB** of a decay reproduce an exponential tail within
  0.15 dB. The first segment after a peak starts flat, because the peak has
  zero slope, and that rounds the top of a decay by 1–5 dB. For a crack, put a
  **shoulder knot** within about 0.1 ms after the peak, as the references and
  `bursts` do.
- **Every knot is a kink.** Very fast segments under loud noise alias. The
  report tells you when a knot matters and which one.

### Body

| Field | Range | What it does |
|---|---|---|
| `tune_hz` | 40 – 4,000 | The settled frequency of the first mode. Up to 4 kHz, so a snap or rim click can be a single high ping. |
| `level_db` | −inf, −90 – +24 | The body's level. `-inf` silences it: the body then still gates rattle, but never lengthens the file. |
| `damping` | 0 – 3 | Mode i decays ratio_i^damping times faster than the fundamental, from the body's peak on. 0: all modes decay together. |
| `modes` | 1 – 4 | `{ ratio, level_db }`. The first ratio is exactly 1; ratios increase strictly and stay ≤ 8. The loudest mode is exactly 0 dB: mode levels are a spectral shape, the body's level is its gain. The ideal membrane gives 1, 1.593, 2.136 and 2.295. |
| `pitch` | 2 – 32 knots | A ratio to `tune_hz`, starting at 0 ms and ending at 1. Every mode follows it: tension modulation raises every mode of a head together. |
| `amp` | 3 – 32 knots | The body shape. Its first knot at gain 1 is the peak, where damping starts. |

Every body frequency, f₀ · ratio · glide, must stay within 10 Hz – 20 kHz at
every pitch knot. `halyard tune` refuses a tuning that would take a mode out of
that range; it never clamps.

### Noise layers

One to four `[[noise]]` tables. Every layer uses the same noise, weighted by
its own band, so layers are coherent: identical bands give identical noise,
two identical layers at equal level sum to +6 dB, and a crossfade between two
layers is a spectral morph of one noise.

| Field | Range | What it does |
|---|---|---|
| `level_db` | −inf, −90 – +24 | The layer's level. |
| `band_hz` | 20 ≤ low < high ≤ 20,000 | Second-order skirts, half power at the edges. Editing the band reweights the same noise. |
| `rattle` | 0 – 1 | Gates the layer at the body's fundamental, following its glide, while the body shape sounds: snare wires. 0 is off, 1 is full depth. |
| `amp` | 3 – 32 knots | The layer's shape; it ends by 1 s, because the noise repeats every second. |

A typical snare has a short, wide **stick** layer (the click) and a longer
**wires** layer, and perhaps a darker, longer layer so the tail darkens.
Layers are how the noise changes colour over time: a bright, short layer over
a darker, longer one.

### Drive

`drive.db`, 0 – 48 dB, is the small-signal gain of a tanh saturator on the sum
of everything. 0 dB is exact bypass. On noise, what matters is **noise level +
drive**: a layer at −6 dB with 24 dB of drive saturates like one at 0 dB with
18 dB.

### Output

| Field | Values | What it does |
|---|---|---|
| `sample_rate` | 44100, 48000, 88200, 96000 | The file's rate. The 96 kHz file is exactly the 48 kHz one with samples filled in. |
| `format` | `float32`, `pcm24`, `pcm16` | PCM uses the plain PCM tag that hardware samplers read, with seeded triangular dither that leaves exact zeros alone. |
| `mode` | `gain`, `peak`, `true-peak` | Apply `db` as a gain, or scale so the sample peak, or the true peak, is at `db` dBFS. |
| `db` | gain −60 – +24; peak −60 – 0 | See `mode`. |
| `polarity` | 1, −1 | Inverts the output. |

A gain-mode render over full scale is refused, with the reduction it needs;
`--allow-over` writes it anyway as a float file. Nothing is ever clipped. A
PCM render must peak 2 LSB below full scale so the dither never clips; peak
mode lowers its target by those 2 LSB. The true peak is the largest value of
the file's band-limited reconstruction, the level a converter reaches between
the samples, measured exactly from the render; use `true-peak` when a
limiter or a streaming target counts in dBTP.

### Seed and takes

`seed` picks the noise. The noise is the instrument, so each seed is a
**take** of the same sound, slightly different, as real hits are.
`halyard render … --takes 8` writes seeds seed, seed + 1, … as
`snare_rr1.wav` … `snare_rr8.wav`. The takes of a set share one gain: in peak
mode, the gain that brings the loudest take to the target. The others keep
their natural levels, because takes differ more in peak (1.5–7 dB) than in
loudness (0.05–3 dB). For sounds whose takes vary a lot, such as short snaps,
`--level-match` scales each take to the loudness of the first; the report
flags any set whose loudness spreads over more than 1 dB.

## Editing commands

Each prints a complete patch; redirect it to a file. They rewrite points and
never add parameters, so you can always keep editing the result by hand. They
fail, and say why, rather than clamp.

- **`halyard tune PATCH NOTE`** sets `tune_hz` from D3, F#3, Eb4 or a frequency
  in Hz. Modes, glide and rattle follow, being ratios.
- **`halyard stretch PATCH CURVE... FACTOR [--after MS]`** rescales knot times
  after each shape's peak (0 ms for pitch), or after `--after`. `all` names
  every amplitude shape, so `halyard stretch snare.toml all 0.8` shortens a whole
  sound; `--after 0` also scales the rises, which tightens a clap.
- **`halyard bursts PATCH LAYER --count N --spacing MS …`** rewrites a layer's
  shape as a clap train: N bursts (1–10), each rising in `--rise` ms (default
  0.1) to its peak, a shoulder 0.1 ms after, and down to `--dip` (default 0.1,
  over the lower of the two peaks) before the next. Each gap is `--growth`
  times the previous; peaks follow `--taper` with the loudest exactly 1; the
  last burst decays over `--tail` ms (default 80) with a knot 20 dB down a
  third of the way. `--start MS` delays the first burst, so two layers with
  different bands and starts make a stacked clap.
- **`halyard softer PATCH DB [--glide P] [--contact P]`** gives the same drum
  struck DB softer, by three rules from the physics of a struck head:
  - every source level falls by DB, before the drive, so a softer hit is also
    cleaner;
  - the glide's excess over 1 scales by v^glide (default 2: tension modulation
    follows the vibration's energy), with v = 10^(−DB/20);
  - every rise lengthens by v^(−contact) and every layer's upper band edge
    scales by v^contact (default 0.2: Hertz's contact law, a slower strike
    excites less high end).

  At 12 dB softer, a 1.08 glide becomes 1.005, a 0.05 ms rise 0.066 ms and a
  16 kHz edge 12.1 kHz. An exponent of 0 turns its rule off. `softer` by a then
  b equals `softer` by a + b up to the decimals it writes.

## Recipes

These start from a reference and take at most three field edits or one
command; the test suite checks that each moves the sound the right way.

| To | Edit |
|---|---|
| Change the tuning | `halyard tune snare.toml F3`, or edit `body.tune_hz` |
| Shorten the tail | `halyard stretch snare.toml all 0.7` |
| Add crack | raise the stick layer's `level_db` (the short, wide layer) by 4–6 dB, or shorten its rise |
| Add body | raise `body.level_db` |
| Add wires | raise the wires layer's `level_db` and its `rattle` |
| Darken | lower each layer's upper band edge (`band_hz[2]`) |
| Add aggression | raise `drive.db` by 6 and lower `output.db` to taste. In a driven sound, the edge lives in the saturated source: raising the stick adds less than in a clean one. |
| Make a ghost note | `halyard softer snare.toml 18` |
| Tighten a clap | `halyard stretch clap.toml all 0.7 --after 0`, or re-run `bursts` with a shorter `--spacing` |
| Darken a clap | lower both edges of each layer's band |
| Shorten or brighten a snap | `halyard stretch snap.toml all 0.6`; raise the burst's band and level |
| Move a snap's ping | edit `body.tune_hz` |

**Pitched buzz.** Rattle reads the body's shape and phase, not its level, so a
silent body can still gate a layer: a buzz at a pitch with no tone of its own.

## Reading the report

```
tuning   185.000 Hz, F#3 +0.0 cents; modes 185.0, 294.7, 395.2 Hz
length   260.0 ms source, 261.9 ms exported (12571 samples at 48000 Hz, float32), seed 1
peak     -4.05 dBFS sample, -3.72 dBTP true
loudness -19.0 LUFS, the largest 400 ms momentary loudness (BS.1770)
domain   inside the validated domain: predicted aliasing -137 dB, margin 37.0 dB
         mostly kinks, 64% at noise[1].amp[2] (0.05 ms); drive headroom +21.6 dB
render   86 ms
```

- **tuning** is `tune_hz` as a note and cents, and each mode's settled
  frequency.
- **length** is the source length (the latest end among the audible sources'
  shapes: silent sources never lengthen a file) and the exported length, which
  includes the decimator's decaying tail. Every file starts at exactly zero and
  ends in that tail, so a sampler needs no fades.
- **peak** is the sample peak and a 4× oversampled true-peak estimate.
- **loudness** is the largest K-weighted 400 ms momentary loudness over every
  window position (ITU-R BS.1770), in LUFS. Short sounds read low, because the
  window spreads their energy over 400 ms.
- **domain** is the renderer's own prediction of its aliasing error, relative
  to the peak. Inside means at most −100 dB: the grid has demonstrated that
  every render inside measures below −100 dB (VALIDATION.md). Outside, the
  render is still defined and exact, but unvalidated; `halyard verify` measures
  its actual error against a reference render.
- **The line under it** says what dominates the prediction:
  - *mostly kinks, 64% at noise[1].amp[2] (0.05 ms)*: a fast segment around
    that knot under a loud carrier. Lengthen the rise around it, or lower that
    layer.
  - *mostly saturation*: the drive on a loud, bright source. Lower the drive,
    the level, or the layer's upper band edge.

  The **drive headroom** is how much more drive the take stands before its
  status changes. Outside, it is negative and the line says what to lower
  `drive.db` to. It is exact or cautious, never optimistic. Margin and
  headroom differ: near the edge the prediction moves 11–12 dB per dB of
  drive, so a large margin can mean little headroom.
- **takes** (with `--takes`): the set gain, the loudness and peak spread, and
  each take's own levels and domain status. Each take is judged on its own
  render.

## Audition

`halyard audition PATCH... [--watch] [--bpm BPM] [--beats 2,4] [--with BED] [--against SAMPLE] [--blind] [--takes N]`

- It plays exactly the buffer `render` writes; PCM formats add only their
  quantization. It opens the device at the patch's sample rate and never
  resamples.
- `--watch` re-renders on every save. Each edit is a numbered generation;
  stale renders are dropped, an invalid edit keeps the last valid buffer
  playing, and a new buffer takes over at the next hit.
- Without a tempo, hits repeat every rendered length plus 0.5 s. With
  `--bpm`, they land on beats 2 and 4 of each 4/4 bar, tails overlapping;
  `--beats 1,2,3,4` hits every beat (a clap), `--beats 3` is half time.
- `--with BED` loops a bed under the hits. It must be at the patch's rate and
  within 1 ms of a whole number of bars; it restarts on the tempo grid, so it
  never drifts. A stereo bed opens a stereo stream with the snare centred.
- `--against SAMPLE` alternates the render with a sample, hit by hit, the
  sample matched to the render's loudness and rematched on every re-render.
  With `--blind` each pair plays in random order, revealed when you stop.
- Several patches cycle hit by hit (velocity layers, or A/B), and `--takes N`
  cycles takes. The first plays as soon as it is ready; the others render
  behind it.
- It warns when the matched sample, or the hits over the bed, would clip the
  device.

## Packs

`halyard pack PATCH... --out DIR [--takes N] [--key K] [--level-match]` renders
a multisample: each patch is one velocity layer, loudest first, and each gets
N takes (take j of a layer plays its seed + j − 1).

- The layers must share their output settings, and their loudness must fall
  from each layer to the next.
- Every file of the pack shares one gain, so the layers keep their natural
  levels.
- Files are named `<stem>_v<top velocity>_rr<take>.wav`, so they also map by
  hand in samplers without SFZ.
- `<stem>.sfz` maps them. Velocity follows a square law, 40·log₁₀(v/127) dB:
  each layer sits where the law reaches its loudness relative to the loudest,
  neighbours meet halfway in dB, and each layer's group carries a curve point
  at every velocity in its range, so the level follows the law across layer
  boundaries. Takes cycle in sequence; the drum never transposes. The key
  defaults to 38, the General MIDI snare, or 39, the hand clap, when no layer
  has a body.
- The pack report lists each layer's loudness, velocity range, take spread and
  domain status.

Neighbouring layers more than 12 dB apart need a group volume above +6 dB,
which some players cap; the pack report says so.

## Outside the family

Acoustic snares, rimshots, cross-stick and brushes; rolls and flams;
processing chains such as gated reverb; and stereo width are outside the
model. Two takes panned apart widen a snare's noise while its body, identical
in every take, stays centred.
