# Halyard

A snare and clap synthesizer that renders a short text patch to a sampler-ready WAV, the same way every time. A modal
body, up to four layers of one seeded noise, up to two rims and a brush are summed into one saturator, computed at 32
times the sample rate and band-limited once by a frozen minimum-phase filter. Each render states whether its aliasing is
demonstrated below −100 dB, what dominates it and how much more drive it would take to leave that domain, and every file
carries the patch that made it.

Hear the eighteen presets, with their figures, reports and patches, in [docs/presets.md](docs/presets.md).

![The crack preset: its level over time and its spectrum](docs/figures/crack.svg)

## Install

Halyard needs Python 3.12 or later.

```
uv tool install "halyard-synth[audition]"
```

or `pipx install "halyard-synth[audition]"`. The `audition` extra adds live playback through sounddevice and PortAudio;
without it every other command works. From a checkout, `uv tool install ".[audition]"`; to work on Halyard, see
[CONTRIBUTING.md](CONTRIBUTING.md).

## Use

```
halyard init crack --tune D3 > snare.toml               # a complete starting patch (writes tune_hz = 146.832)
halyard audition snare.toml --watch --bpm 92 --with loop.wav     # on the backbeat, over the track's loop
halyard audition snare.toml --watch --against reference.wav --blind
halyard render snare.toml --out snare.wav --takes 8      # round-robin takes snare_rr1.wav … snare_rr8.wav
halyard verify snare.toml --takes 8                      # each take against its reference render
halyard extract snare_rr3.wav > take3.toml               # the exact patch of any exported file
halyard tune snare.toml F3 > snare-f3.toml               # retune: modes, glide and rattle follow
halyard stretch snare.toml all 0.8 > tight.toml          # fit every shape to a tempo
halyard init clap > clap.toml
halyard bursts clap.toml 'noise[1]' --count 4 --spacing 11 > clap-4.toml   # a clap train as ordinary knots
halyard softer snare.toml 12 > snare-12.toml             # the same drum struck 12 dB softer
halyard softer snare.toml 24 > snare-24.toml
halyard flam snare.toml --gap 25 > flam.toml             # a grace note 12 dB softer, 25 ms ahead
halyard roll snare.toml --count 4 --spacing 60 --crescendo 9 > roll.toml   # four strokes, louder to the last
halyard stack snare.toml clap.toml --db -4 > layered.toml   # a snare and a clap, one patch, one drive
halyard pack snare.toml snare-12.toml snare-24.toml --takes 6 --out kit/
```

A test runs these commands, all but `audition`, in this order.

The eighteen presets are `tone`, `crack`, `fat`, `driven`, `clap` and `snap`; `machine`, `house`, `short-clap`,
`rim-click`, `big-clap` and `long-snare`; the played `rimshot`, `cross-stick` and `brush`; and the swept `zap`,
`reverse` and `closing`. They are matched in loudness (−27 LUFS), peak below −1 dBFS over seeds 1–16, and every take of
them is inside the validated domain with at least 16 dB of drive headroom. They are starting points: their names never
select hidden algorithms.

Every command is also a Python function; the guide's [From Python](docs/guide.md#from-python) shows them, in examples a
test runs.

The editing commands print a complete, canonical patch and never add a parameter, so every change is visible and can be
edited further. Knots, layers and modes are numbered from 1, as the parser's messages and the report number them:
`noise[2].amp[3].ms` is the time of the third knot of the second layer.

## What a render reports

```
tuning   185.000 Hz, F#3 +0.0 cents; modes 185.0, 294.7, 395.2 Hz
length   260.0 ms source, 261.9 ms exported (12571 samples at 48000 Hz, float32), seed 1
peak     -4.05 dBFS sample, -3.72 dBTP true
loudness -19.0 LUFS, the largest 400 ms momentary loudness (BS.1770)
domain   inside the validated domain: predicted aliasing -137 dB, margin 37.0 dB
         mostly kinks, 64% at noise[1].amp[2] (0.05 ms); drive headroom +21.5 dB
render   86 ms
```

The domain line is the renderer's own prediction of its aliasing error, relative to the output peak; inside means at
most −100 dB. The line under it says what dominates the prediction (saturation, or the kinks with the knot that
contributes most) and how much more drive the take stands before its status changes. Outside, the headroom is negative
and the line says what to lower `drive.db` to. For several takes the report gives each take's own judgement, the
loudness and peak spread, and flags a set whose loudness spreads over more than 1 dB. The
[user guide](docs/guide.md#reading-the-report) explains every line.

## Documentation

- [User guide](docs/guide.md): the workflow, the editing commands, recipes, reading the report, audition, packs, use in
  a DAW, and the Python API.
- [Presets](docs/presets.md): every preset, with its sound, figure, report and patch.
- [Patch reference](docs/patch.md): every field, its range and what it does.
- [Design](docs/design.md): every stage of the sound, the validated domain, what makes a render reproducible, and a map
  of the code.
- [Validation](docs/validation.md): the measurements, from the numerical criteria to the adversarial grid and
  performance.
- [Changelog](CHANGELOG.md).

## Contributing

Contributions are welcome. [CONTRIBUTING.md](CONTRIBUTING.md) explains how to report a bug, set up, test and propose a
change.

## Licence

Halyard is free software, under the [GNU General Public License](LICENSE), version 3 or later.

The sounds you make with it are yours. An additional permission in [NOTICE](NOTICE) lets you use the sounds Halyard
renders, the patches and SFZ maps it writes, and its presets under any terms you choose, in a track or in a sample pack
you sell.
