# Changelog

All notable changes to Halyard are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Within a version, a patch renders the same samples bit for
bit on one platform with the same NumPy and SciPy, and the same sound on any other, the same samples to within a
millionth of full scale; a change to the sound of an existing patch is a new version.

## [0.1.0] - Unreleased

The first public release. The presets' voicing may change in later versions, and each such change will be recorded here.

### Added

- A one-voice snare and clap synthesizer: a modal body of up to four modes, tuned in Hz or by note, with a pitch glide
  and damping that makes higher modes decay faster, and up to four layers of one seeded noise, coherent across their
  bands, with a rattle that gates a layer at the body's fundamental as snare wires do; all summed into a tanh drive and
  computed at 32 times the sample rate. A render depends only on its patch and its seed.
- Up to two rims, struck with the head or instead of it, for rimshots and cross-sticks: each a set of up to eight modes
  at their own frequencies in Hz, untouched by the tuning and the glide, with its own damping and shape.
- A brush: a noise layer whose band moves during the hit by a sweep of up to an octave, as one noise filtered rather
  than a crossfade, coherent with the layers and gated by the rattle like them.
- Takes that differ as played: an optional `[takes]` section strikes each take up to a set strength harder or softer and
  moves its later rises, such as a clap's bursts, by up to a set time, drawn from the seed alone.
- The validated domain: every render reports whether it lies where its aliasing is demonstrated below −100 dB, from a
  rule refit on an adversarial grid of 1,490 fixtures, with the cause of any predicted aliasing and the drive headroom
  left before the status changes. `halyard verify` checks a render against a reference rendered at 128 times the base
  rate, itself checked against one at 256.
- Frozen minimum-phase decimators for the 44.1 and 48 kHz families, verified by checksum.
- Eighteen presets to start from, matched in loudness: `tone`, `crack`, `fat`, `driven`, `clap` and `snap`; `machine`,
  `house`, `short-clap`, `rim-click`, `big-clap` and `long-snare`; the played `rimshot`, `cross-stick` and `brush`; and
  the swept `zap`, `reverse` and `closing`.
- Patches in TOML, read as UTF-8 or UTF-16, with every field required, every problem reported with its field path, and a
  canonical form.
- Editing commands that print a complete patch: `tune` to a note or a frequency, `stretch` of chosen curves, `bursts`
  that write a clap train as ordinary knots, `softer`, the same drum struck softer, `flam`, a softer grace note before
  the main stroke, `roll`, the drum struck in turn with a crescendo and a speeding up, also as ordinary knots, and
  `stack`, two hits such as a snare and a clap in one patch through one drive.
- Exports at 44.1, 48, 88.2, 96, 176.4 and 192 kHz, as float32 or as 24- or 16-bit PCM with seeded TPDF dither, at a
  gain, a sample peak or a true peak, with the root key in a `smpl` chunk and the patch and its provenance in a `snar`
  chunk, which `halyard extract` gives back. Overs are reported, never clipped.
- The true peak, measured exactly as the crest of the output's band-limited reconstruction at 32 times the sample rate,
  and a `true-peak` output mode that brings it to a target.
- Round-robin sets of takes sharing one gain, with optional level matching.
- `halyard audition`: live playback that re-renders on every save, on the backbeat or any beats of a tempo, over a bed,
  or alternating with a sample matched in loudness, in a blind order revealed at the end. It opens the device at the
  patch's rate or says it cannot: through WASAPI on Windows, with Core Audio told to switch the rate on macOS. Dropouts
  are counted and reported.
- `halyard pack`: velocity layers on a square law with takes as round robins, mapped by an SFZ file, and written only if
  every file passes its gates: a true peak at or below 0 dBTP, no sample over the format's ceiling, and first and last
  samples below −90 dBFS.
- Exit statuses a script can rely on: 0 when done, 1 for a refusal or a failure, 2 for bad input, 130 after Ctrl+C; a
  reader that goes away, as `| head` does, ends a command quietly.
- Input that cannot be misread: a pack whose name a sampler would read as part of its SFZ map is refused;
  `halyard extract` never prints control characters, which a terminal would act on; WAVs, embedded chunks and patches
  are read with every size checked, so a damaged or crafted file is refused with its reason; beds and samples holding
  values that are not numbers are refused before they reach the device. Releases are published with attestations of
  where they were built.
