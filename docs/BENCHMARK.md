# The sample-replacement benchmark (plan §8.5)

This is where "no more snare or clap samples" becomes a measurement. It needs
your own samples and your ears, so it is a protocol to run by hand.

## Result

Not yet run.

## Protocol

1. **Collect.** Gather the 20–30 snare and clap samples you actually use that
   fall inside the family: tonal electronic snares, tight bright snares, big
   modern snares, driven snares, claps, snaps and electronic rim clicks. Keep
   their velocity layers where they have them. Acoustic snares, rimshots,
   cross-stick, brushes, rolls and flams, and sounds made by processing chains
   (gated reverb, multi-stage distortion with EQ between stages, layered
   samples) are out of family. List them anyway, marked out of family.
2. **Fix the time limit before starting.** 15 minutes per sample is a
   reasonable choice. Start each from the nearest reference (`halyard init`),
   and edit with audition watching:
   `halyard audition snare.toml --watch --bpm 92 --with groove.wav --against sample.wav`.
3. **Compare blind and level-matched, in context,** with the kick and the
   groove, over several takes:
   `halyard audition snare.toml --watch --bpm 92 --with groove.wav --against sample.wav --blind --takes 6`.
   Audition matches the sample to the render's largest 400 ms momentary
   loudness (BS.1770) and prints each pair's order when stopped. Record the
   matching method if you use another. Where the source has velocity layers,
   derive yours with `halyard softer`, pack them, and compare packs in a groove
   with dynamics in your sampler.
4. **Judge and sort.** A pass means you would use the render in place of the
   sample. Sort each failure as a missing control, a voicing-skill issue, or
   out of family.

The claim holds for the fraction that passes.

## Pre-registered additions

Six additions are pre-registered. Add one only if it fixes recorded failures:

- a post-drive low-pass for the fizz of heavy drive;
- per-mode decay times, if one damping tilt cannot place a mode;
- impulsive wire noise with a density control, if the Gaussian sizzle lacks crackle;
- an extended band at 88.2 and 96 kHz, if repitched takes sound dull;
- a separate drive for the body, if one saturator on the sum cannot put a driven body under clean wires;
- seeded take-to-take variation of body level, tuning and decay, if takes that differ only in their noise sound too alike.

Anything else needs its own evidence.

## Record

| # | Sample | Family | Start | Minutes | Matching | Result | Failure category | Notes |
|---|---|---|---|---|---|---|---|---|
| 1 | | | | | | | | |

Pass rate: _ / _ in family. Out of family: _.
