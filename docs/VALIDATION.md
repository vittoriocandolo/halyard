# Validation record

This records the numerical criteria of the plan (§8.1), the reference method
(§8.2), the adversarial grid (§8.3), the frozen rendering policy, the refit of
the validated-domain rule, the references by measurement and performance. The
listening half of the plan (§8.4 by ear, and §8.5) needs a person; its state is
recorded at the end.

**Machine:** Intel Xeon @ 2.10 GHz, 4 cores, 15 GB, Linux.
**Software:** Python 3.11.15, NumPy 2.4.6, SciPy 1.17.1.

## Numerical criteria (§8.1)

Every criterion is a test in `tests/`. The *measured* column gives the value on
this machine.

| Area | Pass criterion | Measured |
|---|---|---|
| Curves | 10⁻¹² relative; compressed shapes keep their peak at exactly 1 and their segments monotone | Interior slopes agree with SciPy's PCHIP to 1.6·10⁻¹⁶ of the largest secant (2,000 random knot sets); integrals agree with adaptive quadrature to 2.7·10⁻¹⁴. A compressed shape equals the shape at the compressed time to the rounding of that time; its peak is exactly 1, its segments stay monotone within [0, 1], and its jumps scale by s² and s³ (hundreds of random shapes, s up to 512). |
| Tuning and modes | Multi-sine fit on the settled body, tunings up to 4 kHz: every mode within 0.001 cent of f₀·r_i | 3.8·10⁻¹³ cent, four modes at 40, 185, 1000 and 4000 Hz |
| Damping | Exponential body shape: each mode's fitted decay rate over the fundamental's equals r_i^β within 0.1% | Within 0.013% for (1.593, β 0.5 and 1), (2.136, 2), (2.295, 3) and (4, 1) |
| Noise | Common instants ≤ 10⁻¹²; 200-seed mean square 0.5 ± 0.5%; spectra within statistical bounds; identical bands bit-identical; disjoint bands uncorrelated | Common instants agree to 1.3·10⁻¹⁵ between the families at 32× and 1.1·10⁻¹⁵ against the reference rate. Mean square 0.50023 (+0.05%); the per-seed spread is 0.955% against 0.922% predicted. Every 100-component band mean lies within 6σ of the shape; the marginal is Gaussian (KS over 2,000 seeds). Identical bands give the same array, and a table never depends on which others it was computed with. Over one period two layers correlate exactly by their ρ²-weighted band overlap, and across the ensemble disjoint bands correlate by their skirts' overlap only (1.9% four octaves apart). |
| Carriers at knots | Values and slopes from the tables, covered by the kink bound's 1% margin | At 1.4112 and 1.536 MHz, cubic interpolation is within 1.6·10⁻⁶ of the noise's RMS in value and 1.0·10⁻⁴ of its RMS slope, for the widest band. The plan quoted 1.3·10⁻⁷ and 1.2·10⁻⁵; either is far inside the 1%. |
| Rattle | Flat shapes: modulation at the fundamental with depth d within 1%; none when d = 0 or the body shape is 0 | The demodulated layer envelope equals 1 − d(1 − cos φ₁)/2 to 10⁻¹², glide included; depth within 7·10⁻¹⁶ relative. Exactly none at d = 0 and outside the body shape. A silent body gates exactly as a sounding one. |
| Drive | Identities | 0 dB is the identity; S(±1) = ±1 exactly; odd by construction; small-signal gain 10^(drive/20) to 10⁻⁶ at x = 10⁻⁹; S′ and S″ match numerical derivatives. |
| Dynamics | `softer` on every reference, ladders of 0–36 dB: levels, glide v², rises v^(−1/5), upper edges v^(1/5), curves exact to 10⁻¹², composition | Every rule holds on all six references at 0, 6, … 36 dB, up to the decimals written (1 µs, 10⁻⁶ in ratio, 0.1 Hz). Each rewritten segment is the original read at a linearly mapped time to 10⁻¹² (bursts included), and the glide's excess scales on the whole curve to 10⁻¹⁵. `softer` by a then b equals a + b within the written rounding: a knot may move by 1 µs more for every rise before it. |
| Decimators | Frozen coefficients, as the kick | See the decimator table below. |
| Kink bound | Never exceeded by the measured error on an undriven grid fixture | Never exceeded, at 16× and 32×: on 298 undriven fixtures at 32× it sits 5.4–41.7 dB above the measured error (median 12.9 dB). Under drive it is never exceeded either (below). |
| Rendering accuracy | E_max and E_rel ≤ −100 dB for every fixture inside the validated domain | All 654 fixtures inside at 32× pass; the worst is −113.9 dB (below). |
| Drive headroom | The report's value against full renders on every grid fixture; dense 0.05 dB scans on a subset | Never optimistic beyond the brute force's 0.00001 dB resolution, on 957 fixtures and 92,524 full renders; no change of status between the candidates in 169 dense scans (below). |
| Loudness | EBU Tech 3341; K-weighting against BS.1770's 48 kHz table | The 48 kHz coefficients match to 8.9·10⁻¹⁶, the other rates match its response within 0.03 dB. The −23 dBFS 1 kHz sine reads −22.99 to −23.01 LUFS at all four rates. |
| Export | Independent readers; takes carry their seed, gains and patch; PCM reads back exactly; TPDF dither; never full scale | soundfile, `scipy.io.wavfile`, Python's `wave` and an independent chunk walker read identical samples and headers. Each take's `snar` chunk holds its own patch (its seed) and its gains, and its samples follow from them. `smpl` is left out when the body is silent. The dither is triangular (KS), bit-identical per seed and prefix-stable, keeps exact zeros, and leaves an error power of 1/4 LSB² at every offset between steps. |
| Packs | The SFZ mapping, parsed back | Every velocity 1–127 and every take maps to exactly one existing file. The level, measured from the files, follows 40·log₁₀(v/127) within 0.01 dB, and steps between neighbouring velocities never exceed one velocity step. A pack re-renders byte-identically. |
| Takes | Seeds 1–8 of each reference: one gain per set; loudness spread ≤ 1 dB | One gain per set; spreads 0.15–0.77 dB (below). |
| Performance | 0.5 s hit with three layers ≤ 300 ms at any rate; 2 s ≤ 1.5 s; 32-file pack ≤ 15 s; ≤ 300 MB | 186–234 ms with new noise tables, 130–162 ms cached; 467–550 ms; 1.3 s; at most 222 MiB (below). |

### Decimators

Designed once by `tools/design_decimator.py` and shipped with SHA-256
checksums: a Kaiser-window linear-phase prototype at 116 dB, converted with
`minimum_phase(method='homomorphic', half=False)` at 2²⁰ points and
renormalized so the exactly rounded sum of the taps is 1. The 16× and 64×
designs ship for the grid; the renderer uses 32×.

| Family and rate | Taps | Length | 99% of energy within | Passband deviation | Stopband |
|---|---|---|---|---|---|
| 44.1 kHz, 16× | 2,593 | 3.67 ms | 0.85 ms | 1.7·10⁻⁵ dB | 115.7 dB |
| 48 kHz, 16× | 1,447 | 1.88 ms | 0.52 ms | 1.5·10⁻⁵ dB | 115.9 dB |
| **44.1 kHz, 32×** | **5,183** | 3.67 ms | 0.85 ms | 3.7·10⁻⁵ dB | 115.6 dB |
| **48 kHz, 32×** | **2,891** | 1.88 ms | 0.52 ms | 1.8·10⁻⁵ dB | 115.1 dB |
| 44.1 kHz, 64× | 10,363 | 3.67 ms | 0.85 ms | 1.9·10⁻⁵ dB | 115.4 dB |
| 48 kHz, 64× | 5,781 | 1.88 ms | 0.52 ms | 1.6·10⁻⁵ dB | 115.2 dB |

These are the plan's figures (§3). The 96 and 88.2 kHz exports equal the 48
and 44.1 kHz ones with samples filled in, bit for bit, and parallel decimation
is bit-identical to one `upfirdn` call.

## Reference method (§8.2)

`src/halyard_synth/reference.py` renders the take at an absolute reference rate, 128 ×
base (6.144 MHz for the 48 kHz family), with noise from exact FFT tables at
that rate. It applies a zero-phase mask flat to 0.4F and a raised cosine to
zero at 0.5F, keeps every (128/K)-th sample, and runs the result through the
same frozen decimator, at unity gain, with no time shift or gain matching. A
reference counts only if it agrees with the 256 × base render to −120 dB
(10⁻⁸ absolute for near-silent fixtures), 20 dB below the threshold.

- 812 of the 957 grid references converged, with a median agreement of
  −206 dB.
- The 512 × base check (24.6 MHz) on 90 fixtures agreed at least as well as
  the 128/256 pair in every case.
- The 145 unconverged references all have drive ≥ 12 dB and are extreme
  corners, such as four layers at +24 dB or noise at +24 dB under 48 dB of
  drive. Their saturated signal still aliases at 12.3 MHz. The rule puts none
  of them inside.

## The adversarial grid (§8.3)

`tools/grid.py` builds 957 fixtures from combined corners, not one knob at a
time, and runs each at 16× and 32× (`docs/grid/k16.csv`, `docs/grid/k32.csv`).

| Family | Fixtures | Covers |
|---|---|---|
| body | 160 | tuning 40–4000 Hz × one mode, the membrane's four, or four up to ratio 8 × deep glide with damping 3, or flat at +24 dB with damping 0 × drive 0–48 dB |
| noise | 147 | the lowest, narrowest, 2–9 kHz, widest, top 1 kHz, stick and clap bands × 0, +12, +24 dB × drive 0–48 dB |
| layers | 13 | four coherent overlapping layers at +24 dB; crossfades at 0 and +12 dB |
| rattle | 120 | depth 0.25–1 × fundamental 40–4000 Hz × flat or deep glide × drive, the body sounding or silent |
| clicks | 66 | 10, 20 and 50 µs rises under full-scale wide, stick and top-band noise × level × drive; pings at 2 and 4 kHz with damping 3 |
| bursts | 51 | 2, 4 and 10 bursts × spacing 1–30 ms × rises of 10 µs – 1 ms × drive 0 and 18 dB; stacked trains |
| snaps | 27 | a single mode at 1, 2 and 4 kHz under bursts of 1, 10 and 30 ms × drive |
| dynamics | 42 | `softer` ladders of 0–36 dB from every reference |
| seeds | 90 | six representative fixtures, one per family, across seeds 2–16 |
| rates | 18 | 44.1, 88.2 and 96 kHz exports |
| random | 220 | the whole valid domain |
| longest | 3 | 2 s of body and 1 s of noise, at 96 kHz |

### Results

| | 16× | 32× |
|---|---|---|
| Pass (E ≤ −100 dB) | 612 | 710 |
| Converged references | 812 | 812 |
| Inside the validated domain | 546 | 654 |
| Inside and failing or unconverged | **0** | **0** |
| Worst error inside | −110.9 dB | −113.9 dB |
| Undriven fixtures passing | 283 of 298 | 298 of 298 |
| Kink bound exceeded | never | never |
| Saturation constant for 6 dB of margin | 1.11 | 0.419 |
| Outside but passing (the rule's caution) | 66 | 56 |

Per family at 32× (inside / pass / unconverged / worst error inside): body
113 / 118 / 25 / −125.2 dB; noise 51 / 53 / 70 / −119.1; layers 4 / 4 / 7 /
−154.7; rattle 76 / 89 / 0 / −120.3; clicks 39 / 55 / 5 / −113.9; bursts 33 /
51 / 0 / −144.1; snaps 27 / 27 / 0 / −144.5; dynamics 42 / 42 / 0 / −143.4;
seeds 90 / 90 / 0 / −131.7; rates 9 / 9 / 3 / −147.4; random 167 / 169 / 35 /
−115.2; longest 3 / 3 / 0 / −148.0. The random family draws each level as
silence half the time; a draw that would silence every source sounds its first
layer instead, so all 220 random fixtures sound.

### The choice of K

The plan's criterion (§3) is the smallest candidate that meets the performance
budget and for which every reference is inside the validated domain with at
least 3 dB of drive headroom, for seeds 1–16. At 16× the crack reference, a 30
µs stick under bright wires, keeps a minimum margin of 1.0 dB and only 1.7 dB
of drive headroom, and the driven reference exactly 3.0 dB. At 32× every
reference keeps at least 16 dB of headroom (below), and the budget holds
(performance). **K = 32 is frozen.** 64× was not needed.

### The rule's constant

The rule is the kick's, `C · e^(−π²/q_knee) · P + B_kink ≤ 10⁻⁵ · P` (10⁻⁷
when P < 10⁻²). The plan starts C at the kick's 2 and refits it as the
smallest value that covers every converged fixture with at least 6 dB of
margin: wherever the measured error exceeds the kink bound, the saturation
term must cover the rest twice over. At 32× that needs **0.419** (the worst
fixture is a 2–9 kHz layer at 18 dB of drive, seed 10); **C = 0.5 is frozen**,
7.6 dB of margin on it. The domain barely depends on C: near its edge the
prediction moves 11–12 dB per dB of drive, so C = 2 would admit 648 fixtures
instead of 654, and none fails at either.

### The kink bound

No grid fixture, undriven or driven, measures more than its kink bound, so
the rule gains no term. The bound is tightest, 3.8 dB above the measured
error, on fixtures dominated by one isolated jump in the third derivative,
where its images add in phase at the knot and the bound is nearly exact.
What remains is its band B = base/2 against the decimator's 20 kHz passband.
The same 3.8 dB appears across unrelated fixtures (pings, bodies at +24 dB,
damped modes), which shows it is structural, not a mechanism the bound misses.

### Rule of thumb

For a 260 ms noise layer alone at 32×, over seeds 1–16, the first change of
status comes at:

- 20.3–23.3 dB of noise level + drive (median 22.1) for a 2–9 kHz band;
- 17.7–21.4 dB (median 19.6) for 1–16 kHz;
- 27.9–33.9 dB (median 31.9) for 800–2500 Hz.

The plan's prototype, with C = 2 and three seeds, gave 19.9–20.9, 17.9–19.1
and 34.4 dB.

## Drive headroom

`tools/headroom.py` compares the headroom each report computes from its one
render with brute force on every grid fixture (`docs/grid/headroom.csv`). The
brute force renders every candidate the report considers, every 0.5 dB and
both sides of every drop of the knee slew, from the fixture's own drive until
the status changes, then bisects that bracket with full renders to 0.00001 dB.
On fixtures whose edge is within 6 dB, it also scans every 0.05 dB up to the
edge. In all, 957 fixtures and 92,524 full renders:

- **Never optimistic.** On the 796 fixtures whose status changes within
  0–48 dB, the report is at most 0.00001 dB beyond the brute force, its
  resolution. It is exact to 0.001 dB on 611, conservative by 0.001–0.1 dB on
  170, by 0.1–1 dB on 11 and by 1–2.1 dB on 4.
- **Exact where it matters.** Within 3 dB of the edge (103 fixtures) it is at
  most 0.066 dB conservative.
- **No change of status left.** Where full renders never change status within
  0–48 dB (161 fixtures), the report says so on 160. On the other it places an
  edge at 47.8 dB where the status stays inside: cautious, never a claimed
  way back in that is not there.
- **No islands missed.** The 169 dense scans found no change of status
  between the candidates.

The larger gaps come from an output peak that moves to a position no probe
covers, which only lowers the estimate. The report's share of a render,
headroom search included, is about 40 ms for the example at 88.2 kHz
(performance, below).

## References by measurement (§8.4)

`tools/references.py` at 32×, over seeds 1–16. Output gains are set so every
reference reads −27 LUFS at its own seed: the snap, a 40 ms sound, has a crest
of 25.6 dB, so that is the loudest common level at which every take of every
reference peaks below −1 dBFS.

| Reference | LUFS | Loudness spread, takes 1–8 | Largest peak, seeds 1–16 | Inside, seeds 1–16 | Smallest margin | Smallest drive headroom |
|---|---|---|---|---|---|---|
| tone | −27.0 | 0.15 dB | −8.38 dBFS | yes | 56.4 dB | 28.1 dB |
| crack | −27.0 | 0.28 dB | −6.72 dBFS | yes | 19.1 dB | 22.8 dB |
| fat | −27.0 | 0.16 dB | −11.83 dBFS | yes | 33.4 dB | 20.6 dB |
| driven | −27.0 | 0.15 dB | −13.35 dBFS | yes | 20.8 dB | 16.3 dB |
| clap | −27.0 | 0.77 dB | −8.63 dBFS | yes | 29.2 dB | 27.4 dB |
| snap | −27.0 | 0.64 dB | −1.39 dBFS | yes | 25.3 dB | 27.1 dB |

The loudness range across the references is 0.08 dB. Every take spread is
within 1 dB, so none needs level matching.

The usability tasks (`tests/test_usability.py`) each take at most three field
edits or one command and move a measurement the expected way, from every
reference: change the tuning (within 5 cents of a whole tone), shorten the
tail, add crack, add body, add wires, darken, add aggression and make a ghost
note; tighten, shorten, darken and roughen the clap; shorten and brighten the
snap and move its ping. Two lessons from the kick held: measurements fade
their windows rather than cut them, and in the driven reference the edge lives
in the saturated source, so the stick adds less crack there (1.6× the high end
of the first millisecond, against 2–3× in the clean snares).

## Performance

`tools/bench.py`, each case in a fresh interpreter, report included. A render
with a new seed computes its three noise tables, as every take does; a repeated
render finds them cached, as audition's re-renders do. The budget applies to
the first.

| Sound | Rate | New seed | Cached | Budget | Memory above baseline |
|---|---|---|---|---|---|
| 0.5 s hit, three layers | 44.1 kHz | 186 ms | 130 ms | 300 ms | 165 MiB |
| | 48 kHz | 234 ms | 155 ms | 300 ms | 179 MiB |
| | 88.2 kHz | 225 ms | 162 ms | 300 ms | 165 MiB |
| | 96 kHz | 198 ms | 150 ms | 300 ms | 181 MiB |
| 2 s body, 1 s of noise | 44.1 kHz | 495 ms | 400 ms | 1.5 s | 165 MiB |
| | 48 kHz | 467 ms | 417 ms | 1.5 s | 222 MiB |
| | 88.2 kHz | 536 ms | 520 ms | 1.5 s | 165 MiB |
| | 96 kHz | 550 ms | 442 ms | 1.5 s | 199 MiB |
| Pack: 4 layers × 8 takes, 32 files | 48 kHz | 1.3 s | | 15 s | (worker processes) |

Where the time goes, for the example patch at 88.2 kHz (260 ms, two layers,
tables cached): synthesis 32 ms, decimation 27 ms, the kink terms 4 ms, the
probes 9 ms, the drive-headroom search 19 ms, loudness 6 ms and the true peak
2 ms. Keeping the probes' saturator input once over the union of their
windows, which overlap almost entirely, took the headroom search from 46 to
19 ms. A new seed adds its tables, computed in parallel, one transform each:
about 25 ms each when warm.

## Listening

Not yet done; it needs a person:

- approval of the six references by ear, dry and in context, across seeds 1–8
  and in packs of four strengths (§8.4);
- a 16th-note roll at 120 bpm with six takes against one take;
- a pack loaded in two SFZ players, and audition through a real device;
- the sample-replacement benchmark (BENCHMARK.md).
