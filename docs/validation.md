# Validation record

This records the measurements behind [the design](design.md): the numerical criteria, the reference method, the
adversarial grid, the frozen rendering policy, the refit of the validated-domain rule, the presets by measurement,
performance, and packs in two SFZ players.

**Machine:** Qualcomm Oryon, 12 cores, under WSL 2 (Linux 6.18). **Software:** Python 3.15.0, NumPy 2.5.4, SciPy 1.18.1.

Every measurement below was made there. The grid at 32× (`docs/grid/k32.csv`) took 17 minutes and the drive-headroom
brute force (`docs/grid/headroom.csv`) 46, each on 6 workers; the grid at 16× (`docs/grid/k16.csv`) took 19 minutes on
4\. Under Python 3.11 on an Intel Xeon, the first 957 fixtures, also rendered there, give every error within 0.0009 dB
and every knee slew, kink bound and peak within 10⁻¹² relatively.

## Numerical criteria

Every criterion is a test in `tests/`; the *measured* column gives its value.

| Area | Pass criterion | Measured |
| -- | -- | -- |
| Curves | 10⁻¹² relative; compressed shapes keep their peak at exactly 1 and their segments monotone | Interior slopes agree with SciPy's PCHIP to 1.6·10⁻¹⁶ of the largest secant (2,000 random knot sets); integrals agree with adaptive quadrature to 2.7·10⁻¹⁴. A compressed shape equals the shape at the compressed time to the rounding of that time; its peak is exactly 1, its segments stay monotone within [0, 1], and its jumps scale by s² and s³ (hundreds of random shapes, s up to 512). |
| Tuning and modes | Multi-sine fit on the settled body, tunings up to 4 kHz: every mode within 0.001 cent of f₀·r_i | 3.8·10⁻¹³ cent, four modes at 40, 185, 1000 and 4000 Hz |
| Damping | Exponential body shape: each mode's fitted decay rate over the fundamental's equals r_i^β within 0.1% | Within 0.013% for (1.593, β 0.5 and 1), (2.136, 2), (2.295, 3) and (4, 1) |
| Noise | Common instants ≤ 10⁻¹²; 200-seed mean square 0.5 ± 0.5%; spectra within statistical bounds; identical bands bit-identical; disjoint bands uncorrelated | Common instants agree to 1.3·10⁻¹⁵ between the families at 32× and 1.1·10⁻¹⁵ against the preset rate. Mean square 0.50023 (+0.05%); the per-seed spread is 0.955% against 0.922% predicted. Every 100-component band mean lies within 6σ of the shape; the marginal is Gaussian (KS over 2,000 seeds). Identical bands give the same array, and a table never depends on which others it was computed with. Over one period two layers correlate exactly by their ρ²-weighted band overlap, and across the ensemble disjoint bands correlate by their skirts' overlap only (1.9% four octaves apart). |
| Carriers at knots | Values and slopes from the tables, covered by the kink bound's 1% margin | At 1.4112 and 1.536 MHz, cubic interpolation is within 1.6·10⁻⁶ of the noise's RMS in value and 1.0·10⁻⁴ of its RMS slope, for the widest band. Both are far inside the 1%. |
| Rattle | Flat shapes: modulation at the fundamental with depth d within 1%; none when d = 0 or the body shape is 0 | The demodulated layer envelope equals 1 − d(1 − cos φ₁)/2 to 10⁻¹², glide included; depth within 7·10⁻¹⁶ relative. Exactly none at d = 0 and outside the body shape. A silent body gates exactly as a sounding one. |
| Rims | Each mode at its Hz within 0.001 cent, whatever the tuning and glide; the renderer equals the model to 10⁻¹² | A three-mode rim fits within 0.001 cent at 1630, 4210 and 6950 Hz, and renders the same samples under two tunings and glides. With rims, x equals the model summed directly to 10⁻¹², and every jump of x″ and x‴ matches one-sided derivatives of the model. |
| Brush | The blended band within 0.5 dB of the written band down to −30 dB, its level within 0.1 dB, at every position of the sweep; the renderer equals the model to 10⁻¹² | On 300 random bands at 25 positions each: at most 0.48 dB from the written shape, 0.29 dB within its −3 dB band, and 0.094 dB in level; 40 random bands at 6 positions in the tests. The renderer equals the model, its B-spline written by its definition, to 10⁻¹²; every jump, at the sweep's knots and where u crosses a table, matches one-sided derivatives. |
| Flam | From the main stroke's peak on, every shape is the main stroke's moved by the gap; before it, the grace's at its level | On all eighteen presets: exact to 10⁻¹² from the main peak on, the grace's knots and gains to the written 10⁻⁶ before it, and the pitch curve the main stroke's from the body's peak. |
| Drive | Identities | 0 dB is the identity; S(±1) = ±1 exactly; odd by construction; small-signal gain 10^(drive/20) to 10⁻⁶ at x = 10⁻⁹; S′ and S″ match numerical derivatives. |
| Dynamics | `softer` on every preset, ladders of 0–36 dB: levels, glide v², rises v^(−1/5), upper edges v^(1/5), curves exact to 10⁻¹², composition | Every rule holds on all eighteen presets at 0, 6, … 36 dB, up to the decimals written (1 µs, 10⁻⁶ in ratio, 0.1 Hz). Each rewritten segment is the original read at a linearly mapped time to 10⁻¹² (bursts included), and the glide's excess scales on the whole curve to 10⁻¹⁵. `softer` by a then b equals a + b within the written rounding: a knot may move by 1 µs more for every rise before it, and 1 µs more inside a rise, whose first rounding the second edit scales. |
| Decimators | Frozen coefficients, checked by SHA-256 | See the decimator table below. |
| Kink bound | Never exceeded by the measured error on an undriven grid fixture | Never exceeded, at 16× and 32×: on 458 undriven fixtures at 32× it sits 5.4–45.0 dB above the measured error (median 12.1 dB). Under drive it is never exceeded either (below). |
| Rendering accuracy | E_max and E_rel ≤ −100 dB for every fixture inside the validated domain | All 654 fixtures inside at 32× pass; the worst is −113.9 dB (below). |
| Drive headroom | The report's value against full renders on every grid fixture; dense 0.05 dB scans on a subset | Never optimistic beyond the brute force's 0.00001 dB resolution, on 1,490 fixtures and 146,464 full renders; no change of status between the candidates in 269 dense scans (below). |
| Loudness | EBU Tech 3341; K-weighting against BS.1770's 48 kHz table | The 48 kHz coefficients match to 8.9·10⁻¹⁶, and the other rates match its response within 0.008 dB: the high-pass's numerator is scaled to the 48 kHz passband gain, which the standard's 1, −2, 1 would otherwise let drift by up to 0.032 dB at 192 kHz. The −23 dBFS 1 kHz sine reads −22.99 LUFS at all six rates. |
| True peak | The highest crest of the output's band-limited reconstruction, within about 10⁻⁶ (0.00001 dB); a true-peak target reached exactly | Against the reconstruction evaluated directly a thousandth of an internal sample apart, Hann-shaped sines at 1, 12, 19, 19.9 and 20 kHz in any phase read within 1.05·10⁻⁶, the worst near 20 kHz, every crest that could be the highest compared. `true-peak` mode puts the true peak at its target to 10⁻⁹ dB. |
| Export | Independent readers; takes carry their seed, gains and patch; PCM reads back exactly; TPDF dither; never full scale | soundfile, `scipy.io.wavfile`, Python's `wave` and an independent chunk walker read identical samples and headers. Each take's `snar` chunk holds its own patch (its seed) and its gains, and its samples follow from them. `smpl` is left out when the body is silent. The dither is triangular (KS), bit-identical per seed and prefix-stable, keeps exact zeros, and leaves an error power of 1/4 LSB² at every offset between steps. |
| Packs | The SFZ mapping, parsed back | Every velocity 1–127 and every take maps to exactly one existing file. The level, measured from the files, follows 40·log₁₀(v/127) within 0.01 dB, and steps between neighbouring velocities never exceed one velocity step. A pack re-renders byte-identically. |
| Takes | Seeds 1–8 of each preset: one gain per set; loudness spread ≤ 1 dB | One gain per set; spreads 0.15–0.77 dB (below). |
| Performance | From 44.1 to 96 kHz: 0.5 s hit with three layers ≤ 300 ms; 2 s ≤ 1.5 s; 32-file pack ≤ 15 s; ≤ 300 MB | 121–151 ms with new noise tables, 85–108 ms cached; 312–361 ms; 1.5 s; at most 286 MiB (below). |

### Decimators

Designed once by `tools/design_decimator.py` and shipped with SHA-256 checksums: a Kaiser-window linear-phase prototype
at 116 dB, converted with `minimum_phase(method='homomorphic', half=False)` at 2²⁰ points and renormalized so the
exactly rounded sum of the taps is 1. The 16× and 64× designs ship for the grid; the renderer uses 32×.

Designing them again (`uv run python -m tools.design_decimator --check`) gives the same filters, not the same bits. On
an Oryon laptop under NumPy 2.5.4 and SciPy 1.18.1 every design has the shipped length, passband and stopband to
1.1·10⁻⁶ dB, and its response lies 131–176 dB from the shipped one's, below every stopband; the taps differ by up to
9.8·10⁻¹⁰. Under the shipped designs' own NumPy 2.4.6 and SciPy 1.17.1 on that machine they differ by up to 1.4·10⁻¹¹:
the homomorphic method takes the logarithm of a spectrum 116 dB down, which amplifies the FFTs' platform rounding.
Renders load the shipped taps, checked by SHA-256, and never a fresh design.

| Family and rate | Taps | Length | 99% of energy within | Passband deviation | Stopband |
| -- | -- | -- | -- | -- | -- |
| 44.1 kHz, 16× | 2,593 | 3.67 ms | 0.85 ms | 1.7·10⁻⁵ dB | 115.7 dB |
| 48 kHz, 16× | 1,447 | 1.88 ms | 0.52 ms | 1.5·10⁻⁵ dB | 115.9 dB |
| **44.1 kHz, 32×** | **5,183** | 3.67 ms | 0.85 ms | 3.7·10⁻⁵ dB | 115.6 dB |
| **48 kHz, 32×** | **2,891** | 1.88 ms | 0.52 ms | 1.8·10⁻⁵ dB | 115.1 dB |
| 44.1 kHz, 64× | 10,363 | 3.67 ms | 0.85 ms | 1.9·10⁻⁵ dB | 115.4 dB |
| 48 kHz, 64× | 5,781 | 1.88 ms | 0.52 ms | 1.6·10⁻⁵ dB | 115.2 dB |

The 96 and 88.2 kHz exports equal the 48 and 44.1 kHz ones with samples filled in, bit for bit, and parallel decimation
is bit-identical to one `upfirdn` call.

## Reference method

`src/halyard_synth/reference.py` renders the take at an absolute reference rate, 128 × base (6.144 MHz for the 48 kHz
family), with noise from exact FFT tables at that rate. It applies a zero-phase mask flat to 0.4F and a raised cosine to
zero at 0.5F, keeps every (128/K)-th sample, and runs the result through the same frozen decimator, at unity gain, with
no time shift or gain matching. A reference counts only if it agrees with the 256 × base render to −120 dB (10⁻⁸
absolute for near-silent fixtures), 20 dB below the threshold.

- 1,276 of the 1,490 grid references converged, with a median agreement of −201 dB.
- The 512 × base check (24.6 MHz) on 149 fixtures agreed at least as well as the 128/256 pair in every case.
- The 214 unconverged references all have drive ≥ 12 dB and are extreme corners, such as four layers at +24 dB or noise
  at +24 dB under 48 dB of drive. Their saturated signal still aliases at 12.3 MHz. The rule puts none of them inside.

## The adversarial grid

`tools/grid.py` builds 1,490 fixtures from combined corners, not one knob at a time, and runs each at 32×
(`docs/grid/k32.csv`) and, for comparison, at 16× (`docs/grid/k16.csv`).

| Family | Fixtures | Covers |
| -- | -- | -- |
| body | 160 | tuning 40–4000 Hz × one mode, the membrane's four, or four up to ratio 8 × deep glide with damping 3, or flat at +24 dB with damping 0 × drive 0–48 dB |
| noise | 147 | the lowest, narrowest, 2–9 kHz, widest, top 1 kHz, stick and clap bands × 0, +12, +24 dB × drive 0–48 dB |
| layers | 13 | four coherent overlapping layers at +24 dB; crossfades at 0 and +12 dB |
| rattle | 120 | depth 0.25–1 × fundamental 40–4000 Hz × flat or deep glide × drive, the body sounding or silent |
| clicks | 66 | 10, 20 and 50 µs rises under full-scale wide, stick and top-band noise × level × drive; pings at 2 and 4 kHz with damping 3 |
| bursts | 51 | 2, 4 and 10 bursts × spacing 1–30 ms × rises of 10 µs – 1 ms × drive 0 and 18 dB; stacked trains |
| snaps | 27 | a single mode at 1, 2 and 4 kHz under bursts of 1, 10 and 30 ms × drive |
| rims | 197 | a single mode at 100 Hz, a four-mode hoop, two modes at 19–20 kHz, or eight modes over 1–17.8 kHz × rises of 10 µs, 50 µs and 1 ms × damping 0 and the most the 512 limit allows (3, or 2.16 for the eight modes) × 0 and +12 dB × drive 0–36 dB; two rims over the body and wires × drive |
| brushes | 144 | wires, stick, clap and the narrowest bands × a slow sweep, a fast one (an octave in 0.5 ms) and a zigzag of 31 segments × rises of 20 µs and 20 ms × 0 and +12 dB × drive 0–24 dB, rattling over the body |
| dynamics | 126 | `softer` ladders of 0–36 dB from every preset |
| seeds | 90 | six representative fixtures, one per family, across seeds 2–16 |
| rates | 30 | 44.1, 88.2, 96, 176.4 and 192 kHz exports |
| random | 300 | the whole valid domain: 220 drawn without rims or a brush, then 80 with up to two rims and, four times in five, a brush |
| takes | 16 | takes that differ as played, at 3 dB and 5 ms, from `clap`, `crack`, `rimshot` and `brush` across four seeds |
| longest | 3 | 2 s of body and 1 s of noise, at 96 kHz |

### Results

|  | 16× | 32× |
| -- | -- | -- |
| Pass (E ≤ −100 dB) | 940 | 1,092 |
| Converged references | 1,276 | 1,276 |
| Inside the validated domain (C = 0.54) | 798 | 1,010 |
| Inside and failing or unconverged | **0** | **0** |
| Worst error inside | −110.9 dB | −110.4 dB |
| Undriven fixtures passing | 434 of 458 | 458 of 458 |
| Kink bound exceeded | never | never |
| Saturation constant for 6 dB of margin | 7.25 | 0.539 |
| Outside but passing (the rule's caution) | 142 | 82 |

At 16× the rule's constant would have to grow more than thirteenfold, set by eight rim modes rising in 10 µs under 24 dB
of drive, so the constant belongs to the K it was fitted at.

Per family at 32× (inside / pass / unconverged / worst error inside): body 113 / 118 / 25 / −125.2 dB; noise 51 / 53 /
70 / −119.1; layers 4 / 4 / 7 / −154.7; rattle 76 / 89 / 0 / −120.3; clicks 39 / 55 / 5 / −113.9; bursts 33 / 51 / 0 /
−144.1; snaps 27 / 27 / 0 / −144.5; rims 120 / 123 / 36 / −110.4; brushes 78 / 97 / 17 / −125.7; dynamics 126 / 126 / 0
/ −143.4; seeds 90 / 90 / 0 / −131.7; rates 15 / 15 / 5 / −147.2; random 219 / 225 / 49 / −115.2; longest 3 / 3 / 0 /
−148.0; takes 16 / 16 / 0 / −141.0. The random family draws each level as silence half the time; a draw that would
silence every source sounds its first layer instead, so every random fixture sounds.

A rim's modes may be up to 200 times apart. At damping 3 its highest mode would then decay up to eight million times
faster than its first, compressing knots 10 µs apart into nanoseconds, where the 512 × base check agrees less well with
the 256 × base reference than the 128/256 pair does. A body mode, its ratios at most 8, decays at most 512 times faster,
and rims have the same limit: the parser refuses a rim beyond it. The grid's 24 eight-mode rims at damping 3, which the
limit refuses, run at 2.16, the most their 17.8 : 1 spread allows, and every one of them inside the domain passes.

### The choice of K

The criterion is the smallest candidate that meets the performance budget and for which every preset is inside the
validated domain with at least 3 dB of drive headroom, for seeds 1–16. At 16× the crack preset, a 30 µs stick over wires
up to 12 kHz, keeps a minimum margin of 1.0 dB and only 1.7 dB of drive headroom, and the driven preset exactly 3.0 dB.
At 32× every preset keeps at least 16 dB of headroom (below), and the budget holds (performance). **K = 32 is frozen.**
64× was not needed.

### The rule's constant

The rule is `C · e^(−π²/q_knee) · P + B_kink ≤ 10⁻⁵ · P` (10⁻⁷ when P < 10⁻²)
([design](design.md#the-validated-domain)). C is the smallest value that covers every converged fixture with at least 6
dB of margin: wherever the measured error exceeds the kink bound, the saturation term must cover the rest twice over.
That is **0.539**: the worst fixture is a rim with modes at 19 and 20 kHz, rising in 1 ms, under 12 dB of drive, whose
error is −170 dB, far inside the limit, but which the saturation term must still cover twice over. **C = 0.54 is
frozen.** Without the rims, the first 957 fixtures need 0.419 (a 2–9 kHz layer at 18 dB of drive, seed 10). The domain
barely depends on C: near its edge the prediction moves 11–12 dB per dB of drive, so 0.5 and 0.54 admit the same 1,010
fixtures, C = 2 would admit 999, and none fails at any of them.

### The kink bound

No grid fixture, undriven or driven, measures more than its kink bound, so the rule gains no term. The bound is
tightest, 3.8 dB above the measured error, on fixtures dominated by one isolated jump in the third derivative, where its
images add in phase at the knot and the bound is nearly exact. What remains is its band B = base/2 against the
decimator's 20 kHz passband. The same 3.8 dB appears across unrelated fixtures (pings, bodies at +24 dB, damped modes),
which shows it is structural, not a mechanism the bound misses.

### Rule of thumb

For a 260 ms noise layer alone at 32×, over seeds 1–16, the first change of status comes at:

- 20.3–23.3 dB of noise level + drive (median 22.1) for a 2–9 kHz band;
- 17.7–21.4 dB (median 19.6) for 1–16 kHz;
- 27.9–33.9 dB (median 31.9) for 800–2500 Hz.

## Drive headroom

`tools/headroom.py` compares the headroom each report computes from its one render with brute force on every grid
fixture (`docs/grid/headroom.csv`). The brute force renders every candidate the report considers, every 0.5 dB and both
sides of every drop of the knee slew, from the fixture's own drive until the status changes, then bisects that bracket
with full renders to 0.00001 dB. On fixtures whose edge is within 6 dB, it also scans every 0.05 dB up to the edge. In
all, 1,490 fixtures and 146,464 full renders, with C = 0.54:

- **Never optimistic.** On the 1,227 fixtures whose status changes within 0–48 dB, the report is at most 0.00001 dB
  beyond the brute force, its resolution. It is exact to 0.001 dB on 928, conservative by 0.001–0.1 dB on 253, by 0.1–1
  dB on 36 and by 1–2.2 dB on 8.
- **Exact where it matters.** Within 3 dB of the edge (146 fixtures) it is at most 0.147 dB conservative.
- **No change of status left.** Where full renders never change status within 0–48 dB (263 fixtures), the report says so
  on 262. On the other it places an edge at 47.8 dB where the status stays inside: cautious, never a claimed way back in
  that is not there.
- **No islands missed.** The 269 dense scans found no change of status between the candidates.

The larger gaps come from an output peak that moves to a position no probe covers, which only lowers the estimate. The
report's share of a render, headroom search included, is about 40 ms for the example at 88.2 kHz (performance, below).

## Presets by measurement

`tools/presets.py` at 32×, over seeds 1–16. Output gains are set so every preset reads −27 LUFS at its own seed: the
snap, a 40 ms sound, has a crest of 25.6 dB, so that is the loudest common level at which every take of every preset
peaks below −1 dBFS.

| Preset | LUFS | Loudness spread, takes 1–8 | Largest peak, seeds 1–16 | Inside, seeds 1–16 | Smallest margin | Smallest drive headroom |
| -- | -- | -- | -- | -- | -- | -- |
| tone | −27.0 | 0.15 dB | −8.38 dBFS | yes | 56.4 dB | 28.0 dB |
| crack | −27.0 | 0.28 dB | −6.72 dBFS | yes | 19.1 dB | 22.7 dB |
| fat | −27.0 | 0.16 dB | −11.83 dBFS | yes | 33.4 dB | 20.5 dB |
| driven | −27.0 | 0.15 dB | −13.35 dBFS | yes | 20.8 dB | 16.3 dB |
| clap | −27.0 | 0.77 dB | −8.63 dBFS | yes | 29.2 dB | 27.3 dB |
| snap | −27.0 | 0.64 dB | −1.39 dBFS | yes | 25.3 dB | 27.1 dB |
| machine | −27.0 | 0.17 dB | −5.36 dBFS | yes | 67.9 dB | 27.7 dB |
| house | −27.0 | 0.30 dB | −9.87 dBFS | yes | 30.5 dB | 18.6 dB |
| short-clap | −27.0 | 0.87 dB | −3.95 dBFS | yes | 36.3 dB | 27.1 dB |
| rim-click | −27.0 | 0.03 dB | −4.64 dBFS | yes | 23.0 dB | 28.4 dB |
| big-clap | −27.0 | 0.28 dB | −12.96 dBFS | yes | 31.2 dB | 23.5 dB |
| long-snare | −27.0 | 0.17 dB | −14.26 dBFS | yes | 31.4 dB | 21.4 dB |
| rimshot | −27.0 | 0.23 dB | −6.35 dBFS | yes | 19.5 dB | 24.2 dB |
| cross-stick | −27.0 | 0.03 dB | −6.03 dBFS | yes | 31.8 dB | 29.3 dB |
| brush | −27.0 | 0.33 dB | −12.75 dBFS | yes | 95.3 dB | 26.5 dB |
| zap | −27.0 | 0.00 dB | −13.66 dBFS | yes | 29.5 dB | 31.2 dB |
| reverse | −27.0 | 0.26 dB | −13.35 dBFS | yes | 106.0 dB | 19.0 dB |
| closing | −27.0 | 0.23 dB | −9.49 dBFS | yes | 33.4 dB | 18.1 dB |

The loudness range across the presets is 0.08 dB. Every take spread is within 1 dB, so none needs level matching.

The usability tasks (`tests/test_usability.py`) each take at most three field edits or one command and move a
measurement the expected way, from every preset: change the tuning (within 5 cents of a whole tone), shorten the tail,
raise the stick, the body and the wires, lower the bands, add drive and make a ghost note; shorten the clap's bursts and
its tail, lower its bands and add drive; shorten the snap, raise its band and move its mode; the same snare and clap
tasks from the new snares (`machine`, `house`, `long-snare`, `rimshot`) and claps (`short-clap`, `big-clap`); and lower
the rimshot's rim damping, shorten the cross-stick and the rim click, raise the cross-stick rim's damping, move the rim
click three semitones, widen the brush's sweep and slow it. The measurements fade their windows rather than cut them. In
the driven preset the stick is saturated too, so raising it adds less there (1.6× the high-frequency energy of the first
millisecond, against 2–3× in the undriven snares).

## Performance

`tools/bench.py`, each case in a fresh interpreter, report included. A render with a new seed computes its three noise
tables, as every take does; a repeated render finds them cached, as audition's re-renders do. The budget applies to the
first.

| Sound | Rate | New seed | Cached | Budget | Memory above baseline |
| -- | -- | -- | -- | -- | -- |
| 0.5 s hit, three layers | 44.1 kHz | 123 ms | 91 ms | 300 ms | 204 MiB |
|  | 48 kHz | 121 ms | 85 ms | 300 ms | 166 MiB |
|  | 88.2 kHz | 151 ms | 108 ms | 300 ms | 214 MiB |
|  | 96 kHz | 129 ms | 103 ms | 300 ms | 203 MiB |
| 2 s body, 1 s of noise | 44.1 kHz | 323 ms | 277 ms | 1.5 s | 249 MiB |
|  | 48 kHz | 312 ms | 261 ms | 1.5 s | 286 MiB |
|  | 88.2 kHz | 361 ms | 320 ms | 1.5 s | 216 MiB |
|  | 96 kHz | 359 ms | 308 ms | 1.5 s | 236 MiB |
| The brush preset, five tables | 44.1 kHz | 167 ms | 128 ms | 300 ms | 207 MiB |
|  | 48 kHz | 151 ms | 122 ms | 300 ms | 233 MiB |
|  | 88.2 kHz | 194 ms | 145 ms | 300 ms | 277 MiB |
|  | 96 kHz | 174 ms | 126 ms | 300 ms | 270 MiB |
| The rimshot preset, a hoop of four modes | 44.1 kHz | 77 ms | 56 ms | 300 ms | 109 MiB |
|  | 48 kHz | 67 ms | 44 ms | 300 ms | 116 MiB |
|  | 88.2 kHz | 82 ms | 57 ms | 300 ms | 133 MiB |
|  | 96 kHz | 70 ms | 44 ms | 300 ms | 132 MiB |
| Pack: 4 layers × 8 takes, 32 files | 48 kHz | 1.5 s |  | 15 s | (worker processes) |

The heaviest a patch can be is four layers and a brush whose sweep spans an octave between two tables, which reads
seven: 11 tables of 12 MiB at the 48 kHz family's internal rate. It renders a new seed in 230 ms at 48 kHz and 247 ms at
96 kHz, 294 and 270 MiB above the baseline, inside the budget with little to spare; the limit of one brush and of one
octave of sweep keeps it there. A rim costs no table: its modes are sines.

The exact true peak convolves the internal signal with the decimator once, by overlap-add: 9.5 ms for the example patch
at 88.2 kHz. A pack's workers share the cores rather than each taking a thread per core.

## Packs in two SFZ players

`tools/sfz_players.py` writes a pack of the crack preset struck 0, 12 and 24 dB softer, three takes each, and 24 notes:
eight velocities across all three layers, each played three times so the round robins cycle, and each held only 10 ms,
as a programmed hit often is. A pack's map makes every hit a one-shot (`loop_mode=one_shot`); without it both players
released the sample at the note's end, cutting a clap of 220 ms to 15 ms in sfizz and 37 ms in liquidsfz.
`tools/sfz_players.sh` builds sfizz 1.2.3 (`sfizz_render`) and liquidsfz 0.4.1 (through `tools/liquidsfz_render.cc`,
since liquidsfz ships only real-time clients) from source in a throwaway Debian 13 container, renders the notes and
removes the image. The check then identifies every note by the file it matches best, piece by piece, and measures its
gain against that file below 2 kHz. Run on the Oryon:

| Player | Notes on their layer and round robin | Worst match | Gain against the mapping | Spread |
| -- | -- | -- | -- | -- |
| sfizz 1.2.3 | 24 of 24, the round robins in order | 0.9893 | −11.52 dB | 0.048 dB |
| liquidsfz 0.4.1 | 24 of 24, the round robins in order | 0.9975 | −7.22 dB | 0.0001 dB |

So both play every velocity on its layer, cycle the takes in sequence, and apply the group's amplitude curve exactly:
velocity follows the square law across the layers' boundaries, up to each player's own overall gain. What the players do
differently, measured:

- **sfizz** renders deterministically, freewheeling, to 16-bit PCM, which limits the spread on the softest notes, near
  −44 dBFS; the loud notes agree within ±0.008 dB. It preloads a sample's first 1,024 frames and streams the rest, and
  at the seam it plays one frame twice or skips one, so a note matches its file bit for bit at one delay before the seam
  and at the next after it. That is the player's, not the pack's: the files carry no seam.
- **liquidsfz** interpolates even at the files' own rate, which rolls the top off by about 3 dB at 20 kHz and lowers the
  broadband gain of a layer with more high-frequency energy by up to 0.12 dB; below 2 kHz it is flat to 0.0001 dB.
