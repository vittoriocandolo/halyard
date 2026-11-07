# Snare/Clap Synth — Design and Implementation Plan

*Revision 5. Follows the Kick Synth (plan revision 2), which was implemented, validated and approved by ear. This is an independent codebase on its own branch: it carries over the kick's designs and findings, not its code.*

The kick's architecture held up in practice. Every synthesis stage is a pure function of time, curves and noise are exact, one saturator feeds one frozen decimator, patches are strict and complete, and the validated domain is measured. This plan keeps all of it and adapts it to an instrument made mostly of noise, for professionals who program snares and claps in a DAW or a sampler.

Values marked *measured* come from two sources:

- the kick's implementation;
- prototypes run on its engine. A snare-like noise layer is the kick's attack with the body silent and the noise mix at 1, so it can be rendered and verified exactly as the kick is. A body under one noise layer, like a snare's body under its wires, renders the same way.

Values marked *estimated* are derived from those, and this project's own grid confirms or replaces them.

What makes this a state-of-the-art instrument is not a bigger model:

- **Measured aliasing.** Every render states whether it lies where its aliasing is demonstrated below −100 dB, by how much, and how much more drive it would take to leave.
- **Exact reproducibility.** A render is exact at every sample rate and depends only on the patch and its seed.
- **Few controls, each physically motivated.** They cover the family: a modal head, wires that buzz, a noise whose colour evolves, clap trains, and the way all of it changes when the drum is struck softer.
- **Built for production.** It delivers velocity layers and round-robin takes that keep their natural levels, as packs a sampler loads directly, in formats that hardware samplers read. You audition in context and against your own samples.

On the goal of never needing another snare or clap sample: no plan can guarantee that. What a plan can do is make the claim testable for a declared family of sounds and show exactly where it fails. That test is the final gate (§8.5).

## Revision 5

The instrument now serves a declared musical aim: producers whose tracks are led by the kick, in house, tech house and trance, along an axis from the programmed end, where a snare is a machine's, to the played end, where it is a drummer's. The machine end and the middle were already in the family; the played end and the genres' particular hits were not.

| Change | Why |
|---|---|
| Rims: up to two `[[rim]]` sources, each up to eight modes at their own frequencies in Hz, with a damping and a shape (§2.3) | A hoop and a stick ring with no harmonic relation to the head, so a rimshot or a cross-stick needs modes that the tuning and the glide do not move |
| A brush: one `[[brush]]`, a noise layer whose band moves by a sweep of up to an octave (§2.4) | Brushes sweep across the head. One noise blended by a cubic B-spline across third-octave copies keeps the written band's shape within 0.5 dB and its level within 0.1 dB all along the sweep, measured on random bands; the blend is C², so the kink bound gains the sweep's knots and the crossings where B‴ steps |
| `flam`: a softer grace note before the main stroke, as ordinary knots (§4) | A flam is played, and producers program it; like `bursts`, it adds no parameter |
| Nine more references, fifteen in all (§1, §8.4) | The genres' hits and the played end: `machine`, `house`, `short-clap`, `rim-click`, `big-clap`, `long-snare`, `rimshot`, `cross-stick` and `brush`, measured like the six |

Both new sources are optional arrays: a patch without them reads and renders as before, so the schema stays 1 for the first release. Acoustic realism stays outside: these are synthesized rimshots, cross-sticks and brushes, not models of a shell, a stick or a room.

## Revision 4

The kick has since implemented this plan's report and delivery additions: the drive headroom and cause, loudness, `tune`, PCM export, and audition in context. What that taught is now in the plan. Revision 3 carried only the first row.

| Finding on the kick | Change here |
|---|---|
| The validated-domain rule is not monotone in drive: just below the drive where a step stops counting, the domain can have an outside island. Scaling the knee slew with α, as revision 2 proposed, cannot see it. | The kick's method: one render records the knee slew, the kink bound and a lower bound on the output peak for every drive, and the search visits every jump of the knee slew (§5) |
| Dithering every sample makes the first sample non-zero, which breaks the promise that every file starts at exactly zero | PCM dither leaves exact zeros alone (§7.2) |
| A bed exported from a DAW can be a sample or two off whole bars, so looping it at its own length drifts away from the hits | The bed restarts on the tempo grid, and one more than 1 ms off whole bars is refused (§7.5) |
| Matching a quiet sample to a loud render can push it past full scale, and a loud bed under the hits can clip the device | Audition warns of both, and rematches the sample on every re-render (§7.5) |
| The report's additions cost 4–17 ms per render at 16×, about 10–15% of the longest kick | The budget counts them, at about twice that at 32× (§3) |
| The report names a knot by its path, as the parser does, and the kick's paths count from 0, while this plan counted layers from 1 | One convention: paths count from 1 in parser errors, the report and the commands (§4) |

## Revision 2

Revision 1 was reviewed as a professional user and a DSP reviewer would read it, and its open claims were measured on the kick engine. The changes:

| Change | Why |
|---|---|
| Dynamics: `softer` derives velocity layers by three physical rules, and `pack` delivers layers and takes with an SFZ mapping (§2.7, §7.3) | Snares are played with dynamics, and a ghost note is darker and flatter, not just quieter |
| One gain per set of takes, and level matching on request (§7.1) | Measured: takes of a snare differ 1.6–2.7 dB in sample peak but at most 0.1 dB in loudness, so normalizing each take to its own peak would make round robins uneven |
| Short noise events vary with the seed (§2.4) | Measured: snap-like bursts vary 1.4–3.1 dB in loudness between seeds |
| A sixth reference, Snap, and tuning up to 4 kHz (§1, §5) | Snaps and rim clicks are in the family but had no reference, and a single high ping needs a high fundamental |
| Drive headroom in every report, from one render (§5) | Measured: never optimistic, and exact near the boundary |
| The rule's boundaries at 32×, measured (§3) | 20–21 dB of noise level + drive for a 2–9 kHz layer, not the 18 dB assumed |
| Audition on the backbeat, over a loop, and against a sample (§7.5) | Snares are judged in the mix and against references |
| PCM output at 24 and 16 bits, with seeded dither (§7.2) | Hardware samplers |
| Loudness in the report, and references matched in loudness (§7.4, §8.4) | A kit needs consistent levels |
| `tune`, `stretch` over several curves, and `bursts --start` (§4) | The most common professional edits, and stacked claps |
| Silent sources no longer set a file's length (§3) | A one-shot should end with its sound |
| A schema stability promise (§6) | An old project must render the same |
| A user guide (§9, §10) | Professionals need recipes, not this plan |
| The third membrane ratio is 2.295, not 2.296 (§2.3) | An error |

## What the kick taught us

| Finding, measured on the kick | Consequence here |
|---|---|
| **Curve kinks alias with no drive at all.** A 10 µs rise under a full-scale carrier reached −93 dB at 16× (−74 dB at 8×, −114 dB at 32×). The error falls 12 dB per doubling of the segment and 18 dB per doubling of F. A first-principles bound was never exceeded on 205 undriven fixtures, and sits 6–21 dB above the measured error. | Snares and claps are fast envelopes on noise. The bound is part of the design from the start, extended to the modal body and the rattle (§5). |
| **The largest step of the saturator's argument is the wrong statistic.** The step where tanh bends is the right one: a 0 dB attack at 24 dB of drive stepped 1.34 overall but aliased at only −126 dB. | Noise-driven saturation is this instrument's main source of aliasing, and the knee slew measures it (§3, §5). |
| **The final rule admitted 367 of 737 grid fixtures**, `2·exp(−π²/q_knee)·P + kink bound ≤ 10⁻⁵·P`, and every one passed; the worst was −110 dB. | The same rule form starts here, with its constant refit (§5). |
| **8× failed even undriven fast-knot patches.** 32× had the right physics but broke the kick's budget: 281 ms for a 1 s kick at 88.2 kHz. | Snares are short, so 32× is affordable, and the prototype says it is needed (§3). |
| **Explicit points beat generator parameters.** The kick's long tail needed `stretch`, a command that rewrites points. | Clap trains and softer hits are written as knots by commands, not described by parameters (§4). |
| **PCHIP gives a peak zero slope**, so the first segment after it is rounder: 1.1, 2.0 and 5.1 dB above an exponential for knots every 6, 10 and 20 dB. | A crack needs a shoulder knot just after its peak. The templates and `bursts` place one (§2.2). |
| **In a driven sound the edge comes from the saturated source**, not the transient layer: softening the driven kick needed its sweep too. | The usability tasks expect it (§8.4). |
| **Measurement proxies need windows that keep the onset.** A rectangular cut leaked more than the attack itself. | Usability measurements fade out; they never cut (§8.4). |
| **The validated-domain rule is not monotone in drive.** A step near tanh's knee stops counting once α·reach > 1, so just below that drive the domain can have an outside island: a full-scale noise attack is outside at 19.3 dB, inside from 19.5 dB, and outside for good above 20.05 dB. | The drive headroom searches every jump of the knee slew, and the knee slew, the kink bound and a lower bound on the peak are recorded for every drive in one render (§5). |
| **The render's time went to a few places.** Decimation was half of it; parallel chunks, bit-identical to one `upfirdn` call, took a 1 s render at 88.2 kHz from 230–260 ms to 170 ms, back within its 250 ms budget. A noise table took 14–17 ms, 7–23% of a reference render, and curve evaluation was 52% of synthesis. | Parallel decimation, cached noise tables and segment-sliced curves from the start (§3). |

## Changes from the kick

| Kick | Snare/clap | Why |
|---|---|---|
| Body: one sine, a second harmonic and a pitch sweep | Up to four modes at fixed ratios, one pitch glide, higher modes decaying faster | A drum head is inharmonic, and its upper modes die first |
| Attack: tone and noise in a constant-power blend | Up to four layers of one noise realization; the click is a short layer | Noise is the instrument, and layers give a time-varying spectrum exactly |
| — | Rattle: noise gated at the body's fundamental | Snare wires buzz against the head |
| Onset and tail phase modes | Onset only; no phase control | Bass alignment does not apply |
| Noise for at most 1 s, in the attack only | Noise everywhere, for at most 1 s | The same noise period suits snare and clap noise |
| Body up to 8 s | Body up to 2 s | A snare's tone is short |
| K = 16 | K = 32, provisional | Noise-driven saturation and 20 µs clicks (§3) |
| One render per patch | Round-robin takes by seed, and velocity layers as derived patches | A noise instrument repeated identically sounds machine-gunned, and snares are played with dynamics |
| `stretch` | `stretch`, `tune`, `bursts` and `softer` | Common edits, clap trains and softer hits as explicit points |
| Float WAV | Float WAV, PCM at 24 or 16 bits, and packs with an SFZ mapping | Software and hardware samplers |
| Audition: repeating, or on every beat | Also on the backbeat, over a loop, and against a sample | Snares are judged in context |
| Report: predicted aliasing and margin | Also the dominant cause, the drive headroom and the loudness | It tells you what to change |

---

## 1. Scope

The instrument has one voice. A modal body and up to four noise layers are summed into one saturator, band-limited once, and rendered offline to a mono WAV.

It is built for producers and sound designers who program snares and claps in a DAW or a sampler. It renders one-shots, sets of round-robin takes, and velocity-layered packs that a sampler plays.

Its family is synthesized snares and claps: tonal electronic snares, tight bright snares, big modern snares, driven snares, claps, snaps and electronic rim clicks, and, at the played end, synthesized rimshots, cross-sticks and brushes. Fifteen reference patches define acceptance:

- **Tone:** two tonal modes and a soft click over bright snappy noise; round and clearly pitched.
- **Crack:** a quick glide, a sharp click and bright wires with a short decay; tight.
- **Fat:** a lower tuning, a longer body, buzzing wires and a tail that darkens; big.
- **Driven:** body and noise saturated together; dense, still pitched.
- **Clap:** four bursts and a diffuse tail, with no body.
- **Snap:** a short, bright burst over a high ping that dies fast; a finger snap or an electronic rim click.
- **Machine:** two tuned tones and a hiss, no wires and almost no glide; the programmed end.
- **House:** a punchier drum-machine snare, its tone gliding a little under a bright noise that outlasts it.
- **Short clap:** three tight bursts and a short tail, dry; tech house.
- **Rim click:** two short pings and a tick, with no head; the machine's rimshot.
- **Big clap:** two trains in different bands, staggered, over a long diffuse tail; trance.
- **Long snare:** a snare whose wires buzz out past half a second, for builds; trance.
- **Rimshot:** the hoop ringing over a hard hit on the head; the played end.
- **Cross-stick:** the stick's shoulder knocking the rim, the head barely moving.
- **Brush:** a swish across the head, the wires answering, under a soft tap.

These are starting patches. Their names never select hidden algorithms.

Five things are outside the model:

- Acoustic snares, whose sound depends on wire physics, shell, stick position and room; the synthesized rimshots, cross-sticks and brushes of revision 5 do not model them.
- Performance: rolls. One patch is one hit, or one flam written by `flam`; dynamics are delivered as velocity layers (§2.7).
- Sounds defined by processing chains: reverb and gated reverb, multi-stage distortion with EQ between stages, layered samples.
- Stereo width, which belongs to the DAW. Two takes panned apart widen a snare's noise, while its body, identical in every take, stays centred.
- A real-time plugin. The renderer is a pure function of the patch and takes about a quarter of a second per hit (estimated), so a later frontend could render on every edit and play cached buffers from MIDI. That frontend is not part of this plan.

The priority order stays Correctness → Minimalism → Elegance.

## 2. Sound model

The whole instrument fits in three lines:

$$x(t)=b(t)+\sum_{j=1}^{J}w_j(t)+\sum_{k}c_k(t)+v(t),\qquad y(t)=S_\alpha\big(x(t)\big),\qquad \mathrm{out}[m]=p\,g_{\mathrm{out}}\sum_{k=0}^{L-1}h[k]\,y\!\left(\tfrac{mK-k}{F}\right)$$

The symbols are the kick's:

- b is the body and w_j the noise layers, J ≤ 4; since revision 5, c_k are the rims, up to two, and v the brush, if any (§2.3, §2.4).
- S_α is the saturator.
- h is the frozen decimator, of length L.
- F is the internal rate and K the decimation factor.
- g_out is the output gain and p = ±1 the polarity.

Everything before the decimator is a pure function of time, with no state.

### 2.1 Time

As in the kick: the trigger is at t = 0, knot times are milliseconds ≥ 0, and internal sample n lies at t_n = n/F, computed by division and never accumulated.

### 2.2 Curves

Unchanged from the kick. Every curve is a cubic Hermite curve through its knots, with PCHIP slopes: the weighted harmonic mean at interior knots, and the shape-preserving three-point rule at the ends. Overrides are applied first. A curve is constant outside its knots and integrates exactly.

- **Amplitude curves** start and end at 0 with zero end slopes, and peak at exactly 1.
- **The pitch curve** starts at 0 ms, ends at ratio 1 with a zero last slope, and keeps its first slope.

The monotonicity proof carries over, with its consequences: amplitudes stay within [0, 1], and every body frequency is extreme only at pitch knots, so domain checks need only the knots.

**Authoring guidance for transients.** The peak of a shape always has zero slope, because its secants change sign. So the first segment after a peak starts flat. Measured on the kick, that segment rises 1.1, 2.0 and 5.1 dB above an exponential when knots are spaced every 6, 10 and 20 dB. A crack wants the opposite, so place a shoulder knot within about 0.1 ms after the peak; the starting patches and `bursts` do this. Inside a tail, knots every 10 dB of decay stay within 0.15 dB of an exponential (measured).

**Every knot is a kink.** The second and third derivatives jump at knots, and a fast segment under a large carrier aliases (§5). Measured on the prototype, a 20 µs click under a full-scale noise carrier is inside the validated domain at 32× with 12.6 dB of margin, and outside it at 16× by 5.5 dB.

### 2.3 Body

$$b(t)=g_b\sum_{i=1}^{M}a_i\,e_b\big(\tau_i(t)\big)\,\sin\phi_i(t),\qquad \phi_i(t)=2\pi f_0\,r_i\,R(t),\qquad R(t)=\int_0^t r$$

$$\tau_i(t)=\begin{cases}t & t\le t_{\mathrm{pk}}\\ t_{\mathrm{pk}}+s_i\,(t-t_{\mathrm{pk}}) & t>t_{\mathrm{pk}}\end{cases},\qquad s_i=r_i^{\,\beta}$$

Here g_b = 10^(level_db/20), f₀ is `tune_hz`, r(t) is the pitch ratio curve, e_b is the body shape, and t_pk is the time of its first knot at gain 1.

**Modes.** There are one to four.

- The first has ratio exactly 1 and defines the tuning.
- Ratios increase strictly and stay ≤ 8. The ideal circular membrane's ratios, 1, 1.593, 2.136 and 2.295, are a natural start; a tonal electronic snare uses two.
- Mode levels a_i = 10^(level_db/20) form a spectral shape whose loudest mode is exactly 0 dB, just as an amplitude shape peaks at exactly 1. Levels are gains; mode levels are a shape.

**One glide.** All modes follow the same pitch ratio curve, because tension modulation raises every mode of a head together. Phase is integrated exactly, as in the kick. Every mode starts at zero phase, which is zero displacement at the strike. There is no phase control: the kick's tail mode existed for bass alignment, which does not apply here.

**Damping.** A head's upper modes die first. With `damping` β, mode i plays the body shape's decay s_i = r_i^β times faster, from the peak on. The strike itself, the rise to the peak, stays common to all modes. For an exponential decay this gives exactly a decay rate proportional to r_i^β. This construction is chosen over the more obvious e_b^p because it stays clean:

- it is C¹, since the shape's slope at its peak is zero;
- each mode's envelope is still a shape that peaks at exactly 1;
- its jumps simply scale by s_i² and s_i³, so the kink bound covers it with the same formula. Powers of a shape are not piecewise polynomial: where the shape touches zero, at its start for instance, their third derivative can be singular, and the bound's jump terms do not cover that;
- the fundamental (s₁ = 1) lasts longest, so the body's span is the last knot of its shape.

**Harmonics** come from the modes and from the drive. The kick's h2 is not needed; a mode at ratio 2 does its job.

**Tuning.** `tune_hz` is the settled frequency of the first mode. It reaches 4 kHz, so that a snap or a rim click can be a single high ping.

**Rims** (revision 5). A rimshot strikes the hoop with the head; a cross-stick strikes only the hoop, with the stick laid across the head. A hoop and a stick ring with no harmonic relation to the head, so each rim is its own modal source:

$$c_k(t)=g_k\sum_{i=1}^{M_k}a_{ki}\,e_k\big(\tau_{ki}(t)\big)\,\sin 2\pi f_{ki}t,\qquad s_{ki}=(f_{ki}/f_{k1})^{\beta_k}$$

with τ_ki as for the body, from the rim shape's peak. There are up to two rims, each with one to eight modes in Hz from 100 Hz to 20 kHz, strictly increasing, the loudest exactly 0 dB. A rim has no glide and ignores `tune_hz`: the hoop's ring is not the head's. It never gates the rattle. Its shape's knots enter the kink bound exactly as the body's do, with a constant frequency in place of φ₁′. A patch without rims has none, so it renders as before.

### 2.4 Noise layers

$$w_j(t)=g_j\,e_j(t)\,\Big(1-d_j\,e_b(t)\,\tfrac{1-\cos\phi_1(t)}{2}\Big)\,n_j(t)$$

$$n_j(t)=\gamma_j\sum_{k=1}^{20000}\lvert H_j(f_k)\rvert\,\rho_k\cos(2\pi f_kt+\theta_k),\qquad f_k=k\ \mathrm{Hz},\qquad \gamma_j^2\sum_k\lvert H_j(f_k)\rvert^2=\tfrac12$$

**One realization, several spectral windows.** Every layer uses the same components, made as in the kick: SHAKE-256 over a domain-separated key, `snare/noise/v1` plus the seed, gives Rayleigh amplitudes ρ_k with mean square 2 and uniform phases θ_k. Each layer weights them by its own band |H_j|, using the kick's second-order Butterworth skirts with their half-power points at the band edges. The layers are therefore coherent:

- identical bands give bit-identical noise;
- a crossfade between two layers is a spectral morph of one noise;
- editing a band reweights the same components instead of rerolling them;
- two identical layers at equal level sum to +6 dB, not +3 dB.

**Why layers and not a filter envelope.** A snare's noise darkens as it decays. A time-varying filter has no exact form in this component structure: every sample would need its own 20,000-term sum. Layers keep one exact table each and express the same motion, a bright, short layer over a darker, longer one. A band that moves is the brush's, below, and it too is a finite sum of exact tables.

**The click is a layer.** A short, wide layer is the stick; the kick's separate tone-and-noise attack is gone. A tonal ping is a high body mode with strong damping.

**Rattle.** Snare wires slap the resonant head, so their noise is gated at the head's fundamental. The rattle factor follows the first mode's phase φ₁, glide included, and the body shape, because a head that has stopped moving drives no wires. The factor stays within [1 − d_j, 1] and is smooth, so it adds no kinks of its own; the jumps it inherits, from e_b and through φ₁ from the pitch curve, are accounted for in §5. Rattle reads the body's shape and phase, not its level, so a silent body can still gate a layer. That makes a pitched buzz, which is allowed and documented.

**Calibration.** Each n_j has the stationary mean square of a unit sine, ½, as in the kick. Realizations are never renormalized.

**Span.** The noise repeats every second, so noise shapes end by 1000 ms. Snare and clap noise fits in that; longer tails are reverb, which is outside the family.

**Exact samples at any rate** come, as in the kick, from one inverse real FFT per layer, of length F × 1 s. Tables are cached across renders while a layer's seed and band stay the same.

**The seed matters.** Here noise is the instrument, not a garnish on the attack. Near the validated domain's boundary, the measured error spread 6–23 dB across three seeds (prototype), and every seed sounds slightly different. So each seed is a take of the same sound: §7 renders the takes as round robins, §5 judges each one, and §8.4 voices across them.

**Short events vary most.** A short, narrow noise event has few independent components, so its level is itself random. Measured over seeds 1–8, in K-weighted loudness (§7.4):

- snare-like hits vary by 0.05–0.2 dB;
- a four-burst clap varies by 0.7 dB;
- a snap-like burst of 12–30 ms varies by 1.4–3.1 dB, and a 20 µs click alone by 3 dB.

A ping 12 dB above a snap's burst brought its spread down to 0.8 dB, since the ping is the same in every take. Delivery therefore keeps one gain per set, offers level matching, and reports the spread (§7.1).

**Clap bursts** are ordinary shapes with explicit knots, written by the `bursts` command (§4).

**The brush** (revision 5). Bristles swept across a head make a noise whose band moves. One `[[brush]]` is a layer with a sweep s(t), a ratio that multiplies both band edges:

$$v(t)=g\,e(t)\,q(t)\sum_k B\big(u(t)-k\big)\,n_k(t),\qquad u=3\log_2 s(t)$$

where n_k is the brush's noise in its band moved by k third-octaves, B is the cubic B-spline and q the rattle. Because the n_k are one noise, the sum is that noise weighted by Σ_k B(u − k)|H_k|: a filter, not a crossfade. Measured on 300 random bands, it keeps the written band's shape within 0.5 dB down to 30 dB below its peak, and its level within 0.1 dB, at every position of the sweep. The sweep's ratios lie within 0.5 … 2 and an octave apart at most, so a brush reads at most seven tables. The sweep is a curve held before its first knot and after its last, with zero end slopes, so u is C¹ and the blend C²: the kink bound gains the jumps of u″ and u‴ at the sweep's knots, and of B‴ where u crosses a whole number, by u′³ times B‴'s steps of 1, −4, 6, −4 and 1.

### 2.5 Drive

$$S_\alpha(x)=\frac{\tanh\alpha x}{\tanh\alpha}\ \ (\alpha>0),\qquad S_0(x)=x,\qquad \frac{\alpha}{\tanh\alpha}=10^{\,\mathrm{drive}/20}$$

Unchanged from the kick:

- drive in dB is the saturator's small-signal gain, and 0 dB is exact bypass;
- the saturator is odd by construction, and S(1) = 1 exactly;
- α is solved by bracketed root finding on [G − 1, G].

What changes is where the drive bites. On noise, the saturator's aliasing depends on the product α·g_j, so for strong drive the quantity to reason with is **noise level + drive**, in dB. Measured at 64×: a layer at −6 dB with 24 dB of drive gives the same error, within 0.001 dB, as a layer at 0 dB with 18 dB.

### 2.6 Output

Unchanged from the kick. The decimated signal is scaled by the output gain (gain mode) or to a sample-peak target (peak mode). It is then multiplied by the polarity and converted to float32 once, or quantized once to PCM (§7.2).

### 2.7 Dynamics

A patch is one hit, and its levels are those of the hardest hit it will be played at. A softer hit of the same drum is a different patch. `halyard softer PATCH DB` derives it and prints it like any other patch, so every change is visible and can be edited.

With v = 10^(−DB/20), the rewrite applies three rules from the physics of a struck head:

| Rule | Edit | Physics |
|---|---|---|
| Level | Every source level falls by DB, before the drive | The head responds linearly, so every source scales together; entering the drive lower, a softer hit is also cleaner |
| Glide | The pitch curve's excess over 1 scales by v² | Tension modulation raises the pitch in proportion to the vibration's energy, the square of its amplitude |
| Contact | Every rise lengthens by v^(−1/5), and later knots shift by the added time; every layer's upper band edge scales by v^(1/5) | Hertz's contact law: a slower strike stays longer on the head, so it excites less high frequency |

Nothing else changes: the tuning, the decays, the rattle depth, the drive and the output. At 12 dB softer, a 1.08 glide becomes 1.005, a 0.05 ms rise becomes 0.066 ms, and a 16 kHz upper edge becomes 12.1 kHz.

A rise is the run of increasing knots that ends at a local maximum, so each burst of a clap has its own. The rules are exact on the whole curve, not only at its knots (measured to 10⁻¹⁵):

- PCHIP slopes scale with the data, so scaling the knots' excess scales the whole glide.
- Every rise runs between two zero-slope knots, so lengthening it leaves the rest of the shape unchanged.

The exponents are options, `--glide 2` and `--contact 0.2`, and 0 turns a rule off. Voicing decides whether the physical defaults stay (§11). The command fails, rather than clamping, when an edit would leave the valid domain; for example, an upper band edge might reach its lower one, or a shape might pass its span limit.

## 3. Rendering policy

**Internal rate and families.** F = K × base, where base is 44.1 or 48 kHz. The 88.2 and 96 kHz exports reuse their family's F and filter with factor K/2, so a 96 kHz file is exactly the 48 kHz file with the in-between samples filled in.

**Bandwidth.** It is fixed at 20 kHz for every export rate, with the kick's specifications: passband within 0.001 dB, taps summing to exactly 1, and a stopband of at least 110 dB from base/2 upward. The trade-off is recorded: a 96 kHz snare pitched down an octave keeps only 10 kHz. An extended band at the 2× rates is pre-registered (§8.5), not adopted.

**Decimators.** They are designed once by the kick's procedure: a Kaiser prototype at 116 dB, converted to minimum phase by the full-length homomorphic method at 2²⁰ points, then renormalized. Minimum phase keeps every transient free of pre-ringing. They ship as data with checksums. Measured designs:

| Family and rate | Taps | Length | 99% of energy within | Passband deviation | Stopband |
|---|---|---|---|---|---|
| 44.1 kHz, 32× | 5,183 | 3.67 ms | 0.85 ms | 3.7·10⁻⁵ dB | 115.6 dB |
| 48 kHz, 32× | 2,891 | 1.88 ms | 0.52 ms | 1.8·10⁻⁵ dB | 115.1 dB |
| 44.1 kHz, 64× | 10,363 | 3.67 ms | 0.85 ms | 1.8·10⁻⁵ dB | 115.4 dB |
| 48 kHz, 64× | 5,781 | 1.88 ms | 0.52 ms | 1.6·10⁻⁵ dB | 115.2 dB |

The 16× designs match the kick's. A filter's duration is set by its transition width in Hz, so the durations stay the same at every K and only the tap counts grow.

**Timing contract.** As in the kick, but T_src is now the latest end among the shapes of the audible sources: the body shape if the body's level is finite, and the shape of each layer whose level is finite. A silent body still gates rattle, but only inside the layers it gates, so it never lengthens a file.

$$\mathrm{out}[m]=\sum_{k=0}^{L-1}h[k]\,y[mK-k],\quad y[n]=y(n/F)\ \text{for}\ 0\le n<N,\ \text{else } 0;\qquad N=\lfloor T_{\mathrm{src}}F\rfloor+1,\qquad N_{\mathrm{out}}=\Big\lfloor\tfrac{N+L-2}{K}\Big\rfloor+1$$

Nothing trims, fades, high-passes or removes DC. Every file starts at exactly zero, since every shape starts at 0, and ends in the filter's decayed tail, so a sampler needs no fades. PCM dither leaves exact zeros alone, so this holds for PCM too (§7.2).

**Evaluation.**

- The internal signal is computed in blocks of 2¹⁶ samples aligned to n = 0, each source only over its support. A longer render never changes earlier samples.
- Curves are evaluated per segment slice rather than by per-sample lookup, since lookups were 52% of the kick's synthesis time.
- Noise tables are computed together for each render, and cached.
- Decimation runs in parallel chunks that are bit-identical to a single `upfirdn` call, as in the kick.

**Choosing K.** These are prototype measurements: a 260 ms noise layer at 0 dB, band 2–9 kHz, worst of three seeds.

| Noise level + drive | 16× | 32× | 64× |
|---|---|---|---|
| 12 dB | −144 dB, inside | −224 dB, inside | — |
| 18 dB | −86 dB | −150 dB, inside | −244 dB, inside |
| 24 dB | −57 dB | −90 dB | −155 dB, inside |
| 30 dB | −44 dB | −62 dB | −95 dB |

Each doubling of K buys 6 dB of noise level + drive, as the knee-slew law predicts (q ∝ α·g/F). Bisection on drive places the rule's boundary at 32× (measured, seeds 1–3; the rule can have a narrow outside island just below such a boundary, §5):

- 19.9–20.9 dB of noise level + drive for the 2–9 kHz layer;
- 17.9–19.1 dB for a 1–16 kHz layer;
- 34.4 dB for a clap in 800–2500 Hz (seed 1), since a lower band slews less.

A wider 1–16 kHz layer at 18 dB and 32× shows the rule's caution: it measures −125 dB but is predicted outside by 1.9 dB. The same prototype gives two more results:

- a 20 µs click (1–20 kHz, 0 dB) measures −108 dB at 16× but is predicted outside by 5.5 dB, and at 32× it is inside with 12.6 dB of margin;
- a four-burst clap (0.1 ms rises, 800–2500 Hz, 0 dB) is inside at 16× up to 18 dB of drive, with at least 9.8 dB of margin.

The measured cost of a 0.5 s hit with one noise layer, on the kick engine:

| | 16× | 32× | 64× |
|---|---|---|---|
| Export at 48 kHz | 84 ms | 134 ms | 266 ms |
| Export at 88.2 kHz | 94 ms | 184 ms | 346 ms |
| Each extra noise table | 12 ms | 25 ms | 92 ms |

**K = 32 is provisional.** It is the smallest rate that validates 20 µs clicks and 18 dB of noise level + drive, within budget.

The final K is the smallest candidate that meets the performance budget and for which every reference is inside the validated domain, with at least 3 dB of drive headroom (§5), for seeds 1–16. Headroom, not margin, is the criterion, because near the boundary the predicted error moves 11–12 dB per dB of drive. In the prototype, the 2–9 kHz layer at 32× lost 66–73 dB of margin between 18 and 24 dB of drive, depending on the seed.

If voicing puts the Driven reference's wires within 3 dB of that boundary, beyond about 17 dB of noise level + drive for a 2–9 kHz band, 64× is measured against a renegotiated budget. Antiderivative anti-aliasing stays out unless 64× fails too, because it would change what a sample means, and the reference method with it.

**Performance budget.**

- A 0.5 s hit with three noise layers renders in ≤ 300 ms at any export rate, report included. The estimate is about 234 ms of synthesis and decimation at 88.2 kHz, from the measured parts above, and 10–35 ms for the report: the kick's knee-slew record, drive-headroom search and loudness take 4–17 ms at 16×, and the filter windows and sample counts double at 32×.
- The longest sound, 2 s of body with 1 s of noise, renders in ≤ 1.5 s.
- A pack of four velocity layers with eight takes, 32 files, renders in ≤ 15 s, in parallel processes (estimated).
- Memory stays within 300 MB above the interpreter baseline.
- Audition re-renders reuse cached noise tables.

## 4. Patch

The patch is the complete, readable description of the sound. Every field is required and the renderer holds no defaults, so a future default can never change an old patch. `halyard init` writes complete patches. The values below are illustrative, not voiced.

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

[[noise]]                   # noise[1], the stick: short and wide
level_db = -10.0
band_hz  = [1500.0, 16000.0]
rattle   = 0.0
amp = [
  { ms = 0.0,  gain = 0.0 },
  { ms = 0.05, gain = 1.0 },
  { ms = 0.15, gain = 0.7 },  # a shoulder knot keeps the crack sharp
  { ms = 3.0,  gain = 0.1 },
  { ms = 8.0,  gain = 0.0 },
]

[[noise]]                   # noise[2], the wires
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

The parser rejects everything the kick's parser rejects:

- unknown or missing fields;
- wrong types (integers are accepted for floats, booleans never);
- non-finite numbers, except `level_db = -inf` on the body or a layer, which means exact silence;
- knot times that repeat, decrease, or are closer than 10 µs, compared on the decimal values as written;
- amplitude shapes that do not start and end at 0, or whose highest point is not exactly 1;
- pitch curves that do not start at 0 ms or end at ratio 1;
- anything outside the valid domain, and unsupported schema versions.

It also rejects:

- a missing or different `instrument`;
- modes whose first ratio is not exactly 1, whose ratios do not strictly increase, or whose loudest level is not exactly 0 dB;
- fewer than one or more than four noise layers.

It never clamps or repairs a patch. Its messages name fields by path, and paths count from 1 everywhere, in the report and the commands too: `noise[2].amp[3].ms` is the time of the third knot of the second layer.

Every control has a single role, and the model stays small. Still, a three-mode body with three layers has 26 numeric fields and five curves, against the kick's 14 and three, so §8.4 tests that common edits stay short.

**Editing commands** rewrite points visibly and never add a parameter. Each prints a complete, canonical patch, so comments other than the canonical ones are not kept. Layers and knots are numbered from 1, in patch order, as the parser and the report number them.

- `halyard tune PATCH NOTE` sets `tune_hz` from a note name such as D3, or from a frequency in Hz. Modes, glide and rattle follow, since they are ratios. It fails, rather than clamping, when a mode or the glide would leave 10 Hz – 20 kHz.
- `halyard stretch PATCH CURVE... FACTOR [--after MS]` rescales the knot times of one or more curves after their peaks, as in the kick. `all` names every amplitude shape, so one command fits a whole sound to a tempo.
- `halyard bursts PATCH LAYER --count N --spacing MS [--start MS] [--growth G] [--rise MS] [--dip GAIN] [--taper T] [--tail MS]` rewrites a layer's shape as a clap train:
  - N bursts, from 1 to 10, the first starting at `start`;
  - each rises to its peak in `rise`, with a shoulder knot just after the peak, and dips to `dip` before the next;
  - each gap is `growth` times the previous one;
  - peaks follow `taper`, with the loudest exactly 1;
  - the last burst decays to 0 over `tail`.

  The result is ordinary knots: at most 3N + 2, or 3N + 3 with a start, and the command fails beyond the 32-knot limit. Two layers with different bands and starts make a stacked clap.
- `halyard softer PATCH DB [--glide P] [--contact P]` prints the same drum struck DB softer (§2.7).
- `halyard flam PATCH --gap MS [--grace DB] [--glide P] [--contact P]` (revision 5) prints the drum struck twice: a grace note `grace` dB softer by `softer`'s rules, its level written into its shapes so the drive hears it softer, then the main stroke `gap` ms later. Each shape is the grace's until the main stroke starts, takes the grace's value there, and from the main peak on is exactly the main stroke moved by the gap; the pitch curve restarts at the main body's peak. It fails rather than drop knots.

## 5. Domains

There are three separate promises, as in the kick:

- **Valid domain.** What the parser accepts. The renderer promises defined, finite output with bounded time and memory.
- **Validated domain.** Where the numerical criteria of §8 are demonstrated.
- **Voiced regions.** The recorded neighbourhoods of the fifteen references, where musical usefulness is promised.

"Excellent everywhere" is not claimed. The hard limits are:

| Quantity | Valid range |
|---|---|
| `tune_hz` | 40 – 4,000 |
| modes | 1 – 4; the first ratio exactly 1; strictly increasing, ≤ 8 |
| mode levels | −60 – 0 dB, with the loudest exactly 0 |
| body frequency f₀·r_i·r(t) at every pitch knot | 10 Hz – 20 kHz |
| `damping` | 0 – 3 |
| body and layer `level_db` | −inf, or −90 – +24 |
| noise layers | 1 – 4 |
| `band_hz` | 20 ≤ low < high ≤ 20,000 |
| `rattle` | 0 – 1 |
| drive | 0 – 48 dB |
| knots per curve | pitch 2 – 32, amplitude 3 – 32 |
| knot spacing | ≥ 10 µs |
| body curves / noise curves | end by 2 s / by 1 s |
| output | 44.1, 48, 88.2 or 96 kHz; float32, pcm24 or pcm16; gain −60 – +24 dB; peak target −60 – 0 dBFS |
| `seed` | integer, 0 to 2⁶³ − 1, takes included |
| rims (revision 5) | 0 – 2, each 1 – 8 modes in 100 Hz – 20 kHz, strictly increasing, the loudest exactly 0 dB; `damping` 0 – 3, with (f_max / f_1)^damping ≤ 512, the body's largest compression; shape by 2 s |
| brush (revision 5) | 0 – 1; a layer's fields, its shape by 1 s; `sweep` 2 – 32 knots by 1 s, ratios 0.5 – 2 and an octave apart at most, both band edges within 20 Hz – 20 kHz at every knot |

**The validated-domain rule** takes the form measured on the kick, where P is the output peak at unity gain:

$$\text{predicted}=C\,e^{-\pi^2/q_{\mathrm{knee}}}\,P+B_{\mathrm{kink}}\;\le\;10^{-5}\,P\qquad(\text{or }10^{-7}\text{ when }P<10^{-2})$$

- **q_knee** is the kick's knee slew: the largest step of αx between internal samples that crosses zero or touches |αx| ≤ 1.
- **B_kink** is the kick's first-principles bound on the kinks' aliasing:

  $$B_{\mathrm{kink}}=\sum_{\text{knots}}\;\sum_{n=2,3}\frac{4\,\zeta(n+1)\,B\,\lvert J_n\rvert}{(2\pi F)^{n+1}},\qquad B=\tfrac{\text{base}}{2}$$

  Each J_n is a jump in the n-th derivative of y at a knot. The bound is computed exactly from each carrier and its slope at every knot, and from the saturator's local slope S′ and curvature S″. Here it is extended in four places:
  1. **Body modes.** Each mode's amplitude knots after the peak are mapped by its decay compression, so their jumps scale by s_i² and s_i³. At the peak, where the compression starts, the jump joins the shape's unscaled rise to its compressed decay. At pitch knots, as in the kick, the third derivative jumps; for mode i the jump scales with r_i.
  2. **Rattle.** A layer with rattle also jumps wherever the body shape does, through e_b, and at every pitch knot, through φ₁.
  3. **Rims and the brush** (revision 5). A rim's modes jump at its shape's knots as the body's do, at a constant frequency. A brush jumps at its shape's knots as a layer does, with its blend for the noise; at its sweep's knots through u″ and u‴; and where u crosses a whole number, through B‴.
  4. **Carriers at knots.** Noise values and slopes at knot times come from the exact tables by cubic interpolation. At 1.536 MHz this is accurate to 1.3·10⁻⁷ in value and 1.2·10⁻⁵ in slope, where direct summation at 160 knots took 297 ms (measured). The bound is inflated by 1% to cover the interpolation.
- **C** starts at the kick's frozen value of 2. It is refit on this grid as the smallest value that covers every converged fixture with at least 6 dB of margin.
- **If the grid finds a mechanism the rule misses**, as the kick's grid found the kinks, the rule gains a term before anything is frozen.
- **The rule is evaluated per render**, and therefore per take.

**Drive headroom** is how much the drive can change before the validated-domain status does; it is negative outside. Drive does not change the saturator's input x, so one render can record what the rule needs at every other drive. The kick now does exactly this, and this plan inherits its method:

- **The knee slew at every α, exactly.** q(α) = α·M(1/α), where M(m) is the largest step of x that crosses zero or has an end within |x| ≤ m. The render keeps M's small upper frontier: the largest crossing step, and the few steps that beat it, each with its reach.
- **The kink bound at every α, exactly.** The jumps at every knot are computed once, and the bound follows for any α from S′ and S″ at the knots.
- **A lower bound on the output peak.** Every sample is a pure function of time, so the exact output at a few positions follows for any α. The render keeps three kinds: its own loudest outputs; the loudest undriven outputs just after the crest of x, where a weak drive peaks; and the loudest outputs at full drive just after the steepest zero crossing, where a saturated edge overshoots.

The prediction is then exact or overestimated at every drive, so the headroom can only err on the safe side.

**The rule is not monotone in drive.** A step near the knee counts only while α·reach ≤ 1. Just below the α where it stops counting, the domain can have an outside island. On the kick, a full-scale noise attack is outside at 19.3 dB, inside again from 19.5 dB, and outside for good only above 20.05 dB. Revision 2 of this plan proposed scaling the knee slew in proportion to α, which cannot see such islands, and neither can a bisection that assumes one crossing. The search therefore visits every 0.5 dB of drive and both sides of every α at which the knee slew drops, then refines the first change of status.

Measured on the kick's grid, against full renders at the same candidate drives (737 fixtures, 51,295 renders):

- the report is never optimistic beyond the brute force's resolution of 0.00001 dB;
- it is exact to 0.001 dB on 540 of the 673 fixtures whose status changes within 0–48 dB, and within 3 dB of the edge it is at most 0.027 dB conservative;
- where full renders never change status within 0–48 dB, on 64 fixtures, it agrees on every one;
- dense 0.05 dB scans on 147 fixtures found no change of status between the candidates.

The larger gaps, up to 2.3 dB on 3 fixtures, come from an output peak that moves to a position no probe covers, which only makes the headroom smaller. The search costs 4–17 ms per render on the kick.

Once measured, the domain is also published as a rule of thumb. The prototype's figures at 32× are about 20 dB of noise level + drive for a 2–9 kHz layer, 18 dB for 1–16 kHz, and 34 dB for a clap's 800–2500 Hz band. They come from bisection, so an island can bring the first change of status slightly lower; the grid gives the real figures.

`halyard verify` runs the reference comparison on any single patch or set of takes, so settings outside the declared domain can still be checked one by one.

## 6. Repeatability

**Contract.** The same patch, renderer version and frozen decimator reproduce the same sound. On one machine and environment the output is bit-identical. Across machines and library versions it agrees within floating-point rounding, so cross-platform tests use tolerances, not hashes.

**Stability.** Within a schema version, a new release never changes the sound of an existing patch beyond floating-point rounding, and the golden renders enforce this (§10). A change that would alter old patches becomes a new schema version, and older schemas keep rendering as they did.

**Randomness.** As in the kick, the noise coefficients are a pure function of the seed, through SHAKE-256 and our own Box–Muller map, and never come from a library generator. The integer stage is platform-independent and prefix-stable. PCM dither comes from the same construction under its own key, `snare/dither/v1`, so a PCM file is as reproducible as a float one.

**Takes.** The seeds seed, seed + 1, and so on are the round-robin takes of a patch. A patch is one sound; takes are a delivery feature, never a patch parameter. In a pack, take j of every velocity layer uses that layer's seed + j − 1.

**Invariants under test:**

- Repeating a hit changes nothing.
- Editing the body never changes the noise realization. It does change the rattle, by design.
- Editing a band reweights the same noise, and identical bands give bit-identical layers.
- Output gain and polarity never touch synthesis.
- Longer renders never change earlier samples.
- The 48 kHz export equals every second sample of the 96 kHz export.
- A take differs from its patch only in the seed, and its file records that seed and its gains.
- `softer` by 0 dB changes nothing, and `softer` by a then by b equals `softer` by a + b, up to the decimal rounding of the values it writes.
- A pack re-renders bit-identically.
- Every exported WAV records the patch text, renderer version, decimator checksum and dependency versions.

## 7. Delivery

### 7.1 Levels and sets

As in the kick:

- gain mode applies the output gain;
- peak mode scales to a sample-peak target of at most 0 dBFS, and silence stays silence;
- when a gain-mode render exceeds full scale, `render` refuses to write it and `audition` keeps playing the last valid buffer. Both state the peak and the reduction needed, and `--allow-over` writes a float file anyway;
- nothing is ever clipped, and the final buffer is checked for non-finite values.

**A set shares one gain.** The takes of a render, and every file of a pack, are scaled together. In peak mode that gain brings the loudest file of the set to the target. Normalizing each take to its own peak would make the takes uneven, because they differ more in peak than in loudness (measured, seeds 1–8):

| Sound | Loudness spread | Sample-peak spread |
|---|---|---|
| Snare: body and wires | 0.10 dB | 1.6 dB |
| Snare: body and a 50 µs crack | 0.05 dB | 2.7 dB |
| Clap: four bursts | 0.7 dB | 1.5 dB |
| Snap-like bursts, 12–30 ms | 1.4–3.1 dB | 4.2–7.1 dB |
| A 20 µs click alone | 3.0 dB | 7.2 dB |

**Level matching**, with `--level-match`, instead scales each take to the loudness of its layer's first take, for sounds whose takes vary too much, such as snaps. The report flags any set whose loudness spread exceeds 1 dB. Each file records its take gain, so it stays reproducible.

### 7.2 Files

Output is mono WAV from our own RIFF writer:

- **float32** by default;
- **PCM** at 24 or 16 bits for hardware samplers. It uses the plain PCM format tag, which hardware reads most widely, and triangular (TPDF) dither of ±1 LSB from the seeded generator. Exact zeros stay zero: the silent first sample and any digital silence are not dithered. A PCM render must peak at least 2 LSB below full scale, so that dither can never clip it; peak mode lowers its target by those 2 LSB, and `--allow-over` does not apply.

A file contains:

- `fmt` and `data` chunks, and a `fact` chunk for float;
- a `smpl` chunk with the root note from `tune_hz`, omitted when the body is silent;
- a private `snar` chunk with the patch text and provenance: the seed, the take and set gains, the renderer version and the decimator checksum.

`halyard extract` recovers the exact patch.

### 7.3 Packs

`halyard pack PATCH... --out DIR [--takes N] [--key K] [--level-match]` renders a multisample. Each patch is one velocity layer, loudest first, and each layer gets N takes.

- The patches must share their output settings, and their measured loudness must fall from each layer to the next.
- Files are named `<stem>_v<top velocity>_rr<take>.wav`, such as `snare_v127_rr1.wav`, so they also map by hand in samplers that do not read SFZ.
- `<stem>.sfz` maps them:
  - velocity ranges follow a square-law curve, 40·log₁₀(v/127) dB, placed from each layer's loudness relative to the loudest;
  - one `<group>` per layer carries its range and a curve point at every velocity in it, so the level follows the square law across layer boundaries (`amp_velcurve_N`);
  - takes cycle in sequence (`seq_length`, `seq_position`);
  - the drum never transposes (`pitch_keytrack=0`);
  - the key defaults to 38, the General MIDI snare; a clap takes 39.
- The pack report lists each layer's loudness, velocity range, take spread and domain status.

### 7.4 Report

After each render the report shows:

- tuning in Hz and as note ± cents, and the mode frequencies;
- source and exported duration, sample rate, format and seed;
- sample peak in dBFS, a 4× oversampled true-peak estimate (information only), and the loudness: the largest K-weighted momentary loudness (ITU-R BS.1770, 400 ms) over every window position, counting windows that run past the end of the file, in LUFS;
- validated-domain status and margin, the dominant cause (saturation, or kinks, with the knot that contributes most, named by its path), and the drive headroom; outside, the drive to lower to;
- for sets, the loudness and peak spread of the takes;
- render time.

### 7.5 Audition

As in the kick:

- it plays the exact float buffer that `render` would write; PCM formats add only their quantization;
- each edit starts a render tagged with an increasing generation number, and stale renders are dropped;
- an invalid edit leaves the last valid buffer playing;
- a new buffer takes over at the next hit boundary;
- without a tempo, hits repeat every rendered length plus 0.5 s;
- the stream opens at the patch's sample rate and never resamples.

For a snare:

- **`--bpm`** places hits on beats 2 and 4 of each 4/4 bar, where overlapping tails sum. `--beats` chooses others, such as `1,2,3,4` for a clap on every beat or `3` for half time.
- **`--with FILE`** loops a bed under the hits, such as the track's kick-and-hat loop. The bed must be at the patch's sample rate and within 1 ms of a whole number of bars at the given tempo. It restarts on the tempo grid rather than looping at its own length, so it never drifts from the hits. A stereo bed opens a stereo stream, with the snare centred.
- **`--against FILE`** alternates the render with a sample, hit by hit, matched in loudness and rematched on every re-render; the sample is scaled, never the render. With `--blind`, each pair plays in random order, revealed when audition stops. A stereo sample plays as its mid channel. This is the benchmark's comparison (§8.5).
- **Clipping.** Audition warns when the matched sample, or the hits over the bed, would clip the device, so a distorted comparison is never mistaken for the sound.
- **Several patches** cycle hit by hit, for velocity layers or for A/B between patches, and `--takes N` cycles takes. The first plays as soon as it is ready, and the others render behind it.

### 7.6 Commands

```
halyard init crack --tune D3 > snare.toml              # writes tune_hz = 146.832; the patch stays in Hz
halyard audition snare.toml --watch --bpm 92 --with groove.wav
halyard audition snare.toml --watch --against favourite.wav [--blind]
halyard render snare.toml --out snare.wav [--takes 8] [--level-match]
halyard verify snare.toml [--takes 8]
halyard extract snare.wav > snare.toml
halyard tune snare.toml F3
halyard stretch snare.toml all 0.8
halyard bursts clap.toml 'noise[1]' --count 4 --spacing 11
halyard softer snare.toml 12 > snare-12.toml
halyard flam snare.toml --gap 25 > flam.toml
halyard pack snare.toml snare-8.toml snare-16.toml snare-24.toml --takes 6 --out kit/
```

A typical session goes like this:

1. Start from the nearest reference.
2. Voice it on the backbeat, over the track's loop.
3. Match it against the sample it should replace.
4. Derive softer layers.
5. Pack them with takes for the sampler.

## 8. Validation

### 8.1 Numerical criteria

| Area | Method | Pass |
|---|---|---|
| Curves | As the kick, plus decay compression | 10⁻¹² relative; compressed shapes keep their peak at exactly 1 and their segments monotone |
| Tuning and modes | Multi-sine fit on the settled internal body, at tunings up to 4 kHz | Every mode within 0.001 cent of f₀·r_i |
| Damping | Exponential body shape; fitted decay rate per mode | Ratio to the fundamental's rate equals r_i^β within 0.1% |
| Noise | As the kick, plus coherence | Common instants ≤ 10⁻¹²; 200-seed mean square 0.5 ± 0.5%; band spectra within statistical bounds; identical bands bit-identical; disjoint bands uncorrelated within statistical bounds |
| Rattle | Flat shapes; demodulated layer envelope | Modulation at the fundamental with depth d within 1%; none when d = 0 or the body shape is 0 |
| Drive | Identities | As the kick |
| Dynamics | `softer` on every reference, over ladders of 0–36 dB | Levels fall by exactly DB; glide excess scales by v²; rises by v^(−1/5) and upper band edges by v^(1/5); the curves are exact to 10⁻¹²; `softer` composes |
| Decimators | Frozen coefficients | As the kick, for every shipped K |
| Kink bound | Every undriven grid fixture | Never exceeded by the measured error |
| Rendering accuracy | §8.2 over the §8.3 grid | E_max and E_rel ≤ −100 dB for every fixture inside the validated domain, for every seed tested |
| Drive headroom | The report's value against full renders at the same candidate drives, on every grid fixture; dense 0.05 dB scans on a subset | Never optimistic beyond the brute force's resolution; no change of status between the candidates |
| Loudness | EBU Tech 3341 test signals, at every export rate; the K-weighting against BS.1770's 48 kHz table | Within 0.1 LU; coefficients to 10⁻¹⁴, and the other rates within 0.03 dB of its response, as on the kick |
| Export | Independent readers | As the kick; each take carries its own seed, gains and patch text; PCM reads back exactly; dither is TPDF within statistical bounds, bit-identical per seed, and never reaches full scale |
| Packs | The SFZ mapping, parsed back | Every velocity 1–127 and every take maps to exactly one file; the level follows the square law across layer boundaries within one velocity step; a pack re-renders bit-identically |
| Takes | Seeds 1–8 of each reference | One gain per set; loudness spread ≤ 1 dB, with level matching where the report flags it |
| Performance | Named machine, with CPU model and library versions recorded | 0.5 s hit with three layers in ≤ 300 ms at any rate; 2 s in ≤ 1.5 s; a 32-file pack in ≤ 15 s; ≤ 300 MB; render latency reported separately from the wait for the next hit |

### 8.2 Reference method

The method is unchanged from the kick:

1. Render the same patch at M·F.
2. Remove everything above F/2 with a zero-phase FFT mask: flat up to 0.4F, then a raised cosine down to zero at 0.5F.
3. Keep every M-th sample.
4. Pass the result through the same frozen filter.

The mask is applied in segments with guard bands, and the band-limited signal keeps its ringing before the trigger. There is no time shifting, no gain matching, and the output gain is fixed at unity. Filter leakage is bounded and reported separately, and near-silent fixtures use the absolute bound of 10⁻⁷.

Two things adapt:

- **Reference rates are set by absolute rate, not by M.** For the 48 kHz family, the reference renders at 6.1 MHz and its convergence check at 12.3 MHz. That makes M = 4 and 8 at 32×, 8 and 16 at 16×, and 2 and 4 at 64×. A reference counts only if the pair agrees 20 dB below the threshold, and a 24.6 MHz check on a subset confirms the trend. The prototype's references all converged this way.
- **Noise at the reference rate** uses exact FFT tables, one per layer, at about 200 MB each at 24.6 MHz. The grid runs within about 2 GB per worker.

### 8.3 Adversarial grid

The validated domain comes from combined corners, not from moving one knob at a time:

| Family | Covers |
|---|---|
| body | tuning 40–4000 Hz × modes (1–4, ratios up to 8) × glide depth × damping 0–3 × level × drive |
| noise | bands (the lowest, the narrowest, 2–9 kHz, the widest, the top 1 kHz) × level × drive 0–48 dB |
| layers | four coherent overlapping layers at +24 dB; crossfades |
| rattle | depth 0–1 × fundamental × glide × drive |
| clicks | 10–50 µs rises under full-scale noise; high, strongly damped modes |
| bursts | 2–10 bursts × spacing 1–30 ms × rises of 10 µs – 1 ms × drive; stacked trains |
| snaps | a single mode at 1–4 kHz under bursts of 1–30 ms |
| dynamics | `softer` ladders of 0–36 dB from every reference |
| seeds | one fixture per family across 16 seeds |
| rates | 44.1, 88.2 and 96 kHz exports |
| random | at least 200 fixtures over the valid domain |
| longest | 2 s of body and 1 s of noise, at 96 kHz |

The grid runs at 16× and 32×, and at 64× if §3's criterion calls for it. It produces the final K, the rule's constant and the frozen decimators, and it checks the drive-headroom estimate on every fixture. Its results are committed, and a test re-checks the frozen rule against them.

### 8.4 Listening and usability

The six references become recorded acceptance material:

- approved patches, matched in loudness within 1 dB and peaking below −1 dBFS;
- measured useful neighbourhoods in tuning, glide, body/noise balance, crack, tail length, rattle and drive;
- renders both dry and in context, with a kick and a simple groove, since snares and claps are judged in the mix;
- blind, level-matched comparisons, with the matching method recorded.

**Voicing holds across takes and strengths.**

- A reference is approved only if seeds 1–8 all sound like it. One lucky realization does not count.
- From each snare reference, a pack of four layers, 0 to 24 dB softer, sounds like one drum struck four ways when played as a groove with ghost notes.
- A 16th-note roll at 120 bpm with six takes has no machine-gun effect, heard next to the same roll with one take.

Usability is measured as tasks that start from a reference:

- **From each snare:** change the tuning, shorten the tail, add crack, add body, add wires, darken, add aggression, and make a ghost note.
- **From the clap:** tighten it, shorten its tail, darken it, and add aggression.
- **From the snap:** shorten it, brighten it, and move its ping.

Each task should take at most three field edits or one command, and move a measurement in the expected direction. Two lessons from the kick apply: measurements use windows that preserve the onset, and they check direction rather than arbitrary magnitudes. In driven references, expect the edge to live in the saturated source. A task that fails gets an editing command that visibly rewrites the points, never a second parameter competing with them.

### 8.5 The sample-replacement benchmark

This is where "no more snare or clap samples" becomes a measurement.

1. Collect the 20–30 snare and clap samples you actually use that fall inside the family, with their velocity layers where they have them.
2. For each one, voice a patch within a fixed time limit, using `audition --against`.
3. Compare the two blind and level-matched, in context with the kick and the groove, over several takes. Where the source has velocity layers, compare packs in a groove with dynamics.

The claim holds for the fraction that passes. Each failure is sorted as a missing control, a voicing-skill issue, or out of family.

Six additions are pre-registered and may be added only if they fix recorded failures:

- a post-drive low-pass for the fizz of heavy drive;
- per-mode decay times, if one damping tilt cannot place a mode;
- impulsive wire noise with a density control, if the Gaussian sizzle lacks crackle;
- an extended band at 88.2 and 96 kHz, if repitched takes sound dull;
- a separate drive for the body, if one saturator on the sum cannot put a driven body under clean wires;
- seeded take-to-take variation of body level, tuning and decay, if takes that differ only in their noise sound too alike.

Anything else needs its own evidence.

## 9. Implementation shape

**An independent codebase, on its own branch.** Nothing is imported from the kick. What carries over is the design and the measured findings. The kick's validated algorithms are re-implemented from this plan and re-verified by this project's own tests: curves, the noise construction, decimator design, the reference method, the RIFF writer and the audition scheduler.

**Dependencies.**

- Python 3.11+ (for `tomllib`).
- Runtime: NumPy; SciPy for `upfirdn`, a bracketed root finder, FFTs and the K-weighting filters; optional `sounddevice` for audition.
- Standard library for TOML, argument parsing and SHAKE-256.
- Development only: SciPy's filter design.
- Tests: adaptive quadrature and an independent WAV reader such as `soundfile`.

It installs with pip as one package with a `halyard` command, and its tests run on Linux, macOS and Windows.

**Modules.**

| Module | Responsibility |
|---|---|
| `patch.py` | Schema, parsing, validation, canonical TOML; `tune`, `stretch`, `bursts` and `softer`; tuning helpers |
| `curve.py` | Hermite curves, slopes, exact integrals, derivatives, knot jumps, decay compression |
| `noise.py` | Seeded noise, coherent layer tables (cached), values and slopes at knots, seeded dither |
| `voice.py` | Body modes, noise layers with rattle, drive, slew statistics, the kink bound |
| `render.py` | Rates, frozen decimators, parallel decimation, output stage, domain rule and drive headroom, report, takes |
| `loudness.py` | K-weighting at every export rate, momentary loudness, true peak |
| `pack.py` | Velocity layers, takes, set gains and the SFZ mapping |
| `reference.py` | The reference renderer and comparison behind `verify` |
| `wav.py` | RIFF writing and reading, float and PCM |
| `audition.py` | Hits, tempo and beats, beds, A/B and blind order, watching, takes, playback |
| `cli.py` | Commands |
| `data/` | Frozen decimators and checksums, the six references |
| `tools/` | `design_decimator.py`, `grid.py`, `headroom.py` (the drive headroom against full renders on the grid), `bench.py`, `golden.py` |
| `tests/` | The test suite, including the grid-evidence test |
| `docs/` | `GUIDE.md` for users: the workflow, every control, recipes, reading the report; `VALIDATION.md` and `BENCHMARK.md` for the evidence |

The central call is `render(patch) → (audio, report)`. There is no synthesis graph, plugin system or drum framework.

## 10. Milestones

| Milestone | Delivers | Exit criterion |
|---|---|---|
| 1. Complete path | Curves with decay compression, the modal body, coherent noise layers with rattle, drive, provisional 32× decimators, the WAV writer (float and PCM), `render`, `tune`, `stretch`, `bursts`, `softer` | All §8.1 tests except rendering accuracy, drive headroom and packs pass; a pitch-accurate snare and a clap import into a DAW |
| 2. Measure and freeze | Reference renderer, metrics, adversarial grid at 16× and 32× (64× if needed), the drive-headroom estimate | K, the rule's constant and the decimators frozen; the estimate checked on every fixture; golden renders begin |
| 3. Audition and delivery | Watching, playback, tempo and beats, beds, A/B, takes, loudness, set gains, packs | Plays the exact render buffer; stale renders dropped; swaps at hit boundaries; patches and takes cycle hit by hit; the bed never drifts from the hits; a pack loads in two SFZ players with the right velocity and take mapping |
| 4. Voicing | Six references across seeds and strengths, neighbourhoods, usability tasks, the user guide | Acceptance material recorded; controls that did not earn their place removed. Rattle, damping, the fourth layer and `softer`'s defaults must each earn theirs. |
| 5. Benchmark | Sample-replacement benchmark | Pass rate and failure categories documented; domains published |

**Regression contract.** From milestone 1, the invariants are those of the continuous-time model and of the rewrites: curves, tuning, modes, damping, noise, rattle, the drive identities and `softer`'s rules. Sample-identical golden files begin only at milestone 2, when the rendering policy is frozen.

## 11. Open decisions

Six decisions remain, each with its settling method:

| Decision | Settled by |
|---|---|
| Final K (evidence points to 32×) and the rule's constant | The adversarial grid, with §3's criterion |
| Three or four noise layers; whether rattle and damping earn their place | Voicing and the usability tasks |
| Voiced ranges of drive, damping, glide and rattle | The six references |
| `softer`'s default exponents, and whether contact should darken every layer or only the short ones | Voicing packs from the references |
| Maximum lengths (2 s of body, 1 s of noise) | Use and render budgets |
| Whether any pre-registered addition earns its place | The benchmark |

Everything else above is specified.

---

## References

- F. N. Fritsch and R. E. Carlson (1980). Monotone piecewise cubic interpolation. *SIAM Journal on Numerical Analysis*.
- F. N. Fritsch and J. Butland (1984). A method for constructing local monotone piecewise cubic interpolants. *SIAM Journal on Scientific and Statistical Computing*.
- N. H. Fletcher and T. D. Rossing (1998). *The Physics of Musical Instruments*, 2nd edition. Springer. Membrane modes and the snare drum.
- T. D. Rossing, I. Bork, H. Zhao and D. O. Fystrom (1992). Acoustics of snare drums. *Journal of the Acoustical Society of America* 92(1).
- T. D. Rossing (2000). *Science of Percussion Instruments*. World Scientific.
- S. Bilbao (2012). Time domain simulation and sound synthesis for the snare drum. *Journal of the Acoustical Society of America* 131(1).
- F. Avanzini and R. Marogna (2010). A modular physically based approach to the sound synthesis of membrane percussion instruments. *IEEE Transactions on Audio, Speech, and Language Processing* 18(4). Tension modulation and pitch glide.
- K. L. Johnson (1985). *Contact Mechanics*. Cambridge University Press. Hertzian impact.
- B. H. Repp (1987). The sound of two hands clapping: an exploratory study. *Journal of the Acoustical Society of America* 81(4).
- L. Peltola, C. Erkut, P. R. Cook and V. Välimäki (2007). Synthesis of hand clapping sounds. *IEEE Transactions on Audio, Speech, and Language Processing* 15(3).
- S. P. Lipshitz, R. A. Wannamaker and J. Vanderkooy (1992). Quantization and dither: a theoretical survey. *Journal of the Audio Engineering Society* 40(5).
- NIST FIPS 202 (2015). SHA-3 Standard: Permutation-Based Hash and Extendable-Output Functions (defines SHAKE-256).
- ITU-R BS.1770. Algorithms to measure audio programme loudness and true-peak audio level.
- EBU Tech 3341. Loudness metering: "EBU mode" metering to supplement loudness normalisation.
- The SFZ format specification, sfzformat.com.
- The Kick Synth plan, revision 2, and its validation record, `docs/VALIDATION.md` in the kick's repository.
