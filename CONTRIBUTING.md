# Contributing

Thank you for helping with Halyard.

Report a bug or propose an idea in the [issues](https://github.com/vittoriocandolo/halyard/issues/new/choose), where a
form asks for what is needed, and ask a question in the
[discussions](https://github.com/vittoriocandolo/halyard/discussions). Report a vulnerability privately, as
[SECURITY.md](SECURITY.md) explains.

## Set up

Halyard uses [uv](https://docs.astral.sh/uv/):

```
uv sync --all-extras
```

## Check a change

```
uv run coverage run -m pytest
uv run coverage report
uv run ruff check
uv run ruff format --check
uv run mypy
uv run typos
git ls-files -z '*.md' | xargs -0 uv run mdformat --check
uv run zizmor --persona auditor .
```

Coverage runs the tests and requires every line and every branch of `halyard_synth` to run, ruff runs every stable rule,
mypy checks all the code strictly, as Python 3.12 would run it, pytest turns any warning into an error, typos checks the
spelling of code and documentation, in British English with -ize (Oxford spelling: the -ize words in use are listed with
typos's settings in `pyproject.toml`), mdformat checks that the Markdown is formatted, wrapped at 120 as the code is,
and zizmor audits the GitHub workflows. `uv run pytest` alone runs the tests faster.
`git ls-files -z '*.md' | xargs -0 uv run mdformat` formats the Markdown.

`uv run ruff check --fix`, then `uv run ruff format`, fix most of what ruff finds; in that order, the formatter keeps
the trailing commas the linter adds. The formatter warns that the trailing comma rule, COM812, may conflict with it: on
this code they agree, so the rule stays on.

CI runs mypy and the tests on every platform and Python that Halyard supports, and the tests again with the oldest
supported NumPy and SciPy.

CI also fuzzes the patch reader and the WAV reader with [Atheris](https://github.com/google/atheris), two minutes each,
from the presets and a file of every sample format: any input must be read or refused with the reason, and a patch read
must come back unchanged from its canonical text. To fuzz locally, on Linux on x86-64 where Atheris has wheels:

```
uv run --group fuzz python fuzz/patch.py -max_total_time=120
uv run --group fuzz python fuzz/wav.py -max_total_time=120
```

## Rules

- **Correct, simple, minimal, elegant, in that order.** A change that makes the code shorter but less obviously correct
  is not an improvement.
- **Renders stay bit for bit the same.** On one platform with the same NumPy and SciPy, a change to any rendered sample
  changes the sound of existing patches. It needs a reason, a changelog entry, a new version and new golden renders
  (`uv run python -m tools.golden`). Across platforms and library versions, samples may differ by rounding, which is no
  change: the golden renders allow a millionth of full scale.
- **Edits print the same patch in every version.** A patch an editing command printed is kept and rendered later, and a
  take is drawn anew at every render, so `tests/test_edits.py` pins the canonical text each writes by its SHA-256. Their
  numbers are rounded decimals, the same on every platform, unlike a render's samples. A change needs a reason, a
  changelog entry, a new version and new pins.
- **Patches have no defaults.** A new field means a new schema version, and older schemas keep rendering exactly as
  before.
- **Seeds and names are forever.** The keys that seed the noise and the dither, `snare/noise/v1` and `snare/dither/v1`,
  the patch's `instrument = "snare"` and the WAV chunk `snar` never change: a new behaviour takes a new key. They keep
  the name the instrument was designed under, so that the evidence in `docs/` stays the evidence for this code.
- **Every render is judged.** A change to the synthesis keeps the validated domain's rule honest: `docs/validation.md`
  records the evidence, and the adversarial grid in `docs/grid` is checked by the tests.
- **Every behaviour has a test,** written as a property that holds on every platform, such as "rendering in blocks never
  changes a sample", rather than as a stored hash of a render, which rounding would break. Coverage checks that the
  tests run every line and every branch; code no test can reach is removed, not excluded.
- **Errors say where and what.** They name the file or the field path, and the problem.
- **NumPy and SciPy are the only dependencies,** and `sounddevice` only for audition.
- **Every check stays on.** Fix what a linter finds instead of silencing it. The one exception is `assert` in tests,
  which is how pytest checks.

## Documentation

The figures, the sounds and the presets' gallery in `docs/` are generated from the code: `uv run python -m docs.build`
writes them, and a test fails when any is out of date, as when a change to the sound or to a preset changes what they
show. The gallery describes each preset by the user guide's table of presets, so a new preset needs a row there. The
README's quick start runs as a test, and its example report must be what the code prints.

## Measurements

The tools behind [the validation record](docs/validation.md), each run with `python -m`:

```
uv run pytest                                    # the full suite (about a minute)
uv run python -m tools.design_decimator --check
uv run python -m tools.grid run --k 32 --workers 6  # the adversarial grid, about 10 minutes; a brush's
                                                    # references hold up to seven tables of 98–197 MB, so 6 fit in 16 GB
uv run python -m tools.grid analyse docs/grid/k16.csv docs/grid/k32.csv
uv run python -m tools.headroom run --dense      # the report's drive headroom against full renders (about 20 minutes on 12 cores)
uv run python -m tools.headroom analyse docs/grid/headroom.csv
uv run python -m tools.presets                # the eighteen presets over seeds 1–16
uv run python -m tools.bench                     # render time and memory on this machine
uv run python -m tools.sfz_players prepare DIR   # a pack and notes; then tools/sfz_players.sh DIR (Docker)
uv run python -m tools.sfz_players check DIR     # and what sfizz and liquidsfz played
uv run python -m tools.golden                    # regenerate golden renders (a deliberate sound change only)
uv run python -m docs.build                      # the figures, sounds and gallery in docs/, from the code
uv run python -m docs.build --check              # what is out of date (a test checks it too)
```

## Frozen filters

The decimators ship as data in `src/halyard_synth/data`, checked by SHA-256.
`uv run python -m tools.design_decimator --check` designs them again with the current NumPy and SciPy and checks that
each design is the shipped filter: the same length, and the same passband and stopband to 10⁻⁴ dB. The taps themselves
differ by up to about 10⁻⁹ between platforms and library versions, since the minimum-phase method takes the logarithm of
a spectrum 116 dB down; the shipped data, not a fresh design, is what renders use. Without `--check` it replaces the
shipped data, which changes every render.

## Releasing

1. Set the version in `pyproject.toml` and `src/halyard_synth/__init__.py`; a test checks that they agree.
2. Move the changelog's entries under the new version, with the date.
3. Tag the commit `vX.Y.Z` and push the tag.

The release workflow then checks that the tag matches the version, runs the tests, builds the package and publishes it
to PyPI as `halyard-synth`, through PyPI's trusted publishing: configure this repository and its `pypi` environment as a
trusted publisher on PyPI once, and no token is needed. The tests and the build run in a job that never holds the
publishing token; a second job, which runs no project code, uploads what the first built, with attestations of where it
was built. Give the `pypi` environment required reviewers, so that a pushed tag publishes only once someone approves it.
The package is `halyard_synth`; the command is `halyard`.

## Commits and pull requests

Write commit subjects in the imperative, such as "Add a 24-bit export", with a body that explains why. Keep a pull
request to one change, with its tests and documentation.

## Licence

Halyard is licensed under the GNU GPL version 3 or later, with the additional permission in [NOTICE](NOTICE) for the
files it writes. By contributing, you agree to license your contribution under the same terms, so that the sounds people
make with Halyard stay theirs.
