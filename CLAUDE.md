# state_space

Pure Dart package on pub.dev: exact Gaussian process regression for irregular time series in linear time, as a Kalman filter and RTS smoother over structural components. No runtime dependencies; SDK `^3.9.0`. Written for the weight diary trale (`../trale`), its main consumer.

## Tasks

| Task | Command |
|---|---|
| Dependencies | `dart pub get` |
| Format | `dart format .` |
| Analyze | `dart analyze --fatal-infos` |
| Tests | `dart test` |
| One test file | `dart test test/damped_trend_test.dart` |
| Examples (as CI) | `for f in example/*.dart; do dart run "$f" > /dev/null; done` |
| Package check (as CI) | `dart pub publish --dry-run` |
| API docs | `dart doc` (into the ignored `doc/api/`) |
| Fixtures, figures, benchmark tables, calibration | `CONTRIBUTING.md`, Regenerating things |

**Format, analyze, test and run the examples once, on the finished change.** They are the gate before commit and PR, not a step after every edit. In between, run only what answers a concrete question — one test file, usually.

CI (`ci.yaml`) runs format, analyze, test and the examples at SDK 3.9.0 and stable, compiles the benchmarks, and runs `dart pub publish --dry-run` and pana on every PR. Language features must exist in Dart 3.9.

## Branching and release

- `main` is the only long-lived branch; feature branches `feat/<topic>`, `fix/<topic>`, `docs/<topic>`, `ci/<topic>`, `chore/<topic>` off `main`, back by merge commit. Never commit to `main` directly.
- Every commit lands on `main` as it is (no squash), so every commit is a conventional commit — see `/commit`. The PR title uses the same format.
- Versions are prepared with `/release`: a `release/prepare_vX.Y.Z` branch with one `chore: prepare vX.Y.Z` commit that bumps `version:` in `pubspec.yaml`, renames `## Unreleased` in `CHANGELOG.md` to `## X.Y.Z` and updates the `^X.Y.Z` install line in the README and Getting started. After the merge, `/release` pushes the tag `vX.Y.Z`, which runs `publish.yaml` and publishes to pub.dev. Never tag or `dart pub publish` by hand.

## Changelog

`CHANGELOG.md` is read on pub.dev by developers who call the package. Every change to the public API or to what it returns adds a bullet to `## Unreleased`, the topmost heading, created if it is missing.

- What a caller can now do or will now get, naming the API in backticks. Not the implementation or how the change was found
- Bug fixes start with `Fixed …`
- Refactoring, tests, CI, tooling, benchmarks and docs get no line

## Unreleased changes

Released means listed under a version heading in `CHANGELOG.md` and on pub.dev. Everything else — `## Unreleased` on `main`, any branch, your own earlier commits in a PR — is not published. Changing or removing it needs no deprecation, no compatibility shim and no test asserting it is gone: delete the old code together with its tests, and say so in one line of the PR body.

Released public API is different: under 1.0, a change that breaks callers bumps the minor version, because pub reads `^0.2.0` as `<0.3.0`, and its changelog line says it breaks.

## Dev loop

1. Branch off `main`.
2. Implement the whole change in `lib/`, its docs in `doc/` and the README where they describe it, and its `CHANGELOG.md` line. Tests only where the Tests section below asks for one.
3. Once, on the finished change: format, analyze, test, examples. Fix what they report.
4. `/commit`, then `/pr`.

## Project structure

- `lib/state_space.dart` — the public API; `lib/authoring.dart` — that plus `MatrixBlock`, `samplingResolution` and `checkComponent`, for writing components
- `lib/src/components/` — one file per component; `engine/` — filter, smoother, diffuse initialisation, 2×2 fast path; `fit/` — profile likelihood, optimisers, penalty; `stats/` — distributions
- `test/support/dense_reference.dart` — the dense `O(N³)` Gaussian process the references compare against; `test/fixtures/` — statsmodels output from `tool/generate_fixtures.py`
- `doc/` — guides, published as dartdoc categories through `dartdoc_options.yaml`; `doc/validation.md` lists every reference test and its tolerance
- `example/` — run by CI; `benchmark/`, `tool/` — maintainer scripts, not published (`.pubignore`)

## Code

- Strict analysis (`strict-casts`, `strict-inference`, `strict-raw-types`) with `--fatal-infos`: relative imports inside `lib/`, `final` locals, sorted directives
- No runtime dependencies. `matrices` is a dev dependency for the dense references only
- Doc comments are the API reference on pub.dev (`public_member_api_docs`): what the member is, its units and ranges, what it throws. A component's class doc carries its SDE and discretisation, as in `DampedLinearTrend`
- Arguments are checked in constructors through `lib/src/arguments.dart` and throw `ArgumentError`; data that does not determine the model throws `UnderdeterminedModelException`
- A new component extends `Component` from `authoring.dart`, passes `checkComponent`, and gets a section in `doc/components.md` and in `doc/validation.md`
- No fallback logic unless explicitly asked; a silent default hides the bug. A fit that is not determined says so in `warnings`
- Real weight diaries are personal data: never commit one or anything derived from it. Read exports where they are, keep outputs out of the repository, and give tests a synthetic series with a fixed seed

### Comments

Default is none: names and doc comments say what the code does. Any other comment exists only for a **why** the reader cannot see in the code — a numerical trap, the reason for a tolerance, the trigger of a non-obvious branch — and is one or two lines. Never restate the code, narrate steps, or mark the obvious. The same holds in tests.

```dart
// Bad
final integral = total * h / 3; // Simpson's rule

// Good
// Cancellation can push it a hair below zero when the process barely moves.
return variance <= 0 ? 0 : math.sqrt(variance);
```

### Docs

State what the package does and how to use it, not why a choice was made. A passage earns its place when it stops a reader from filing an issue or making a mistake. A claim that several documents need has one home and links from the rest.

### Tests

The suite must stay small enough to maintain by hand. The reference tests are its spine; any other test is the exception, not part of every change. These rules limit what a change adds: an existing test is removed only when another check already covers it.

- The references — `dense_gp_reference_test.dart`, the other `*_reference_test.dart`, `golden_test.dart`, `fast_path_equivalence_test.dart`, `analytic_limits_test.dart` — pass unchanged after an engine change. A tolerance moves only together with `doc/validation.md`
- New component: one test against its dense kernel through `test/support/`, one for the behaviour it exists for, and an entry in the `checkComponent` list in `api_test.dart`. Not every limit, invariance or branch
- Promises to callers — read-only results, value equality, sending across isolates, a literal `name` — are checked once each in `api_test.dart` and `isolate_test.dart`. A new type joins those lists rather than getting a test of its own
- New feature: test its core calculation against a closed form or a reference. Extend an existing test file before creating a new one
- Bug fix: at most one regression test, and only when the bug was in a calculation
- Recovery and coverage on simulated data, with a fixed seed, only where no deterministic reference exists; they are the slowest and most fragile tests
- Never: argument checks, parameter-vector round trips (`checkComponent` covers them), defaults, the wording of a warning
- When unsure whether a test earns its place, leave it out
- Through the public API, readable on its own; engine references may import `src/engine/`. Shared setup only in `test/support/`
- On failure, assume the implementation is wrong before the test

## Working with the user

- Concise and direct: lead with the answer or action, no preamble or restatement. One sentence if possible.
- Investigate before concluding — read the source, trace the call path. No "the issue is…" before evidence.
- Non-trivial tasks (several files, an interface change, more than one reasonable design): state the approach in a few lines and get approval first.
- Commit and PR text is short: a conventional subject and at most 2–3 lines of why. `/commit` and `/pr` carry the details.
