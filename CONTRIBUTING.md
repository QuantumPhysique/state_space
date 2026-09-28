# Contributing

Issues and pull requests are welcome. For anything larger than a fix, open an
issue first so the design can be agreed before the code is written.

## Checks

CI runs these, at the minimum supported SDK and at the current one:

```sh
dart format --output=none --set-exit-if-changed .
dart analyze --fatal-infos
dart test
for f in example/*.dart; do dart run "$f" > /dev/null; done
```

Every exported member has a doc comment, and `public_member_api_docs` keeps it
that way.

## What the tests hold the code to

The `O(N³)` Gaussian process is the specification and the linear-time recursion
the implementation; [Validation](doc/validation.md) lists every reference and its
tolerance. A change to the engine should leave `dense_gp_reference_test.dart`,
`golden_test.dart` and `fast_path_equivalence_test.dart` passing unchanged. A
new component should come with a dense reference test and pass
`checkComponent` from `package:state_space/authoring.dart`.

## Regenerating things

| what | how |
|---|---|
| statsmodels fixtures in `test/fixtures/` | `uv run --with numpy --with statsmodels tool/generate_fixtures.py` |
| the statsmodels exact-diffuse comparison | `uv run --with numpy --with statsmodels tool/statsmodels_diffuse_repro.py` |
| performance tables in `doc/validation.md` | `dart compile exe benchmark/<name>.dart -o /tmp/b && /tmp/b` for `scaling_benchmark` and `components_benchmark` |
| `doc/images/trend.png` | `dart run tool/figure/figure.dart > /tmp/trend.csv && uv run --with matplotlib tool/figure/plot.py /tmp/trend.csv doc/images/trend.png` |
| `doc/images/icon.png`, `icon.svg`, `screenshot.png`, `social-preview.png` | `dart run tool/figure/icon.dart > /tmp/icon.csv && dart run tool/figure/figure.dart > /tmp/trend.csv && uv run --with matplotlib tool/figure/icon.py /tmp/icon.csv /tmp/trend.csv doc/images` |
| calibration tables | `dart run tool/calibration/calibrate.dart --all --today=2026-09-17` |

Record the machine and `dart --version` beside any timings you change.

## Releases

Each release has a `CHANGELOG.md` entry and a `vX.Y.Z` tag on the commit that
was published.
