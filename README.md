# state_space

![Social preview](https://raw.githubusercontent.com/QuantumPhysique/state_space/main/doc/images/social-preview.png)

**state_space** estimates smoothed trends from noisy readings taken at
irregular times. It returns the trend, its slope and a credible band, across
gaps and into the future. It was built for the body-weight diary
[trale](https://github.com/QuantumPhysique/trale), where days are skipped and
readings are sporadic, but works on any time series. Written in pure Dart, it
runs on every Flutter platform.

Under the hood it is Gaussian process regression, computed exactly in linear
time with a Kalman filter and RTS smoother. The hyperparameters are fitted by
maximising the marginal likelihood.



![Readings with a gap, the smoothed trend with its 95% credible band, and a forecast](https://raw.githubusercontent.com/QuantumPhysique/state_space/main/doc/images/trend.png)
## Install

```yaml
dependencies:
  state_space: ^0.1.0
```

The API is pre-1.0 and may still change.

## Usage

```dart
import 'package:state_space/state_space.dart';

final data = [
  Observation(0, 81.3),
  Observation(2, 81.0),   // a gap is not a special case
  Observation(3, 80.5),
  Observation(7, 80.0),
  Observation(8, 80.0),
  Observation(8, 79.8),   // neither are two readings at the same instant
  // ... four weeks of readings in all
];

// fit estimates the variances; the value passed here is only a placeholder
final fitted = fit(StructuralModel.localLinearTrend(processVariance: 1), data);
fitted.warnings;                 // empty here

final everyDay = [for (var d = 0; d <= 30; d++) d.toDouble()];
final trend = fitted.model.smooth(data, grid: everyDay);

trend.mean;                      // the smoothed curve
trend.trendSlope;                // its rate of change
trend.credibleBand();            // lower and upper band, as two arrays

final ahead = [for (var d = 31; d <= 60; d++) d.toDouble()];
fitted.model.forecast(data, ahead);
```

There is no interpolation or resampling. A missing day is a step without an
update, an irregular gap is a different `dt`, and a repeated timestamp is
`dt = 0`.

Time is a plain `double` with no units, calendars or locales, and the process
variances are per its unit. The default search brackets suit days, and
`TimeAxis` turns calendar dates into days.

The fit is Gaussian, so one mistyped reading distorts it. `FitResult.warnings`
names such a reading.

[Getting started](https://github.com/QuantumPhysique/state_space/blob/main/doc/getting-started.md)
has the full series, `TimeAxis`, and
[how to set a bad reading aside](https://github.com/QuantumPhysique/state_space/blob/main/doc/getting-started.md#bad-readings).

## Components

| component | use it for | states |
|---|---|---|
| `LocalLinearTrend` | a smooth trend that keeps its direction; the default | 2 |
| `LocalLevel` | a level with no persistent direction | 1 |
| `TrigonometricSeasonal` | a pattern with a known period, such as a week or a year | 2 per harmonic |
| `RegressionComponent` | dated events and known covariates | 1 per column |
| `Matern` | short-lived correlated deviations beside a trend | 1, 2 or 3 |
| `StochasticCycle` | an approximate rhythm whose period is estimated | 2 |

Each is a Gaussian process kernel with an exact state-space form. The default
trend's posterior mean is a natural cubic smoothing spline. [Components](https://github.com/QuantumPhysique/state_space/blob/main/doc/components.md)
has the kernels, and how to write your own.

## Combining components

Components are added together. One-off events, such as a fortnight over
Christmas or a course of medication, go in as indicator columns. Their
coefficients are states rather than parameters, so they add no dimension to
the fit:

```dart
final diary = ...;               // a year of daily readings

final model = StructuralModel([
  LocalLinearTrend(processVariance: 1e-4),
  TrigonometricSeasonal(period: 7, harmonics: 2, processVariance: 1e-3),
  RegressionComponent([
    IndicatorRegressor('christmas', [(from: 350, to: 364)]),
  ]),
]);   // two parameters to fit

final fitted = fit(model, diary);
final posterior = fitted.model.smooth(diary);
posterior.componentMean(0);      // the trend
posterior.componentMean(1);      // the weekly pattern, separately
posterior.coefficients.first;    // the Christmas effect, with its error

fitted.model                     // residual checks
    .diagnose(diary)
    .ljungBox(lags: 14, fittedParameters: model.parameterCount);
```

`example/events_example.dart` fits a trend and two such indicators to a
simulated year and prints the recovered effects beside the truth they were
generated from.

## Performance

A full `smooth` with a trend takes about 170–190 ns per observation, from a
hundred readings to a hundred thousand, so ten years of daily readings take
under a millisecond. The tables are in [Validation](https://github.com/QuantumPhysique/state_space/blob/main/doc/validation.md#performance).

Results are plain data (`Float64List`s, doubles and small records), so a
model and its posterior can be sent between isolates.

## Documentation

| | |
|---|---|
| [Getting started](https://github.com/QuantumPhysique/state_space/blob/main/doc/getting-started.md) | dates, grids, slopes, forecasts, unequal and bad readings, noise floors, errors, isolates |
| [Choosing a model](https://github.com/QuantumPhysique/state_space/blob/main/doc/choosing-a-model.md) | which components, and how to compare them |
| [Components](https://github.com/QuantumPhysique/state_space/blob/main/doc/components.md) | what each one is, which kernels are reachable, and writing your own |
| [How it works](https://github.com/QuantumPhysique/state_space/blob/main/doc/how-it-works.md) | the Gaussian process, the SDE, the filter, and the numerical choices |
| [Validation](https://github.com/QuantumPhysique/state_space/blob/main/doc/validation.md) | what is checked, against what, and how closely, plus benchmarks |
| [Calibration](https://github.com/QuantumPhysique/state_space/blob/main/tool/calibration/README.md) | how the models behave on weight diaries |
| [Roadmap](https://github.com/QuantumPhysique/state_space/blob/main/doc/roadmap.md) | what might come next, and the references |
| [API reference](https://pub.dev/documentation/state_space/latest/) | every class and method |
| [Changelog](https://github.com/QuantumPhysique/state_space/blob/main/CHANGELOG.md) | what changed in each release |
| [Contributing](https://github.com/QuantumPhysique/state_space/blob/main/CONTRIBUTING.md) | the checks CI runs, and regenerating fixtures, tables and figures |

## Credits

Developed with Claude Code as a pair programmer.

## Licence

[MIT](LICENSE).
