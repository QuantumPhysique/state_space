# state_space

Exact Gaussian process regression for irregular time series, in linear time.
Kalman filtering, RTS smoothing, and marginal-likelihood hyperparameter
estimation. Pure Dart, no runtime dependencies.

**What comes back** is the posterior as a time trace: a mean, a variance and an
interval at every output time, at the observation times or on any grid you ask
for, including between readings and beyond both ends of the data.

```dart
final data = [
  Observation(0, 81.2),
  Observation(1, 80.9),
  Observation(9, 80.4),   // a gap is not a special case
  Observation(9, 80.6),   // neither are two readings at the same instant
];

final fitted = fit(StructuralModel.localLinearTrend(processVariance: 1), data);

final everyDay = Float64List.fromList([for (var d = 0; d <= 20; d++) d + 0.0]);
final trend = fitted.model.smooth(data, grid: everyDay);

trend.level;                  // the smoothed curve
trend.slope;                  // its rate of change, for free
trend.credibleInterval(12);   // and how sure it is, at output point 12

final ahead = Float64List.fromList([for (var d = 21; d <= 50; d++) d + 0.0]);
fitted.model.forecast(data, ahead);            // and where it is heading
```

Components add up, and the posterior comes apart the same way. Events that are
not periodic — a fortnight over Christmas, a course of medication — go in as
indicator columns and cost the optimiser nothing at all, because their
coefficients are states rather than parameters:

```dart
final model = StructuralModel([
  const LocalLinearTrend(processVariance: 1e-4),
  TrigonometricSeasonal(period: 7, harmonics: 2, processVariance: 1e-3),
  RegressionComponent([
    IndicatorRegressor('christmas', [(from: 350, to: 364)]),
  ]),
]);   // still a two-dimensional fit: the coefficient is a state

final fitted = fit(model, diary);
final posterior = fitted.model.smooth(diary);
posterior.componentMean(0);      // the trend
posterior.componentMean(1);      // the weekly pattern, separately
posterior.coefficients.first;    // what that fortnight was worth, +/- its error

fitted.model                     // and whether to believe any of it
    .diagnose(diary)
    .ljungBox(lags: 14, fittedParameters: model.parameterCount);
```

`example/events_example.dart` fits a trend and two such indicators to a
simulated year and prints the recovered effects beside the truth they were
generated from.

Nothing is interpolated, nothing is resampled, and no gap is filled. Missing data
is a step with no update, an irregular gap is a different `dt`, and a repeated
timestamp is `dt = 0`. All three fall out of the recursion rather than being
handled.

## Which kernels

Each component is a Gaussian process kernel written in the form that makes it
fast, and they compose block-diagonally.

| component | kernel | states | prior |
|---|---|---|---|
| `LocalLevel` | Brownian motion | 1 | diffuse |
| `LocalLinearTrend` | cubic spline, `min(s,t)³/3 + min(s,t)²\|s−t\|/2` | 2 | diffuse |
| `TrigonometricSeasonal` | `min(s,t) Σⱼ cos λⱼ(s−t)` | 2 per harmonic | diffuse |
| `RegressionComponent` | a constant per column | 1 per column | diffuse |
| `Matern` | Matérn, ν = 1/2, 3/2, 5/2 | 1, 2, 3 | stationary |
| `StochasticCycle` | `ρ^\|τ\| cos(2πτ/p)`, period estimated | 2 | stationary |

You implement `Component` rather than passing a covariance function, because
linear time comes from the Markov property and only kernels with a
finite-dimensional state-space form have it. So the useful question is not "can I
supply a kernel" but **is my kernel reachable** — and the table is the answer.

For the default `LocalLinearTrend` the implied kernel is the cubic spline kernel,
so the posterior mean is a natural cubic smoothing spline: as a Bayesian
posterior, with honest uncertainty, computed in linear time.

## Documentation

| | |
|---|---|
| [Getting started](https://github.com/QuantumPhysique/state_space/blob/main/doc/getting-started.md) | grids, slopes, forecasts, unequal readings, noise floors, isolates |
| [Choosing a model](https://github.com/QuantumPhysique/state_space/blob/main/doc/choosing-a-model.md) | which components, and how to tell whether one earned its place |
| [Components](https://github.com/QuantumPhysique/state_space/blob/main/doc/components.md) | what each one is, when to reach for it, and which kernels are reachable |
| [How it works](https://github.com/QuantumPhysique/state_space/blob/main/doc/how-it-works.md) | the Gaussian process, the SDE, the filter, and the numerical choices |
| [Validation](https://github.com/QuantumPhysique/state_space/blob/main/doc/validation.md) | what is checked, against what, and how closely — plus benchmarks |
| [Roadmap](https://github.com/QuantumPhysique/state_space/blob/main/doc/roadmap.md) | what shipped when, what might come next, and the references |
| [Calibration](https://github.com/QuantumPhysique/state_space/blob/main/tool/calibration/README.md) | does the right answer actually help, measured on weight diaries |

## Performance

Around 210–230 ns per observation for a full `smooth`, flat across three orders
of magnitude, on an M-series Mac. Ten years of daily readings smooth in about a
millisecond. The tables are in [Validation](https://github.com/QuantumPhysique/state_space/blob/main/doc/validation.md#performance).

## What it does not do

It knows nothing about your domain: no dates, no units, no calendars, no locales.
Time is a `double` in whatever unit you like, and the process variances are
expressed per that unit. There is no configuration object and no strategy enum.

Nor is there a matrix type in anything a caller receives — every result is
`Float64List`s and doubles, which is what lets a whole model and its posterior
cross an isolate boundary unchanged.

## Used by

[trale](https://github.com/QuantumPhysique/trale), a privacy-respecting body
weight diary, for its trend curve and uncertainty band.

## Licence

[MIT](LICENSE).

Deliberately permissive, and deliberately different from the application it was
written for. trale is AGPL, which is a reasonable position for something people
run; a library is not, because that licence would be inherited by everything
built on top of it. An AGPL application can use an MIT library freely, so nothing
is lost in the direction that matters here.
