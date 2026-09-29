# Getting started

The examples use a body-weight diary with skipped days, two readings on one
morning and a fortnight with no readings. Nothing in the package is specific to
weight.

## Install

```yaml
dependencies:
  state_space: ^0.1.0
```

No runtime dependencies, no platform channels, no native code.

## A first fit

```dart
import 'package:state_space/state_space.dart';

final data = [
  Observation(0, 81.3),
  Observation(2, 81.0),
  Observation(3, 80.5),
  Observation(7, 80.0),
  Observation(8, 80.0),
  Observation(8, 79.8),
  Observation(10, 79.7),
  Observation(11, 79.7),
  Observation(12, 79.5),
  Observation(17, 79.3),
  Observation(18, 79.3),
  Observation(19, 79.2),
  Observation(20, 79.2),
  Observation(21, 79.4),
  Observation(22, 79.2),
  Observation(24, 79.3),
  Observation(27, 79.1),
];

final fitted = fit(StructuralModel.localLinearTrend(processVariance: 1), data);
print(fitted.warnings);         // []
final trend = fitted.model.smooth(data);

trend.mean[0];                  // the smoothed curve at the first reading
trend.credibleInterval(0);      // (lo: ..., hi: ...) around it
```

`fit` estimates how flexible the curve should be and `smooth` computes the
posterior. `fit` ignores the `processVariance` you pass: it scans a fixed
bracket, thirteen decades wide, in the ratio of the process variance to the
noise (about `1e-8.7` to `1e4.3`). `lowerLogRatio` and `upperLogRatio` move it.
The default suits time in days.

Check `fitted.warnings` before using a fit. Four readings, for instance, do
not determine a smoothing level, and the fit says so;
[Choosing a model](https://github.com/QuantumPhysique/state_space/blob/main/doc/choosing-a-model.md#read-the-warnings)
explains each warning.

## What comes back

`smooth` returns a [`SmoothingResult`][r]: parallel read-only arrays, one entry
per output time, in the order you asked for them.

| | |
|---|---|
| `times` | the output times |
| `mean` | the posterior mean of the signal, all components together |
| `variance` | its variance: uncertainty about the *signal*, not about the next reading |
| `trendSlope`, `trendSlopeVariance` | the trend component's rate of change, or null if no component has one |
| `componentMean(i)`, `componentVariance(i)` | what component `i` contributed |
| `componentSlope(i)`, `componentSlopeVariance(i)` | its rate of change, if it has one |
| `coefficients` | regression coefficients, with standard errors |
| `logMarginalLikelihood` | see [Choosing a model](https://github.com/QuantumPhysique/state_space/blob/main/doc/choosing-a-model.md) before comparing these |

`trendSlope` is the trend's own rate. With a weekly seasonal in the model it is
not the derivative of `mean`, which includes the weekly wiggle.

There are two kinds of interval:

```dart
trend.credibleInterval(i);    // where the underlying trend is
trend.predictiveInterval(i);  // where the next reading would fall
trend.credibleBand();         // the first at every output time, as lo and hi arrays
```

Draw the credible band around the curve. It is narrow, and most individual
readings fall outside it, because it describes the trend rather than the next
reading. About 95 % of the observations should sit inside the 95 % predictive
band.

## Dates

Time is a `double`. `TimeAxis` converts calendar dates to days since an origin
and back, counting every calendar day as one, including the 23- and 25-hour
days at a change of clocks:

```dart
final axis = TimeAxis.days(readings.first.date);
final data = [for (final r in readings) axis.observation(r.date, r.kg)];
// ... smooth, then map each output time back:
final dates = [for (final t in trend.times) axis.dateAt(t)];
```

Subtracting two `DateTime`s and dividing by 24 hours is off by an hour after a
change of clocks, so readings taken at 07:00 every day no longer land on whole
days.

## Irregular data

```dart
final data = [
  Observation(0, 81.2),
  Observation(1, 80.9),
  Observation(9, 80.4),   // an eight-day gap
  Observation(9, 80.6),   // and two readings at the same instant
];
```

There is no interpolation or resampling, and duplicates do not need averaging.
A gap is a larger `dt` in the recursion, a repeated timestamp is `dt = 0`, and a
day with no reading is a step without an update.

**Observations must be sorted by time.** Results come back in the order you
supplied.

## Asking for output between the readings

Pass a grid. Grid points are steps with no observation attached, so each one
costs one more step of the recursion.

```dart
final everyDay = [for (var d = 0; d <= 30; d++) d.toDouble()];
final daily = fitted.model.smooth(data, grid: everyDay);
```

A grid may extend past the data at either end; the band widens there.
Where a grid time coincides exactly with an observation, the reported state
includes that observation.

## Slopes and forecasts

The trend carries its own rate of change:

```dart
daily.trendSlope![10];          // signal units per time unit
daily.trendSlopeVariance![10];  // and how sure it is
```

A forecast is the same recursion with no more observations:

```dart
final horizon = [for (var d = 31; d <= 60; d++) d.toDouble()];
final ahead = fitted.model.forecast(data, horizon);

ahead.credibleInterval(0);    // where the signal is going
ahead.predictiveInterval(0);  // where an actual reading would fall
```

The band widens quickly. For a local linear trend the forecast variance grows
like the cube of the horizon, because the slope itself is uncertain and keeps
drifting. This is expected behaviour.

If the first horizon time equals the last observation time, the forecast there
is the filtered state: conditioned on everything up to and including that
reading. Use it for a value that should not change once it has been shown.

## Readings you trust differently

`relativeVariance` scales one observation's noise against the model's noise
level, so an average of two weighings and a single quick one can be in the
same series:

```dart
Observation(4, 80.4);                          // an ordinary reading
Observation(5, 80.5, relativeVariance: 0.5);   // trusted twice as much
Observation(6, 79.1, relativeVariance: 4.0);   // trusted half as much
```

## Bad readings

The fit is Gaussian, so one mistyped value (801 for 80.1, or 8.01) inflates
the noise estimate and moves the curve for months around it. `fit` measures
how far each reading is from what the rest of the data predicts, and names the
worst one in `warnings` when it is more than six typical errors away:

```dart
var fitted = fit(model, data);
final worst = fitted.largestResidual;
if (worst != null && worst.score.abs() > FitResult.outlierScore) {
  final screened = [
    for (final o in data)
      o.time == worst.time
          ? Observation(o.time, o.value, relativeVariance: 1e6)
          : o
  ];
  fitted = fit(model, screened);
}
```

A very large `relativeVariance` effectively ignores the reading but keeps it in
the list, so the output times do not change. Removing it works too.

## Noise floors

By default the noise level is whatever explains the data best. On a run of
nearly identical readings that can be smaller than any real scale could
achieve, with a band that is far too narrow.

```dart
fit(model, data, minimumMeasurementVariance: 0.029 * 0.029);  // rounds to 100 g
fit(model, data, fixedMeasurementVariance: 0.2 * 0.2);        // known noise
```

`minimumMeasurementVariance` is a floor. The fit runs normally and is only
redone with the noise pinned if the free estimate is below the floor, so it
costs nothing otherwise. A display that rounds to 100 g adds a rounding error
with standard deviation `0.1 / sqrt(12)`, about 0.029 kg. Use a floor whenever
a `Matern` is in the model.

`fixedMeasurementVariance` pins the noise level for when it is known, and
estimates the smoothing given it.

## Choosing the smoothing yourself

If the stiffness of the curve is a user setting rather than an estimate, skip
`fit`. Build the model at the ratio you want and let the data set the scale:

```dart
// A stiffness from a user setting: process variance = ratio * noise variance.
final chosen = StructuralModel.localLinearTrend(processVariance: ratio)
    .withEstimatedScale(data);
final trend = chosen.smooth(data, grid: everyDay);
```

`withEstimatedScale` keeps every variance ratio and sets the noise level to its
restricted maximum likelihood estimate, in one forward pass. The curve depends
only on the ratio; the noise level only changes the band. Pass
`minimumMeasurementVariance` to put a floor under the estimate.

## When there is not enough data

A trend has two flat directions, a level and a slope, so it needs readings at
two distinct times. Each harmonic of a seasonal adds two more, and each
regression column one. `fit` needs one reading more than that to
estimate a noise level from. With fewer, the package throws an
`UnderdeterminedModelException` whose message says what is missing:

```dart
try {
  final fitted = fit(model, data);
} on UnderdeterminedModelException catch (e) {
  // Too little data, or two components the data cannot tell apart.
}
```

All the package's data-dependent failures are `StateSpaceException`s:
`UnderdeterminedModelException`, and `NumericalBreakdownException` for a
component whose process noise is not a covariance. Invalid arguments throw
`ArgumentError`.

`ApproximateDiffuse` returns a very wide band where exact initialisation
throws. It is sound only with a time unit that keeps rates of change
near order one, such as days.

A band computed from a handful of readings treats the estimated noise level as
known, so in the first week or two of a diary it is too narrow. A noise floor
at a realistic spread, rather than at the instrument's resolution, helps.

## Refitting as data arrives

A diary that gains one reading a day rarely moves its optimum far, so there is
no need to repeat the full scan:

```dart
var fitted = fit(model, data);                          // cold, once
fitted = fit(fitted.model, longerData,                  // warm, after
    start: SearchStart.previousParameters);
```

On two years of daily readings with a trend, a weekly seasonal and a Matérn,
adding one observation took 573 filter passes cold and 232 warm, with the same
likelihood to 1e-4. A warm start cannot jump to a different basin, so run a
cold fit when the data changes character rather than just growing.

## Isolates

A model and its posterior are plain data, so they can be sent to another
isolate:

```dart
final posterior = await Isolate.run(() => model.smooth(data));
```

In Flutter, `compute` does the same. `test/isolate_test.dart` sends a model
across, uses it there, and brings back a posterior, a forecast and a set of
diagnostics.

## Where to go next

* [Choosing a model](https://github.com/QuantumPhysique/state_space/blob/main/doc/choosing-a-model.md): which components, and how to
  compare them
* [Components](https://github.com/QuantumPhysique/state_space/blob/main/doc/components.md): what each one is, and which kernels are reachable
* [How it works](https://github.com/QuantumPhysique/state_space/blob/main/doc/how-it-works.md): the Gaussian process, the SDE and the filter
* [Validation](https://github.com/QuantumPhysique/state_space/blob/main/doc/validation.md): what is checked, against what, and how closely

[r]: https://pub.dev/documentation/state_space/latest/state_space/SmoothingResult-class.html
