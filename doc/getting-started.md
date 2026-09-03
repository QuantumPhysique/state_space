# Getting started

Everything here uses one running example: a body-weight diary, because that is
what the package was written for and because it has every awkward property real
data has — readings that skip days, two readings on one morning, a fortnight
away with none at all.

Nothing about the package is specific to it. Time is a `double` in whatever unit
you find natural and values are whatever you are measuring.

## Install

```yaml
dependencies:
  state_space: ^0.5.0
```

No runtime dependencies, no platform channels, no native code.

Grids and horizons are `Float64List`s, so anything that builds one also needs
`import 'dart:typed_data';` — the library does not re-export it.

## The shortest thing that works

```dart
import 'package:state_space/state_space.dart';

final data = [
  Observation(0, 81.2),
  Observation(1, 80.9),
  Observation(4, 80.4),
  Observation(7, 80.6),
];

final fitted = fit(StructuralModel.localLinearTrend(processVariance: 1), data);
final trend = fitted.model.smooth(data);

trend.level[0];                 // the smoothed curve at the first reading
trend.credibleInterval(0);      // (lo: ..., hi: ...) around it
```

Two calls. `fit` estimates how flexible the curve should be; `smooth` computes
the posterior. The `processVariance: 1` you pass in is a starting point and is
irrelevant to the answer — `fit` searches thirteen decades either side of it.

## What comes back

`smooth` returns a [`SmoothingResult`][r]: parallel arrays, one entry per output
time, in the order you asked for them.

| | |
|---|---|
| `times` | the output times |
| `level` | the posterior mean of the signal |
| `levelVariance` | its variance — uncertainty about the *signal*, not about the next reading |
| `slope`, `slopeVariance` | the rate of change, when the model has one component that has a rate |
| `componentMean(i)`, `componentVariance(i)` | what each component contributed |
| `coefficients` | regression coefficients, with standard errors |
| `logMarginalLikelihood` | see [Choosing a model](https://github.com/QuantumPhysique/state_space/blob/main/doc/choosing-a-model.md) before comparing these |

Two interval helpers, and the difference between them matters:

```dart
trend.credibleInterval(i);    // where the underlying trend is
trend.predictiveInterval(i);  // where the next reading would fall
```

The first is the band to draw around a curve. It is narrow, and most individual
measurements fall outside it — that is not a defect, it is the difference
between "where is the trend" and "where would the next reading land". About 95 %
of the observations should sit inside the 95 % version of the second.

Everything is `Float64List`s and doubles. There is no matrix type in anything a
caller receives, which is also what lets a model and its posterior cross an
isolate boundary unchanged.

## Irregular data is not a special case

```dart
final data = [
  Observation(0, 81.2),
  Observation(1, 80.9),
  Observation(9, 80.4),   // an eight-day gap
  Observation(9, 80.6),   // and two readings at the same instant
];
```

Nothing is interpolated, nothing is resampled, no gap is filled, and you do not
have to average the duplicates. A gap is a larger `dt` in the recursion, a
repeated timestamp is `dt = 0`, and a day with no reading is a step with no
update. All three fall out of the same code path rather than being handled.

The one rule: **observations must be sorted by time**. The package will not sort
them for you, because results come back in the order you supplied.

## Asking for output between the readings

Pass a grid. Grid points are steps with no observation attached, so a value
between two readings costs one more step in the same linear recursion — there is
still no interpolation anywhere.

```dart
final everyDay = Float64List.fromList([for (var d = 0; d <= 30; d++) d + 0.0]);
final daily = fitted.model.smooth(data, grid: everyDay);
```

A grid may extend past the data at either end, and the band widens accordingly.
Where a grid time coincides exactly with an observation, the reported state
includes that observation.

## The slope, and where it is heading

The trend carries its own rate of change, so you get it without differencing
anything:

```dart
daily.slope![10];          // signal units per time unit
daily.slopeVariance![10];  // and how sure it is
```

Forecasting is the same recursion with no observations left to update on:

```dart
final horizon = Float64List.fromList([for (var d = 31; d <= 60; d++) d + 0.0]);
final ahead = fitted.model.forecast(data, horizon);

ahead.credibleInterval(0);    // where the signal is going
ahead.predictiveInterval(0);  // where an actual reading would fall
```

The band widens quickly, and for a local linear trend the variance of the level
grows like the cube of the horizon. That is a statement about the model rather
than a defect of it: a trend whose slope is free to wander really does become
unknowable, and a band that stayed narrow would be lying.

A horizon whose first entry *is* the last observation time reports the filtered
state there — conditioned on everything up to and including that reading and
nothing after it. That is the number to show when it must not move once shown.

## Readings you trust differently

`relativeVariance` weights one observation against the model's own noise level,
so an average of two weighings and a hurried single one can sit in the same
series:

```dart
Observation(4, 80.4);                          // an ordinary reading
Observation(5, 80.5, relativeVariance: 0.5);   // trusted twice as much
Observation(6, 79.1, relativeVariance: 4.0);   // trusted half as much
```

It is relative rather than absolute so that one number still sets the scale of
the noise and these weights only say how the readings differ from one another.

## Telling it what the instrument can do

Left alone, the noise level is whatever explains the data best. On a run of
nearly identical readings that can be a number no real scale could deliver, and
a band far too narrow to believe.

```dart
fit(model, data, minimumMeasurementVariance: 0.05 * 0.05);  // reads to 100 g
fit(model, data, fixedMeasurementVariance: 0.05 * 0.05);    // and no arguing
```

`minimumMeasurementVariance` is a floor and is the one to reach for. The fit runs
normally and is redone with the noise pinned only if the free estimate lands
below the floor, so it costs nothing when the data agrees. `fixedMeasurementVariance`
pins the level outright, which is what you want when the smoothing is being
chosen rather than estimated and the noise level still has to come from
somewhere.

A floor cannot be a clamp applied afterwards. Pinning one variance in absolute
units breaks the scale equivariance that lets everything else be searched as a
ratio, so the other variances have to be found again against it — which is what
the second fit does.

## Refitting as data arrives

A diary that gains a reading a day does not move its optimum, and rediscovering
the same basin costs hundreds of filter passes.

```dart
var fitted = fit(model, data);                          // cold, once
fitted = fit(fitted.model, longerData,                  // warm, after
    start: SearchStart.previousParameters);
```

On two years of daily readings with a trend, a weekly seasonal and a Matérn,
adding one observation: 317 filter passes cold, 117 warm, same answer. Use it
only when the surface is one a local search can be trusted on, and run a cold
fit whenever the data changes character rather than merely grows.

## Off the interface thread

A whole model and its posterior are plain data, so they cross an isolate
boundary unchanged:

```dart
final posterior = await compute(_smoothOffThread, (model, data));
```

`test/isolate_test.dart` sends a model across, uses it there, and brings back a
posterior, a forecast and a set of diagnostics.

## Where to go next

* [Choosing a model](https://github.com/QuantumPhysique/state_space/blob/main/doc/choosing-a-model.md) — which components, and how to tell
  whether one earned its place
* [Components](https://github.com/QuantumPhysique/state_space/blob/main/doc/components.md) — what each one is, and which kernels are reachable
* [How it works](https://github.com/QuantumPhysique/state_space/blob/main/doc/how-it-works.md) — the Gaussian process, the SDE and the filter
* [Validation](https://github.com/QuantumPhysique/state_space/blob/main/doc/validation.md) — what is checked, against what, and how closely

[r]: https://pub.dev/documentation/state_space/latest/state_space/SmoothingResult-class.html
