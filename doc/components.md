# Components

A model is a sum of components:

```text
y(t) = sum_i H_i(t) x_i(t) + eps(t),   eps ~ N(0, measurementVariance)
```

Each is a Gaussian process kernel in state-space form and owns a few
consecutive states. The blocks are stacked block-diagonally, so a component only
adds its own cost.

| component | kernel | states | prior | parameters |
|---|---|---|---|---|
| `LocalLevel` | Brownian motion, `min(s,t)` | 1 | diffuse | 1 |
| `LocalLinearTrend` | cubic spline, `min(s,t)³/3 + min(s,t)²\|s−t\|/2` | 2 | diffuse | 1 |
| `TrigonometricSeasonal` | `min(s,t) Σⱼ cos λⱼ(s−t)` | 2 per harmonic | diffuse | 1 |
| `RegressionComponent` | a constant per column | 1 per column | diffuse | **0** |
| `Matern` | Matérn, ν = 1/2, 3/2, 5/2 | 1, 2, 3 | stationary | 2 |
| `StochasticCycle` | `ρ^\|τ\| cos(2πτ/p)`, period estimated | 2 | stationary | 3 |

The last column is how many search dimensions each adds to `fit`. The
measurement variance is concentrated out and adds none, so a trend plus twenty
holiday indicators is a one-dimensional search.

## Supported kernels

Only kernels with a finite-dimensional state-space form are supported, since
the linear cost relies on the Markov property; the table above is the list. A
new one is added by writing a `Component`, not by passing `k(s, t)`, which
would cost `O(N³)`.

The squared exponential is not supported: it has no exact finite-state form.

## The non-stationary ones

These have no proper prior: the level, and the phase of a pattern, come from
the data alone. The engine treats them as flat directions; see
[exact diffuse initialisation](https://github.com/QuantumPhysique/state_space/blob/main/doc/how-it-works.md#exact-diffuse-initialisation).

### `LocalLevel`

A level following a Wiener process. One state, `A(dt) = 1`, `Q(dt) = σ² dt`. The
posterior mean is a linear interpolant through shrunken observations, the
continuous-time counterpart of simple exponential smoothing.

Use it when the series has no persistent direction. It will not extrapolate.

### `LocalLinearTrend`

The default. The state is `(level, slope)`; the slope is driven by white noise
and the level is its integral:

```text
d(mu) = nu dt,   d(nu) = sigma dB
```

The implied kernel is the cubic spline kernel, so **the posterior mean is a
natural cubic smoothing spline** (Wahba 1978), here with a credible band and in
linear time.
[How it works](https://github.com/QuantumPhysique/state_space/blob/main/doc/how-it-works.md#as-a-kalman-filter-and-rts-smoother)
has the kernel and the smoothing parameter.

Unlike `LocalLevel` it carries a slope, and extrapolates it.

### `TrigonometricSeasonal`

A repeating pattern of known period whose shape can drift. Each harmonic is a
pair of states that rotate into one another over time, and the observation
reads their sum.

```dart
TrigonometricSeasonal(period: 7, harmonics: 2, processVariance: 1e-3)
```

Two or three harmonics resolve a weekly shape on daily readings. At and past
the Nyquist frequency, `2 · harmonics · gap ≥ period`, a harmonic aliases onto
a lower one or leaves a state the data can never see, so `fit` and `smooth`
refuse it with an `UnderdeterminedModelException`. The check uses the typical
gap between readings, so the period can be in any time unit: `period: 1` with
time in years and monthly readings is fine.

`processVariance` is the rate at which the pattern may change shape, not its
amplitude. All harmonics share it (Harvey's specification), so the component
has one parameter.

Two properties worth knowing:

* **Averaged over a full period each harmonic integrates to zero**, so the
  component has no level of its own and does not compete with a trend for one.
  Over a stretch shorter than a period the two are confounded.
* **Shrinking the variance to zero does not remove the component.** It only
  stops the pattern changing, leaving a rigid Fourier series whose starting
  coefficients have a flat prior. Over less than one period a rigid sinusoid is
  very nearly a constant plus a slope, so an annual component on 180 days of
  data draws a cycle of 1.14 peak to trough in a series that has none, while
  reporting its variance at the floor. The component's own posterior standard
  deviation shows the problem: it is 1.27 there, larger than the pattern it drew
  and eighteen times the 0.07 of the total signal. With a year of data it is
  0.065. See [Choosing a model](https://github.com/QuantumPhysique/state_space/blob/main/doc/choosing-a-model.md#what-competes-with-what).

Dummy-variable seasonality is not available, since it has no sensible `A(dt)`
for a non-integer gap.

### `RegressionComponent`

Coefficients on known functions of time, such as a fortnight over Christmas, a
conference, a course of medication or a dose:

```dart
RegressionComponent([
  IndicatorRegressor('christmas', [(from: 350, to: 364)]),
  StepRegressor('dose', knots, values),
])
```

`IndicatorRegressor` is one while something is happening and zero otherwise;
`StepRegressor` holds a value between known instants. A regressor is data rather
than a closure, so it can be sent to an isolate.

Each coefficient is one state with `A = I` and `Q = 0` under a flat prior, so
**`parameterCount` is zero**: a coefficient is a flat direction, which exact
diffuse initialisation integrates out anyway. Coefficients add no search
dimension and come with posterior standard errors from the same pass as the
trend:

```dart
posterior.coefficients.first;   // christmas: 1.121 +/- 0.091
```

Because such a state never moves, the backward pass skips it; see
[`Component.isStatic`](https://github.com/QuantumPhysique/state_space/blob/main/doc/how-it-works.md#the-smoother).
The forward pass still carries every column as a flat direction, and its cost
grows roughly with the square of the number of columns: on two years of daily readings
a likelihood evaluation of a trend and a weekly seasonal takes 0.4 ms, with one
indicator 0.5 ms, and with twenty 20 ms, so a fit with twenty indicators costs
about fifty times one without.

## The stationary ones

These have a proper prior: they stay around zero and forget their past. That
suits a *deviation* from a trend rather than a level, and they add no flat
directions.

### `Matern`

The Matérn kernel at ν = 1/2, 3/2 and 5/2, exactly, with one, two and three
states.

```dart
final model = StructuralModel([
  LocalLinearTrend(processVariance: 1e-4),
  Matern.oneHalf(variance: 0.1, lengthScale: 3),
]);
fit(model, data, minimumMeasurementVariance: 0.029 * 0.029);
```

Use it for short-lived correlated deviations that the trend should not follow.
`ν = 1/2` is an Ornstein–Uhlenbeck process, the continuous-time AR(1). It takes
up correlated day-to-day variation that a trend-only model would count as
noise.

The order sets how rough the path is: `LocalLinearTrend` has a continuous
derivative, and `ν = 1/2` allows corners.

Give `fit` a `minimumMeasurementVariance` whenever a Matérn is in the model.
With correlated day-to-day variation in the data, the likelihood can prefer a
large Matérn and a noise level near zero. `fit` searches for the alternative
and `warnings` reports it, but a floor at the instrument's resolution avoids
the problem.

### `StochasticCycle`

An approximate rhythm: a damped cosine, rather than a pattern that repeats
exactly. A seasonal of period 7 repeats every seven days forever; a cycle of
period 7 tends to come round after about a week, drifts out of step, and comes
back.

Its period is estimated, and the likelihood is multimodal in it: a cycle at
half the period explains every second peak and has its own maximum. `fit` scans
that axis much more finely than the others before the local search.

It needs a lot of data. Below about four complete cycles the period cannot be
estimated, and below eight it is very noisy, although the fit still returns a
value.

**Check the damping before the period.** On a series with no cycle the damping
goes to the top of its bracket, and the width reported for the period becomes
tiny and meaningless. `warnings` reports both, and the `StochasticCycle` API
documentation has the details.

## A floor from the sampling

Below the sampling interval, a shape parameter measured in time units no longer
describes a different model. A Matérn with `ν = 1/2` and a length scale shorter
than the gap between readings is effectively white noise, so it competes with
the measurement error instead of the trend, and the likelihood slightly prefers
it. Allowed down to a length scale of 0.01 on daily readings with a true noise
level of 0.3, it reports the noise as **0.002**, with a band covering every
point and a trend that interpolates the noise.

So `fit` raises the bottom of `lengthScaleBounds` to the typical gap between
visits, and a cycle's `periodBounds` to twice it, which is the Nyquist limit.
Readings much closer together than the typical gap, such as two weighings on one
morning, count as one visit. A higher lower bound of your own is kept, and the
upper bound is used as given.

## Writing your own

Import `package:state_space/authoring.dart` and extend `Component`. It is a
`base` class, so the subclass must be declared `final`, `base` or `sealed`:

```dart
final class MyComponent extends Component {
  const MyComponent();

  @override
  String get name => 'MyComponent';

  // stateDim, parameterCount, transition, processNoise, observationAt,
  // diffuseStates, properPrior, parameters, withParameters
}
```

Nine members have no default: `stateDim`, `parameterCount`, `transition` and
`processNoise` (`A(dt)` and `Q(dt)`, exact for *any* non-negative gap including
zero), `observationAt`, `diffuseStates`, `properPrior` for the states that are
not diffuse, `parameters` and `withParameters`. Override `name` with a literal,
and `parameterSpecs` if any parameter is not a variance.

`MatrixBlock` is the matrix type components write into: a view onto the
engine's buffer, so a component fills its own block in place without knowing
where that block sits.

`checkComponent` tests what the engine relies on and the type system cannot: the
declared lengths, `A(0) = I` and `Q(0) = 0`, a symmetric positive semi-definite
`Q`, consistency across gaps (`Q(s + t) = A(t) Q(s) A(t)' + Q(t)`), the
`withParameters` round trip, and an honest `isStatic`. It returns one sentence
per problem:

```dart
test('my component', () => expect(checkComponent(MyComponent()), isEmpty));
```
