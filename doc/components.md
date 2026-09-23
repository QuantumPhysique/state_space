# Components

A model is a sum of components:

```text
y(t) = sum_i H_i(t) x_i(t) + eps(t),   eps ~ N(0, measurementVariance)
```

Each one is a Gaussian process kernel written in the form that makes it fast, and
each owns a few consecutive states. They compose block-diagonally, so adding a
component costs what that component costs and nothing more.

| component | kernel | states | prior | parameters |
|---|---|---|---|---|
| `LocalLevel` | Brownian motion, `min(s,t)` | 1 | diffuse | 1 |
| `LocalLinearTrend` | cubic spline, `min(s,t)³/3 + min(s,t)²\|s−t\|/2` | 2 | diffuse | 1 |
| `TrigonometricSeasonal` | `min(s,t) Σⱼ cos λⱼ(s−t)` | 2 per harmonic | diffuse | 1 |
| `RegressionComponent` | a constant per column | 1 per column | diffuse | **0** |
| `Matern` | Matérn, ν = 1/2, 3/2, 5/2 | 1, 2, 3 | stationary | 2 |
| `StochasticCycle` | `ρ^\|τ\| cos(2πτ/p)`, period estimated | 2 | stationary | 3 |

The last column is the number of search dimensions each adds to `fit`, which
concentrates the measurement variance out on top of that. A trend plus twenty
holiday indicators is a one-dimensional search.

## Supported kernels

Only kernels with a finite-dimensional state-space form are supported, because
linear time comes from the Markov property; the table above is the list. A new
one is added by writing a `Component`, not by passing `k(s, t)`, which would
cost `O(N³)`.

The squared exponential is not supported: it has no exact finite-state form.

## The non-stationary ones

These have no prior of their own. Where the level sits, and where a pattern sits
in its cycle, are questions only the data can answer, and the engine handles them
as flat directions — see [exact diffuse initialisation](https://github.com/QuantumPhysique/state_space/blob/main/doc/how-it-works.md#exact-diffuse-initialisation).

### `LocalLevel`

A level following a Wiener process. One state, `A(dt) = 1`, `Q(dt) = σ² dt`. The
posterior mean is a linear interpolant through shrunken observations — the
continuous-time counterpart of simple exponential smoothing.

Use it when the series has no persistent direction. It will not extrapolate.

### `LocalLinearTrend`

The default, and the one most series want. The state is `(level, slope)`, the
slope is driven by white noise and the level is its integral:

```text
d(mu) = nu dt,   d(nu) = sigma dB
```

The implied kernel is the cubic spline kernel, so **the posterior mean is a
natural cubic smoothing spline**, computed as the exact posterior of a Gaussian
process with a credible band, in linear time (Wahba 1978).
[How it works](https://github.com/QuantumPhysique/state_space/blob/main/doc/how-it-works.md#as-a-kalman-filter-and-rts-smoother)
has the kernel and the smoothing parameter.

Choosing this over `LocalLevel` is a claim that "still going down" is a
meaningful sentence about your data. It carries a slope and will extrapolate it.

### `TrigonometricSeasonal`

A repeating pattern of known period whose shape drifts. Each harmonic is a pair
of states that rotate into one another as time passes; the observation reads the
sum.

```dart
TrigonometricSeasonal(period: 7, harmonics: 2, processVariance: 1e-3)
```

Two or three harmonics resolve a weekly shape on daily readings. At and past
the Nyquist frequency, `2 · harmonics · gap ≥ period`, a harmonic aliases onto
a lower one or leaves a state the data can never see, so `fit` and `smooth`
refuse it with an `UnderdeterminedModelException`. The check uses the typical
gap between readings, so the period can be in any time unit: `period: 1` with
time in years and monthly readings is fine.

`processVariance` is the rate at which the pattern is allowed to change shape,
not its amplitude. All harmonics share it, which is Harvey's specification and
keeps the component to one parameter.

Two things about it that surprise people:

* **Averaged over a full period each harmonic integrates to zero**, so the
  component carries no level of its own and does not compete with a trend for
  one. Over a stretch shorter than a period they *are* confounded, but that is a
  statement about the data.
* **Shrinking the variance to zero does not remove the component.** It only
  stops the pattern evolving; what is left is a rigid Fourier series whose
  starting coefficients have a flat prior that nothing shrinks. Over less than
  one period a rigid sinusoid is very nearly a constant plus a slope, so an
  annual component on 180 days of data draws a cycle of 1.14 peak to trough out
  of a series that has none, while reporting its variance at the floor. It does
  say so, in the place worth looking: that component's own posterior standard
  deviation there is 1.27 — larger than the pattern it drew, and eighteen times
  the 0.07 the total signal is known to. By a year it is 0.065. See
  [Choosing a model](https://github.com/QuantumPhysique/state_space/blob/main/doc/choosing-a-model.md#what-competes-with-what).

The usual alternative, dummy-variable seasonality, has no sensible `A(dt)` for a
non-integer gap, which rules it out here.

### `RegressionComponent`

Coefficients on known columns of time — a fortnight over Christmas, a
conference, a course of medication, a dose:

```dart
RegressionComponent([
  IndicatorRegressor('christmas', [(from: 350, to: 364)]),
  StepRegressor('dose', knots, values),
])
```

`IndicatorRegressor` is one while something is happening and zero otherwise;
`StepRegressor` holds a value between known instants. A regressor is data rather
than a closure, so that it survives an isolate boundary whatever it was built
from.

Each coefficient is one state with `A = I` and `Q = 0` under a flat prior, so
**`parameterCount` is zero**: the exact diffuse machinery already integrates out
flat directions, and a coefficient is one. They add no search dimension and
arrive with posterior standard errors from the same recursion that produced the
trend:

```dart
posterior.coefficients.first;   // christmas: 1.121 +/- 0.091
```

Because such a state never moves, the backward pass skips it; see
[`Component.isStatic`](https://github.com/QuantumPhysique/state_space/blob/main/doc/how-it-works.md#the-smoother).
The forward pass still carries every column as a flat direction, and its cost
grows roughly with the square of their number: on two years of daily readings
a likelihood evaluation of a trend and a weekly seasonal takes 0.4 ms, with one
indicator 0.5 ms, and with twenty 20 ms, so a fit with twenty indicators costs
about fifty times one without.

## The stationary ones

These have a prior of their own: they hover around zero and forget where they
have been. That makes them the right shape for a component that is a *deviation*
rather than a level, and it means they contribute no flat directions at all.

### `Matern`

The standard Gaussian process kernel, in state-space form, at ν = 1/2, 3/2 and
5/2 — one, two and three states, exactly rather than approximately.

```dart
final model = StructuralModel([
  LocalLinearTrend(processVariance: 1e-4),
  Matern.oneHalf(variance: 0.1, lengthScale: 3),
]);
fit(model, data, minimumMeasurementVariance: 0.029 * 0.029);
```

Reach for it when a series has structure the trend should not be chasing. `ν = 1/2`
is an Ornstein–Uhlenbeck process, the exact continuous-time AR(1), and it absorbs
the correlated wobble a trend-only model has nowhere to put but the noise.

The order also decides how rough the path may be: the cubic spline of
`LocalLinearTrend` assumes a trend with a continuous derivative, and `ν = 1/2`
allows corners.

Give `fit` a `minimumMeasurementVariance` whenever a Matérn is in the model.
With correlated day-to-day variation in the data, the likelihood can prefer a
large Matérn and a noise level near zero; `fit` searches for the alternative
and `warnings` reports it, but a floor at the instrument's resolution rules the
corner out.

### `StochasticCycle`

The *approximate* rhythm: a cosine that fades and finds its way back, rather than
a pattern that repeats forever. A seasonal of period 7 insists that Tuesdays are
seven days apart forever; a cycle of period 7 says the series tends to come back
round after about a week, drifts out of step, and returns.

Its period is estimated rather than given, which makes it the one component here
whose likelihood is multimodal — a cycle at half the period explains every second
peak and sits on its own maximum. `fit` scans that axis far more finely than the
others before anything local runs.

It needs a lot of data. Below about four complete cycles the period is not
estimable at all and below eight it is very noisy, and the fit will still return
a number.

**Read the damping before the period.** On a series with no cycle the damping
goes to the top of its bracket, and the width reported for the period becomes
tiny and meaningless. The `StochasticCycle` API documentation has the details,
and `warnings` reports both.

## Both stationary components have a floor you did not set

A shape parameter measured in time units stops being a different model below the
sampling interval. A Matérn with `ν = 1/2` and a length scale shorter than the gap
between readings *is* white noise, so it competes with the measurement error
rather than with the trend, and the likelihood mildly prefers it that way.
Allowed down to a length scale of 0.01 on daily readings with a true noise level
of 0.3, it reports the noise as **0.002**, draws a band covering every point,
and hands back a trend interpolating the noise.

So `fit` raises the bottom of `lengthScaleBounds` to the typical gap between
visits, and a cycle's `periodBounds` to twice it, which is the Nyquist limit.
Readings much closer together than the typical gap, such as two weighings on one
morning, count as one visit. Your own lower bound is respected when it is
higher, and the upper bound is honoured as given.

## Writing your own

Import `package:state_space/authoring.dart` and extend `Component`. Nine members
have no default: `stateDim`, `parameterCount`, `transition` and `processNoise`
(`A(dt)` and `Q(dt)`, exact for *any* non-negative gap including zero),
`observationAt`, `diffuseStates`, `properPrior` for the states that are not
diffuse, `parameters` and `withParameters`. Override `name` with a literal, and
`parameterSpecs` if any parameter is not a variance.

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
