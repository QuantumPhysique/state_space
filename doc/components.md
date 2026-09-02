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

The last column is what each costs the optimiser, and `fit` concentrates the
measurement variance out on top of that. A trend plus twenty holiday indicators
is a one-dimensional search.

## Why you cannot hand it a covariance function

You implement `Component`; you do not pass `k(s, t)`. That is the whole premise
rather than a limitation of the API: linear time comes from the Markov property,
and only kernels with a finite-dimensional state-space form have it. A general
`k(s, t)` puts you back at `O(N³)`, which is the thing being replaced.

So the useful question is not "can I supply a kernel" but **is my kernel
reachable**, and the table above is the answer.

**The squared exponential is not on it.** It has no exact finite-state form —
only a Padé approximation of its spectral density costing about six states for a
few digits, which is a lot of machinery for a kernel whose infinite smoothness is
rarely what anyone actually believes.

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
natural cubic smoothing spline** — as a Bayesian posterior, with honest
uncertainty, in linear time (Wahba 1978). The smoothing parameter is
`lambda = measurementVariance / processVariance`, and `fit` estimates it by
maximum marginal likelihood rather than by cross-validation.

Choosing this over `LocalLevel` is a claim that "still going down" is a
meaningful sentence about your data. It carries a slope and will extrapolate it.

### `TrigonometricSeasonal`

A repeating pattern of known period whose shape drifts. Each harmonic is a pair
of states that rotate into one another as time passes; the observation reads the
sum.

```dart
TrigonometricSeasonal(period: 7, harmonics: 2, processVariance: 1e-3)
```

Two or three harmonics resolve a weekly shape. The fourth is resolving detail
finer than seven daily readings support, and `harmonics` must in any case stay
under half the period — at and past the Nyquist frequency a harmonic either
aliases onto a lower one or leaves a state the data can never see, so the
constructor refuses it.

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
than a closure — not because a closure cannot cross an isolate boundary, but
because what it *captures* might not, and the failure would surface in the
caller's code over something the type system never showed them.

Each coefficient is one state with `A = I` and `Q = 0` under a flat prior, so
**`parameterCount` is zero**: the exact diffuse machinery already integrates out
flat directions, and a coefficient is one. They cost the optimiser nothing and
arrive with posterior standard errors from the same recursion that produced the
trend:

```dart
posterior.coefficients.first;   // christmas: 1.121 +/- 0.091
```

Because such a state never moves, the backward pass skips it too — see
[`Component.isStatic`](https://github.com/QuantumPhysique/state_space/blob/main/doc/how-it-works.md#the-smoother).

## The stationary ones

These have a prior of their own: they hover around zero and forget where they
have been. That makes them the right shape for a component that is a *deviation*
rather than a level, and it means they contribute no flat directions at all.

### `Matern`

The standard Gaussian process kernel, in state-space form, at ν = 1/2, 3/2 and
5/2 — one, two and three states, exactly rather than approximately.

```dart
StructuralModel([
  const LocalLinearTrend(processVariance: 1e-4),
  Matern.oneHalf(variance: 0.1, lengthScale: 3),
]);
```

Reach for it when a series has structure the trend should not be chasing. `ν = 1/2`
is an Ornstein–Uhlenbeck process, the exact continuous-time AR(1), and it absorbs
the correlated wobble a trend-only model has nowhere to put but the noise.

The order also decides how rough the path may be, which is the modelling choice
worth making deliberately: the cubic spline of `LocalLinearTrend` assumes a trend
with a continuous derivative, and `ν = 1/2` assumes nothing of the sort.

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

**Check `atBracketEdge` first, and specifically the damping.** A cycle fitted to
a series with no cycle pushes the damping to the top of its bracket, where the
component is a rigid sinusoid and can chase noise — and then the width reported
for the *period* becomes tiny, because a rigid sinusoid's likelihood in frequency
is as sharp as a periodogram spike. On white noise the period width comes back at
a thousandth of a decade while the answer is meaningless. The width is
conditional on the damping, so it is worth reading only once the damping is
interior. See [Choosing a model](https://github.com/QuantumPhysique/state_space/blob/main/doc/choosing-a-model.md#read-the-warnings).

## Both stationary components have a floor you did not set

A shape parameter measured in time units stops being a different model below the
sampling interval. A Matérn with `ν = 1/2` and a length scale shorter than the gap
between readings *is* white noise, so it competes with the measurement error
rather than with the trend — and the likelihood mildly prefers it that way. Left
free on daily readings with a true noise level of 0.3, it will report the noise as
**0.002**, draw a band covering every point, and hand back a trend interpolating
the noise.

So `fit` raises the bottom of `lengthScaleBounds` to the median gap between
readings, and a cycle's `periodBounds` to twice it, which is the Nyquist limit.
It is the same refusal `TrigonometricSeasonal` already makes about harmonics,
moved to where the limit depends on the data rather than on the component alone.
Your own lower bound is respected when it is higher, and the upper bound is
honoured as given.

## Writing your own

Implement `Component`. You supply the state dimension, `A(dt)` and `Q(dt)` —
which must be exact for *any* non-negative gap, including zero — an observation
row, which of your states are diffuse, and a proper prior for the ones that are
not. Everything else has a default.

`MatrixBlock` is the one matrix type in the API and appears only here: it is a
view onto the engine's buffer, so your component fills its own block in place
without knowing where that block sits. Implementing a component is the one place
the engine's internals are visible, and that is deliberate.
