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

fitted.model.forecast(data, nextThirtyDays);   // and where it is heading
```

Components add up, and the posterior comes apart the same way:

```dart
final model = StructuralModel([
  const LocalLinearTrend(processVariance: 1e-3),
  TrigonometricSeasonal(period: 7, harmonics: 2, processVariance: 1e-3),
]);

final posterior = fit(model, data).model.smooth(data);
posterior.componentMean(0);   // the trend
posterior.componentMean(1);   // the weekly pattern, separately

model.diagnose(data).ljungBox(lags: 14);   // and whether to believe any of it
```

Events that are not periodic — a fortnight over Christmas, a conference, a
course of medication — go in as indicator columns, and cost the optimiser
nothing at all, because their coefficients are states rather than parameters:

```dart
StructuralModel([
  const LocalLinearTrend(processVariance: 1e-4),
  RegressionComponent([
    IndicatorRegressor('christmas', [(from: 350, to: 364)]),
  ]),
]);   // still a one-dimensional fit

posterior.coefficients.first;   // christmas: 1.121 +/- 0.091
```

Nothing is interpolated, nothing is resampled, and no gap is filled. Missing
data is a step with no update; an irregular gap is a different `dt`; a repeated
timestamp is `dt = 0`. All three fall out of the recursion rather than being
handled.

Readings need not be equally trustworthy, either. `relativeVariance` weights a
single observation against the model's noise level, so an average of two
weighings and a hurried single one can sit in the same series and be given
their due:

```dart
Observation(4, 80.4);                          // an ordinary reading
Observation(5, 80.5, relativeVariance: 0.5);   // trusted twice as much
Observation(6, 79.1, relativeVariance: 4.0);   // trusted half as much
```

It is relative rather than absolute so that one number still sets the scale of
the noise and these weights only say how the readings differ from each other.

And when you know something about the instrument, you can say so. Left alone,
the noise level is whatever explains the data best, which on a run of nearly
identical readings can be a number no real scale could deliver and a band far
too narrow to believe:

```dart
fit(model, data, minimumMeasurementVariance: 0.05 * 0.05);   // reads to 100 g
fit(model, data, fixedMeasurementVariance: 0.05 * 0.05);     // and no arguing
```

The floor costs nothing when the data agrees — the fit runs normally, and only
if the estimate lands below the floor is it redone with the noise held there.
It cannot be a clamp applied afterwards, because pinning one variance in
absolute units breaks the scale equivariance that lets everything else be
searched as a ratio; the other variances have to be found again against it.

## Which kernels

The components are additive blocks of a structural model, and each one is a
Gaussian process kernel written in the form that makes it fast.

| component | kernel | states | prior |
|---|---|---|---|
| `LocalLevel` | Brownian motion | 1 | diffuse |
| `LocalLinearTrend` | cubic spline, `min(s,t)³/3 + min(s,t)²|s−t|/2` | 2 | diffuse |
| `TrigonometricSeasonal` | `min(s,t) Σⱼ cos λⱼ(s−t)` | 2 per harmonic | diffuse |
| `RegressionComponent` | a constant per column | 1 per column | diffuse |
| `Matern` | Matérn, ν = 1/2, 3/2, 5/2 | 1, 2, 3 | stationary |
| `StochasticCycle` | `ρ^\|τ\| cos(2πτ/p)`, period estimated | 2 | stationary |

You do not hand the package a covariance function; you implement `Component`
and it composes block-diagonally with everything else. The reason is the whole
premise: linear time comes from the Markov property, and only kernels with a
finite-dimensional state-space form have it. A general `k(s, t)` puts you back
at `O(N³)`, which is the thing being replaced. So the useful question is not
"can I supply a kernel" but **is my kernel reachable**, and the table is the
answer. The squared exponential is not on it: it has no exact finite-state
form, only a Padé approximation of its spectral density costing about six
states, which is a lot of machinery for a kernel whose infinite smoothness is
rarely what anyone actually believes.

`Matern` is the one to reach for when a series has structure the trend should
not be chasing. Put `ν = 1/2` — an Ornstein-Uhlenbeck process, the exact
continuous-time AR(1) — alongside a trend and it absorbs the correlated wobble
that a trend-only model has nowhere to put but the noise:

```dart
StructuralModel([
  const LocalLinearTrend(processVariance: 1e-4),
  Matern.oneHalf(variance: 0.1, lengthScale: 3),
]);
```

The order also decides how rough the curve may be, which is the modelling
choice worth making deliberately: the cubic spline assumes a trend with a
continuous derivative, and `ν = 1/2` assumes nothing of the sort.

`StochasticCycle` is the *approximate* rhythm — a cosine that fades and finds
its way back, rather than a pattern that repeats forever. Its period is
estimated rather than given, which makes it the one component here whose
likelihood is multimodal: a cycle at half the period explains every second peak
and sits on its own maximum. `fit` scans that axis far more finely than the
others before anything local runs, and `plateauDecadesByParameter` on the
period is the number that says whether to believe the answer.

## One object, three descriptions

The whole value of the package is that these are the same thing, and it
computes the third to get the first.

**As a Gaussian process.** Put a prior `f ~ GP(0, k)` on the trend, observe it
with noise, and the posterior is the textbook expression

```text
E[f(t*) | y] = k*' C^-1 y,     Var[f(t*) | y] = k(t*,t*) - k*' C^-1 k*
log p(y) = -0.5 (y' C^-1 y + log|C| + N log 2 pi),     C = K + sigma_eps^2 I
```

Exact, and `O(N^3)` in time and `O(N^2)` in memory. This is the specification.

**As a stochastic differential equation.** For a Markovian kernel the same
prior is a linear SDE. Let the *rate of change* be a Wiener process and the
level be its integral:

```text
d(mu) = nu dt,   d(nu) = sigma dB
```

which discretises **exactly**, over any gap `dt`, to

```text
A(dt) = [[1, dt],      Q(dt) = sigma^2 [[dt^3/3, dt^2/2],
         [0,  1]]                      [dt^2/2, dt    ]]
```

The off-diagonal term in `Q` is the whole story about irregular sampling: over
a gap, uncertainty about the slope integrates into uncertainty about the level,
and the two come out correlated. A discrete local linear trend with a diagonal
`Q` silently drops it, and is exact only for unit steps.

**As a Kalman filter and RTS smoother.** Which computes the same posterior and
the same log marginal likelihood in `O(N)` time and memory, and gives the slope
and the decomposition on the way past.

The implied kernel is the cubic spline kernel `k(t,t') = sigma^2 (m^3/3 +
m^2 |t - t'| / 2)` with `m = min(t, t')`, so **the trend curve is a natural
cubic smoothing spline, computed as the exact posterior of a Gaussian process,
in linear time** (Wahba 1978). The smoothing parameter is
`lambda = measurementVariance / processVariance`, and `fit` estimates it by
maximum marginal likelihood rather than cross-validation.

## Validation

Four independent checks, none of which subsumes the others. This is the part
worth reading.

**The linear-time answer equals the cubic-time definition, in-language, on
every commit.** `test/dense_gp_reference_test.dart` builds the `N x N` spline
kernel from the equations above, adds the noise, and computes the textbook
`O(N^3)` posterior with a Cholesky factorisation from the `matrices` package —
a dev dependency used for nothing else. The smoothed mean and the log likelihood agree to
1e-9, and the posterior variance to 1e-9 absolute, at observation times and at
grid points between them. This validates the *model*, not just the implementation.

**The seasonal component equals its own closed-form kernel.** The rotation is
orthogonal and its driving noise isotropic, so `A(t-u) Q A(t'-u)'` does not
depend on `u` at all and the integral over the noise collapses to
`k(t, t') = sigma^2 min(t, t') sum_j cos(lambda_j (t - t'))` — Brownian motion
multiplied by a comb of cosines. `test/seasonal_reference_test.dart` builds that
matrix densely and checks the filter against it, then does the same for a trend
plus two seasonals of different periods and checks each component's share
against the dense per-component posterior. Confounding between a trend and a
weekly pattern leaves the total fit intact and shows up only in the
decomposition.

**A regression coefficient equals its generalised least-squares estimate.** A
coefficient is a flat direction like any other, so the exact diffuse machinery
estimates it as a by-product. `test/regression_reference_test.dart` is what
makes "by-product" mean exactly rather than approximately: the smoothed
coefficient matches the dense GLS estimate to 1e-9 and its variance matches the
corresponding diagonal of `(B' C^-1 B)^-1` to 1e-11, with the trend's own level
and slope checked from the same solve to show the regression columns have not
disturbed them.

**The diffuse likelihood equals the restricted likelihood.** Exact diffuse
initialisation leaves a `log|M|` term behind when the flat prior is integrated
out. Get its sign or scale wrong and every likelihood shifts by a constant that
nothing else would notice — fits still converge, to the same place, but the
number reported is not the restricted likelihood it claims to be. Densely, the
same quantity is REML, so the test computes
`-2 log L = (N-d) log 2pi + log|C| + log|B' C^-1 B| + y' P y` and checks both
the total and `log|M|` on its own.

Being REML is also the limit of what the number is good for, and this was
overclaimed here until it was measured. A restricted likelihood is comparable
across models that integrate out the *same* flat directions and no further. The
integral is against an improper prior of unit density, so `d` carries units and
so does the answer: writing a regression column in grams rather than kilograms
shifts `logMarginalLikelihood` by exactly `log 1000` while the fit, the
posterior and the coefficient are unchanged, and measuring time in half-days
rather than days shifts it by exactly `log 2` per diffuse direction that carries
a time dimension. Both are pinned in `comparability_test.dart`.
`FitResult.diffuseDimension` and `FitResult.isComparableWith` are there so the
check can be made rather than assumed; to choose between a model with a weekly
component and one without, use the fitted noise level, an out-of-sample error,
or `diagnose`.

**Cross-language golden fixtures, for both initialisations.**
`tool/generate_fixtures.py` builds the same model in statsmodels as an
`MLEModel` with time-varying system matrices, once with a wide proper prior and
once with `initialization='diffuse'`, and dumps the filtered, predicted and
smoothed states, their covariances, and the per-observation likelihood to JSON.
Script and both fixture sets are committed, so the claim that exact
initialisation changes the first few steps and nothing else is testable rather
than asserted.

Under the exact prior everything agrees to 1e-10 at every step including the
first. Under the wide one, the first two steps agree to about four digits,
because there the smoothed covariance is the difference of two quantities of
order 1e5 giving an answer of order 1e-3 — asserted as a floor rather than
papered over.

Two things worth knowing about the reference. `UnobservedComponents(level=
'local linear trend')` is the *discrete* model and is not what this package
implements, so comparing against it would be comparing against a different
model. And statsmodels' exact diffuse smoother disagrees with a dense
generalised-least-squares computation about the smoothed slope at the very
first step when the transition matrix is genuinely time-varying: it agrees to
1e-14 whenever the step is constant — at unit steps, at 2.5, at 0.5 — and
diverges only once the steps vary, while this package agrees with the dense
form in every case. Those four numbers are pinned against the dense form
instead, which is a sharper test anyway.

**The fast path is held to the engine it specialises.** A single two-state
component runs its forward pass in unrolled scalars.
`fast_path_equivalence_test.dart` asserts agreement with the generic engine to
1e-12 across irregular gaps, a repeated timestamp, a two-month hole, missing
observations, unequal weights and all three initialisations. The generic engine
came first and remains the definition of the answer; if the two disagree, the
fast path is wrong.

**Analytic limits.** Sending the process variance to zero reproduces ordinary
least squares for the two-state model and the precision-weighted mean for the
one-state model — and the agreement improves as `1/kappa` with the width of the
diffuse prior, which is asserted rather than assumed. Shifting all times by a
constant changes nothing. Scaling every variance by `c` scales the posterior
variance by `c` and leaves the posterior mean alone; that is the invariance the
profile likelihood rests on, so it is tested directly. Reversing time reverses
the answer. Posterior variance never grows when data is added.

**Parameter recovery.** Simulate from known variances using the exact
discretisation, fit, and check both that the estimate is close and that it gets
closer as the series grows.

Plus the edge cases: no observations, one observation, repeated timestamps, a
five-year gap, constant data, zero-variance readings, and rejection of NaN,
infinities and unsorted input.

## Performance

`benchmark/scaling_benchmark.dart`, AOT-compiled, median of five samples, on an
M-series Mac. `smooth` is filter plus smoother plus the reported posterior.

| N | `logLikelihood` | `smooth` | per observation | `O(N^2)` kernel smoother |
|---|---|---|---|---|
| 100 | 0.00 ms | 0.02 ms | 206 ns | 0.04 ms |
| 1 000 | 0.02 ms | 0.21 ms | 208 ns | 3.67 ms |
| 10 000 | 0.23 ms | 2.21 ms | 221 ns | 368 ms |
| 100 000 | 2.41 ms | 23.21 ms | 232 ns | — |

Flat cost per observation across three orders of magnitude, which is what
linear means. Ten years of daily readings smooth in about a millisecond.

The `logLikelihood` column is where the two-state fast path shows up — about
four times faster than the generic engine, steadily, from a thousand points to
a hundred thousand:

| N | generic engine | fast path | |
|---|---|---|---|
| 1 000 | 0.09 ms | 0.02 ms | 4.0x |
| 10 000 | 0.93 ms | 0.23 ms | 4.1x |
| 100 000 | 9.37 ms | 2.43 ms | 3.9x |

`smooth` barely moves, because the backward pass dominates it and is
deliberately left generic. That is the point of specialising only the forward
pass: `fit` runs one per likelihood evaluation, some fifty per call, while the
backward pass runs once.

### On the linear algebra dependency

`benchmark/dependency_comparison.dart` measures one covariance prediction,
`P- = A P A' + Q`, hand-rolled against `matrices`:

| size | hand-rolled | `matrices` |
|---|---|---|
| 2x2 | 28 ns | 87 ns |
| 6x6 | 457 ns | 294 ns |
| 18x18 | 10.8 us | 3.5 us |

The hand-rolled loop wins at 2x2 by a factor of three, because per-operation
overhead dominates when there are eight multiplications to do. It **loses** from
6x6 upward, because `matrices` multiplies with `Float64x2` SIMD and four
accumulators. That is the opposite of what was expected, and it is in the
repository because a documented evaluation is worth more than either adopting
the dependency or quietly avoiding it.

The engine still does not take it, for two reasons that benchmark cannot show:
`A` and `Q` are block diagonal, which `matrices` has no way to express and
would multiply the zeros of; and there is no in-place or out-parameter path, so
a pass would allocate a result object per operation — roughly seventy thousand
short-lived objects for a decade of daily data. The decision is "not yet", and
the thing to re-measure is a whole pass rather than one product.

## What it does not do

It knows nothing about your domain. No dates, no units, no calendars, no
locales. Time is a `double` in whatever unit you like, and the process
variances are expressed per that unit. There is no configuration object and no
strategy enum.

Nor is there a matrix type in anything a caller receives: every result is
`Float64List`s and doubles, which is also what lets a whole model and its
posterior cross an isolate boundary unchanged. There is exactly one matrix type
in the API, `MatrixBlock`, and it appears only where a component writes its own
transition and noise blocks — a view onto the engine's buffer, so a component
fills its block in place without knowing where that block sits. Implementing a
`Component` is the one place the engine's internals are visible, and that is
deliberate.

## Numerical notes

* The measurement update is in **Joseph form**. For a scalar observation it
  expands to a rank-two symmetric update that costs `O(n^2)` rather than
  `O(n^3)`, is symmetric and positive semi-definite by construction for any
  gain, and is what the code computes — the textbook `P = (I - KH) P-` is the
  same expression algebraically and worse numerically.
* The smoother solves `P- G' = A P` by Cholesky rather than forming an inverse,
  and jitters the diagonal before giving up.
* Non-stationary states get **exact diffuse initialisation** by default, done
  by augmentation rather than by a second set of recursions. The state is
  written `x(0) = a + B d` with `d` unknown and flat; everything downstream is
  affine in `d`, so the filter carries `dx/dd` alongside the state — one extra
  mean propagation per flat direction, no extra covariance work at all — and
  the flat directions are integrated out in closed form at the end of the pass.
  The smoother reuses the same gains and recombines by the law of total
  variance.
* Because it is exact, the data has to actually determine those directions. A
  two-state trend needs readings at two distinct times; given one, the package
  says so rather than returning a variance whose size is an artefact of a
  prior. `ApproximateDiffuse` is still available for callers who would rather
  have the large number, and its error falls as `1/kappa` until rounding takes
  over around `1e7`.
* `fit` concentrates the measurement variance out analytically, so a
  `k`-component model is a `k`-dimensional search rather than a `k + 1`
  dimensional one — the saving is exact and does not shrink as components are
  added. A coordinate scan finds the basin, then golden section for one
  parameter or Nelder-Mead with a restart for several.
* `FitResult.plateauDecadesByParameter` reports how far each parameter can move
  on its own before the fit loses half a nat. If the likelihood cannot tell a
  stiff curve from a flexible one, the result says so rather than returning a
  confident number. Each width is conditional on the other parameters, so when
  two components trade off against each other the joint region is wider than
  any of these slices.
* Standardised residuals are the **recursive** ones. Under a flat prior the
  innovation is an affine function of the unknown starting point rather than a
  number, and substituting the final estimate would condition every residual on
  the whole series — including its own future. Instead the starting point is
  re-estimated from what came strictly before each observation, which gives
  errors that are exactly independent under the model. Their sum of squares
  reproduces the likelihood's own quadratic form to 6e-11 relative, which it
  must: the restricted likelihood factorises into precisely these predictive
  densities.
* Regression coefficients are states with `A = I` and `Q = 0` under a flat
  prior, so `parameterCount` for a regression component is zero. A trend plus
  twenty holiday indicators is still a one-dimensional fit, and the twenty
  coefficients arrive with posterior standard errors from the same recursion
  that produced the trend.
* `FitResult.parameterStatus` says whether each variance was estimated, shrunk
  out at the bottom of the bracket, or pushed past the top of it. A variance of
  zero is the edge of the parameter space rather than an interior point, so the
  half-nat width reported for such a parameter is one-sided and is not an error
  bar.
* **Shrinking a seasonal's variance to zero does not remove the component**, it
  only stops it evolving — what is left is a rigid Fourier series whose
  starting coefficients have a flat prior that nothing shrinks. Over less than
  one period a rigid sinusoid is very nearly a constant plus a slope, so an
  annual component on 180 days of data draws a cycle of 1.14 peak to trough out
  of a series that has none, while reporting its variance at the floor. It does
  say so, in the place worth looking: that component's own posterior standard
  deviation there is 1.27 — larger than the pattern it drew, and eighteen times
  the 0.07 the total signal is known to. By a year it is 0.065.
* A penalised-complexity penalty on the variances is available and is **off by
  default**, which was not the plan. It was expected to stabilise the
  trend/seasonal split on short histories; measured against the simulated paths
  it generates, over twelve replications, it makes no difference to the
  decomposition at any sample size (seasonal RMSE 0.0867 against 0.0870 at
  N = 60, 0.0759 against 0.0759 at N = 500) and is clearly worse at recovering
  the variances themselves. What it does do reliably is drive a component's
  drift parameter to the floor when there is no drift to find. The numbers are
  in `ComplexityPenalty`'s documentation.

## Roadmap

0.1 was the engine, two components, the output grid, one-parameter fitting, and
the validation harness above. 0.2 added exact diffuse initialisation, the scalar
two-state fast path, and `forecast()`. 0.3 added trigonometric seasonality,
fitting over several variance ratios at once, and innovation diagnostics. 0.4
added regression components for events and holidays, and annual seasonality. 0.5
adds the first stationary components — Matérn and a damped cycle — along with
the noise floor and the parameter machinery both of them needed.

* **0.6** — the rigid periodic kernel, if anyone wants it. It is the same
  rotation blocks `TrigonometricSeasonal` already builds, with no process noise
  and a Bessel stationary prior, so it is small now that the stationary work is
  done. Whether it earns its place is a real question: the drifting seasonal is
  the more defensible model for most data, since a pattern identical every week
  for three years is a strong claim.

Not planned: calendar-monthly seasonality, EKF/UKF, particle filters,
multivariate observations.

## Calibration

Computing the right thing and being useful are different claims, and only the
first is a test. `tool/calibration/` is the second: it fits several candidate
models to a weight diary — a real export, or four synthetic ones — and prints
the fitted smoothing, how well determined it is, the noise level and a
portmanteau test, both for the whole history and at each stage of its growth.

```sh
dart run tool/calibration/calibrate.dart --all
dart run tool/calibration/calibrate.dart --all my-export.txt
```

It is how the advice in this README about which components to use was arrived
at, and it found one thing worth repeating here: adding a Matérn deviation
*and* leaving the trend free is worse than not adding it at all, because the
two compete for the same slow variation and the trend's variance ends up
undetermined over five decades. Whitening the residuals is not free.

## References

* Kalman (1960), *J. Basic Eng.* 82(1): 35–45
* Rauch, Tung & Striebel (1965), *AIAA J.* 3(8): 1445–1450
* Wahba (1978), *JRSS-B* 40(3): 364–372 — the spline/GP equivalence
* Harvey (1989), *Forecasting, Structural Time Series Models and the Kalman
  Filter*, CUP — profile likelihood, ch. 3–4
* Durbin & Koopman (2012), *Time Series Analysis by State Space Methods*, 2nd
  ed. — exact diffuse initialisation, ch. 5
* Hartikainen & Särkkä (2010), *IEEE MLSP*: 379–384 — GP to state space
* Särkkä & Solin (2019), *Applied Stochastic Differential Equations*, CUP
* Solin & Särkkä (2014), *AISTATS*: 904–912 — periodic covariance functions as
  state-space models
* Silverman (1984), *Ann. Statist.* 12(3): 898–916 — the spline's equivalent
  kernel, which is how the calibration tool turns a variance ratio into a
  bandwidth in days

## Used by

[trale](https://github.com/QuantumPhysique/trale), a privacy-respecting body
weight diary, for its trend curve and uncertainty band.

## Licence

[MIT](LICENSE).

Deliberately permissive, and deliberately different from the application it was
written for. trale is AGPL, which is a reasonable position for something people
run; a library is not, because that licence would be inherited by everything
built on top of it. An AGPL application can use an MIT library freely, so
nothing is lost in the direction that matters here.
