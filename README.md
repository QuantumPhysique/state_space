# state_space

Exact Gaussian process regression for irregular time series, in linear time.
Kalman filtering, RTS smoothing, and marginal-likelihood hyperparameter
estimation. Pure Dart, no runtime dependencies.

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
```

Nothing is interpolated, nothing is resampled, and no gap is filled. Missing
data is a step with no update; an irregular gap is a different `dt`; a repeated
timestamp is `dt = 0`. All three fall out of the recursion rather than being
handled.

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

**Cross-language golden fixtures.** `tool/generate_fixtures.py` builds the same
model in statsmodels as an `MLEModel` with time-varying system matrices, and
dumps the filtered, predicted and smoothed states, their covariances, and the
per-observation likelihood to JSON. Script and fixtures are both committed.
Filtered and predicted states agree to 1e-10 relative and their covariances to
1e-9; smoothed states to 1e-9 and smoothed covariances to 1e-11, once the
diffuse prior has washed out. Over the first two steps the smoothed covariance
is the difference of two quantities of order 1e5 giving an answer of order
1e-3, so the two implementations agree there to about four digits — which the
tests assert as a floor rather than paper over.

Note that `UnobservedComponents(level='local linear trend')` is the *discrete*
model and is not what this package implements — comparing against it would be
comparing against a different model.

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
| 100 | 0.01 ms | 0.02 ms | 228 ns | 0.04 ms |
| 1 000 | 0.07 ms | 0.22 ms | 223 ns | 3.67 ms |
| 10 000 | 0.70 ms | 2.22 ms | 222 ns | 368 ms |
| 100 000 | 7.09 ms | 23.35 ms | 234 ns | — |

Flat cost per observation across three orders of magnitude, which is what
linear means. Ten years of daily readings smooth in about a millisecond.

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
variances are expressed per that unit. There is no configuration object, no
strategy enum, and no `Matrix` in the public API.

## Numerical notes

* The measurement update is in **Joseph form**. For a scalar observation it
  expands to a rank-two symmetric update that costs `O(n^2)` rather than
  `O(n^3)`, is symmetric and positive semi-definite by construction for any
  gain, and is what the code computes — the textbook `P = (I - KH) P-` is the
  same expression algebraically and worse numerically.
* The smoother solves `P- G' = A P` by Cholesky rather than forming an inverse,
  and jitters the diagonal before giving up.
* The diffuse prior is `1e6` times the *measurement variance*, not `1e6`
  absolute. Scaling it with the model is what keeps the whole thing
  scale-equivariant, and that is what makes the profile likelihood in `fit`
  exact rather than merely close. Exact diffuse initialisation lands in 0.2.
* `fit` concentrates the measurement variance out analytically, so a
  one-component model is a one-dimensional search over `log q`: a coarse scan
  for the right basin, then golden section inside it, around fifty filter
  passes. With `k` components you will search `k` ratios instead of `k + 1`
  variances — the saving survives to every future version.
* `FitResult.plateauDecades` reports how wide the near-maximum region is. If
  the likelihood cannot tell a stiff curve from a flexible one, the result says
  so rather than returning a confident number.

## Roadmap

0.1 is the engine, two components, the output grid, one-parameter fitting, and
the validation harness above.

* **0.2** — exact diffuse initialisation; a scalar 2x2 fast path *with an
  equivalence test against the generic engine*; `forecast()`.
* **0.3** — trigonometric seasonal components; Nelder-Mead over several
  variance ratios; penalised ML; innovation diagnostics.
* **0.4** — regression components for holidays and tagged events; annual
  seasonality.
* **0.5** — a damped stochastic cycle with an *estimated* period, a grid scan
  for its multimodal likelihood, and an identifiability report.

Not planned: calendar-monthly seasonality, EKF/UKF, particle filters,
multivariate observations.

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

## Licence

[GNU AGPLv3+](LICENSE), the same licence as
[trale](https://github.com/QuantumPhysique/trale), the application this package
was written for.
