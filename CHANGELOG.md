## 0.5.0

The first stationary components, the noise floor issue #1 asked for, and a
calibration harness that turned out to contradict what the roadmap expected.

* **`Matern`** at ν = 1/2, 3/2 and 5/2 — one, two and three states, exactly.
  The standard Gaussian process kernel and the obvious gap in the component
  list. `ν = 1/2` is an Ornstein-Uhlenbeck process, the exact continuous-time
  AR(1), which is the shape of hydration wobble in a weight series and of a
  great many other things a trend should not be chasing. The order also chooses
  how rough the curve may be, which matters: the cubic spline of
  `LocalLinearTrend` assumes a differentiable trend, and plenty of series are
  not.
* **`StochasticCycle`**, the quasi-periodic kernel `σ² ρ^|τ| cos(2πτ/p)`, with
  the period estimated rather than given. Unlike a `TrigonometricSeasonal` it
  says the rhythm is *approximate* — the cycle drifts out of step and finds its
  way back — which is what a physiological rhythm actually does.
* **Both are stationary**, which is new. Everything shipped before this is
  diffuse, and `properPrior` had existed since 0.1 as a hook nothing
  implemented. That is why the two arrive together: getting a stationary prior
  right once serves both, and it is what saves a component whose variance
  shrinks towards nothing from still drawing a pattern that is not there.
* Both are pinned against dense `O(N³)` Gaussian process references to 1e-9 on
  the likelihood and 1e-10 on the posterior, the same way the trend and the
  seasonal are. For a model of stationary components alone there are no flat
  directions at all, so the comparison is against the plain textbook likelihood
  with no restricted-likelihood correction in the way.

* **A noise floor.** `fit(model, data, minimumMeasurementVariance: ...)` runs
  normally and redoes the fit with the noise pinned only if the estimate lands
  below the floor; `fixedMeasurementVariance` pins it outright. Requested in
  issue #1, and a real gap: a run of nearly identical readings would let the
  model claim a precision no kitchen scale can deliver, and draw a band to
  match. A floor cannot be a clamp applied afterwards — pinning one variance in
  absolute units breaks the scale equivariance that lets everything else be
  searched as a ratio, so the rest must be re-estimated against it, which moves
  the process variance by 0.6 decades on the test series.
* **`ParameterSpec`** tells `fit` which parameters are variances. A length
  scale must not be multiplied back up by the fitted noise level at the end,
  and a period has no business being searched over a bracket meant for variance
  ratios. `Component.parameterSpecs` defaults to all-variances, so nothing
  outside this package needs changing.
* **The coordinate scan runs twice** for a model with more than one parameter.
  The first sweep scans each axis against arbitrary values of everything it has
  not reached yet, which is harmless when the axes barely interact and wrong
  for a cycle whose period is scanned against a cycle that is not there. The
  period axis is also scanned far more finely than the others, because its
  likelihood is multimodal — a cycle at half the period explains every second
  peak and sits on its own maximum.
* `FitResult` gains `parameterSpecs` and `measurementVariancePinned`.
  `varianceRatios` now reports `NaN` for a parameter that is not a variance,
  since a ratio of a period to a variance is not a quantity; `varianceRatio`
  looks for a single *variance* rather than a single parameter, so it still
  works for a `Matern` with its two.

* **`tool/calibration/`**, a harness that fits candidate models to a real weight
  diary — a trale export, or any file of date-and-weight lines — and prints what
  each says, both over the whole history and at each stage of its growth. It is
  offline and read-only.
* It contradicted the plan, which is why it exists. The roadmap expected a
  hydration component to stabilise a trend fit by giving the autocorrelation
  somewhere to go. It does whiten the residuals — Ljung-Box from under 0.0001 to
  0.02–0.06 — and it makes the trend's own variance *worse*, undetermined over
  five decades or more, because the two compete for the same slow variation.
  What actually stabilises the fit is a weekly seasonal: with one, the fitted
  bandwidth lands at 1.8 to 2.2 days on all four synthetic diaries with a
  plateau of 0.4 to 0.85 decades, where a trend alone gives 2.0 to 3.5 days and
  half a decade to over a decade of plateau.

## 0.4.0

**Relicensed to MIT**, from AGPLv3+. AGPL is a reasonable position for an
application and the wrong one for a library, where it is inherited by
everything built on top, network clause and all. Nothing is lost in the
direction that matters: an AGPL application may use an MIT library freely, so
[trale](https://github.com/QuantumPhysique/trale) is unaffected. The 0.1.0 entry
below still says AGPLv3+ and stays that way, because it was true when it was
written.

Events that are not periodic, annual seasonality, and a fit that says when a
variance has been shrunk out rather than estimated.

* **`RegressionComponent`**, with `IndicatorRegressor` for events with a start
  and an end — a fortnight over Christmas, a conference, a course of medication
  — and `StepRegressor` for a covariate that changes at known instants. Asking
  a sum of sinusoids for a two-week rectangle costs a dozen harmonics and rings
  on both sides of it; an indicator represents it exactly with one state.
* **Coefficients cost the optimiser nothing.** They are states with `A = I` and
  `Q = 0` under a flat prior, which is exactly the sort of flat direction exact
  diffuse initialisation already integrates out. `parameterCount` is zero, a
  trend plus twenty indicators is still a one-dimensional fit, and `fit` now
  handles a model with no free variance at all: one forward pass, and the
  measurement variance in closed form.
* `SmoothingResult.coefficients` reports each one with a posterior standard
  error and a credible interval. Pinned against dense generalised least squares
  to 1e-9 on the estimate and 1e-11 on the variance.
* **Annual seasonality** needs no new component, only a period of 365.25. Three
  years of daily data separates a trend, a weekly pattern and an annual one to
  within 0.15 on every amplitude.
* **`FitResult.parameterStatus`** distinguishes a variance that was estimated
  from one shrunk out at the bottom of the bracket and one pushed past the top.
  A variance of zero is the edge of the parameter space rather than an interior
  point, so the half-nat width reported there is one-sided and is not an error
  bar.
* Regressors are data rather than closures, and are defined by knots rather
  than samples, so an output grid can ask for the signal between readings.
* An **isolate round-trip test**, for a promise the package had been making
  without checking: a model goes across, a posterior, a forecast and a set of
  diagnostics come back, and all of them match what the same code computes in
  process. Writing it corrected a misconception in two doc comments — a plain
  closure crosses an isolate boundary perfectly well, since Dart 2.15 closures
  are sendable within an isolate group. What fails is anything holding a port
  or a file handle, including a closure that captured one, which is a runtime
  failure in the caller's code over something the type system never showed
  them. That, rather than sendability, is why regressors are data.
* `relativeVariance` is documented in the README rather than only in the API
  docs, and the README no longer claims there is no matrix type in the public
  API — there is one, `MatrixBlock`, and it is how a component writes its own
  transition and noise blocks.

Breaking changes:

* `FitResult.atBracketEdge` is now derived from `parameterStatus` rather than
  stored. It still means what it did; `parameterStatus` says which parameter
  and which edge, which is what determines whether the answer is wrong or
  merely uninteresting.
* `Component.identifiabilityHint` takes the two endpoints of the series rather
  than its duration. An event indicator whose occurrences all fall outside the
  data contributes a column of zeros, and a duration cannot tell it that. This
  matters only if you have written a `Component` of your own.

Another piece of roadmap advice did not survive contact with a measurement. It
said to always include the annual component and let the marginal likelihood
shrink it to zero rather than gating on how much history there is. Shrinking
the variance does not remove the component, it only stops it evolving; what is
left is a rigid Fourier series whose starting coefficients have a flat prior
that nothing shrinks, and over less than one period a rigid sinusoid is very
nearly a constant plus a slope. On 180 days of data with no annual cycle in it
at all, the annual component draws 1.14 peak to trough while reporting its
variance at the floor.

The advice survives, for a different reason than the one given. The component
says plainly that it does not know its own share: its posterior standard
deviation there is 1.27, larger than the pattern it drew and eighteen times the
0.07 the total signal is known to. Read the component's band rather than its
mean and the spurious cycle is obviously consistent with nothing. By a year
that figure is 0.065 and by two it is 0.026.

## 0.3.0

Seasonality, fitting several variances at once, and a way to tell whether the
model deserves to be believed.

* **`TrigonometricSeasonal`**, in rotation form. Dummy-variable seasonality has
  no transition matrix for a non-integer gap, which rules it out of a package
  whose whole design rests on being exact over an arbitrary one; the rotation
  has one, and `A(a) A(b) = A(a + b)` is asserted directly. Harmonics at or past
  the Nyquist frequency are refused rather than diagnosed later, because at
  `lambda = pi` the rotation degenerates to `-I` on unit steps and leaves a
  state nothing can ever observe.
* **Multi-parameter fitting.** The measurement variance is still concentrated
  out, so a `k`-component model is a `k`-dimensional search. A coordinate scan
  finds the basin and Nelder-Mead with one restart refines it. On five hundred
  simulated readings of a trend plus an evolving weekly pattern, both variance
  ratios come back within a factor of `e` and the decomposition is recovered to
  about a quarter of the measurement noise.
* **Innovation diagnostics.** `StructuralModel.diagnose` returns the
  standardised prediction errors, their autocorrelations and a Ljung-Box test
  with a p-value. On four hundred daily readings carrying a weekly cycle, a
  trend alone looks like a perfectly reasonable fit and is caught immediately:
  lag-7 autocorrelation 0.70 and `p = 1.2e-225`.
* **The residuals are the recursive ones**, with the starting point estimated
  from what came strictly before each observation rather than from the whole
  series. Substituting the final estimate is the tempting shortcut and it
  conditions every residual on its own future.
* A closed-form kernel for the seasonal component,
  `k(t, t') = sigma^2 min(t, t') sum_j cos(lambda_j (t - t'))`, and a dense
  `O(N^3)` test against it — including a three-component decomposition checked
  against the dense per-component posterior.
* `ComplexityPenalty`, a penalised-complexity penalty on the variances.

Breaking changes:

* `FitResult.varianceRatio` now throws for a model with more than one variance,
  where there is no single ratio to report. Use `varianceRatios`, which is in
  component order.
* `FitResult.plateauDecades` is measured differently and its value will change.
  It used to be read off the coarse scan grid, which quantised it to the scan
  step — 1.25 decades with the default bracket, coarse enough that `isFlat` was
  closer to a coin toss than a diagnostic. Each parameter is now probed along
  its own axis, and `plateauDecadesByParameter` reports them separately.
* `fit` no longer refuses models with more than one free parameter, which is
  the point of the release. A model whose components cannot be told apart now
  fails in the engine, with a message explaining which failure it is.
* `Component` gains two members, which matters only if you have written one of
  your own: `wanderOver`, giving the spread of the component's contribution
  over a window, and `identifiabilityHint`, which has a default.

One thing did not work out as planned. The roadmap expected the penalty to be
on by default, on the grounds that plain maximum likelihood gives unstable
trend/seasonal splits on short histories. Measured against the paths that
generated the data, over twelve replications at three sample sizes, it makes no
difference to the decomposition — 0.0867 against 0.0870 at `N = 60`, 0.0759
against 0.0759 at `N = 500` — and it is clearly worse at recovering the
variances, with the root-mean-square error of the log ratio going from 0.37 to
0.59 at `N = 500`. It does not even reduce the spread of the estimates, which
is the usual consolation. So it ships, off by default, with the measurements
recorded next to it. What it does do reliably is drive a component's drift
parameter to the floor when there is no drift to find, which is a different and
narrower thing to want.

## 0.2.0

Numerical maturity: the flat prior directions are handled exactly, the pass
that fitting hammers is four times faster, and the model can project forward.

* **Exact diffuse initialisation**, by augmentation rather than by a second
  set of recursions. The state is written `x(0) = a + B d` with `d` unknown and
  flat; everything downstream is affine in `d`, so the filter carries `dx/dd`
  beside the state and the flat directions are integrated out in closed form at
  the end of the pass. The smoother reuses the same gains and recombines by the
  law of total variance. This is now the default; `ApproximateDiffuse` remains
  available by name.
* **`forecast()`**, which is prediction with the update skipped — the same
  thing the recursion already does over a gap inside a series. `O(N + H)` time,
  no history kept, and it agrees with smoothing on a trailing grid to 1e-9.
* **A scalar two-state fast path** for the forward pass, about four times
  faster than the generic engine, with `fast_path_equivalence_test.dart`
  holding it to that engine at 1e-12. The generic engine came first and remains
  the definition of the answer.
* Golden fixtures for both initialisations, and a dense restricted-likelihood
  check on the `log|M|` term that exact initialisation introduces.

Breaking changes:

* `StructuralModel`'s `diffuseVariance` parameter is replaced by
  `initialization`, taking `ExactDiffuse()` (the default) or
  `ApproximateDiffuse(variance: ...)`. The old constant said how wide the prior
  was but not what kind of prior it was, which left nowhere to put the exact
  case except a second, mutually exclusive knob.
* Two calls that used to return an answer now raise, because the answer they
  returned was an artefact. A single observation under a two-state trend, and
  an output grid with no observations behind it, do not determine the flat
  directions; exact initialisation says so. Pass `ApproximateDiffuse()` to get
  the old behaviour.

Everything else gets quietly sharper. The closed-form limits — a rigid trend is
ordinary least squares, reversing time reverses the answer — used to hold to
1e-4 as the prior widened. They now hold outright, to between 1e-9 and 1e-12.

One note on the reference implementation, since it took some pinning down.
statsmodels' exact diffuse smoother disagrees with a dense
generalised-least-squares computation about the smoothed slope at the very
first step when the transition matrix is genuinely time-varying. It agrees to
1e-14 whenever the step is constant — at unit steps, at 2.5, at 0.5 — and
diverges only once the steps vary; this package agrees with the dense form in
every case. Those four numbers are asserted against the dense form instead.

## 0.1.0

First release, under the GNU AGPLv3+. A generic dense state-space engine, two components, and the
validation harness that makes the rest believable.

* `StructuralModel` composes a list of `Component`s block-diagonally and
  exposes `smooth`, `logLikelihood`, and an optional output grid.
* `LocalLinearTrend` and `LocalLevel`, both continuous-time and exact for any
  gap, including zero.
* Kalman forward pass with Joseph-form scalar updates, missing-data skipping,
  and diffuse burn-in; RTS backward pass solving by Cholesky rather than
  inverting, and smoothing the filtered moments in place.
* `fit` estimates the process-to-measurement variance ratio by maximum profile
  marginal likelihood — coarse scan, then golden section — and reports how flat
  the surface was.
* `SmoothingResult` gives the level, its variance, the slope, per-component
  contributions, and both credible and predictive intervals. The component
  accessors are there from the start so that adding seasonal and regression
  components in later versions is not a breaking change.
* Validated against a dense `O(N^3)` Gaussian process built from the same
  kernel, against statsmodels fixtures generated by a committed script, against
  closed-form limits, and by parameter recovery from simulated series.

Two deliberate deviations from the design document, both to make the API say
what it means:

* `Observation.relativeVariance` rather than `variance`, because the number is
  a multiplier on the model's measurement variance rather than an absolute
  variance, and calling it `variance` invites the wrong reading.
* `levelVariance` / `componentVariance` / `credibleInterval(i, coverage:)`
  rather than the abbreviated names, and `parameters` rather than
  `initialParameters`, since a component returns its current values.

Known limitations, all of them scheduled:

* Diffuse initialisation is approximate. It costs about four digits in the
  smoothed covariance over the first few steps and nothing thereafter; the
  golden tests assert exactly that. Exact diffuse initialisation is 0.2.
* `fit` handles one free parameter. A multi-component model needs a
  multivariate optimiser, which arrives with the seasonal components in 0.3.
* No `forecast()` yet — 0.2.
