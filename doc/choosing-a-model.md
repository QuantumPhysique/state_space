# Choosing a model

Six components is enough to have to think about it. The short answer: start with
the smallest model that could be true, add one component at a time, and let the
residuals rather than the likelihood tell you whether it earned its place.

## Start here

| what you have | what to use |
|---|---|
| noisy readings of something that drifts | `LocalLinearTrend` — the default, and a smoothing spline |
| a level with no persistent direction | `LocalLevel` |
| a pattern on a calendar you know: a week, a year | `TrigonometricSeasonal(period: 7, harmonics: 2, ...)`, or `period: 365.25` |
| named, dated events: a holiday, a course of medication | `RegressionComponent` with `IndicatorRegressor`s |
| a covariate that changes at known instants | `RegressionComponent` with a `StepRegressor` |
| correlated wobble the trend should not be chasing | `Matern.oneHalf(...)` alongside the trend |
| a rhythm that is approximate, or whose period you do not know | `StochasticCycle` |

`LocalLinearTrend` against `LocalLevel` is a claim about whether the thing has a
direction that persists. A trend carries a slope and will extrapolate it; a level
will not. For anything where "still going down" is a meaningful sentence, the
trend is the one.

[Components](https://github.com/QuantumPhysique/state_space/blob/main/doc/components.md) has the detail on each. This page is about putting
them together.

## Add one at a time, and let the residuals ask

```dart
final d = model.diagnose(data);
d.ljungBox(lags: 14, fittedParameters: model.parameterCount);
d.autocorrelation(7);
```

A pattern the model has not accounted for shows up as autocorrelation long before
it shows up as a visibly bad fit. Add the component that explains the spike rather
than the one you had in mind when you started.

Note what a lag is here: it counts **observations, not days**. Lag one is the
previous reading, whenever that happened to be. That is the right notion — under
the model the residuals are independent however unevenly they are spaced — but it
does mean a weekly pattern appears at lag seven only when the sampling is roughly
daily. On thinner data, look wherever a period's worth of readings falls.

Pass `fittedParameters` when the variances were estimated on the same data. Each
estimated parameter costs a degree of freedom, and ignoring that makes the test
optimistic.

## How to tell whether it earned its place — and how not to

**Do not compare `logMarginalLikelihood` across models with different diffuse
structure.**

Under exact diffuse initialisation it is a *restricted* likelihood: the flat
directions have been integrated out against an improper prior of unit density, so
the result carries their units. Two consequences, both measured and both pinned
in `comparability_test.dart`:

* Writing a regression column in grams rather than kilograms shifts it by exactly
  `log 1000`, while the fit, the posterior and the coefficient are unchanged.
* Measuring time in half-days rather than days shifts it by exactly `log 2` per
  diffuse direction that carries a time dimension — for the same stochastic
  process on the same data.

Either is enough to reverse a verdict. This is not a defect of the
implementation; it is what a restricted likelihood is, and the same warning
applies to REML anywhere else you meet it.

`FitResult.isComparableWith` is the check, and `diffuseDimension` is what has to
match:

| component | flat directions |
|---|---|
| `LocalLevel` | 1 |
| `LocalLinearTrend` | 2 |
| `TrigonometricSeasonal` | 2 per harmonic |
| `RegressionComponent` | 1 per column |
| `Matern`, `StochasticCycle` | 0 |

So a trend *can* be compared with a trend plus a Matérn, and *cannot* be compared
with a trend plus a weekly seasonal.

Three things can be compared across anything:

1. **The fitted noise level.** A model missing a real component has to explain
   that component as noise, and reports a scale far noisier than it is. This is
   the most useful single number the package produces.
2. **The Ljung–Box p-value**, which asks the question directly.
3. **Out-of-sample error.** Fit on a prefix, `forecast` the rest, score it.
   Slower, and the only one of the three that is not a within-sample argument.

The first two answer cleanly on two years of daily readings built from a trend, a
weekly sinusoid and noise of standard deviation 0.3:

| model | fitted noise sd | Ljung–Box p (14 lags) |
|---|---|---|
| trend only | 0.472 | 3e-49 |
| trend + weekly | **0.302** | **0.31** |

The noise level recovers the truth to three decimal places and the portmanteau
test goes from certain rejection to unremarkable. The two likelihoods differ by
six flat directions against two and cannot be subtracted at all.

## Read the warnings

```dart
fitted.atBracketEdge;   // did anything finish on a bound?
for (final warning in fitted.warnings) print(warning);
```

`atBracketEdge` is the one-line version and the first thing worth checking.
`warnings` is empty when there is nothing to say, and reports three things:

* a parameter that finished on a bound — its estimate is a boundary rather than
  an interior optimum, and the width beside it is one-sided;
* a parameter the data barely constrains, meaning a plateau over two decades
  wide;
* **a width that was measured while another parameter of the same component sat
  on a bound**, which is the one that is easy to miss.

That third case is worth understanding. Every width in
`plateauDecadesByParameter` is conditional on the other parameters, so when one
of them has finished on a bound the slice is taken *at* that bound. A
`StochasticCycle` fitted to a series with no cycle in it drives the damping to
the top of its bracket, where the component is a rigid sinusoid whose likelihood
in frequency is as sharp as a periodogram spike — and then reports the period as
pinned to a thousandth of a decade. Nothing about that is wrong arithmetically
and all of it is misleading.

`plateauDecadesByParameter` is `NaN` for a parameter that is not searched on a
log scale; a damping factor is a logit, and a width in logits over `ln 10` is not
decades of anything. `plateauWidthByParameter` carries the raw number for every
parameter.

## What competes with what

Components that can draw the same shape will trade off against each other, and
the fit will not always tell you which it chose.

* **A trend and a seasonal, over less than one period.** A rigid sinusoid over
  half a cycle is very nearly a constant plus a slope, so an annual component on
  180 days will draw a swing out of a series that has none. It does say so, in
  the place worth looking: that component's own posterior standard deviation
  comes back *larger than the amplitude it drew*. Check `componentVariance`
  alongside `componentMean`. Shrinking the variance to zero does not remove the
  component — it only stops the pattern evolving.
* **A trend and a Matérn.** Both are slow. Leaving both free is measurably worse
  than adding neither: the two compete for the same slow variation and the
  trend's variance ends up undetermined over five decades. If you add a Matérn
  deviation, consider fixing the trend's smoothing rather than fitting it.
  Whitening the residuals is not free.
* **A Matérn and the measurement noise.** A length scale below the sampling
  interval *is* white noise. `fit` floors it at the median gap between readings
  for that reason; without the floor the likelihood prefers the corner where the
  Matérn takes the noise and the reported precision becomes fiction.
* **A cycle and a free level.** A random walk whose variance is free can draw any
  wiggle, so a `StochasticCycle` beside an unconstrained `LocalLevel` often
  loses. It fails in two directions and `warnings` names both: on a series with a
  real cycle the level's variance runs to the top of its bracket and interpolates
  the data, and on a series with none the level shrinks away while the cycle's
  own variance and period come back flat over eight and two decades. Either way
  the reading is the same — nothing here is determined.

## When to stop

When adding a component does not reduce the fitted noise level, when the
Ljung–Box p-value is unremarkable, and when `warnings` is empty. A model whose
components each have something to do is worth more than one that fits slightly
better and cannot say which part did the work.

## A worked comparison on real data

`tool/calibration/` fits several candidate models to a weight diary — a real
export, or four synthetic ones — and prints the fitted smoothing, how well
determined it is, the noise level and a portmanteau test, both for the whole
history and at each stage of its growth.

```sh
dart run tool/calibration/calibrate.dart --all
dart run tool/calibration/calibrate.dart --all my-export.txt
```

Nothing leaves the machine. It is how the advice on this page was arrived at, and
[its own README](https://github.com/QuantumPhysique/state_space/blob/main/tool/calibration/README.md) is worth reading for how to read
the tables — in particular why the likelihood column may only be compared down a
run of equal diffuse dimension.
