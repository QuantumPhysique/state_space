# Choosing a model

Start with the smallest plausible model, add one component at a time, and use
the residuals rather than the likelihood to decide whether each one helps.

## Start here

| what you have | what to use |
|---|---|
| noisy readings of something that drifts | `LocalLinearTrend` (the default; a smoothing spline) |
| a level with no persistent direction | `LocalLevel` |
| a pattern on a calendar you know: a week, a year | `TrigonometricSeasonal(period: 7, harmonics: 2, ...)`, or `period: 365.25` |
| named, dated events: a holiday, a course of medication | `RegressionComponent` with `IndicatorRegressor`s |
| a covariate that changes at known instants | `RegressionComponent` with a `StepRegressor` |
| correlated wobble the trend should not be chasing | `Matern.oneHalf(...)` alongside the trend, with a noise floor |
| a rhythm that is approximate, or whose period you do not know | `StochasticCycle` |

The choice between `LocalLinearTrend` and `LocalLevel` is about whether the
series has a direction that persists. A trend carries a slope and extrapolates
it; a level does not. If "still going down" makes sense for your data, use the
trend.

[Components](https://github.com/QuantumPhysique/state_space/blob/main/doc/components.md) has the detail on each. This page is about putting
them together.

## Add one component at a time

```dart
final d = model.diagnose(data);
d.ljungBox(lags: 14, fittedParameters: model.parameterCount);
d.autocorrelation(7);
```

A pattern the model is missing shows up as autocorrelation long before it
shows up as a visibly bad fit. Add the component that explains the spike, which
is not always the one you expected.

A lag counts **observations, not days**: lag one is the previous reading,
whenever that was. Under the model the residuals are independent however
unevenly they are spaced, but a weekly pattern appears at lag seven only when
the sampling is roughly daily. On thinner data, look wherever a period's worth
of readings falls.

Pass `fittedParameters` when the variances were estimated on the same data.
Each estimated parameter costs a degree of freedom, and leaving them out makes
the test optimistic. Pass `parameterCount`: it counts the searched variances
but not the concentrated-out measurement variance, which follows Box and
Jenkins in not charging for the residual scale.

## Comparing models

**Do not compare `logMarginalLikelihood` across models with different diffuse
structure.**

Under exact diffuse initialisation it is a *restricted* likelihood: the flat
directions have been integrated out against an improper prior of unit density, so
the result carries their units. `comparability_test.dart` checks both of these:

* Writing a regression column in grams rather than kilograms shifts it by exactly
  `log 1000`, while the fit, the posterior and the coefficient are unchanged.
* Measuring time in half-days rather than days shifts it by exactly `log 2` per
  diffuse direction that carries a time dimension, for the same process on the
  same data.

Either can reverse a comparison. The same caveat applies to REML in general.

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

These can be compared across any models:

1. **The fitted noise level.** A model missing a real component has to explain
   it as noise, and reports a larger noise level than the truth. This is
   usually the most useful single number.
2. **The Ljung–Box p-value.**
3. **Out-of-sample error.** Fit on a prefix, `forecast` the rest, score it.
   Slower, but the only one of the three that is not computed in-sample.

On two years of daily readings built from a trend, a weekly sinusoid and noise
with standard deviation 0.3, the first two give a clear answer:

| model | fitted noise sd | Ljung–Box, 14 lags |
|---|---|---|
| trend only | 0.472 | statistic 1794 on 13 df, p below 1e-300 |
| trend + weekly | **0.302** | statistic 13.9 on 12 df, p = **0.31** |

The noise level matches the truth to three decimal places, and the Ljung–Box
test goes from certain rejection to unremarkable. The two likelihoods have six
and two flat directions and cannot be compared.

## Read the warnings

```dart
fitted.atBracketEdge;   // did anything finish on a bound?
for (final warning in fitted.warnings) print(warning);
```

`atBracketEdge` is the quick check. `warnings` is empty when there is nothing
to report, and otherwise lists:

* **a reading far from what the rest of the data predicts**, more than six
  typical errors away; see
  [Bad readings](https://github.com/QuantumPhysique/state_space/blob/main/doc/getting-started.md#bad-readings);
* **a parameter that finished on a bound.** Its estimate is a boundary rather
  than an interior optimum, and the width beside it is one-sided. At the bottom,
  the variance of a trend or seasonal means the component is fitted as fixed
  over time, and the variance of a stationary component means it contributes
  nothing;
* **a stationary component's variance at the top of its bracket**, which means
  it has taken over the measurement noise: pass `minimumMeasurementVariance`;
* **a parameter the data barely constrains**, meaning a plateau over two
  decades wide;
* **a width measured while another parameter of the same component sat on a
  bound.** Every width in `plateauDecadesByParameter` is conditional on the
  other parameters, so it is then taken *at* that bound. The common case is a
  `StochasticCycle` on a series with no cycle in it; its documentation works it
  through.

`plateauDecadesByParameter` is `NaN` for a parameter that is not searched on a
log scale; a damping factor is a logit, and a width in logits over `ln 10` is not
decades of anything. `plateauWidthByParameter` carries the raw number for every
parameter.

## What competes with what

Components that can produce the same shape trade off against each other, and
the fit does not always tell you which one took it.

* **A trend and a seasonal, over less than one period.** A rigid sinusoid over
  half a cycle is very nearly a constant plus a slope, so an annual component on
  180 days will draw a swing in a series that has none. The component's own
  posterior standard deviation then comes back *larger than the amplitude it
  drew*, so check `componentVariance` alongside `componentMean`. Shrinking the
  variance to zero does not remove the component; it only stops the pattern
  changing.
* **A trend and a Matérn.** Both are slow. With both free the fit is measurably
  worse than with neither: the two compete for the same slow variation and the
  trend's variance ends up undetermined over five decades. If you add a Matérn
  deviation, consider fixing the trend's smoothing rather than fitting it.
* **A Matérn and the measurement noise.** A length scale below the sampling
  interval *is* white noise, so `fit` floors it at the typical gap between
  visits. Above the floor the two can still trade: when readings carry
  correlated day-to-day variation, the likelihood can prefer a large Matérn
  and a noise level near zero. `fit` searches again from a larger noise share
  when a Matérn's variance ends at the top of its bracket and keeps the better
  optimum, and `warnings` says when the variance is still there; a
  `minimumMeasurementVariance` at the instrument's resolution rules the corner
  out.
* **A cycle and a free level.** A random walk with a free variance can follow any
  wiggle, so a `StochasticCycle` beside an unconstrained `LocalLevel` often
  loses. It fails in two ways and `warnings` reports both: on a series with a
  real cycle the level's variance runs to the top of its bracket and interpolates
  the data, and on a series without one the level shrinks away while the cycle's
  variance and period come back flat over eight and two decades. In both cases
  the fit has not determined anything.

## When to stop

Stop when adding a component no longer reduces the fitted noise level, the
Ljung–Box p-value is unremarkable, and `warnings` is empty. A model in which
every component has a clear job is more useful than one that fits slightly
better but cannot say which part did the work.

## A worked comparison on real data

[`tool/calibration/`](https://github.com/QuantumPhysique/state_space/tree/main/tool/calibration)
in the GitHub repository fits several candidate models to a weight diary (a
real export, or four synthetic ones) and prints the fitted smoothing, how well
it is determined, the noise level and a Ljung–Box test, for the whole history
and at several earlier lengths. It is not part of the published package; run it
from a clone of the repository:

```sh
dart run tool/calibration/calibrate.dart --all
dart run tool/calibration/calibrate.dart --all my-export.txt
```

Everything runs locally. [Its README](https://github.com/QuantumPhysique/state_space/blob/main/tool/calibration/README.md)
explains how to read the tables, including why the likelihood column can only
be compared between rows of equal diffuse dimension.
