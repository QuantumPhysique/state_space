## 0.2.0

A month without stepping on the scale shouldn't turn into a forecast that runs
off the chart, at least not in [trale](https://github.com/QuantumPhysique/trale),
the weight diary I write this package for. This release adds a trend that
levels off when the readings stop.

- `DampedLinearTrend` and `StructuralModel.dampedLinearTrend`: a trend whose
  slope reverts to zero over `timeScale`. Across a gap several time scales
  long the curve is close to a straight line between its ends, and a forecast
  levels off at `level + timeScale * slope`, with a variance growing linearly
  in the horizon rather than like its cube.
- A `DampedLinearTrend` gives a flat line from a single reading, and its
  likelihood can be compared with a `LocalLevel`'s but not with a
  `LocalLinearTrend`'s.
- `fit` estimates `timeScale` along with the process variance, within
  `timeScaleBounds` and never below the typical gap between readings.

## 0.1.0

First release.

- Exact Gaussian process regression for irregular time series in linear time:
  a Kalman filter and RTS smoother over any gaps, repeated timestamps and
  missing data, reporting the posterior at the observation times or on any
  output grid.
- Components: `LocalLevel`, `LocalLinearTrend` (a natural cubic smoothing
  spline), `TrigonometricSeasonal`, `RegressionComponent` with indicator and
  step regressors, `Matern` at ν = 1/2, 3/2 and 5/2, and `StochasticCycle`.
- `fit`: maximum marginal likelihood with the noise level concentrated out, a
  bracket scan before local search, plateau widths per parameter, a
  measurement-variance floor, warm starts, and warnings in sentences for
  bounds, flat directions and bad readings.
- `StructuralModel.withEstimatedScale` for smoothing chosen by the caller, with
  the noise level estimated from the data.
- `forecast`, credible and predictive intervals and bands, per-component means
  and slopes, and regression coefficients with standard errors.
- Exact diffuse initialisation by default, refusing with
  `UnderdeterminedModelException` when the data does not determine the model.
- Innovation diagnostics: recursive residuals, autocorrelation and Ljung–Box.
- `TimeAxis` for converting calendar dates to days and back across changes of
  clock.
- `package:state_space/authoring.dart` for writing components, with
  `checkComponent`.
- Validated against a dense `O(N³)` Gaussian process, generalised least squares,
  closed-form kernels and statsmodels fixtures; see `doc/validation.md`.
