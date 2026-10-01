## Unreleased

- `DampedLinearTrend` and `StructuralModel.dampedLinearTrend`: a trend whose
  slope reverts to zero over a time scale, so that the curve straightens across
  long gaps and a forecast levels off. The level is diffuse and the slope
  starts from its stationary distribution, so one reading is enough.

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
