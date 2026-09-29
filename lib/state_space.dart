/// Exact Gaussian process regression for irregular time series, in linear
/// time.
///
/// The prior is a Gaussian process with a Markovian kernel, written as a
/// linear SDE and discretised exactly over each gap, so a Kalman filter and
/// RTS smoother compute the same posterior and log marginal likelihood as the
/// `O(N^3)` textbook expressions in `O(N)`. With the default
/// [LocalLinearTrend] the posterior mean is a natural cubic smoothing spline.
///
/// ```dart
/// final data = [
///   Observation(0, 81.3),
///   Observation(2, 81.0),   // gaps are not a special case
///   Observation(3, 80.5),
///   Observation(7, 80.0),
///   Observation(8, 80.0),
///   Observation(8, 79.8),   // neither are duplicate times
///   // ... four weeks of readings in all
/// ];
///
/// final fitted = fit(
///   StructuralModel.localLinearTrend(processVariance: 1e-3),
///   data,
/// );
/// print(fitted.warnings);               // empty when the fit is determined
/// final posterior = fitted.model.smooth(data);
/// print(posterior.mean);                // the trend
/// print(posterior.credibleInterval(0)); // and how sure it is
/// ```
///
/// Missing data needs no filling, irregular sampling needs no resampling, and
/// two readings at the same instant need no averaging.
///
/// ## Documentation
///
/// * [Getting started](https://github.com/QuantumPhysique/state_space/blob/main/doc/getting-started.md):
///   grids, slopes, forecasts, unequal readings, noise floors, isolates
/// * [Choosing a model](https://github.com/QuantumPhysique/state_space/blob/main/doc/choosing-a-model.md):
///   which components, and how to compare them
/// * [Components](https://github.com/QuantumPhysique/state_space/blob/main/doc/components.md):
///   what each one is, and writing your own
/// * [How it works](https://github.com/QuantumPhysique/state_space/blob/main/doc/how-it-works.md):
///   the filter, the smoother and the fit
/// * [Validation](https://github.com/QuantumPhysique/state_space/blob/main/doc/validation.md):
///   what is checked, against what, and how closely
library;

export 'src/component.dart';
export 'src/components/local_level.dart';
export 'src/components/local_linear_trend.dart';
export 'src/components/matern.dart';
export 'src/components/regression.dart';
export 'src/components/stochastic_cycle.dart';
export 'src/components/trigonometric_seasonal.dart';
export 'src/diagnostics.dart';
export 'src/exceptions.dart';
export 'src/fit/fit.dart' show fit, SearchStart;
export 'src/fit/penalty.dart';
export 'src/initialization.dart';
export 'src/model.dart';
export 'src/observation.dart';
export 'src/parameter_spec.dart';
export 'src/result.dart'
    hide newCoefficient, newFitResult, newForecastResult, newSmoothingResult;
export 'src/time_axis.dart';
