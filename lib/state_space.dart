/// Exact Gaussian process regression for irregular time series, in linear
/// time.
///
/// The package computes one object under three descriptions, and its whole
/// value is that they agree:
///
/// 1. **A Gaussian process.** `f ~ GP(0, k)` observed with noise. The
///    posterior mean and variance are the textbook expressions involving
///    `(K + sigma^2 I)^-1`, which cost `O(N^3)` time and `O(N^2)` memory.
/// 2. **A stochastic differential equation.** For Markovian kernels the same
///    prior is a linear SDE, discretised *exactly* over an arbitrary gap.
/// 3. **A Kalman filter and RTS smoother.** Which computes the same posterior
///    and the same log marginal likelihood in `O(N)` time and memory.
///
/// For the default [LocalLinearTrend] the implied kernel is the cubic spline
/// kernel, so the posterior mean is the natural cubic smoothing spline — as a
/// Bayesian posterior, with honest uncertainty, computed in linear time.
///
/// ```dart
/// final data = [
///   Observation(0, 81.2),
///   Observation(1, 80.9),
///   Observation(4, 80.4),   // gaps are not a special case
///   Observation(4, 80.6),   // neither are duplicate times
/// ];
///
/// final result = fit(
///   StructuralModel.localLinearTrend(processVariance: 1e-3),
///   data,
/// );
/// final posterior = result.model.smooth(data);
/// print(posterior.level);              // the trend
/// print(posterior.credibleInterval(0)) // and how sure it is
/// ```
///
/// Missing data needs no filling, irregular sampling needs no resampling, and
/// two readings at the same instant need no averaging: all three fall out of
/// the recursion.
///
/// ## Documentation
///
/// * [Getting started](https://github.com/QuantumPhysique/state_space/blob/main/doc/getting-started.md)
///   — grids, slopes, forecasts, unequal readings, noise floors, isolates
/// * [Choosing a model](https://github.com/QuantumPhysique/state_space/blob/main/doc/choosing-a-model.md)
///   — which components, and how to tell whether one earned its place
/// * [Components](https://github.com/QuantumPhysique/state_space/blob/main/doc/components.md)
///   — what each one is, and which kernels are reachable
/// * [How it works](https://github.com/QuantumPhysique/state_space/blob/main/doc/how-it-works.md)
///   — the Gaussian process, the SDE, the filter, and the numerical choices
/// * [Validation](https://github.com/QuantumPhysique/state_space/blob/main/doc/validation.md)
///   — what is checked, against what, and how closely
library;

export 'src/component.dart';
export 'src/components/local_level.dart';
export 'src/components/local_linear_trend.dart';
export 'src/components/matern.dart';
export 'src/components/regression.dart';
export 'src/components/stochastic_cycle.dart';
export 'src/components/trigonometric_seasonal.dart';
export 'src/diagnostics.dart';
export 'src/engine/matrix_block.dart' show MatrixBlock;
export 'src/fit/fit.dart' show fit, SearchStart, samplingResolution;
export 'src/fit/penalty.dart';
export 'src/initialization.dart';
export 'src/model.dart';
export 'src/observation.dart';
export 'src/parameter_spec.dart';
export 'src/result.dart';
