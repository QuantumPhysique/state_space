import 'dart:math' as math;
import 'dart:typed_data';

import '../component.dart';
import '../engine/matrix_block.dart';

/// A smoothly varying level whose rate of change is a Wiener process.
///
/// The state is `(mu, nu)`: level and slope. The slope is driven by white
/// noise of intensity [processVariance], the level is its integral:
///
/// ```text
/// d(mu) = nu dt
/// d(nu) = sigma dB,   E[dB^2] = dt
/// ```
///
/// Discretised exactly over a gap `dt`:
///
/// ```text
/// A(dt) = [[1, dt],       Q(dt) = sigma^2 [[dt^3/3, dt^2/2],
///          [0,  1]]                       [dt^2/2, dt    ]]
/// ```
///
/// The off-diagonal term in `Q` is the part a discrete local linear trend
/// throws away: over a gap, uncertainty about the slope integrates into
/// uncertainty about the level, and the two end up correlated. Keeping it is
/// what makes the model exact for irregular gaps rather than approximately
/// right for unit steps.
///
/// The implied Gaussian process prior is the cubic spline kernel
/// `k(t, t') = sigma^2 (m^3/3 + m^2 |t - t'| / 2)` with `m = min(t, t')`, so
/// the posterior mean is a natural cubic smoothing spline with smoothing
/// parameter `lambda = measurementVariance / processVariance` (Wahba 1978).
///
/// {@category Components}
class LocalLinearTrend extends Component {
  const LocalLinearTrend({required this.processVariance})
      : assert(processVariance > 0, 'processVariance must be positive');

  /// Intensity of the white noise driving the slope, in squared signal units
  /// per cubed time unit. Larger values buy a more responsive trend.
  final double processVariance;

  @override
  int get stateDim => 2;

  @override
  int get parameterCount => 1;

  @override
  void transition(double dt, MatrixBlock out) {
    out.set(0, 0, 1);
    out.set(0, 1, dt);
    out.set(1, 0, 0);
    out.set(1, 1, 1);
  }

  @override
  void processNoise(double dt, MatrixBlock out) {
    final dt2 = dt * dt;
    final covariance = processVariance * dt2 / 2;
    out.set(0, 0, processVariance * dt2 * dt / 3);
    out.set(0, 1, covariance);
    out.set(1, 0, covariance);
    out.set(1, 1, processVariance * dt);
  }

  @override
  void observationAt(double time, Float64List out) {
    out[0] = 1;
    out[1] = 0;
  }

  /// The path spread of integrated Brownian motion works out at
  /// `sigma^2 T^3 / 30`, against `sigma^2 T^3 / 3` for the variance it
  /// reaches at the end of the window: a factor of ten, because most of the
  /// terminal variance is accumulated near the end and the average over the
  /// path never sees it.
  @override
  double wanderOver(double span) =>
      math.sqrt(processVariance * span * span * span / 30);

  @override
  List<bool> get diffuseStates => const [true, true];

  @override
  void properPrior(Float64List mean, MatrixBlock covariance) {
    // Both states are diffuse; nothing to contribute.
  }

  @override
  Float64List get parameters =>
      Float64List.fromList([math.log(processVariance)]);

  @override
  Component withParameters(Float64List theta) =>
      LocalLinearTrend(processVariance: math.exp(theta[0]));

  @override
  int? get rateStateIndex => 1;

  @override
  String toString() => 'LocalLinearTrend(processVariance: $processVariance)';
}
