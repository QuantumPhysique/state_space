import 'dart:math' as math;
import 'dart:typed_data';

import '../arguments.dart';
import '../component.dart';
import '../engine/matrix_block.dart';
import '../parameter_spec.dart';

/// A smoothly varying level whose rate of change is pulled back towards zero:
/// the damped trend, or integrated Ornstein-Uhlenbeck process.
///
/// The state is `(mu, nu)`: level and slope. The slope reverts to zero over
/// [timeScale] and is driven by white noise of intensity [processVariance];
/// the level is its integral:
///
/// ```text
/// d(mu) = nu dt
/// d(nu) = -nu / tau dt + sigma dB,   E[dB^2] = dt
/// ```
///
/// Discretised exactly over a gap `dt`, with `phi = exp(-dt / tau)`:
///
/// ```text
/// A(dt) = [[1, tau (1 - phi)],
///          [0, phi          ]]
///
/// Q(dt) = sigma^2 tau^2 [[dt - 2 tau (1 - phi) + tau (1 - phi^2) / 2, (1 - phi)^2 / 2      ],
///                        [(1 - phi)^2 / 2,                            (1 - phi^2) / (2 tau)]]
/// ```
///
/// Over spans much shorter than [timeScale] this is a [LocalLinearTrend] with
/// the same [processVariance]. Over much longer spans the level moves like a
/// [LocalLevel] of variance `processVariance * timeScale^2`. So across a gap
/// several time scales long the posterior mean bends only near the edges and
/// is otherwise close to a straight line between them, and a forecast
/// continues the current slope for about [timeScale] and then levels off at
/// `level + timeScale * slope`, with a variance growing linearly in the
/// horizon rather than like its cube.
///
/// The level is diffuse. The slope has its stationary prior, mean zero and
/// variance [stationarySlopeVariance], so the component has one flat
/// direction: a single reading determines it, and a likelihood is comparable
/// with a [LocalLevel]'s but not with a [LocalLinearTrend]'s.
///
/// The pull towards zero acts on a sustained slope too, at the ends of the
/// series. Between readings on both sides a steady slope is recovered, but at
/// the last reading it comes out at 0.77, 0.88, 0.92 and 0.96 of its value
/// for a [timeScale] of 5, 10, 15 and 30 smoothing bandwidths, the bandwidth
/// being `(measurementVariance / processVariance)^(1/4)` for one reading per
/// time unit (Silverman 1984).
///
/// Two parameters, and only one of them is a variance: see [parameterSpecs].
/// [timeScale] is in the caller's time unit, so the bracket it is searched
/// over depends on that unit and can be set with [timeScaleBounds].
///
/// {@category Components}
final class DampedLinearTrend extends Component {
  /// A trend whose slope is driven by white noise of intensity
  /// [processVariance] and reverts to zero over [timeScale].
  ///
  /// [processVariance] and [timeScale] must be finite and positive, and
  /// [timeScaleBounds] a positive increasing range: the bracket [fit] searches
  /// the time scale over.
  DampedLinearTrend({
    required this.processVariance,
    required this.timeScale,
    this.timeScaleBounds = const (lower: 1e-2, upper: 1e4),
  }) {
    checkPositive(processVariance, 'processVariance');
    checkPositive(timeScale, 'timeScale');
    final (:lower, :upper) = timeScaleBounds;
    if (!(lower > 0) || !(lower < upper) || !upper.isFinite) {
      throw ArgumentError.value(
        timeScaleBounds,
        'timeScaleBounds',
        'must be a positive increasing range',
      );
    }
  }

  @override
  String get name => 'DampedLinearTrend';

  /// Intensity of the white noise driving the slope, in squared signal units
  /// per cubed time unit, as for [LocalLinearTrend].
  final double processVariance;

  /// Time over which the slope reverts to zero, in the caller's time unit:
  /// after a gap `dt` the expected slope is `exp(-dt / timeScale)` times what
  /// it was.
  final double timeScale;

  /// Range the fit searches [timeScale] over, in the caller's time unit.
  ///
  /// The default spans six decades, which for daily data is a quarter of an
  /// hour to twenty-seven years.
  final ({double lower, double upper}) timeScaleBounds;

  /// Variance of the slope in the long run, `processVariance * timeScale / 2`,
  /// in squared signal units per squared time unit. It is also the prior on
  /// the slope at the first reading.
  double get stationarySlopeVariance => processVariance * timeScale / 2;

  @override
  int get stateDim => 2;

  @override
  int get parameterCount => 2;

  @override
  List<ParameterSpec> get parameterSpecs => [
    const VarianceParameter(),
    ShapeParameter(
      label: 'time scale',
      lower: math.log(timeScaleBounds.lower),
      upper: math.log(timeScaleBounds.upper),
    ),
  ];

  @override
  void transition(double dt, MatrixBlock out) {
    final u = dt / timeScale;
    out.set(0, 0, 1);
    out.set(0, 1, -timeScale * _expm1(-u));
    out.set(1, 0, 0);
    out.set(1, 1, math.exp(-u));
  }

  /// The level entry is `sigma^2 tau^3 (u - 2 (1 - phi) + (1 - phi^2) / 2)`
  /// with `u = dt / tau`: terms of order `u` that cancel to a result of order
  /// `u^3`. Below [_seriesBelow] it is summed as a power series instead, which
  /// is what keeps it right for a gap far shorter than the time scale, and the
  /// two `1 - phi` factors come from an `expm1` for the same reason.
  @override
  void processNoise(double dt, MatrixBlock out) {
    final tau = timeScale;
    final u = dt / tau;
    final decayed = -_expm1(-u);
    final decayedTwice = -_expm1(-2 * u);
    final level = u < _seriesBelow
        ? _levelSeries(u)
        : u - 2 * decayed + decayedTwice / 2;
    final covariance = processVariance * tau * tau * decayed * decayed / 2;
    out.set(0, 0, processVariance * tau * tau * tau * level);
    out.set(0, 1, covariance);
    out.set(1, 0, covariance);
    out.set(1, 1, processVariance * tau * decayedTwice / 2);
  }

  @override
  void observationAt(double time, Float64List out) {
    out[0] = 1;
    out[1] = 0;
  }

  /// The level is diffuse; the slope has its stationary distribution.
  @override
  List<bool> get diffuseStates => const [true, false];

  @override
  void properPrior(Float64List mean, MatrixBlock covariance) {
    mean[0] = 0;
    mean[1] = 0;
    covariance.fill(0);
    covariance.set(1, 1, stationarySlopeVariance);
  }

  /// The path spread with the slope's stationary prior included, as a
  /// [Matern]'s includes its stationary variance:
  ///
  /// ```text
  /// wander^2 = sigma^2 tau^2 T (e^-x - 1 + x - x^2/2 + x^3/6) / x^3,   x = T / tau
  /// ```
  ///
  /// Over a window much shorter than [timeScale] that is a straight line of
  /// random slope, `sigma^2 tau T^2 / 24`, and over a much longer one a random
  /// walk, `sigma^2 tau^2 T / 6`. Unlike [LocalLinearTrend]'s, it grows without
  /// bound as [timeScale] does, because so does the slope's prior.
  @override
  double wanderOver(double span) {
    if (!(span > 0)) return 0;
    final tau = timeScale;
    final x = span / tau;
    return math.sqrt(processVariance * tau * tau * span * _wanderShape(x));
  }

  @override
  int? get rateStateIndex => 1;

  @override
  Float64List get parameters =>
      Float64List.fromList([math.log(processVariance), math.log(timeScale)]);

  @override
  Component withParameters(Float64List theta) => DampedLinearTrend(
    processVariance: math.exp(theta[0]),
    timeScale: math.exp(theta[1]),
    timeScaleBounds: timeScaleBounds,
  );

  @override
  bool operator ==(Object other) =>
      other is DampedLinearTrend &&
      other.processVariance == processVariance &&
      other.timeScale == timeScale &&
      other.timeScaleBounds == timeScaleBounds;

  @override
  int get hashCode => Object.hash(
    DampedLinearTrend,
    processVariance,
    timeScale,
    timeScaleBounds,
  );

  @override
  String toString() =>
      'DampedLinearTrend(processVariance: $processVariance, '
      'timeScale: $timeScale)';
}

/// Where the closed form of the level noise stops being the better of the two.
///
/// The closed form loses about `log10(3 / u^2)` digits to cancellation, under
/// two here; the series converges like `(2u)^n / n!`, in under twenty terms.
const double _seriesBelow = 0.5;

/// `u - 2 (1 - e^-u) + (1 - e^-2u) / 2`, which is
/// `integral_0^u (1 - e^-s)^2 ds`, as its power series
/// `sum_{n >= 2} (-1)^n (2^n - 2) u^(n+1) / (n+1)!`.
double _levelSeries(double u) {
  // u^(n+1) / (n+1)!, with the sign (-1)^n folded in, and 2^n, both at n = 2.
  var power = u * u * u / 6;
  var twos = 4.0;
  var sum = (twos - 2) * power;
  for (var n = 3; n <= 40; n++) {
    power *= -u / (n + 1);
    twos *= 2;
    final term = (twos - 2) * power;
    sum += term;
    if (term.abs() <= sum.abs() * 1e-17) break;
  }
  return sum;
}

/// `(e^-x - 1 + x - x^2/2 + x^3/6) / x^3`, the shape of [DampedLinearTrend]'s
/// squared wander; `x / 24` for small `x` and `1 / 6` for large.
///
/// The bracket is the tail of the exponential series from the fourth power
/// on, so for small `x` it is summed as that tail rather than computed as a
/// difference of terms of order one.
double _wanderShape(double x) {
  if (x >= 2) {
    final x2 = x * x;
    return (math.exp(-x) - 1 + x - x2 / 2 + x2 * x / 6) / (x2 * x);
  }
  // sum_{n >= 4} (-x)^n / n!, divided by x^3: start at x / 24.
  var term = x / 24;
  var sum = term;
  for (var n = 5; n <= 40; n++) {
    term *= -x / n;
    sum += term;
    if (term.abs() <= sum.abs() * 1e-17) break;
  }
  return sum;
}

/// `exp(x) - 1`, accurate near zero, where computing it that way leaves only
/// the rounding error of `exp(x)`.
///
/// Kahan's form: the same rounding error appears in `exp(x) - 1` and in
/// `log(exp(x))`, so it cancels in their ratio.
double _expm1(double x) {
  final e = math.exp(x);
  if (e == 1.0) return x;
  final shifted = e - 1.0;
  if (shifted == -1.0) return -1.0;
  return shifted * x / math.log(e);
}
