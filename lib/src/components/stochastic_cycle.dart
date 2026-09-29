import 'dart:math' as math;
import 'dart:typed_data';

import '../arguments.dart';
import '../component.dart';
import '../engine/matrix_block.dart';
import '../exceptions.dart';
import '../parameter_spec.dart';
import 'stationary.dart';

/// A damped oscillation whose period is estimated from the data.
///
/// ```text
/// A(dt) = damping^dt * [[ cos(lambda dt), sin(lambda dt)],
///                       [-sin(lambda dt), cos(lambda dt)]]
///
/// Q(dt) = stationaryVariance * (1 - damping^(2 dt)) * I,   lambda = 2 pi / period
/// ```
///
/// The implied covariance function is `stationaryVariance * damping^|tau| *
/// cos(2 pi tau / period)`, a cosine that fades: the quasi-periodic kernel of
/// the Gaussian process literature. Unlike a `TrigonometricSeasonal`, which
/// repeats exactly, the cycle drifts out of step and back, as a physiological
/// rhythm does.
///
/// Being stationary it needs no diffuse states; its starting phase comes from
/// the stationary prior.
///
/// The likelihood in [period] is multimodal, with secondary maxima at half and
/// twice the true period, so [fit] scans the period axis finely before
/// refining ([periodScanPoints]). Below about four complete cycles the period
/// is not estimable, and below eight it is noisy; the fit still returns a
/// number.
///
/// **Read the damping before the period.** On a series with no cycle in it the
/// damping goes to the top of its bracket, where the component is a rigid
/// sinusoid whose likelihood in frequency is as sharp as a periodogram spike,
/// and the period's plateau width comes back at a thousandth of a decade
/// while the period means nothing: on white noise the fitted period ranged
/// from 2.2 to 10.4. A period width is only meaningful once the damping is
/// [ParameterStatus.determined]. [FitResult.warnings] reports both.
///
/// {@category Components}
final class StochasticCycle extends Component {
  /// A cycle of the given [period], [damping] and [stationaryVariance].
  ///
  /// [period] and [stationaryVariance] must be finite and positive and
  /// [damping] strictly between 0 and 1. [periodBounds] and [dampingBounds]
  /// are the brackets [fit] searches over: a finite positive increasing range,
  /// and an increasing range inside `(0, 1)`.
  StochasticCycle({
    required this.period,
    required this.damping,
    required this.stationaryVariance,
    this.periodBounds = const (lower: 2.0, upper: 400.0),
    this.dampingBounds = const (lower: 0.05, upper: 0.9999),
    this.periodScanPoints = 120,
  }) {
    checkPositive(period, 'period');
    if (!(damping > 0) || !(damping < 1)) {
      throw ArgumentError.value(
        damping,
        'damping',
        'must lie strictly between 0 and 1: at 1 the cycle never fades and '
            'is not stationary, and at 0 it has no memory to oscillate with',
      );
    }
    checkPositive(stationaryVariance, 'stationaryVariance');
    if (!(periodBounds.lower > 0) ||
        !(periodBounds.lower < periodBounds.upper) ||
        !periodBounds.upper.isFinite) {
      throw ArgumentError.value(
        periodBounds,
        'periodBounds',
        'must be a finite positive increasing range',
      );
    }
    if (!(dampingBounds.lower > 0) ||
        !(dampingBounds.lower < dampingBounds.upper) ||
        !(dampingBounds.upper < 1)) {
      throw ArgumentError.value(
        dampingBounds,
        'dampingBounds',
        'must be an increasing range strictly inside (0, 1)',
      );
    }
    if (periodScanPoints < 3) {
      throw ArgumentError.value(
        periodScanPoints,
        'periodScanPoints',
        'must be at least 3',
      );
    }
  }

  @override
  String get name => 'StochasticCycle';

  /// Length of one turn of the cycle, in the caller's time unit.
  ///
  /// [fit] raises the bottom of [periodBounds] to twice the typical gap
  /// between readings, the shortest period the data can see; see
  /// [parameterSpecsAt].
  final double period;

  /// How much of the oscillation survives one time unit, in `(0, 1)`.
  ///
  /// `0.99` on a daily series leaves half the amplitude after about ten weeks.
  /// The limit `damping -> 1` is the boundary of stationarity, where the cycle
  /// becomes a rigid sinusoid with no stationary distribution at all; the
  /// bracket stops short of it, and a fit that ends there is telling you the
  /// data wants a `TrigonometricSeasonal` instead.
  final double damping;

  /// Variance of the cycle's contribution at any single moment.
  final double stationaryVariance;

  /// Range the fit searches [period] over, in the caller's time unit.
  final ({double lower, double upper}) periodBounds;

  /// Range the fit searches [damping] over.
  final ({double lower, double upper}) dampingBounds;

  /// Resolution of the coordinate scan on the period axis.
  ///
  /// The default places 120 points across [periodBounds], which over the
  /// default two-decade range is about five per cent apart. The peak in the
  /// period is roughly `1 / (2 * cycles observed)` wide in relative terms, so
  /// this resolves a peak from about eight cycles onward, which is also where
  /// the period starts to be estimable.
  final int periodScanPoints;

  /// Angular frequency, `2 pi / period`.
  double get frequency => 2 * math.pi / period;

  /// The covariance function, `k(tau)`: a damped cosine.
  double covariance(double lag) {
    final magnitude = lag.abs();
    return stationaryVariance *
        math.pow(damping, magnitude) *
        math.cos(frequency * magnitude);
  }

  @override
  int get stateDim => 2;

  @override
  int get parameterCount => 3;

  @override
  List<ParameterSpec> get parameterSpecs => [
    const VarianceParameter(),
    ShapeParameter(
      label: 'damping',
      // A logit, so a width in it is not a width in decades.
      isLogarithmic: false,
      lower: _logit(dampingBounds.lower),
      upper: _logit(dampingBounds.upper),
    ),
    ShapeParameter(
      label: 'period',
      lower: math.log(periodBounds.lower),
      upper: math.log(periodBounds.upper),
      scanPoints: periodScanPoints,
      // The fine scan is about multimodality, not about scale; without
      // this the simplex would step thirty times less far along the period
      // axis than along the variance ones.
      searchStep: 0.5,
    ),
  ];

  /// Raises the period bracket to the Nyquist limit of the sampling: a cycle
  /// shorter than twice the gap between readings aliases onto a longer one.
  @override
  List<ParameterSpec> parameterSpecsAt({required double resolution}) {
    if (!(resolution > 0)) return parameterSpecs;
    final floor = math.max(periodBounds.lower, 2 * resolution);
    final upper = periodBounds.upper;
    if (!(floor < upper)) {
      throw UnderdeterminedModelException(
        'a StochasticCycle period is bracketed at '
        '[${periodBounds.lower}, $upper], but the readings are $resolution '
        'apart, so nothing shorter than ${2 * resolution} is above the '
        'Nyquist limit. Widen periodBounds or drop the component.',
      );
    }
    final specs = parameterSpecs;
    return [
      specs[0],
      specs[1],
      ShapeParameter(
        label: 'period',
        lower: math.log(floor),
        upper: math.log(upper),
        scanPoints: periodScanPoints,
        searchStep: 0.5,
      ),
    ];
  }

  @override
  void transition(double dt, MatrixBlock out) {
    final decay = math.pow(damping, dt).toDouble();
    final angle = frequency * dt;
    final c = decay * math.cos(angle);
    final s = decay * math.sin(angle);
    out.set(0, 0, c);
    out.set(0, 1, s);
    out.set(1, 0, -s);
    out.set(1, 1, c);
  }

  /// `Q(dt) = P_inf - A(dt) P_inf A(dt)'`, which for an isotropic stationary
  /// covariance and an orthogonal rotation collapses to a scalar: the rotation
  /// preserves the covariance exactly, and only the damping loses any of it.
  @override
  void processNoise(double dt, MatrixBlock out) {
    final lost = stationaryVariance * (1 - math.pow(damping, 2 * dt));
    out.set(0, 0, lost);
    out.set(0, 1, 0);
    out.set(1, 0, 0);
    out.set(1, 1, lost);
  }

  @override
  void observationAt(double time, Float64List out) {
    out[0] = 1;
    out[1] = 0;
  }

  @override
  List<bool> get diffuseStates => const [false, false];

  @override
  void properPrior(Float64List mean, MatrixBlock covariance) {
    mean[0] = 0;
    mean[1] = 0;
    covariance.set(0, 0, stationaryVariance);
    covariance.set(0, 1, 0);
    covariance.set(1, 0, 0);
    covariance.set(1, 1, stationaryVariance);
  }

  @override
  double wanderOver(double span) => stationaryWander(covariance, span);

  @override
  Float64List get parameters => Float64List.fromList([
    math.log(stationaryVariance),
    _logit(damping),
    math.log(period),
  ]);

  @override
  Component withParameters(Float64List theta) => StochasticCycle(
    stationaryVariance: math.exp(theta[0]),
    damping: _logistic(theta[1]),
    period: math.exp(theta[2]),
    periodBounds: periodBounds,
    dampingBounds: dampingBounds,
    periodScanPoints: periodScanPoints,
  );

  @override
  bool operator ==(Object other) =>
      other is StochasticCycle &&
      other.period == period &&
      other.damping == damping &&
      other.stationaryVariance == stationaryVariance &&
      other.periodBounds == periodBounds &&
      other.dampingBounds == dampingBounds &&
      other.periodScanPoints == periodScanPoints;

  @override
  int get hashCode => Object.hash(
    StochasticCycle,
    period,
    damping,
    stationaryVariance,
    periodBounds,
    dampingBounds,
    periodScanPoints,
  );

  @override
  String toString() =>
      'StochasticCycle(period: $period, damping: $damping, '
      'stationaryVariance: $stationaryVariance)';
}

double _logit(double p) => math.log(p / (1 - p));

double _logistic(double x) => 1 / (1 + math.exp(-x));
