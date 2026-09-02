import 'dart:math' as math;
import 'dart:typed_data';

import '../component.dart';
import '../engine/matrix_block.dart';
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
/// cos(2 pi tau / period)` — a cosine that fades. It is the quasi-periodic
/// kernel of the Gaussian process literature, and it says something a
/// `TrigonometricSeasonal` cannot: that the rhythm is *approximate*. A seasonal
/// of period 7 insists that Tuesdays are seven days apart forever. A cycle of
/// period 7 says the series tends to come back round after about a week, drifts
/// out of step, and finds its way back — which is what a physiological rhythm
/// actually does.
///
/// Being stationary it needs no diffuse states, and where it sits in its cycle
/// at the start of the data comes from the stationary prior rather than from
/// the first few readings.
///
/// **The period is the hard part, and it is worth understanding why.** Unlike
/// every other parameter in the package the likelihood in [period] is
/// multimodal: a cycle fitted at half the true period, or twice it, explains a
/// good deal of the same variation and sits on its own local maximum. A local
/// optimiser started in the wrong basin will converge confidently to the wrong
/// answer. [fit] handles this by scanning the period axis far more finely than
/// the others before it refines anything — see [ParameterSpec] — which is what
/// [periodScanPoints] controls.
///
/// **And it needs a lot of data.** Below about four complete cycles the period
/// is not estimable at all, and below eight it is very noisy; the fit will
/// still return a number.
///
/// **Check the damping first, and only then the period.** The obvious check —
/// `FitResult.plateauDecadesByParameter` on the period axis — is the right one
/// only once the damping has an interior optimum, and it is actively
/// misleading otherwise. On a series with no cycle in it the damping is driven
/// to the top of its bracket, where the component is a rigid sinusoid whose
/// likelihood in frequency is as sharp as a periodogram spike; the period width
/// then comes back at a thousandth of a decade while the answer means nothing.
/// Measured over five runs of four hundred points of white noise, the fitted
/// period ranged from 2.2 to 10.4 and the reported width from 0.001 to 0.025
/// decades. Every width here is conditional on the other parameters, and this
/// is the case where that caveat does the most work.
///
/// `FitResult.warnings` says both things in words: that the damping finished on
/// a bound, and that the period's width was measured there.
///
/// {@category Components}
class StochasticCycle extends Component {
  /// Validates rather than asserts, for the same reason
  /// `TrigonometricSeasonal` does: every failure mode here produces a
  /// plausible-looking number rather than an obvious crash.
  StochasticCycle({
    required this.period,
    required this.damping,
    required this.stationaryVariance,
    this.periodBounds = const (lower: 2.0, upper: 400.0),
    this.dampingBounds = const (lower: 0.05, upper: 0.9999),
    this.periodScanPoints = 120,
  }) {
    if (!(period > 0) || !period.isFinite) {
      throw ArgumentError.value(
          period, 'period', 'must be finite and positive');
    }
    if (!(damping > 0) || !(damping < 1)) {
      throw ArgumentError.value(
          damping,
          'damping',
          'must lie strictly between 0 and 1: at 1 the cycle never fades and '
              'is not stationary, and at 0 it has no memory to oscillate with');
    }
    if (!(stationaryVariance > 0) || !stationaryVariance.isFinite) {
      throw ArgumentError.value(stationaryVariance, 'stationaryVariance',
          'must be finite and positive');
    }
    if (!(periodBounds.lower > 0) ||
        !(periodBounds.lower < periodBounds.upper)) {
      throw ArgumentError.value(
          periodBounds, 'periodBounds', 'must be a positive increasing range');
    }
    if (!(dampingBounds.lower > 0) ||
        !(dampingBounds.lower < dampingBounds.upper) ||
        !(dampingBounds.upper < 1)) {
      throw ArgumentError.value(dampingBounds, 'dampingBounds',
          'must be an increasing range strictly inside (0, 1)');
    }
    if (periodScanPoints < 3) {
      throw ArgumentError.value(
          periodScanPoints, 'periodScanPoints', 'must be at least 3');
    }
  }

  /// Length of one turn of the cycle, in the caller's time unit.
  ///
  /// A period shorter than twice the typical gap between readings is not a
  /// cycle the data can see, whatever the likelihood reports. The component
  /// alone cannot refuse one, because it does not know the sampling interval;
  /// [fit] does, and raises the bottom of [periodBounds] to it through
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
  /// this resolves a peak from eight cycles onward — which is about where the
  /// period becomes estimable in the first place.
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

  /// Raises the period bracket to the Nyquist limit of the actual sampling.
  ///
  /// The class documentation notes that nothing here knows the sampling
  /// interval, so nothing here can refuse a period below Nyquist. [fit] does
  /// know, and passes it in: a cycle of period shorter than twice the gap
  /// between readings is aliased onto a longer one and is not a rhythm the
  /// data can see, whatever the likelihood reports.
  @override
  List<ParameterSpec> parameterSpecsAt({required double resolution}) {
    if (!(resolution > 0)) return parameterSpecs;
    final floor = math.max(periodBounds.lower, 2 * resolution);
    final upper = periodBounds.upper;
    if (!(floor < upper)) {
      throw ArgumentError('a StochasticCycle period is bracketed at '
          '[${periodBounds.lower}, $upper], but the readings are $resolution '
          'apart, so nothing shorter than ${2 * resolution} is above the '
          'Nyquist limit. Widen periodBounds or drop the component.');
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
  String toString() => 'StochasticCycle(period: $period, damping: $damping, '
      'stationaryVariance: $stationaryVariance)';
}

double _logit(double p) => math.log(p / (1 - p));

double _logistic(double x) => 1 / (1 + math.exp(-x));
