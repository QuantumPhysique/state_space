import 'dart:math' as math;
import 'dart:typed_data';

import '../component.dart';
import '../engine/matrix_block.dart';
import '../exceptions.dart';
import '../parameter_spec.dart';

/// A repeating pattern of fixed [period] whose shape drifts over time.
///
/// The pattern is carried by [harmonics] sinusoids at frequencies
/// `lambda_j = 2 pi j / period`, each held in a pair of states `(gamma_j,
/// gamma_j*)` that rotate into one another as time passes:
///
/// ```text
/// A_j(dt) = [[ cos(lambda_j dt), sin(lambda_j dt)],
///            [-sin(lambda_j dt), cos(lambda_j dt)]]
///
/// Q_j(dt) = processVariance * dt * I
/// ```
///
/// The observation reads `sum_j gamma_j`. The starred states are never
/// observed directly; they exist because a sinusoid needs two numbers to
/// describe where it is in its cycle, and the rotation is what moves phase
/// into amplitude and back.
///
/// Dummy-variable seasonality has no `A(dt)` for a non-integer gap; the
/// rotation form does, and reduces to the familiar recursion on evenly spaced
/// data.
///
/// Without process noise the component is an ordinary Fourier series with
/// unknown coefficients — rigid, repeating forever. [processVariance] is what
/// lets it breathe: it is the rate at which the pattern is allowed to change
/// shape, in squared signal units per time unit. All harmonics share it
/// (Harvey's specification), so the component has one free parameter.
///
/// The rotation is orthogonal and the noise isotropic, so the implied
/// Gaussian process kernel is
///
/// ```text
/// k(t, t') = processVariance * min(t, t') * sum_j cos(lambda_j (t - t'))
/// ```
///
/// — Brownian motion multiplied by a cosine comb.
///
/// Averaged over a full period each harmonic integrates to zero, so the
/// component carries no level of its own. Over a stretch of data shorter than
/// a period it is still confounded with a trend.
///
/// {@category Components}
final class TrigonometricSeasonal extends Component {
  /// A pattern repeating every [period] time units, resolved by [harmonics]
  /// sinusoids.
  ///
  /// Weekly data on a daily time base means `period: 7`; two or three
  /// harmonics is usually plenty. [period] and [processVariance] must be
  /// finite and positive and [harmonics] at least one.
  ///
  /// The highest harmonic must also sit below the Nyquist frequency of the
  /// data, `2 * harmonics * gap < period` for readings `gap` apart. That
  /// depends on the data, so [fit] checks it (and throws
  /// [UnderdeterminedModelException]) rather than the constructor.
  TrigonometricSeasonal({
    required this.period,
    required this.harmonics,
    required this.processVariance,
  }) {
    if (!(period > 0) || !period.isFinite) {
      throw ArgumentError.value(
        period,
        'period',
        'must be finite and positive',
      );
    }
    if (harmonics < 1) {
      throw ArgumentError.value(harmonics, 'harmonics', 'must be at least 1');
    }
    if (!(processVariance > 0) || !processVariance.isFinite) {
      throw ArgumentError.value(
        processVariance,
        'processVariance',
        'must be finite and positive',
      );
    }
  }

  @override
  String get name => 'TrigonometricSeasonal';

  /// Length of one full cycle, in the caller's time unit.
  final double period;

  /// How many sinusoids describe the pattern. The component holds two states
  /// per harmonic.
  final int harmonics;

  /// Rate at which the pattern is allowed to change shape, in squared signal
  /// units per time unit. Zero would make the seasonal shape fixed forever;
  /// large values let it be rewritten between one cycle and the next.
  final double processVariance;

  @override
  int get stateDim => 2 * harmonics;

  @override
  int get parameterCount => 1;

  /// Angular frequency of harmonic [j], counting from one.
  double frequency(int j) => 2 * math.pi * j / period;

  @override
  void transition(double dt, MatrixBlock out) {
    out.fill(0);
    for (var j = 1; j <= harmonics; j++) {
      final angle = frequency(j) * dt;
      final c = math.cos(angle);
      final s = math.sin(angle);
      final at = 2 * (j - 1);
      out.set(at, at, c);
      out.set(at, at + 1, s);
      out.set(at + 1, at, -s);
      out.set(at + 1, at + 1, c);
    }
  }

  @override
  void processNoise(double dt, MatrixBlock out) {
    out.fill(0);
    final variance = processVariance * dt;
    for (var i = 0; i < stateDim; i++) {
      out.set(i, i, variance);
    }
  }

  @override
  void observationAt(double time, Float64List out) {
    for (var j = 0; j < harmonics; j++) {
      out[2 * j] = 1;
      out[2 * j + 1] = 0;
    }
  }

  /// `processVariance * span * harmonics / 2`: the harmonics add, so a
  /// pattern resolved by three sinusoids drifts faster than one resolved by a
  /// single sinusoid at the same noise intensity.
  ///
  /// The exact figure subtracts a further
  /// `2 sigma^2 sum_j (T - sin(lambda_j T) / lambda_j) / (lambda_j^2 T^2)`,
  /// which is 5.5 per cent of the total over one period at one harmonic, under
  /// 1.3 per cent by two periods and under 0.2 per cent by five; it is left
  /// out.
  @override
  double wanderOver(double span) =>
      math.sqrt(processVariance * span * harmonics / 2);

  @override
  List<bool> get diffuseStates => List.filled(stateDim, true);

  @override
  void properPrior(Float64List mean, MatrixBlock covariance) {
    // Every state is diffuse: where the pattern sits in its cycle, and how
    // large it is, are both for the data to say.
  }

  @override
  String? identifiabilityHint(double from, double to, {double resolution = 0}) {
    final aliasing = _aliasing(resolution);
    if (aliasing != null) return aliasing;
    final span = to - from;
    if (span >= period) return null;
    return 'a seasonal component of period $period was given a series '
        'spanning only $span, so it has not been round once and over that '
        'stretch looks much like a constant with a slope';
  }

  /// Refuses a harmonic at or past the Nyquist frequency of data sampled
  /// every [resolution] time units, where it aliases onto a lower harmonic or
  /// leaves a state the data can never see.
  @override
  List<ParameterSpec> parameterSpecsAt({required double resolution}) {
    final aliasing = _aliasing(resolution);
    if (aliasing != null) throw UnderdeterminedModelException(aliasing);
    return parameterSpecs;
  }

  String? _aliasing(double resolution) {
    if (!(resolution > 0) || 2 * harmonics * resolution < period) return null;
    return 'harmonic $harmonics of a seasonal component of period $period '
        'sits at or past the Nyquist frequency of readings $resolution apart, '
        'so it cannot be told apart from a lower harmonic; use fewer than '
        '${(period / (2 * resolution)).ceil()} harmonics';
  }

  @override
  Float64List get parameters =>
      Float64List.fromList([math.log(processVariance)]);

  @override
  Component withParameters(Float64List theta) => TrigonometricSeasonal(
    period: period,
    harmonics: harmonics,
    processVariance: math.exp(theta[0]),
  );

  @override
  bool operator ==(Object other) =>
      other is TrigonometricSeasonal &&
      other.period == period &&
      other.harmonics == harmonics &&
      other.processVariance == processVariance;

  @override
  int get hashCode =>
      Object.hash(TrigonometricSeasonal, period, harmonics, processVariance);

  @override
  String toString() =>
      'TrigonometricSeasonal(period: $period, harmonics: '
      '$harmonics, processVariance: $processVariance)';
}
