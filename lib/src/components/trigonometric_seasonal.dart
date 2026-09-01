import 'dart:math' as math;
import 'dart:typed_data';

import '../component.dart';
import '../engine/matrix_block.dart';

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
/// The usual alternative, dummy-variable seasonality, keeps `s - 1` states and
/// shifts them one position per step. That has no sensible `A(dt)` for a
/// non-integer gap, which rules it out here: the whole design rests on being
/// exact over an arbitrary gap. The rotation form has one, and reduces to the
/// familiar recursion when the data happen to be evenly spaced.
///
/// Without process noise the component is an ordinary Fourier series with
/// unknown coefficients — rigid, repeating forever. [processVariance] is what
/// lets it breathe: it is the rate at which the pattern is allowed to change
/// shape, in squared signal units per time unit. All harmonics share it, which
/// is Harvey's specification and keeps the component to one free parameter.
/// Letting each harmonic drift at its own rate is a different model, and one
/// that short histories cannot identify.
///
/// The implied Gaussian process is worth writing down, because it is simpler
/// than the state-space form suggests. The rotation is orthogonal and the
/// noise isotropic, so `A(t-u) Q A(t'-u)'` does not depend on `u` at all, and
/// the integral over the driving noise collapses to
///
/// ```text
/// k(t, t') = processVariance * min(t, t') * sum_j cos(lambda_j (t - t'))
/// ```
///
/// — Brownian motion multiplied by a cosine comb. That identity is what
/// `seasonal_reference_test.dart` checks the filter against, densely and in
/// `O(N^3)`.
///
/// One structural note. Averaged over a full period each harmonic integrates
/// to zero, so the component carries no level of its own and does not compete
/// with a trend for it. The two are still confounded over a stretch of data
/// shorter than a period, which is a statement about the data rather than
/// about the model.
class TrigonometricSeasonal extends Component {
  /// A pattern repeating every [period] time units, resolved by [harmonics]
  /// sinusoids.
  ///
  /// Weekly data on a daily time base means `period: 7`; two or three
  /// harmonics is usually plenty, since the fourth would be resolving detail
  /// finer than the data supports.
  ///
  /// Unlike the other components this validates rather than asserts, so it
  /// cannot be `const`. A period and a harmonic count that do not go together
  /// produce a model that is silently unidentifiable rather than obviously
  /// wrong, and that is not a failure worth shipping to release builds.
  TrigonometricSeasonal({
    required this.period,
    required this.harmonics,
    required this.processVariance,
  }) {
    if (!(period > 0) || !period.isFinite) {
      throw ArgumentError.value(
          period, 'period', 'must be finite and positive');
    }
    if (harmonics < 1) {
      throw ArgumentError.value(harmonics, 'harmonics', 'must be at least 1');
    }
    if (!(processVariance > 0) || !processVariance.isFinite) {
      throw ArgumentError.value(
          processVariance, 'processVariance', 'must be finite and positive');
    }
    if (2 * harmonics >= period) {
      // At lambda = pi the rotation degenerates to -I on unit steps, which
      // leaves the starred state unobservable for evenly spaced data and the
      // diffuse solve singular. Above it the harmonics alias onto lower ones.
      // Both are modelling errors rather than numerical ones, so they are
      // refused here rather than diagnosed later.
      throw ArgumentError.value(
          harmonics,
          'harmonics',
          'must be fewer than half the period ($period): harmonic $harmonics '
              'sits at or past the Nyquist frequency, where it either aliases '
              'onto a lower harmonic or leaves a state the data can never see');
    }
  }

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

  /// The kernel at equal times is `processVariance * span * sum_j cos(0)`, so
  /// the harmonics add rather than average: a pattern resolved by three
  /// sinusoids drifts faster than one resolved by a single sinusoid at the
  /// same noise intensity. Averaged over the path that halves, giving
  /// `processVariance * span * harmonics / 2`.
  ///
  /// The exact figure subtracts a further
  /// `2 sigma^2 sum_j (T - sin(lambda_j T) / lambda_j) / (lambda_j^2 T^2)`,
  /// which the oscillation drives down like `1 / (lambda_j^2 T)`. Over a
  /// window of a single period that term is about three per cent of the
  /// total, and it falls away from there; a component observed over less than
  /// one period is refused anyway by the time anything reads this.
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

  /// A seasonal that has not been round once is not, strictly, degenerate --
  /// sinusoids stay linearly independent on any handful of distinct times.
  /// It is merely so badly conditioned that it is the first thing worth
  /// suspecting when something else tips the solve over, which is what this
  /// hint is for: a lead, offered only once the engine has already failed.
  @override
  String? identifiabilityHint(double from, double to) {
    final span = to - from;
    if (span >= period) return null;
    return 'a seasonal component of period $period was given a series '
        'spanning only $span, so it has not been round once and over that '
        'stretch looks much like a constant with a slope';
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
  String toString() => 'TrigonometricSeasonal(period: $period, harmonics: '
      '$harmonics, processVariance: $processVariance)';
}
