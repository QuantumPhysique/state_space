import 'dart:math' as math;
import 'dart:typed_data';

import '../component.dart';
import '../engine/matrix_block.dart';

/// A level that follows a Wiener process: `d(mu) = sigma dB`.
///
/// One state, `A(dt) = 1`, `Q(dt) = sigma^2 dt`. The implied kernel is
/// Brownian motion, `k(t, t') = sigma^2 min(t, t')`, and the posterior mean is
/// a linear interpolant through shrunken observations — the continuous-time
/// counterpart of simple exponential smoothing.
///
/// Useful on its own for series with no persistent direction, and as the
/// simplest thing that can go wrong when something in the engine breaks.
///
/// {@category Components}
final class LocalLevel extends Component {
  /// A level driven by white noise of intensity [processVariance], which must
  /// be finite and positive.
  LocalLevel({required this.processVariance}) {
    if (!(processVariance > 0) || !processVariance.isFinite) {
      throw ArgumentError.value(
          processVariance, 'processVariance', 'must be finite and positive');
    }
  }

  @override
  String get name => 'LocalLevel';

  /// Intensity of the white noise driving the level, in squared signal units
  /// per time unit.
  final double processVariance;

  @override
  int get stateDim => 1;

  @override
  int get parameterCount => 1;

  @override
  void transition(double dt, MatrixBlock out) => out.set(0, 0, 1);

  @override
  void processNoise(double dt, MatrixBlock out) =>
      out.set(0, 0, processVariance * dt);

  @override
  void observationAt(double time, Float64List out) => out[0] = 1;

  /// `sigma^2 T / 6` for the spread of the path, against `sigma^2 T` at the
  /// end of it.
  @override
  double wanderOver(double span) => math.sqrt(processVariance * span / 6);

  @override
  List<bool> get diffuseStates => const [true];

  @override
  void properPrior(Float64List mean, MatrixBlock covariance) {
    // The level is diffuse; nothing to contribute.
  }

  @override
  Float64List get parameters =>
      Float64List.fromList([math.log(processVariance)]);

  @override
  Component withParameters(Float64List theta) =>
      LocalLevel(processVariance: math.exp(theta[0]));

  @override
  bool operator ==(Object other) =>
      other is LocalLevel && other.processVariance == processVariance;

  @override
  int get hashCode => Object.hash(LocalLevel, processVariance);

  @override
  String toString() => 'LocalLevel(processVariance: $processVariance)';
}
