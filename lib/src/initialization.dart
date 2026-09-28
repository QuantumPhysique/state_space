import 'arguments.dart';

/// How the prior on the state at the first time step is specified.
///
/// Trend and seasonal states are non-stationary: there is no proper prior to
/// put on them, because the level of a random walk has no stationary
/// distribution. The standard answers are to approximate the flat prior with a
/// very wide proper one, or to handle the flat directions exactly and
/// separately. This type is the choice between them.
///
/// {@category How it works}
sealed class Initialization {
  const Initialization();
}

/// A proper prior of [variance] times the model's measurement variance on
/// every diffuse state.
///
/// It never refuses: where the data cannot determine a flat direction the
/// posterior is simply very wide. The price is accuracy. The error shows up as
/// shrinkage of the fitted curve toward zero, and as lost precision in the
/// smoothed covariance over the first few steps, where the answer is the
/// difference of two numbers of order [variance].
///
/// **Use it only with a time unit that keeps rates of change near order one,
/// such as days for a daily series.** Every diffuse state gets the same prior
/// variance, whatever its units, so with time in seconds or milliseconds a
/// slope's prior is many orders of magnitude too wide next to a level's, the
/// covariance updates lose their precision, and the curve can be off by a
/// sizeable fraction of the noise with a band of zero width. No choice of
/// [variance] fixes that; [ExactDiffuse] does not have the problem.
///
/// The prior sits at the first step of the timeline, so output grid points
/// before the first observation move it and change the answer slightly.
///
/// The prior is a multiple of the measurement variance so that scaling every
/// variance in the model scales it too, which keeps the profile likelihood
/// used by fitting exact.
final class ApproximateDiffuse extends Initialization {
  /// A prior of [variance] times the measurement variance, which must be
  /// finite and positive.
  ApproximateDiffuse({this.variance = 1e6}) {
    checkPositive(variance, 'variance');
  }

  /// Prior variance on each diffuse state, as a multiple of the measurement
  /// variance.
  final double variance;

  @override
  bool operator ==(Object other) =>
      other is ApproximateDiffuse && other.variance == variance;

  @override
  int get hashCode => Object.hash(ApproximateDiffuse, variance);

  @override
  String toString() => 'ApproximateDiffuse(variance: $variance)';
}

/// Exact diffuse initialisation: the flat directions are handled exactly,
/// as the limit of an infinitely wide prior, rather than approximated by a
/// wide one (Durbin & Koopman 2012, ch. 5).
///
/// The engine does this by augmentation rather than by a separate diffuse
/// recursion. Write the initial state as `x(0) = a + B d`, where `B` selects
/// the diffuse directions and `d` is an unknown vector with a flat prior.
/// Everything downstream is affine in `d`, so the filter carries the
/// sensitivity `dx(t)/dd` alongside the state, at the cost of one extra mean
/// propagation per diffuse direction and no extra covariance work at all.
/// The flat directions are then integrated out in closed form: a generalised
/// least squares problem of the size of the diffuse dimension, solved once at
/// the end of the pass.
///
/// This is exact, and does not depend on the time unit or where time starts.
/// The diffuse directions must be determined by the data: a two-state trend
/// needs observations at two distinct times, and the engine throws
/// [UnderdeterminedModelException] rather than returning a very large number.
final class ExactDiffuse extends Initialization {
  /// The default initialisation.
  const ExactDiffuse();

  @override
  bool operator ==(Object other) => other is ExactDiffuse;

  @override
  int get hashCode => (ExactDiffuse).hashCode;

  @override
  String toString() => 'ExactDiffuse()';
}
