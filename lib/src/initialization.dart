/// How the prior on the state at the first time step is specified.
///
/// Trend and seasonal states are non-stationary: there is no proper prior to
/// put on them, because the level of a random walk has no stationary
/// distribution. The standard answers are to approximate the flat prior with a
/// very wide proper one, or to handle the flat directions exactly and
/// separately. This type is the choice between them.
sealed class Initialization {
  const Initialization();
}

/// A proper prior of [variance] times the model's measurement variance on
/// every diffuse state.
///
/// Simple, and wrong by about one part in [variance]. The error shows up as
/// shrinkage of the fitted curve toward zero, and as lost precision in the
/// smoothed covariance over the first few steps, where the answer is the
/// difference of two numbers of order [variance].
///
/// The prior is a multiple of the measurement variance rather than an absolute
/// number so that scaling every variance in the model scales the prior too.
/// That keeps the model scale-equivariant, which is what makes the profile
/// likelihood used by fitting exact rather than merely close.
final class ApproximateDiffuse extends Initialization {
  const ApproximateDiffuse({this.variance = 1e6})
      : assert(variance > 0, 'variance must be positive');

  /// Prior variance on each diffuse state, as a multiple of the measurement
  /// variance.
  final double variance;

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
/// This is exact, so the shrinkage and the lost digits of
/// [ApproximateDiffuse] both disappear. The price is that the diffuse
/// directions must actually be determined by the data — a two-state trend
/// needs observations at two distinct times before it means anything, and the
/// engine says so rather than returning a very large number.
final class ExactDiffuse extends Initialization {
  const ExactDiffuse();

  @override
  String toString() => 'ExactDiffuse()';
}
