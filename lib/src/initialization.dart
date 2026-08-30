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
