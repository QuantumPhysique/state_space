/// A model that cannot produce an answer on the data it was given.
///
/// Thrown for conditions that depend on the data rather than on how the API
/// was called, so a caller that feeds in user data should expect to catch it.
/// Invalid arguments — an unsorted list, a negative variance — throw
/// [ArgumentError] instead.
///
/// {@category Getting started}
sealed class StateSpaceException implements Exception {
  const StateSpaceException(this.message);

  /// What went wrong and what to do about it.
  final String message;

  @override
  String toString() => 'StateSpaceException: $message';
}

/// The data does not determine the model: too few observations, or two
/// components that produce the same signal on this data.
///
/// Under [ExactDiffuse], `d` flat directions need observations at `d`
/// distinguishable times before anything is defined: two for a trend, two per
/// harmonic for a seasonal, one per regression column. [fit] additionally
/// needs at least one observation beyond those to estimate a noise level
/// from. [ApproximateDiffuse] returns a very wide posterior instead of
/// throwing, but only makes sense when the time unit keeps rates of change
/// near order one.
final class UnderdeterminedModelException extends StateSpaceException {
  /// An exception explaining what the data does not determine.
  const UnderdeterminedModelException(super.message);

  @override
  String toString() => 'UnderdeterminedModelException: $message';
}

/// The recursion reached a covariance that is not positive definite.
///
/// With the components in this package and valid arguments this does not
/// happen; it points at a component whose process noise is not a covariance,
/// or at an observation with zero [Observation.relativeVariance] that measures
/// a direction the model is already certain about.
final class NumericalBreakdownException extends StateSpaceException {
  /// An exception explaining where the recursion broke down.
  const NumericalBreakdownException(super.message);

  @override
  String toString() => 'NumericalBreakdownException: $message';
}
