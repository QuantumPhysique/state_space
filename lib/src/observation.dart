/// A single scalar measurement at a point in time.
///
/// [time] is a plain double in whatever unit the caller finds natural — days
/// since an epoch, seconds, fractional years. The package never converts it,
/// and the process variances of the components are expressed per that unit.
///
/// {@category Getting started}
class Observation implements Comparable<Observation> {
  /// A reading of [value] taken at [time].
  const Observation(this.time, this.value, {this.relativeVariance = 1.0});

  /// When the reading was taken, in the caller's own time unit.
  final double time;

  /// What was read.
  final double value;

  /// Noise variance of this reading *relative to* the model's measurement
  /// variance, so the effective noise is
  /// `relativeVariance * model.measurementVariance`.
  ///
  /// It is relative rather than absolute so that a model stays scale-free: one
  /// number sets the noise level and these weights say how the readings differ
  /// from each other. Use `0.5` for a reading you trust twice as much (an
  /// average of two weighings, say), `4.0` for one you trust half as much.
  final double relativeVariance;

  @override
  int compareTo(Observation other) => time.compareTo(other.time);

  @override
  String toString() => 'Observation($time, $value, '
      'relativeVariance: $relativeVariance)';
}
