/// A single scalar measurement at a point in time.
///
/// [time] is a plain double in the caller's own unit, and the components'
/// process variances are expressed per that unit. The default search brackets
/// in [fit] suit days; [TimeAxis] converts calendar dates to days.
///
/// {@category Getting started}
final class Observation implements Comparable<Observation> {
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
  /// Use `0.5` for a reading you trust twice as much (an average of two
  /// weighings, say), `4.0` for one you trust half as much, and a large value
  /// to set aside a reading you believe is a mistake. Must be finite and not
  /// negative; zero asks for the reading to be matched exactly.
  final double relativeVariance;

  @override
  int compareTo(Observation other) => time.compareTo(other.time);

  @override
  bool operator ==(Object other) =>
      other is Observation &&
      other.time == time &&
      other.value == value &&
      other.relativeVariance == relativeVariance;

  @override
  int get hashCode => Object.hash(time, value, relativeVariance);

  @override
  String toString() => 'Observation($time, $value, '
      'relativeVariance: $relativeVariance)';
}
