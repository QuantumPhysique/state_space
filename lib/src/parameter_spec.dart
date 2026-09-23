/// What one entry of a component's parameter vector is, as far as fitting is
/// concerned.
///
/// Every parameter the package searches over lives in an unconstrained
/// coordinate (a log for something positive, a logit for something bounded),
/// so that no optimiser has to respect a constraint. A variance and a shape
/// parameter differ in two ways:
///
/// * Scale. [fit] searches the *ratio* of each variance to the measurement
///   variance and multiplies it back up at the end. A length scale or a
///   period is not multiplied.
/// * Bracket. A variance ratio is searched over [fit]'s `lowerLogRatio` to
///   `upperLogRatio`; a shape parameter over the range its component gives.
///
/// {@category Components}
sealed class ParameterSpec {
  const ParameterSpec();

  /// A short name for what this parameter is, used when a fit has something to
  /// say about it. Two or three words, lower case, no component name.
  String get label;

  /// Whether the unconstrained coordinate is a logarithm, so that a width in
  /// it divided by `ln 10` is a width in decades.
  ///
  /// True for a log variance, a log length scale and a log period. False for a
  /// logit, where the same division produces a number in no unit at all — see
  /// [FitResult.plateauDecadesByParameter], which reports [double.nan] rather
  /// than a plausible-looking figure.
  bool get isLogarithmic;

  /// How far the simplex should step along this axis when it starts, or null
  /// to let [fit] choose from the scan resolution.
  ///
  /// Set it when the scan resolution is fine for a reason unrelated to the
  /// parameter's scale, as for a cycle's period.
  double? get searchStep => null;
}

/// A log variance, in squared signal units, and the default for a component
/// that does not say otherwise.
final class VarianceParameter extends ParameterSpec {
  /// A variance, described in [FitResult.warnings] as [label].
  const VarianceParameter({this.label = 'variance'});

  @override
  final String label;

  @override
  bool get isLogarithmic => true;

  @override
  bool operator ==(Object other) =>
      other is VarianceParameter && other.label == label;

  @override
  int get hashCode => Object.hash(VarianceParameter, label);

  @override
  String toString() => 'VarianceParameter($label)';
}

/// A parameter describing shape rather than size: a length scale, a period, a
/// damping factor.
///
/// Unaffected by the scale of the data, and searched over [lower] to [upper] in
/// whatever unconstrained coordinate the component chose — a log for a period,
/// a logit for a damping factor.
///
/// [scanPoints] overrides the resolution of the coordinate scan on this axis
/// alone, for a likelihood that is multimodal in this parameter, as a cycle's
/// period is.
final class ShapeParameter extends ParameterSpec {
  /// A shape parameter searched over [lower] to [upper], a finite non-empty
  /// range. [scanPoints], if given, must be at least 3 and [searchStep]
  /// finite and positive.
  ShapeParameter({
    required this.lower,
    required this.upper,
    this.scanPoints,
    this.searchStep,
    this.label = 'shape',
    this.isLogarithmic = true,
  }) {
    if (!(lower < upper) || !lower.isFinite || !upper.isFinite) {
      throw ArgumentError('empty or infinite bracket [$lower, $upper]');
    }
    final points = scanPoints;
    if (points != null && points < 3) {
      throw ArgumentError.value(points, 'scanPoints', 'must be at least 3');
    }
    final step = searchStep;
    if (step != null && (!(step > 0) || !step.isFinite)) {
      throw ArgumentError.value(step, 'searchStep', 'must be finite and > 0');
    }
  }

  @override
  final String label;

  /// False for a logit coordinate, such as a damping factor.
  @override
  final bool isLogarithmic;

  @override
  final double? searchStep;

  /// Bottom of the search bracket, in the component's own coordinate.
  final double lower;

  /// Top of it.
  final double upper;

  /// How many points the coordinate scan places across the bracket, or null to
  /// use whatever [fit] was told.
  final int? scanPoints;

  @override
  bool operator ==(Object other) =>
      other is ShapeParameter &&
      other.label == label &&
      other.lower == lower &&
      other.upper == upper &&
      other.scanPoints == scanPoints &&
      other.searchStep == searchStep &&
      other.isLogarithmic == isLogarithmic;

  @override
  int get hashCode => Object.hash(ShapeParameter, label, lower, upper,
      scanPoints, searchStep, isLogarithmic);

  @override
  String toString() => 'ShapeParameter($label, lower: $lower, upper: $upper'
      '${scanPoints == null ? '' : ', scanPoints: $scanPoints'})';
}
