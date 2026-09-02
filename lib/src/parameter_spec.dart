/// What one entry of a component's parameter vector is, as far as fitting is
/// concerned.
///
/// Every parameter the package searches over lives in an unconstrained
/// coordinate — a log for something positive, a logit for something bounded —
/// so that no optimiser ever has to respect a constraint. Up to v0.4 that was
/// the whole story, because every parameter in the package was a variance and
/// they could all be treated alike. A length scale and a period cannot be, and
/// the difference shows up in two places.
///
/// The first is scale. [fit] concentrates the measurement variance out of the
/// likelihood by searching the *ratio* of every other variance to it and
/// multiplying the lot back up at the end. A length scale is not a variance and
/// must not be multiplied back up; doing so would rescale a period in days by
/// the noise level, which is nonsense in a way that produces a plausible-looking
/// number rather than an error.
///
/// The second is the search bracket. Twenty decades below the measurement
/// variance is a reasonable place to look for a variance ratio and an absurd
/// place to look for a period. A component knows what range its own shape
/// parameters live in and says so here.
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

  /// How far the simplex should step along this axis when it starts.
  ///
  /// Null lets [fit] choose from the width of the bracket. It is worth setting
  /// where the bracket's width is driven by something other than the scale of
  /// the parameter: a cycle's period is scanned at high resolution because the
  /// likelihood in it is multimodal, and deriving a simplex step from that
  /// resolution would give the period axis a displacement thirty times smaller
  /// than the variance axes for no reason connected to either.
  double? get searchStep => null;
}

/// A log variance, in squared signal units.
///
/// This is what every parameter in the package was until v0.5, and it is still
/// the default for a component that does not say otherwise.
final class VarianceParameter extends ParameterSpec {
  const VarianceParameter({this.label = 'variance'});

  @override
  final String label;

  @override
  bool get isLogarithmic => true;

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
/// alone. It exists for the one case that genuinely needs it: the likelihood in
/// a cycle's period is multimodal, with secondary maxima at half and twice the
/// truth, and a scan coarse enough to step over the real peak will find one of
/// those instead. Everything else is content with the caller's default.
final class ShapeParameter extends ParameterSpec {
  /// Validates rather than asserts, because a bracket that excludes the answer
  /// produces a confident wrong number rather than a crash, and that is not a
  /// failure worth shipping to release builds.
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
  String toString() => 'ShapeParameter($label, lower: $lower, upper: $upper'
      '${scanPoints == null ? '' : ', scanPoints: $scanPoints'})';
}
