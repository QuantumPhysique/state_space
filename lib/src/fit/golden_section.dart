import 'dart:math' as math;

/// Outcome of a one-dimensional golden-section search.
class GoldenSectionResult {
  const GoldenSectionResult({
    required this.argument,
    required this.value,
    required this.evaluations,
    required this.converged,
  });

  /// Location of the maximum.
  final double argument;

  /// The objective there.
  final double value;

  final int evaluations;

  /// Whether the bracket shrank below the tolerance rather than the search
  /// running out of iterations.
  final bool converged;
}

final double _inverseGolden = (math.sqrt(5) - 1) / 2;

/// Maximises a unimodal [objective] on `[lower, upper]` by golden-section
/// search.
///
/// One function evaluation per iteration after the first two, no derivatives,
/// and a bracket that shrinks by a fixed factor every step — which for a
/// likelihood evaluated by a full Kalman pass is exactly the trade one wants.
/// It assumes a single maximum in the bracket; the caller is responsible for
/// having found the right basin first.
GoldenSectionResult maximise(
  double Function(double) objective,
  double lower,
  double upper, {
  double tolerance = 1e-4,
  int maxIterations = 200,
}) {
  if (!(lower < upper)) {
    throw ArgumentError('empty bracket [$lower, $upper]');
  }

  var a = lower;
  var b = upper;
  var c = b - _inverseGolden * (b - a);
  var d = a + _inverseGolden * (b - a);
  var fc = objective(c);
  var fd = objective(d);
  var evaluations = 2;
  var iterations = 0;

  while (b - a > tolerance && iterations < maxIterations) {
    if (fc > fd) {
      b = d;
      d = c;
      fd = fc;
      c = b - _inverseGolden * (b - a);
      fc = objective(c);
    } else {
      a = c;
      c = d;
      fc = fd;
      d = a + _inverseGolden * (b - a);
      fd = objective(d);
    }
    evaluations++;
    iterations++;
  }

  final atC = fc >= fd;
  return GoldenSectionResult(
    argument: atC ? c : d,
    value: atC ? fc : fd,
    evaluations: evaluations,
    converged: b - a <= tolerance,
  );
}
