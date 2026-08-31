import 'dart:typed_data';

import 'package:state_space/src/fit/nelder_mead.dart';
import 'package:test/test.dart';

/// Rosenbrock's function, negated so that the optimum is a maximum at
/// `(1, 1)`. The classic hard case for a derivative-free search: a long
/// curved valley whose floor is nearly flat.
double _rosenbrock(Float64List x) {
  final a = 1 - x[0];
  final b = x[1] - x[0] * x[0];
  return -(a * a + 100 * b * b);
}

/// Beale's function, negated. Maximum at `(3, 0.5)`, with plateaux either
/// side that strand a search which trusts its first descent direction.
double _beale(Float64List x) {
  final a = 1.5 - x[0] + x[0] * x[1];
  final b = 2.25 - x[0] + x[0] * x[1] * x[1];
  final c = 2.625 - x[0] + x[0] * x[1] * x[1] * x[1];
  return -(a * a + b * b + c * c);
}

/// A separable quadratic in four variables, which is the shape a well
/// determined likelihood actually has near its maximum.
double _quadratic(Float64List x) {
  const centre = [1.5, -2.0, 0.25, 3.0];
  var total = 0.0;
  for (var i = 0; i < x.length; i++) {
    final d = x[i] - centre[i];
    total -= (i + 1) * d * d;
  }
  return total;
}

void main() {
  group('the simplex search', () {
    test('walks up the Rosenbrock valley', () {
      var result = maximiseSimplex(_rosenbrock, Float64List.fromList([-1.2, 1]),
          step: 0.5, tolerance: 1e-12);
      // One restart, as the fitting code does: a fresh simplex around the
      // answer is the standard insurance against a collapsed one.
      result = maximiseSimplex(_rosenbrock, result.argument,
          step: 0.1, tolerance: 1e-14);

      expect(result.argument[0], closeTo(1, 1e-5));
      expect(result.argument[1], closeTo(1, 1e-5));
      expect(result.value, closeTo(0, 1e-10));
    });

    test('finds the maximum of Beale from a long way out', () {
      var result = maximiseSimplex(_beale, Float64List.fromList([0, 0]),
          step: 1, tolerance: 1e-12);
      result =
          maximiseSimplex(_beale, result.argument, step: 0.1, tolerance: 1e-14);

      expect(result.argument[0], closeTo(3, 1e-5));
      expect(result.argument[1], closeTo(0.5, 1e-5));
    });

    test('handles four dimensions without help', () {
      final result = maximiseSimplex(
          _quadratic, Float64List.fromList([0, 0, 0, 0]),
          step: 2, tolerance: 1e-12);

      expect(result.converged, isTrue);
      for (final (i, expected) in [1.5, -2.0, 0.25, 3.0].indexed) {
        expect(result.argument[i], closeTo(expected, 1e-5),
            reason: 'coordinate $i');
      }
    });

    test('converges on a one-dimensional problem too', () {
      // The degenerate case is worth a test of its own: a two-vertex simplex
      // is a bracketing search with unusual bookkeeping, and it is what a
      // single-component model would use if it were not better served by
      // golden section.
      final result = maximiseSimplex(
          (x) => -(x[0] - 2.5) * (x[0] - 2.5), Float64List.fromList([0]),
          step: 1, tolerance: 1e-14);
      expect(result.argument[0], closeTo(2.5, 1e-6));
    });

    test('reports that it stopped early when it did', () {
      final result = maximiseSimplex(_rosenbrock, Float64List.fromList([-3, 4]),
          step: 0.5, tolerance: 1e-16, maxEvaluations: 30);
      expect(result.converged, isFalse);
      expect(result.evaluations, lessThanOrEqualTo(40));
    });

    test('refuses an empty parameter vector', () {
      expect(
          () => maximiseSimplex((x) => 0, Float64List(0)), throwsArgumentError);
    });

    test('does not care about the scale of the objective', () {
      // Adding a constant and multiplying by a positive one moves the answer
      // nowhere, which it must not, because a profile likelihood carries an
      // arbitrary additive constant.
      final plain = maximiseSimplex(_beale, Float64List.fromList([1, 1]),
          step: 0.5, tolerance: 1e-12);
      final shifted = maximiseSimplex(
          (x) => 3 * _beale(x) + 1e6, Float64List.fromList([1, 1]),
          step: 0.5, tolerance: 3e-12);
      expect(shifted.argument[0], closeTo(plain.argument[0], 1e-4));
      expect(shifted.argument[1], closeTo(plain.argument[1], 1e-4));
    });
  });
}
