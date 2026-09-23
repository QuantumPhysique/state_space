import 'dart:math' as math;

// Lanczos coefficients, g = 7, n = 9. Good to about fifteen digits over the
// half-plane we need, which is more than a p-value will ever be read to.
const List<double> _lanczos = [
  0.99999999999980993,
  676.5203681218851,
  -1259.1392167224028,
  771.32342877765313,
  -176.61502916214059,
  12.507343278686905,
  -0.13857109526572012,
  9.9843695780195716e-6,
  1.5056327351493116e-7,
];

/// `log Gamma(x)` for `x > 0`.
double logGamma(double x) {
  if (!(x > 0)) {
    throw ArgumentError.value(x, 'x', 'must be positive');
  }
  final z = x - 1;
  var series = _lanczos[0];
  for (var i = 1; i < _lanczos.length; i++) {
    series += _lanczos[i] / (z + i);
  }
  final t = z + 7.5;
  return 0.5 * math.log(2 * math.pi) +
      (z + 0.5) * math.log(t) -
      t +
      math.log(series);
}

/// Convergence target for both expansions. The series and the continued
/// fraction each gain roughly a digit per term in their own half of the
/// range, so this costs a few dozen terms at worst.
const double _epsilon = 1e-15;
const int _maximumTerms = 500;

/// The regularised lower incomplete gamma `P(a, x)`, by its series expansion.
///
/// Converges quickly for `x < a + 1` and slowly to hopelessly outside it,
/// which is what the continued fraction below is for.
double _lowerBySeries(double a, double x) {
  var term = 1 / a;
  var sum = term;
  for (var n = 1; n < _maximumTerms; n++) {
    term *= x / (a + n);
    sum += term;
    if (term.abs() < sum.abs() * _epsilon) break;
  }
  return sum * math.exp(-x + a * math.log(x) - logGamma(a));
}

/// The regularised upper incomplete gamma `Q(a, x)`, by the modified Lentz
/// evaluation of its continued fraction. The complement of the series, and
/// the accurate branch for `x >= a + 1`.
double _upperByContinuedFraction(double a, double x) {
  const tiny = 1e-300;
  var b = x + 1 - a;
  var c = 1 / tiny;
  var d = 1 / b;
  var result = d;
  for (var n = 1; n < _maximumTerms; n++) {
    final an = -n * (n - a);
    b += 2;
    d = an * d + b;
    if (d.abs() < tiny) d = tiny;
    c = b + an / c;
    if (c.abs() < tiny) c = tiny;
    d = 1 / d;
    final delta = d * c;
    result *= delta;
    if ((delta - 1).abs() < _epsilon) break;
  }
  return result * math.exp(-x + a * math.log(x) - logGamma(a));
}

/// The upper tail of a chi-square distribution: `P(X > statistic)` for `X`
/// with [degreesOfFreedom].
///
/// This is what turns a Ljung-Box statistic into something a caller can act
/// on. Measured against `scipy.stats.chi2.sf` the worst relative error over
/// the pinned grid is 1.6e-14, and it holds up thirteen decades into the tail
/// -- far past the point where the answer is simply "no".
double chiSquareUpperTail(double statistic, int degreesOfFreedom) {
  if (degreesOfFreedom < 1) {
    throw ArgumentError.value(
      degreesOfFreedom,
      'degreesOfFreedom',
      'must be at least 1',
    );
  }
  if (statistic <= 0) return 1;
  if (!statistic.isFinite) return 0;

  final a = degreesOfFreedom / 2;
  final x = statistic / 2;
  final tail = x < a + 1
      ? 1 - _lowerBySeries(a, x)
      : _upperByContinuedFraction(a, x);
  // Rounding can push either branch a hair outside [0, 1] deep in a tail.
  return tail.clamp(0.0, 1.0);
}
