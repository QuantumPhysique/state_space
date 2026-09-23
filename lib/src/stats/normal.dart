import 'dart:math' as math;

// Coefficients of Peter Acklam's rational approximation to the inverse normal
// cumulative distribution. Two branches for the tails, one for the centre.
const List<double> _a = [
  -3.969683028665376e+01,
  2.209460984245205e+02,
  -2.759285104469687e+02,
  1.383577518672690e+02,
  -3.066479806614716e+01,
  2.506628277459239e+00,
];
const List<double> _b = [
  -5.447609879822406e+01,
  1.615858368580409e+02,
  -1.556989798598866e+02,
  6.680131188771972e+01,
  -1.328068155288572e+01,
];
const List<double> _c = [
  -7.784894002430293e-03,
  -3.223964580411365e-01,
  -2.400758277161838e+00,
  -2.549732539343734e+00,
  4.374664141464968e+00,
  2.938163982698783e+00,
];
const List<double> _d = [
  7.784695709041462e-03,
  3.224671290700398e-01,
  2.445134137142996e+00,
  3.754408661907416e+00,
];

const double _low = 0.02425;

/// The standard normal quantile function, `Phi^-1(p)`.
///
/// Accurate to about 1.2e-9 in relative terms across the whole range, which is
/// several orders of magnitude finer than the width of any credible interval
/// worth drawing. Used only to turn a coverage level into a multiplier.
double normalQuantile(double p) {
  if (!(p > 0) || !(p < 1)) {
    throw ArgumentError.value(p, 'p', 'must lie strictly between 0 and 1');
  }
  if (p < _low) {
    final q = math.sqrt(-2 * math.log(p));
    return (((((_c[0] * q + _c[1]) * q + _c[2]) * q + _c[3]) * q + _c[4]) * q +
            _c[5]) /
        ((((_d[0] * q + _d[1]) * q + _d[2]) * q + _d[3]) * q + 1);
  }
  if (p > 1 - _low) {
    return -normalQuantile(1 - p);
  }
  final q = p - 0.5;
  final r = q * q;
  return (((((_a[0] * r + _a[1]) * r + _a[2]) * r + _a[3]) * r + _a[4]) * r +
          _a[5]) *
      q /
      (((((_b[0] * r + _b[1]) * r + _b[2]) * r + _b[3]) * r + _b[4]) * r + 1);
}

/// The two-sided multiplier for a central interval of the given coverage:
/// 1.96 for 0.95, 2.576 for 0.99.
double twoSidedZ(double coverage) {
  if (!(coverage > 0) || !(coverage < 1)) {
    throw ArgumentError.value(
      coverage,
      'coverage',
      'must lie strictly between 0 and 1',
    );
  }
  return normalQuantile(0.5 + coverage / 2);
}
