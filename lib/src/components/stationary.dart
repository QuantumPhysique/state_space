import 'dart:math' as math;

/// Root-mean-square deviation of a stationary process from its own average
/// over a window of [span], which is what `Component.wanderOver` asks every
/// component for.
///
/// ```text
/// wander^2 = k(0) - (2 / T^2) integral_0^T (T - tau) k(tau) dtau
/// ```
///
/// The double integral of a stationary kernel over the square collapses to a
/// single integral against the triangular weight `T - tau`, evaluated by
/// Simpson's rule. The closed forms that exist for each kernel have removable
/// singularities where the damping or the frequency goes to zero.
///
/// The error is `O(h^4 k'''')` with `h = T / n`, so what matters is `h`
/// against the kernel's own scale: a length scale of 0.1 over a window of 365
/// gives a relative error of 1e-3, a length scale of 1 gives 5e-5, and a
/// resolved kernel is exact to the last bit.
double stationaryWander(double Function(double lag) covariance, double span) {
  if (!(span > 0)) return 0;
  final h = span / _intervals;
  // Simpson: endpoints once, odd nodes four times, even nodes twice. The
  // weight (T - tau) vanishes at the far end, so that endpoint contributes
  // nothing and is included only for symmetry of the expression.
  var total = span * covariance(0);
  for (var i = 1; i < _intervals; i++) {
    final tau = i * h;
    total += (i.isOdd ? 4 : 2) * (span - tau) * covariance(tau);
  }
  final integral = total * h / 3;
  final variance = covariance(0) - 2 * integral / (span * span);
  // Cancellation can push it a hair below zero when the process barely moves
  // over the window, which is a rounding artefact and not a negative variance.
  return variance <= 0 ? 0 : math.sqrt(variance);
}

/// Enough that the quartic error term is invisible for any kernel here, and
/// still only a few hundred multiplications once per penalty evaluation.
const int _intervals = 256;
