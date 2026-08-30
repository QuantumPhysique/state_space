import 'dart:math' as math;
import 'dart:typed_data';

/// Cholesky factorisation of a small symmetric positive-definite matrix,
/// specialised for the sizes a structural model produces (single digits, at
/// worst a couple of dozen).
///
/// The factor overwrites the lower triangle of [a]; the strict upper triangle
/// is left as it was, so the caller can keep using the original there.
/// Returns false if the matrix is not positive definite, which the smoother
/// treats as "add a little jitter and try again" rather than as an error.
bool choleskyFactor(Float64List a, int n) {
  for (var i = 0; i < n; i++) {
    final rowI = i * n;
    for (var j = 0; j <= i; j++) {
      final rowJ = j * n;
      var sum = a[rowI + j];
      for (var k = 0; k < j; k++) {
        sum -= a[rowI + k] * a[rowJ + k];
      }
      if (i == j) {
        if (!(sum > 0)) return false;
        a[rowI + j] = math.sqrt(sum);
      } else {
        a[rowI + j] = sum / a[rowJ + j];
      }
    }
  }
  return true;
}

/// Solves `A x = b` in place, given the factor produced by [choleskyFactor].
///
/// [b] holds the right-hand side at `b[offset .. offset + n)` and receives the
/// solution in the same place.
void choleskySolve(Float64List l, int n, Float64List b, int offset) {
  // Forward substitution through the lower factor.
  for (var i = 0; i < n; i++) {
    var sum = b[offset + i];
    final row = i * n;
    for (var k = 0; k < i; k++) {
      sum -= l[row + k] * b[offset + k];
    }
    b[offset + i] = sum / l[row + i];
  }
  // Back substitution through its transpose.
  for (var i = n - 1; i >= 0; i--) {
    var sum = b[offset + i];
    for (var k = i + 1; k < n; k++) {
      sum -= l[k * n + i] * b[offset + k];
    }
    b[offset + i] = sum / l[i * n + i];
  }
}
