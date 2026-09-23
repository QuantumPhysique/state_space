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

/// Relative pivot below which [factorInformation] calls a matrix singular.
///
/// A direction the others explain up to one part in ten billion carries no
/// usable information: its estimate would come with a variance ten billion
/// times its own scale. Genuinely singular systems leave pivots of rounding
/// size, around 1e-16, and the most ill-conditioned determined models the
/// test suite builds stay above 1e-8.
const double informationTolerance = 1e-10;

/// Cholesky factorisation of an information matrix, with a rank decision that
/// does not depend on the units of its rows.
///
/// The matrix is equilibrated to unit diagonal first, so each pivot is the
/// fraction of a direction's information that the directions before it do
/// not already carry. A direction is refused when that fraction falls below
/// [informationTolerance]: a matrix that is singular in exact arithmetic
/// leaves a pivot of rounding size, whose sign is an accident, and the bare
/// check in [choleskyFactor] would accept it about half the time.
///
/// On success [a] holds the lower factor of the original matrix, as
/// [choleskyFactor] leaves it, so [choleskySolve] applies unchanged.
bool factorInformation(Float64List a, int n) {
  final scale = Float64List(n);
  for (var i = 0; i < n; i++) {
    final diagonal = a[i * n + i];
    if (!(diagonal > 0) || !diagonal.isFinite) return false;
    scale[i] = math.sqrt(diagonal);
  }
  for (var i = 0; i < n; i++) {
    for (var j = 0; j <= i; j++) {
      a[i * n + j] /= scale[i] * scale[j];
    }
  }
  for (var i = 0; i < n; i++) {
    final rowI = i * n;
    for (var j = 0; j <= i; j++) {
      final rowJ = j * n;
      var sum = a[rowI + j];
      for (var k = 0; k < j; k++) {
        sum -= a[rowI + k] * a[rowJ + k];
      }
      if (i == j) {
        if (!(sum > informationTolerance)) return false;
        a[rowI + j] = math.sqrt(sum);
      } else {
        a[rowI + j] = sum / a[rowJ + j];
      }
    }
  }
  for (var i = 0; i < n; i++) {
    for (var j = 0; j <= i; j++) {
      a[i * n + j] *= scale[i];
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
