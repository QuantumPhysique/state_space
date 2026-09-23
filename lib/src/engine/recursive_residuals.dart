import 'dart:math' as math;
import 'dart:typed_data';

import 'cholesky.dart';

/// What the forward pass keeps about each observation so that standardised
/// residuals can be reconstructed after the fact.
///
/// Under a flat prior the innovation is not a number but an affine function of
/// the unknown starting point, `v(d) = va + vb . d`. Substituting the final
/// estimate of `d` would give something conditioned on the whole series, which
/// is not what a whiteness test wants. Keeping the pieces lets the residuals
/// be rebuilt honestly, one observation at a time, at the cost of `2 + d`
/// doubles per observation and only when they are asked for.
class ResidualPieces {
  /// Room for [count] observations under [diffuseDim] flat directions.
  ResidualPieces(this.count, this.diffuseDim)
    : times = Float64List(count),
      constant = Float64List(count),
      variance = Float64List(count),
      loading = Float64List(count * diffuseDim);

  /// Observations the pieces have room for.
  final int count;

  /// Flat directions in the model.
  final int diffuseDim;

  /// Time of each observation.
  final Float64List times;

  /// `va`: the innovation the filter computed, which under a flat prior is
  /// the part that does not depend on the unknown starting point.
  final Float64List constant;

  /// `S`: the innovation variance, which does not depend on it at all.
  final Float64List variance;

  /// `vb`: how the innovation moves with the starting point, `count` rows of
  /// [diffuseDim].
  final Float64List loading;

  var _filled = 0;

  /// Records one observation's innovation [va], its variance [s] and its
  /// loading [vb] on the flat directions.
  void add(double time, double va, double s, Float64List vb) {
    final at = _filled;
    times[at] = time;
    constant[at] = va;
    variance[at] = s;
    for (var c = 0; c < diffuseDim; c++) {
      loading[at * diffuseDim + c] = vb[c];
    }
    _filled++;
  }
}

/// Standardised one-step-ahead prediction errors.
///
/// With a proper prior these are just `v_t / sqrt(S_t)` past the burn-in.
/// With a flat one they are the *recursive* residuals: at each observation the
/// starting point is estimated from what came before it and nothing else, so
///
/// ```text
/// e_t = (va_t + vb_t . dhat_{t-1}) / sqrt(S_t + vb_t' M_{t-1}^-1 vb_t)
/// ```
///
/// where `M_{t-1}` and `dhat_{t-1}` are the information and the estimate
/// accumulated strictly before `t`. Those are independent standard normals
/// under the model — genuinely so, not approximately — which is the property
/// a Ljung-Box test needs and which substituting the final estimate would
/// quietly destroy.
///
/// Nothing is reported until the flat directions are pinned down, since until
/// then the prediction has infinite variance in some direction. That happens
/// after exactly `d` observations unless the data is degenerate, which is why
/// the count comes out at `N - d` and matches the degrees of freedom the
/// likelihood charges for.
///
/// The cost is a `d x d` factorisation per observation. That is `O(N d^3)`,
/// which for the dozen-odd states a structural model reaches is nothing, and
/// it only runs when a caller asks for diagnostics.
({Float64List times, Float64List values}) recursiveResiduals(
  ResidualPieces pieces, {
  required int burnIn,
}) {
  final n = pieces.count;
  final d = pieces.diffuseDim;

  if (d == 0) {
    final kept = math.max(0, n - burnIn);
    final times = Float64List(kept);
    final values = Float64List(kept);
    for (var i = 0; i < kept; i++) {
      final at = i + burnIn;
      times[i] = pieces.times[at];
      values[i] = pieces.constant[at] / math.sqrt(pieces.variance[at]);
    }
    return (times: times, values: values);
  }

  final information = Float64List(d * d);
  final rhs = Float64List(d);
  final factor = Float64List(d * d);
  final estimate = Float64List(d);
  final solvedLoading = Float64List(d);
  final times = Float64List(n);
  final values = Float64List(n);
  var kept = 0;

  for (var i = 0; i < n; i++) {
    final row = i * d;
    final s = pieces.variance[i];

    // Two guards, and both are needed. The count is the exact one: `i`
    // observations cannot determine more than `i` directions, so before `d`
    // of them the information matrix is singular by construction. The
    // factorisation catches the rest -- repeated timestamps, a component the
    // data cannot see -- but on its own it does not catch the first, because
    // a matrix that is singular in exact arithmetic can still hand back a
    // tiny positive pivot and a residual that means nothing.
    factor.setAll(0, information);
    if (i >= d && factorInformation(factor, d)) {
      estimate.setAll(0, rhs);
      for (var c = 0; c < d; c++) {
        estimate[c] = -estimate[c];
      }
      choleskySolve(factor, d, estimate, 0);

      var predicted = pieces.constant[i];
      for (var c = 0; c < d; c++) {
        predicted += pieces.loading[row + c] * estimate[c];
      }

      for (var c = 0; c < d; c++) {
        solvedLoading[c] = pieces.loading[row + c];
      }
      choleskySolve(factor, d, solvedLoading, 0);
      var spread = s;
      for (var c = 0; c < d; c++) {
        spread += pieces.loading[row + c] * solvedLoading[c];
      }

      times[kept] = pieces.times[i];
      values[kept] = predicted / math.sqrt(spread);
      kept++;
    }

    for (var r = 0; r < d; r++) {
      final weighted = pieces.loading[row + r] / s;
      rhs[r] += weighted * pieces.constant[i];
      for (var c = 0; c < d; c++) {
        information[r * d + c] += weighted * pieces.loading[row + c];
      }
    }
  }

  return (
    times: Float64List.sublistView(times, 0, kept),
    values: Float64List.sublistView(values, 0, kept),
  );
}
