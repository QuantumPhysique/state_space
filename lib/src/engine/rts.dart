import 'dart:typed_data';

import '../component.dart';
import 'cholesky.dart';
import 'kalman.dart';
import 'matrix_block.dart';
import 'timeline.dart';

/// Rauch-Tung-Striebel backward pass.
///
/// Turns the filtered moments `E[x_t | y_1..t]` into the smoothed moments
/// `E[x_t | y_1..T]` by running
///
/// ```text
/// G_t   = P_t A_{t+1}' (P-_{t+1})^-1
/// x^s_t = x_t + G_t (x^s_{t+1} - x-_{t+1})
/// P^s_t = P_t + G_t (P^s_{t+1} - P-_{t+1}) G_t'
/// ```
///
/// backwards from the last step. `G_t` is obtained by solving
/// `P-_{t+1} G_t' = A_{t+1} P_t` with a Cholesky factorisation rather than by
/// forming the inverse: same cost, better conditioning, and it fails loudly
/// instead of quietly when the predicted covariance is singular.
///
/// The results overwrite [FilterResult.filteredMean] and
/// [FilterResult.filteredCovariance] in place. Each smoothed step is read
/// exactly once, by the step before it, so nothing is lost — and a decade of
/// daily data at sixteen states is a few megabytes, which is worth not
/// doubling.
class RtsSmoother {
  RtsSmoother(this.components)
      : stateDim = components.fold(0, (n, c) => n + c.stateDim),
        _offsets = _blockOffsets(components) {
    final n = stateDim;
    _a = Float64List(n * n);
    _gain = Float64List(n * n);
    _factor = Float64List(n * n);
    _left = Float64List(n * n);
    _delta = Float64List(n * n);
    _residual = Float64List(n);

    for (var b = 0; b < components.length; b++) {
      final dim = components[b].stateDim;
      final start = _offsets[b];
      _aBlocks.add(MatrixBlock(_a, start * n + start, n, dim, dim));
    }
  }

  final List<Component> components;
  final int stateDim;
  final List<int> _offsets;
  final List<MatrixBlock> _aBlocks = [];

  late final Float64List _a;
  late final Float64List _gain;
  late final Float64List _factor;
  late final Float64List _left;
  late final Float64List _delta;
  late final Float64List _residual;

  double _cachedGap = double.nan;

  static List<int> _blockOffsets(List<Component> components) {
    final offsets = <int>[];
    var next = 0;
    for (final c in components) {
      offsets.add(next);
      next += c.stateDim;
    }
    return offsets;
  }

  /// Smooths [filtered] in place. The filter must have been run with
  /// `keepHistory: true` over the same [timeline].
  void smoothInPlace(Timeline timeline, FilterResult filtered) {
    final mean = filtered.filteredMean;
    final cov = filtered.filteredCovariance;
    final predMean = filtered.predictedMean;
    final predCov = filtered.predictedCovariance;
    if (mean == null || cov == null || predMean == null || predCov == null) {
      throw ArgumentError('the forward pass was run without a history; '
          'call KalmanFilter.run(..., keepHistory: true)');
    }

    final n = stateDim;
    final square = n * n;
    // As in the forward pass: read the workspace fields once rather than on
    // every array access.
    final gain = _gain, left = _left, delta = _delta, residual = _residual;

    // The last step is already conditioned on everything.
    for (var t = timeline.length - 2; t >= 0; t--) {
      _buildTransition(timeline.gaps[t + 1]);

      final here = t * square;
      final next = (t + 1) * square;

      // gain = P_t A' , then solved against P-_{t+1} row by row.
      _blockRightMultiplyTranspose(cov, here, gain);
      _factorPredicted(predCov, next);
      for (var i = 0; i < n; i++) {
        choleskySolve(_factor, n, gain, i * n);
      }

      for (var i = 0; i < n; i++) {
        residual[i] = mean[(t + 1) * n + i] - predMean[(t + 1) * n + i];
      }
      for (var i = 0; i < n; i++) {
        var sum = mean[t * n + i];
        for (var j = 0; j < n; j++) {
          sum += gain[i * n + j] * residual[j];
        }
        mean[t * n + i] = sum;
      }

      for (var i = 0; i < square; i++) {
        delta[i] = cov[next + i] - predCov[next + i];
      }
      // left = G delta
      for (var i = 0; i < n; i++) {
        for (var j = 0; j < n; j++) {
          var sum = 0.0;
          for (var k = 0; k < n; k++) {
            sum += gain[i * n + k] * delta[k * n + j];
          }
          left[i * n + j] = sum;
        }
      }
      // P^s = P + left G', upper triangle then mirrored.
      for (var i = 0; i < n; i++) {
        for (var j = i; j < n; j++) {
          var sum = 0.0;
          for (var k = 0; k < n; k++) {
            sum += left[i * n + k] * gain[j * n + k];
          }
          final value = cov[here + i * n + j] + sum;
          cov[here + i * n + j] = value;
          cov[here + j * n + i] = value;
        }
      }
    }
  }

  void _buildTransition(double dt) {
    if (dt == _cachedGap) return;
    _a.fillRange(0, stateDim * stateDim, 0);
    for (var b = 0; b < components.length; b++) {
      components[b].transition(dt, _aBlocks[b]);
    }
    _cachedGap = dt;
  }

  /// `out = P A'` for block-diagonal `A`, reading `P` from `source[offset...]`.
  void _blockRightMultiplyTranspose(
      Float64List source, int offset, Float64List out) {
    final n = stateDim;
    for (var b = 0; b < components.length; b++) {
      final start = _offsets[b];
      final dim = components[b].stateDim;
      for (var row = 0; row < n; row++) {
        for (var i = 0; i < dim; i++) {
          var sum = 0.0;
          for (var j = 0; j < dim; j++) {
            sum += source[offset + row * n + start + j] *
                _a[(start + i) * n + start + j];
          }
          out[row * n + start + i] = sum;
        }
      }
    }
  }

  /// Copies the predicted covariance into the scratch factor and decomposes
  /// it, nudging the diagonal if the copy is not quite positive definite.
  void _factorPredicted(Float64List predCov, int offset) {
    final n = stateDim;
    _factor.setRange(0, n * n, predCov.getRange(offset, offset + n * n));
    if (choleskyFactor(_factor, n)) return;

    var trace = 0.0;
    for (var i = 0; i < n; i++) {
      trace += predCov[offset + i * n + i];
    }
    var jitter = 1e-12 * (trace / n).abs();
    if (jitter == 0) jitter = 1e-300;
    for (var attempt = 0; attempt < 6; attempt++) {
      _factor.setRange(0, n * n, predCov.getRange(offset, offset + n * n));
      for (var i = 0; i < n; i++) {
        _factor[i * n + i] += jitter;
      }
      if (choleskyFactor(_factor, n)) return;
      jitter *= 100;
    }
    throw StateError('the predicted covariance is not positive definite and '
        'could not be recovered by jittering; the model is degenerate at this '
        'step');
  }
}
