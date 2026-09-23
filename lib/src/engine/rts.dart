import 'dart:typed_data';

import '../component.dart';
import '../exceptions.dart';
import '../initialization.dart';
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
/// The results overwrite [FilterResult.stateMean] and
/// [FilterResult.stateCovariance] in place — which is why those are not called
/// `filtered`: the same arrays hold the filtered moments before this runs and
/// the smoothed ones after. Each smoothed step is read
/// exactly once, by the step before it, so nothing is lost — and a decade of
/// daily data at sixteen states is a few megabytes, which is worth not
/// doubling.
///
/// ## What it does not smooth
///
/// A state with no dynamics — a regression coefficient, `A = I` and `Q = 0`
/// under a flat prior — has covariance identically zero conditional on the
/// flat directions, at every step. So `G` has zero rows *and* zero columns
/// there: such a state smooths to its filtered value, and contributes nothing
/// to any other state's. The recursion below therefore runs on the states that
/// actually move, gathering them into a compact block and writing the results
/// back where they came from.
///
/// The backward pass is cubic in the state dimension, so for a trend plus
/// twenty holiday indicators this smooths two states rather than twenty-two.
/// See [Component.isStatic].
class RtsSmoother {
  /// [initialization] is what decides whether the reduction below applies, and
  /// omitting it declines the reduction rather than guessing at one: it is
  /// sound only under a flat prior, since with a proper one a static state has
  /// real variance and a real gain like anything else.
  RtsSmoother(this.components, {Initialization? initialization})
      : stateDim = components.fold(0, (n, c) => n + c.stateDim),
        _offsets = _blockOffsets(components),
        _active = _activeStates(components, initialization) {
    final n = _active.length;
    _a = Float64List(n * n);
    _gain = Float64List(n * n);
    _factor = Float64List(n * n);
    _left = Float64List(n * n);
    _delta = Float64List(n * n);
    _residual = Float64List(n);
    _sensitivityResidual = Float64List(n);

    var at = 0;
    for (var b = 0; b < components.length; b++) {
      final component = components[b];
      if (_reduced && component.isStatic) continue;
      final dim = component.stateDim;
      _activeComponents.add(component);
      _globalOffsets.add(_offsets[b]);
      _compactOffsets.add(at);
      _aBlocks.add(MatrixBlock(_a, at * n + at, n, dim, dim));
      at += dim;
    }
  }

  final List<Component> components;

  /// States in the full model, which is what the caller's arrays are laid out
  /// over.
  final int stateDim;

  final List<int> _offsets;

  /// Global indices of the states the backward pass actually has to touch,
  /// ascending. See [Component.isStatic].
  final Int32List _active;

  /// Whether anything was dropped, in which case the reduced path runs.
  bool get _reduced => _active.length != stateDim;

  final List<Component> _activeComponents = [];

  /// Where each active component's block sits in the caller's layout, and
  /// where it sits in the compact one the recursion works in.
  final List<int> _globalOffsets = [];
  final List<int> _compactOffsets = [];
  final List<MatrixBlock> _aBlocks = [];

  late final Float64List _a;
  late final Float64List _gain;
  late final Float64List _factor;
  late final Float64List _left;
  late final Float64List _delta;
  late final Float64List _residual;
  late final Float64List _sensitivityResidual;

  double _cachedGap = double.nan;

  /// The diagonal nudge that last rescued a factorisation, as a fraction of
  /// that step's mean diagonal, or zero.
  double _lastJitterRatio = 0;

  static List<int> _blockOffsets(List<Component> components) {
    final offsets = <int>[];
    var next = 0;
    for (final c in components) {
      offsets.add(next);
      next += c.stateDim;
    }
    return offsets;
  }

  /// The states the backward recursion can change.
  ///
  /// Under a flat prior a [Component.isStatic] component's covariance
  /// conditional on the flat directions is identically zero, so the smoother
  /// gain has zero rows and zero columns there: those states smooth to their
  /// filtered values, and they contribute nothing to any other state's. What
  /// they are worth arrives later, from the flat directions, in
  /// [combineDiffuse].
  ///
  /// Under [ApproximateDiffuse] the same states carry `kappa` and behave like
  /// anything else, so nothing is dropped — and neither is anything when the
  /// caller did not say which prior is in force.
  static Int32List _activeStates(
      List<Component> components, Initialization? initialization) {
    final all = components.fold(0, (int n, c) => n + c.stateDim);
    if (initialization is! ExactDiffuse) {
      return Int32List.fromList([for (var i = 0; i < all; i++) i]);
    }
    final kept = <int>[];
    var next = 0;
    for (final component in components) {
      for (var i = 0; i < component.stateDim; i++) {
        if (!component.isStatic) kept.add(next);
        next++;
      }
    }
    return Int32List.fromList(kept);
  }

  /// Smooths [filtered] in place. The filter must have been run with
  /// `keepHistory: true` over the same [timeline].
  void smoothInPlace(Timeline timeline, FilterResult filtered) {
    final mean = filtered.stateMean;
    final cov = filtered.stateCovariance;
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
    final active = _active;
    final m = active.length;
    final diffuseDim = filtered.diffuseDim;
    final sensitivity = filtered.stateSensitivity;
    final predictedSensitivity = filtered.predictedSensitivity;
    if (diffuseDim > 0 &&
        (sensitivity == null || predictedSensitivity == null)) {
      throw ArgumentError('the forward pass tracked diffuse directions but '
          'kept no sensitivity history');
    }
    // Every state is static: nothing the backward pass could change. The
    // filtered moments already are the smoothed ones, and what the
    // coefficients are worth arrives in [combineDiffuse].
    if (m == 0) return;

    // The last step is already conditioned on everything.
    for (var t = timeline.length - 2; t >= 0; t--) {
      _buildTransition(timeline.gaps[t + 1]);

      final here = t * square;
      final next = (t + 1) * square;

      // gain = P_t A' , then solved against P-_{t+1} row by row.
      _blockRightMultiplyTranspose(cov, here, gain);
      _factorPredicted(predCov, next);
      for (var i = 0; i < m; i++) {
        choleskySolve(_factor, m, gain, i * m);
      }

      for (var i = 0; i < m; i++) {
        final row = active[i];
        residual[i] = mean[(t + 1) * n + row] - predMean[(t + 1) * n + row];
      }
      for (var i = 0; i < m; i++) {
        final row = active[i];
        var sum = mean[t * n + row];
        for (var j = 0; j < m; j++) {
          sum += gain[i * m + j] * residual[j];
        }
        mean[t * n + row] = sum;
      }

      if (diffuseDim > 0) {
        _smoothSensitivity(
            sensitivity!, predictedSensitivity!, diffuseDim, t, gain);
      }

      for (var i = 0; i < m; i++) {
        final row = next + active[i] * n;
        for (var j = 0; j < m; j++) {
          final at = row + active[j];
          delta[i * m + j] = cov[at] - predCov[at];
        }
      }
      // left = G delta
      for (var i = 0; i < m; i++) {
        for (var j = 0; j < m; j++) {
          var sum = 0.0;
          for (var k = 0; k < m; k++) {
            sum += gain[i * m + k] * delta[k * m + j];
          }
          left[i * m + j] = sum;
        }
      }
      // P^s = P + left G', upper triangle then mirrored.
      for (var i = 0; i < m; i++) {
        final rowI = here + active[i] * n;
        for (var j = i; j < m; j++) {
          var sum = 0.0;
          for (var k = 0; k < m; k++) {
            sum += left[i * m + k] * gain[j * m + k];
          }
          final value = cov[rowI + active[j]] + sum;
          cov[rowI + active[j]] = value;
          cov[here + active[j] * n + active[i]] = value;
        }
      }
    }
  }

  /// The same backward recursion, applied to each column of the sensitivity.
  ///
  /// The gains depend only on covariances, and the recursion is affine in the
  /// mean, so smoothing `xa` and each column of `dx/dd` separately and
  /// recombining afterwards gives the same answer as smoothing the combined
  /// state would have.
  void _smoothSensitivity(Float64List sensitivity,
      Float64List predictedSensitivity, int d, int t, Float64List gain) {
    final n = stateDim;
    final active = _active;
    final m = active.length;
    final residual = _sensitivityResidual;
    final here = t * n * d;
    final next = (t + 1) * n * d;

    for (var c = 0; c < d; c++) {
      for (var i = 0; i < m; i++) {
        final at = next + active[i] * d + c;
        residual[i] = sensitivity[at] - predictedSensitivity[at];
      }
      for (var i = 0; i < m; i++) {
        final at = here + active[i] * d + c;
        var sum = sensitivity[at];
        for (var j = 0; j < m; j++) {
          sum += gain[i * m + j] * residual[j];
        }
        sensitivity[at] = sum;
      }
    }
  }

  /// Folds the estimated flat directions back into the smoothed moments.
  ///
  /// Given the flat directions, the smoothed state is normal with mean
  /// `xa + Xb d` and covariance `Ps` — a covariance that does not depend on
  /// `d` at all. The flat directions are themselves normal with mean `dhat`
  /// and covariance `S`. So the law of total variance gives the whole answer
  /// in one line each:
  ///
  /// ```text
  /// E[x]   = xa + Xb dhat
  /// Var[x] = Ps + Xb S Xb'
  /// ```
  ///
  /// Written back in place, so that everything downstream sees an ordinary
  /// smoothed mean and covariance and needs to know nothing about any of this.
  ///
  /// [steps] limits the fold-in to the steps whose moments will actually be
  /// read. This costs `O(n^2 d)` per step, which for a model whose flat
  /// directions are mostly regression coefficients is the largest single term
  /// in the whole backward pass; a caller reporting on a short grid has no
  /// reason to pay it at every observation. Every step is folded when it is
  /// null, and a step left out keeps moments that are conditional on the flat
  /// directions rather than marginal over them.
  void combineDiffuse(FilterResult filtered, {Int32List? steps}) {
    final d = filtered.diffuseDim;
    final estimate = filtered.diffuseMean;
    final spread = filtered.diffuseCovariance;
    // An empty series leaves the flat directions unestimated, and there is
    // nothing to fold them into either.
    if (d == 0 || estimate == null || spread == null) return;

    final n = stateDim;
    final mean = filtered.stateMean!;
    final cov = filtered.stateCovariance!;
    final sensitivity = filtered.stateSensitivity!;
    final scaled = Float64List(n * d);

    final count = steps?.length ?? filtered.stepCount;
    for (var k = 0; k < count; k++) {
      final t = steps == null ? k : steps[k];
      final block = t * n * d;

      for (var i = 0; i < n; i++) {
        var shift = 0.0;
        for (var c = 0; c < d; c++) {
          shift += sensitivity[block + i * d + c] * estimate[c];
        }
        mean[t * n + i] += shift;
      }

      for (var i = 0; i < n; i++) {
        for (var c = 0; c < d; c++) {
          var sum = 0.0;
          for (var e = 0; e < d; e++) {
            sum += sensitivity[block + i * d + e] * spread[e * d + c];
          }
          scaled[i * d + c] = sum;
        }
      }
      final here = t * n * n;
      for (var i = 0; i < n; i++) {
        for (var j = i; j < n; j++) {
          var sum = 0.0;
          for (var c = 0; c < d; c++) {
            sum += scaled[i * d + c] * sensitivity[block + j * d + c];
          }
          cov[here + i * n + j] += sum;
          if (i != j) cov[here + j * n + i] += sum;
        }
      }
    }
  }

  void _buildTransition(double dt) {
    if (dt == _cachedGap) return;
    final m = _active.length;
    _a.fillRange(0, m * m, 0);
    for (var b = 0; b < _activeComponents.length; b++) {
      _activeComponents[b].transition(dt, _aBlocks[b]);
    }
    _cachedGap = dt;
  }

  /// `out = P A'` for block-diagonal `A`, reading `P` from `source[offset...]`
  /// in the caller's layout and writing the compact active block.
  void _blockRightMultiplyTranspose(
      Float64List source, int offset, Float64List out) {
    final n = stateDim;
    final active = _active;
    final m = active.length;
    for (var b = 0; b < _activeComponents.length; b++) {
      final global = _globalOffsets[b];
      final compact = _compactOffsets[b];
      final dim = _activeComponents[b].stateDim;
      for (var row = 0; row < m; row++) {
        final src = offset + active[row] * n + global;
        for (var i = 0; i < dim; i++) {
          final a = (compact + i) * m + compact;
          var sum = 0.0;
          for (var j = 0; j < dim; j++) {
            sum += source[src + j] * _a[a + j];
          }
          out[row * m + compact + i] = sum;
        }
      }
    }
  }

  /// Gathers the active block of the predicted covariance into the scratch
  /// factor and decomposes it, nudging the diagonal if it is not quite
  /// positive definite.
  ///
  /// The nudge is a fraction of the step's own mean diagonal, and once one has
  /// been needed the same fraction is tried first at the steps that follow. A
  /// model carrying a state with no dynamics under a proper prior fails the
  /// plain factorisation at every step, and without the reuse each of them
  /// would pay for the escalation. Reusing a fraction rather than an amount
  /// keeps a step with a large covariance, such as a grid point far past the
  /// data, from imposing its nudge on steps with a small one.
  void _factorPredicted(Float64List predCov, int offset) {
    final m = _active.length;
    var trace = 0.0;
    for (var i = 0; i < m; i++) {
      trace += predCov[offset + _active[i] * stateDim + _active[i]];
    }
    final scale = (trace / m).abs();
    double jitterFor(double ratio) =>
        scale > 0 ? ratio * scale : ratio * 1e-288;

    if (_lastJitterRatio > 0) {
      _gatherActive(predCov, offset, _factor);
      final jitter = jitterFor(_lastJitterRatio);
      for (var i = 0; i < m; i++) {
        _factor[i * m + i] += jitter;
      }
      if (choleskyFactor(_factor, m)) return;
      _lastJitterRatio = 0;
    }

    _gatherActive(predCov, offset, _factor);
    if (choleskyFactor(_factor, m)) return;

    var ratio = 1e-12;
    for (var attempt = 0; attempt < 6; attempt++) {
      _gatherActive(predCov, offset, _factor);
      final jitter = jitterFor(ratio);
      for (var i = 0; i < m; i++) {
        _factor[i * m + i] += jitter;
      }
      if (choleskyFactor(_factor, m)) {
        _lastJitterRatio = ratio;
        return;
      }
      ratio *= 100;
    }
    throw const NumericalBreakdownException('the predicted covariance is not '
        'positive definite and could not be recovered by jittering: a '
        'component\'s process noise is not a valid covariance');
  }

  /// Copies the active rows and columns of a caller-layout matrix into a
  /// compact one. A no-op reshuffle when nothing is static.
  void _gatherActive(Float64List source, int offset, Float64List out) {
    final n = stateDim;
    final active = _active;
    final m = active.length;
    if (m == n) {
      out.setRange(0, n * n, source, offset);
      return;
    }
    for (var i = 0; i < m; i++) {
      final row = offset + active[i] * n;
      final to = i * m;
      for (var j = 0; j < m; j++) {
        out[to + j] = source[row + active[j]];
      }
    }
  }
}
