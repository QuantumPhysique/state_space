import 'dart:math' as math;
import 'dart:typed_data';

import '../component.dart';
import 'matrix_block.dart';
import 'timeline.dart';

const double _log2pi = 1.8378770664093456;

/// Everything the forward pass produces.
///
/// The state arrays are only populated when the filter is asked to keep a
/// history; a likelihood evaluation does not need them and does not pay for
/// them. Each is laid out step-major: state `j` of step `t` lives at
/// `t * stateDim + j`, and covariance entry `(i, j)` at
/// `t * stateDim * stateDim + i * stateDim + j`.
class FilterResult {
  FilterResult({
    required this.stateDim,
    required this.stepCount,
    required this.logLikelihood,
    required this.sumLogInnovationVariance,
    required this.sumWeightedSquares,
    required this.usedObservations,
    required this.measurementVariance,
    this.filteredMean,
    this.filteredCovariance,
    this.predictedMean,
    this.predictedCovariance,
  });

  final int stateDim;
  final int stepCount;

  /// Log marginal likelihood of the observations, diffuse burn-in excluded.
  final double logLikelihood;

  /// `sum log S_t` over the observations that entered the likelihood.
  final double sumLogInnovationVariance;

  /// `sum v_t^2 / S_t` over the same observations.
  final double sumWeightedSquares;

  /// How many observations entered the likelihood, i.e. all of them less the
  /// diffuse burn-in.
  final int usedObservations;

  /// The measurement variance the pass was run with.
  final double measurementVariance;

  final Float64List? filteredMean;
  final Float64List? filteredCovariance;
  final Float64List? predictedMean;
  final Float64List? predictedCovariance;

  /// Maximum-likelihood measurement variance given the *ratios* of all the
  /// other variances to it.
  ///
  /// Scaling every covariance in the model by a constant leaves the Kalman
  /// gains and every innovation `v_t` untouched and scales every `S_t` by that
  /// constant. So the measurement variance can be concentrated out of the
  /// likelihood analytically instead of being searched over — one dimension
  /// less for every fit, no matter how many components there are.
  double get profileMeasurementVariance =>
      measurementVariance * sumWeightedSquares / usedObservations;

  /// The likelihood at [profileMeasurementVariance], as a function of the
  /// variance ratios alone.
  ///
  /// Independent of the [measurementVariance] the pass happened to use: the
  /// scale cancels between `sum log S_t` and the fitted variance.
  double get profileLogLikelihood {
    final n = usedObservations;
    return -0.5 *
        (n * (_log2pi + 1) +
            n * math.log(sumWeightedSquares / n) +
            sumLogInnovationVariance);
  }
}

/// Forward Kalman recursion over a [Timeline].
///
/// Holds the workspace for one pass. Construction is cheap relative to the
/// pass itself (everything is `O(stateDim^2)`), so the fitting code simply
/// builds a new filter per likelihood evaluation rather than mutating one.
class KalmanFilter {
  KalmanFilter(
    this.components, {
    required this.measurementVariance,
    required this.diffuseVariance,
    int? burnIn,
  })  : burnIn = burnIn ?? _diffuseStateCount(components),
        stateDim = components.fold(0, (n, c) => n + c.stateDim),
        _offsets = _blockOffsets(components) {
    final n = stateDim;
    _x = Float64List(n);
    _p = Float64List(n * n);
    _xPred = Float64List(n);
    _pPred = Float64List(n * n);
    _a = Float64List(n * n);
    _q = Float64List(n * n);
    _work = Float64List(n * n);
    _h = Float64List(n);
    _ph = Float64List(n);
    _gain = Float64List(n);

    for (var b = 0; b < components.length; b++) {
      final dim = components[b].stateDim;
      final start = _offsets[b];
      _aBlocks.add(MatrixBlock(_a, start * n + start, n, dim, dim));
      _qBlocks.add(MatrixBlock(_q, start * n + start, n, dim, dim));
      _hSlices.add(Float64List.sublistView(_h, start, start + dim));
    }
  }

  final List<Component> components;
  final double measurementVariance;
  final double diffuseVariance;
  final int stateDim;

  /// How many leading observations are excluded from the likelihood.
  ///
  /// Defaults to the number of diffuse states: with a flat prior, the first
  /// few innovations only serve to locate the state and say nothing about the
  /// parameters, so including them would make the likelihood depend on the
  /// arbitrary size of the prior. Set it to zero when the prior is genuinely
  /// informative, which is what the dense-Gaussian-process cross-check does.
  final int burnIn;

  final List<int> _offsets;
  final List<MatrixBlock> _aBlocks = [];
  final List<MatrixBlock> _qBlocks = [];
  final List<Float64List> _hSlices = [];

  late final Float64List _x;
  late final Float64List _p;
  late final Float64List _xPred;
  late final Float64List _pPred;
  late final Float64List _a;
  late final Float64List _q;
  late final Float64List _work;
  late final Float64List _h;
  late final Float64List _ph;
  late final Float64List _gain;

  /// Innovation and its variance from the most recent [_update]. Kept as
  /// fields rather than returned in a wrapper so that the loop over a long
  /// series allocates nothing at all.
  double _innovation = 0;
  double _innovationVariance = 0;

  /// Gap the cached [_a] and [_q] were built for, or NaN if they are stale.
  /// A uniformly sampled series rebuilds them once and then never again, which
  /// is the common case for callers who ask for a regular output grid.
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

  static int _diffuseStateCount(List<Component> components) {
    var count = 0;
    for (final c in components) {
      for (final flag in c.diffuseStates) {
        if (flag) count++;
      }
    }
    return count;
  }

  /// Runs the forward pass. With [keepHistory] the filtered and predicted
  /// moments are retained for the RTS backward pass.
  FilterResult run(Timeline timeline, {bool keepHistory = false}) {
    final n = stateDim;
    final steps = timeline.length;

    final filteredMean = keepHistory ? Float64List(steps * n) : null;
    final filteredCov = keepHistory ? Float64List(steps * n * n) : null;
    final predictedMean = keepHistory ? Float64List(steps * n) : null;
    final predictedCov = keepHistory ? Float64List(steps * n * n) : null;

    _initialise();

    var seen = 0;
    var used = 0;
    var sumLogS = 0.0;
    var sumWeighted = 0.0;

    for (var t = 0; t < steps; t++) {
      if (t == 0) {
        // The prior is stated at the first step's time, so there is nothing
        // to propagate through yet.
        _pPred.setAll(0, _p);
        _xPred.setAll(0, _x);
      } else {
        _predict(timeline.gaps[t]);
      }

      if (keepHistory) {
        predictedMean!.setRange(t * n, (t + 1) * n, _xPred);
        predictedCov!.setRange(t * n * n, (t + 1) * n * n, _pPred);
      }

      if (timeline.hasObservation(t)) {
        final r = timeline.variances[t] * measurementVariance;
        _update(timeline.times[t], timeline.values[t], r);
        seen++;
        if (seen > burnIn) {
          used++;
          sumLogS += math.log(_innovationVariance);
          sumWeighted += _innovation * _innovation / _innovationVariance;
        }
      } else {
        _x.setAll(0, _xPred);
        _p.setAll(0, _pPred);
      }

      if (keepHistory) {
        filteredMean!.setRange(t * n, (t + 1) * n, _x);
        filteredCov!.setRange(t * n * n, (t + 1) * n * n, _p);
      }
    }

    return FilterResult(
      stateDim: n,
      stepCount: steps,
      logLikelihood: -0.5 * (used * _log2pi + sumLogS + sumWeighted),
      sumLogInnovationVariance: sumLogS,
      sumWeightedSquares: sumWeighted,
      usedObservations: used,
      measurementVariance: measurementVariance,
      filteredMean: filteredMean,
      filteredCovariance: filteredCov,
      predictedMean: predictedMean,
      predictedCovariance: predictedCov,
    );
  }

  /// Prior at the first step: a large multiple of the measurement variance on
  /// the diffuse states, whatever the component asks for on the rest.
  ///
  /// Scaling the prior by the measurement variance rather than fixing it in
  /// absolute terms keeps the whole model scale-equivariant, which is what
  /// makes [FilterResult.profileLogLikelihood] exact rather than merely close.
  void _initialise() {
    final n = stateDim;
    _x.fillRange(0, n, 0);
    _p.fillRange(0, n * n, 0);

    final kappa = diffuseVariance * measurementVariance;
    for (var b = 0; b < components.length; b++) {
      final component = components[b];
      final start = _offsets[b];
      final dim = component.stateDim;
      component.properPrior(
        Float64List.sublistView(_x, start, start + dim),
        MatrixBlock(_p, start * n + start, n, dim, dim),
      );
      final diffuse = component.diffuseStates;
      for (var i = 0; i < dim; i++) {
        if (!diffuse[i]) continue;
        final row = start + i;
        _x[row] = 0;
        for (var j = 0; j < n; j++) {
          _p[row * n + j] = 0;
          _p[j * n + row] = 0;
        }
        _p[row * n + row] = kappa;
      }
    }
  }

  /// `x- = A x`, `P- = A P A' + Q`, exploiting the block-diagonal structure of
  /// `A` and `Q`.
  ///
  /// The covariance itself is dense — the gain is a rank-one update spanning
  /// every state — but the transition is not, so this costs
  /// `2n * sum(n_i^2)` rather than `2n^3`.
  void _predict(double dt) {
    final n = stateDim;
    // Read the workspace fields once. A `late final` field carries an
    // initialisation check on every read, which costs about a third of a bare
    // matrix multiply at these sizes; over a hundred thousand steps it is
    // worth the two extra lines.
    final x = _x, p = _p, xPred = _xPred, pPred = _pPred;
    final a = _a, q = _q, work = _work;

    if (dt == 0) {
      // A = I and Q = 0, so the prediction is the previous posterior. This is
      // the same-timestamp case: two readings on one morning are simply two
      // updates.
      xPred.setAll(0, x);
      pPred.setAll(0, p);
      return;
    }

    if (dt != _cachedGap) {
      a.fillRange(0, n * n, 0);
      q.fillRange(0, n * n, 0);
      for (var b = 0; b < components.length; b++) {
        components[b].transition(dt, _aBlocks[b]);
        components[b].processNoise(dt, _qBlocks[b]);
      }
      _cachedGap = dt;
    }

    for (var b = 0; b < components.length; b++) {
      final start = _offsets[b];
      final dim = components[b].stateDim;
      for (var i = 0; i < dim; i++) {
        var sum = 0.0;
        for (var j = 0; j < dim; j++) {
          sum += a[(start + i) * n + start + j] * x[start + j];
        }
        xPred[start + i] = sum;
      }
    }

    // work = A P, one block row at a time.
    for (var b = 0; b < components.length; b++) {
      final start = _offsets[b];
      final dim = components[b].stateDim;
      for (var i = 0; i < dim; i++) {
        final out = (start + i) * n;
        for (var col = 0; col < n; col++) {
          var sum = 0.0;
          for (var j = 0; j < dim; j++) {
            sum += a[(start + i) * n + start + j] * p[(start + j) * n + col];
          }
          work[out + col] = sum;
        }
      }
    }

    // P- = work A', one block column at a time.
    for (var b = 0; b < components.length; b++) {
      final start = _offsets[b];
      final dim = components[b].stateDim;
      for (var row = 0; row < n; row++) {
        for (var i = 0; i < dim; i++) {
          var sum = 0.0;
          for (var j = 0; j < dim; j++) {
            sum += work[row * n + start + j] * a[(start + i) * n + start + j];
          }
          pPred[row * n + start + i] = sum;
        }
      }
    }

    for (var b = 0; b < components.length; b++) {
      final start = _offsets[b];
      final dim = components[b].stateDim;
      for (var i = 0; i < dim; i++) {
        for (var j = 0; j < dim; j++) {
          pPred[(start + i) * n + start + j] += q[(start + i) * n + start + j];
        }
      }
    }
  }

  /// Scalar measurement update in Joseph form.
  ///
  /// For a scalar observation `(I - KH) P- (I - KH)' + K R K'` expands to
  ///
  /// ```text
  /// P = P- - (P-H') K' - K (P-H')' + S K K'
  /// ```
  ///
  /// which is `O(n^2)` rather than `O(n^3)` and, computed as four symmetric
  /// terms and mirrored across the diagonal, is symmetric and positive
  /// semi-definite by construction for *any* gain — including one degraded by
  /// rounding. The textbook `P = (I - KH) P-` is algebraically the same thing
  /// and numerically worse; there is no reason to prefer it here.
  void _update(double time, double value, double r) {
    final n = stateDim;
    final x = _x, p = _p, xPred = _xPred, pPred = _pPred;
    final h = _h, ph = _ph, gain = _gain;

    for (var b = 0; b < components.length; b++) {
      components[b].observationAt(time, _hSlices[b]);
    }

    var predicted = 0.0;
    for (var i = 0; i < n; i++) {
      predicted += h[i] * xPred[i];
    }
    final v = value - predicted;

    for (var i = 0; i < n; i++) {
      var sum = 0.0;
      for (var j = 0; j < n; j++) {
        sum += pPred[i * n + j] * h[j];
      }
      ph[i] = sum;
    }

    var s = r;
    for (var i = 0; i < n; i++) {
      s += h[i] * ph[i];
    }
    if (!(s > 0) || !s.isFinite) {
      throw StateError('Innovation variance $s at time $time is not positive. '
          'The model has become numerically degenerate; check for a zero '
          'process variance combined with a zero observation variance.');
    }

    for (var i = 0; i < n; i++) {
      gain[i] = ph[i] / s;
      x[i] = xPred[i] + gain[i] * v;
    }

    for (var i = 0; i < n; i++) {
      for (var j = i; j < n; j++) {
        final updated = pPred[i * n + j] -
            ph[i] * gain[j] -
            gain[i] * ph[j] +
            s * gain[i] * gain[j];
        p[i * n + j] = updated;
        p[j * n + i] = updated;
      }
    }

    _innovation = v;
    _innovationVariance = s;
  }
}
