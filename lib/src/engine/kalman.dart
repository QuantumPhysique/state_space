import 'dart:math' as math;
import 'dart:typed_data';

import '../component.dart';
import '../initialization.dart';
import 'cholesky.dart';
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
    required this.diffuseDim,
    this.diffuseLogDeterminant = 0,
    this.diffuseMean,
    this.diffuseCovariance,
    this.filteredMean,
    this.filteredCovariance,
    this.predictedMean,
    this.predictedCovariance,
    this.filteredSensitivity,
    this.predictedSensitivity,
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

  /// Number of flat directions handled exactly, zero under an approximate
  /// prior.
  final int diffuseDim;

  /// `log|M|`, where `M` is the information the data carries about the flat
  /// directions. Zero when there are none.
  ///
  /// This is the term that makes the diffuse likelihood a *marginal*
  /// likelihood: integrating a flat prior out of a Gaussian leaves behind the
  /// determinant of its precision, and dropping it would make likelihoods
  /// incomparable across models with different diffuse dimensions.
  final double diffuseLogDeterminant;

  /// Generalised-least-squares estimate of the flat directions, length
  /// [diffuseDim].
  final Float64List? diffuseMean;

  /// Its covariance, `M^-1`, laid out row-major.
  final Float64List? diffuseCovariance;

  final Float64List? filteredMean;
  final Float64List? filteredCovariance;
  final Float64List? predictedMean;
  final Float64List? predictedCovariance;

  /// Sensitivity of the filtered state to the flat directions, step-major and
  /// `stateDim x diffuseDim` per step. Null under an approximate prior.
  ///
  /// Note what combining this with [diffuseMean] does and does not give. The
  /// estimate of the flat directions uses every observation, so
  /// `filteredMean + filteredSensitivity * diffuseMean` is conditioned on all
  /// the data in those directions and on the data so far in the others. That
  /// is the right combination after the backward pass, and at the last step,
  /// and a mixture of two conditionings anywhere else. A genuine filtered
  /// state under a flat prior needs the estimate rebuilt from the data up to
  /// that step, which nothing in this package currently asks for.
  final Float64List? filteredSensitivity;

  /// The same, before each step's update.
  final Float64List? predictedSensitivity;

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
            sumLogInnovationVariance +
            diffuseLogDeterminant);
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
    required this.initialization,
    int? burnIn,
  })  : diffuseDim =
            initialization is ExactDiffuse ? _diffuseStateCount(components) : 0,
        burnIn = burnIn ??
            (initialization is ExactDiffuse
                ? 0
                : _diffuseStateCount(components)),
        stateDim = components.fold(0, (n, c) => n + c.stateDim),
        _offsets = _blockOffsets(components),
        _diffuseStates = _diffuseStateIndices(components) {
    final n = stateDim;
    final d = diffuseDim;
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
    _xb = Float64List(n * d);
    _xbPred = Float64List(n * d);
    _information = Float64List(d * d);
    _diffuseRhs = Float64List(d);
    _vb = Float64List(d);

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
  final Initialization initialization;
  final int stateDim;

  /// Number of flat directions carried exactly. Zero under an approximate
  /// prior, in which case none of the sensitivity machinery runs.
  final int diffuseDim;

  /// How many leading observations are excluded from the likelihood.
  ///
  /// Defaults to the number of diffuse states: with a flat prior, the first
  /// few innovations only serve to locate the state and say nothing about the
  /// parameters, so including them would make the likelihood depend on the
  /// arbitrary size of the prior. Set it to zero when the prior is genuinely
  /// informative, which is what the dense-Gaussian-process cross-check does.
  final int burnIn;

  final List<int> _offsets;

  /// Global state index of each flat direction, in order.
  final List<int> _diffuseStates;

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

  /// `dx/dd`, the sensitivity of the state to the flat directions, laid out
  /// `stateDim x diffuseDim` row-major.
  late final Float64List _xb;
  late final Float64List _xbPred;

  /// The generalised-least-squares system for the flat directions: `M` and
  /// the right-hand side, accumulated one observation at a time.
  late final Float64List _information;
  late final Float64List _diffuseRhs;
  late final Float64List _vb;

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

  static List<int> _diffuseStateIndices(List<Component> components) {
    final indices = <int>[];
    var next = 0;
    for (final c in components) {
      for (final flag in c.diffuseStates) {
        if (flag) indices.add(next);
        next++;
      }
    }
    return indices;
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
    final d = diffuseDim;
    final steps = timeline.length;

    final filteredMean = keepHistory ? Float64List(steps * n) : null;
    final filteredCov = keepHistory ? Float64List(steps * n * n) : null;
    final predictedMean = keepHistory ? Float64List(steps * n) : null;
    final predictedCov = keepHistory ? Float64List(steps * n * n) : null;
    final filteredSensitivity =
        keepHistory && d > 0 ? Float64List(steps * n * d) : null;
    final predictedSensitivity =
        keepHistory && d > 0 ? Float64List(steps * n * d) : null;

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
        if (d > 0) _xbPred.setAll(0, _xb);
      } else {
        _predict(timeline.gaps[t]);
      }

      if (keepHistory) {
        predictedMean!.setRange(t * n, (t + 1) * n, _xPred);
        predictedCov!.setRange(t * n * n, (t + 1) * n * n, _pPred);
        predictedSensitivity?.setRange(t * n * d, (t + 1) * n * d, _xbPred);
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
        if (d > 0) _xb.setAll(0, _xbPred);
      }

      if (keepHistory) {
        filteredMean!.setRange(t * n, (t + 1) * n, _x);
        filteredCov!.setRange(t * n * n, (t + 1) * n * n, _p);
        filteredSensitivity?.setRange(t * n * d, (t + 1) * n * d, _xb);
      }
    }

    Float64List? diffuseMean;
    Float64List? diffuseCovariance;
    var diffuseLogDeterminant = 0.0;
    // With no steps there is nothing to report and nothing to estimate, so
    // the diffuse system is left unsolved rather than declared singular.
    if (d > 0 && steps > 0) {
      final solved = _solveDiffuseSystem(timeline.observationCount,
          timeline.times[steps - 1] - timeline.times[0]);
      diffuseMean = solved.mean;
      diffuseCovariance = solved.covariance;
      diffuseLogDeterminant = solved.logDeterminant;
      // Substituting the estimate back into the weighted residual sum leaves
      // `rss(dhat) = rss(0) + rhs' dhat`, because `M dhat = -rhs` collapses the
      // quadratic term onto the linear one.
      for (var c = 0; c < d; c++) {
        sumWeighted += _diffuseRhs[c] * diffuseMean[c];
      }
      // Each flat direction costs one degree of freedom, exactly as it would
      // in an ordinary regression.
      used = timeline.observationCount - d;
    }

    return FilterResult(
      stateDim: n,
      stepCount: steps,
      logLikelihood: -0.5 *
          (used * _log2pi + sumLogS + sumWeighted + diffuseLogDeterminant),
      sumLogInnovationVariance: sumLogS,
      sumWeightedSquares: sumWeighted,
      usedObservations: used,
      measurementVariance: measurementVariance,
      diffuseDim: d,
      diffuseLogDeterminant: diffuseLogDeterminant,
      diffuseMean: diffuseMean,
      diffuseCovariance: diffuseCovariance,
      filteredMean: filteredMean,
      filteredCovariance: filteredCov,
      predictedMean: predictedMean,
      predictedCovariance: predictedCov,
      filteredSensitivity: filteredSensitivity,
      predictedSensitivity: predictedSensitivity,
    );
  }

  /// Walks the state forward past the end of the data, with nothing to
  /// update on.
  ///
  /// Must be called on the same filter that produced [result], directly after
  /// [run], because it continues from the workspace that pass left behind.
  /// Forecasting is just prediction with the update skipped, which is the same
  /// thing the recursion does over any gap in the middle of a series -- the
  /// only difference is that no observation ever arrives to close it.
  ({Float64List mean, Float64List variance}) project(
      double from, Float64List horizon, FilterResult result) {
    final n = stateDim;
    final d = diffuseDim;
    final estimate = result.diffuseMean;
    final spread = result.diffuseCovariance;
    final mean = Float64List(horizon.length);
    final variance = Float64List(horizon.length);
    final loading = Float64List(d);

    var previous = from;
    for (var k = 0; k < horizon.length; k++) {
      _predict(horizon[k] - previous);
      _x.setAll(0, _xPred);
      _p.setAll(0, _pPred);
      if (d > 0) _xb.setAll(0, _xbPred);
      previous = horizon[k];

      for (var b = 0; b < components.length; b++) {
        components[b].observationAt(horizon[k], _hSlices[b]);
      }

      var signal = 0.0;
      for (var i = 0; i < n; i++) {
        signal += _h[i] * _x[i];
      }
      var spreadHere = 0.0;
      for (var i = 0; i < n; i++) {
        if (_h[i] == 0) continue;
        var row = 0.0;
        for (var j = 0; j < n; j++) {
          row += _p[i * n + j] * _h[j];
        }
        spreadHere += _h[i] * row;
      }

      if (d > 0) {
        // The flat directions contribute through H dx/dd, exactly as they do
        // to a smoothed step.
        for (var c = 0; c < d; c++) {
          var sum = 0.0;
          for (var i = 0; i < n; i++) {
            sum += _h[i] * _xb[i * d + c];
          }
          loading[c] = sum;
          signal += sum * estimate![c];
        }
        for (var r = 0; r < d; r++) {
          for (var c = 0; c < d; c++) {
            spreadHere += loading[r] * spread![r * d + c] * loading[c];
          }
        }
      }

      mean[k] = signal;
      variance[k] = spreadHere;
    }

    return (mean: mean, variance: variance);
  }

  /// Integrates the flat directions out of the likelihood.
  ///
  /// With a flat prior on `d`, the joint density is Gaussian in `d`, so the
  /// integral is available in closed form: the estimate is the weighted
  /// least-squares solution of `M d = -rhs`, its covariance is `M^-1`, and the
  /// integration leaves `log|M|` behind in the likelihood.
  ({Float64List mean, Float64List covariance, double logDeterminant})
      _solveDiffuseSystem(int observationCount, double span) {
    final d = diffuseDim;
    final factor = Float64List.fromList(_information);
    if (!choleskyFactor(factor, d)) {
      throw StateError(
          singularDiffuseMessage(components, d, observationCount, span));
    }

    var logDeterminant = 0.0;
    for (var i = 0; i < d; i++) {
      logDeterminant += 2 * math.log(factor[i * d + i]);
    }

    final mean = Float64List(d);
    for (var c = 0; c < d; c++) {
      mean[c] = -_diffuseRhs[c];
    }
    choleskySolve(factor, d, mean, 0);

    // M is symmetric, so its inverse is too: solving against each unit vector
    // in turn fills one row, which is also one column.
    final covariance = Float64List(d * d);
    for (var r = 0; r < d; r++) {
      covariance[r * d + r] = 1;
      choleskySolve(factor, d, covariance, r * d);
    }

    return (mean: mean, covariance: covariance, logDeterminant: logDeterminant);
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

    // Under exact initialisation the flat directions get no prior variance at
    // all: they are carried in the sensitivity instead, and integrated out at
    // the end of the pass.
    final kappa = switch (initialization) {
      ExactDiffuse() => 0.0,
      ApproximateDiffuse(:final variance) => variance * measurementVariance,
    };
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

    final d = diffuseDim;
    if (d == 0) return;
    _xb.fillRange(0, n * d, 0);
    for (var c = 0; c < d; c++) {
      _xb[_diffuseStates[c] * d + c] = 1;
    }
    _information.fillRange(0, d * d, 0);
    _diffuseRhs.fillRange(0, d, 0);
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
    final d = diffuseDim;

    if (dt == 0) {
      // A = I and Q = 0, so the prediction is the previous posterior. This is
      // the same-timestamp case: two readings on one morning are simply two
      // updates.
      xPred.setAll(0, x);
      pPred.setAll(0, p);
      if (d > 0) _xbPred.setAll(0, _xb);
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

    // The sensitivity to the flat directions rides along on the same
    // transition. It is d extra mean propagations and no extra covariance
    // work, which is the whole reason exact initialisation is affordable here.
    if (d > 0) {
      final xb = _xb, xbPred = _xbPred;
      for (var b = 0; b < components.length; b++) {
        final start = _offsets[b];
        final dim = components[b].stateDim;
        for (var i = 0; i < dim; i++) {
          final row = (start + i) * n + start;
          for (var c = 0; c < d; c++) {
            var sum = 0.0;
            for (var j = 0; j < dim; j++) {
              sum += a[row + j] * xb[(start + j) * d + c];
            }
            xbPred[(start + i) * d + c] = sum;
          }
        }
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

    if (diffuseDim > 0) _updateSensitivity(v, s);

    _innovation = v;
    _innovationVariance = s;
  }

  /// Carries the sensitivity through the same update, and accumulates this
  /// observation's contribution to the least-squares system for the flat
  /// directions.
  ///
  /// The innovation is affine in the unknown `d`: `v(d) = va + Vb d`, where
  /// `va` is the innovation the filter just computed and `Vb = -H dx/dd`. The
  /// sensitivity updates exactly like the state does, but against zero data,
  /// because the gain does not depend on `d`.
  void _updateSensitivity(double va, double s) {
    final n = stateDim;
    final d = diffuseDim;
    final xb = _xb, xbPred = _xbPred, vb = _vb, h = _h, gain = _gain;
    final information = _information, rhs = _diffuseRhs;

    for (var c = 0; c < d; c++) {
      var sum = 0.0;
      for (var i = 0; i < n; i++) {
        sum += h[i] * xbPred[i * d + c];
      }
      vb[c] = -sum;
    }

    for (var i = 0; i < n; i++) {
      final g = gain[i];
      for (var c = 0; c < d; c++) {
        xb[i * d + c] = xbPred[i * d + c] + g * vb[c];
      }
    }

    for (var r = 0; r < d; r++) {
      final weighted = vb[r] / s;
      rhs[r] += weighted * va;
      for (var c = 0; c < d; c++) {
        information[r * d + c] += weighted * vb[c];
      }
    }
  }
}

/// Explains a singular diffuse information matrix as well as the model
/// allows.
///
/// Shared by both forward-pass implementations so that the two never drift
/// apart on the one error a caller is actually likely to hit. The engine
/// contributes the arithmetic — how many flat directions there are and how
/// much data there was — and the components contribute whatever they know
/// about their own identifiability, which is knowledge the engine is
/// deliberately kept clear of.
String singularDiffuseMessage(List<Component> components, int diffuseDim,
    int observationCount, double span) {
  final reasons = <String>[];
  if (observationCount < diffuseDim) {
    reasons.add('there are only $observationCount observations for '
        '$diffuseDim flat directions, and each one costs a degree of freedom');
  }
  for (final component in components) {
    final hint = component.identifiabilityHint(span);
    if (hint != null) reasons.add(hint);
  }
  if (reasons.isEmpty && components.length > 1) {
    reasons.add('two components can produce the same signal on this data — a '
        'trend and a level both supply a level, and two seasonals that share '
        'a harmonic are the same function of time');
  }

  final buffer = StringBuffer('the data does not determine the model\'s '
      '$diffuseDim diffuse states: $observationCount observations left the '
      'diffuse information matrix singular');
  if (reasons.isEmpty) {
    buffer.write('. ');
  } else {
    buffer.write(', because ${reasons.join('; and ')}. ');
  }
  buffer.write('Either supply more data, drop a component, or fall back to '
      'ApproximateDiffuse, which returns a very large variance instead of '
      'refusing.');
  return buffer.toString();
}
