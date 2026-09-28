import 'dart:math' as math;
import 'dart:typed_data';

import '../component.dart';
import '../exceptions.dart';
import '../initialization.dart';
import 'cholesky.dart';
import 'layout.dart';
import 'matrix_block.dart';
import 'recursive_residuals.dart';
import 'scale.dart';
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
  /// Assembled by [KalmanFilter.run] and `FastPath2x2.run`.
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
    this.stateMean,
    this.stateCovariance,
    this.predictedMean,
    this.predictedCovariance,
    this.stateSensitivity,
    this.predictedSensitivity,
    this.residualTimes,
    this.standardisedResiduals,
  });

  /// States in the model.
  final int stateDim;

  /// Steps in the timeline the pass ran over.
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
  /// determinant of its precision. Densely it is `log|B' C^-1 B|`, and with it
  /// the result is exactly the restricted likelihood.
  ///
  /// It does **not** make likelihoods comparable across models with different
  /// diffuse dimensions. The integral is against an improper flat prior of unit density,
  /// so `d` carries units and the answer carries them too: scaling one column
  /// of `B` by `c` scales `|M|` by `c^2` and shifts the log likelihood by
  /// exactly `-log c`, without changing the model, the data or the posterior.
  /// Rescaling a regression column or changing the time unit both do this. See
  /// [diffuseDim] and `FitResult.isComparableWith`.
  final double diffuseLogDeterminant;

  /// Generalised-least-squares estimate of the flat directions, length
  /// [diffuseDim].
  final Float64List? diffuseMean;

  /// Its covariance, `M^-1`, laid out row-major.
  final Float64List? diffuseCovariance;

  /// The state moments, step-major, populated only when the pass was asked to
  /// keep a history.
  ///
  /// Filtered when the forward pass returns them: `E[x_t | y_1..t]`. The
  /// backward pass overwrites them in place with the smoothed moments, and
  /// `RtsSmoother.combineDiffuse` then folds the flat directions in — so what
  /// these hold depends on how far along the pipeline the caller is, which is
  /// why they are not named for any one stage. The memory argument for
  /// overwriting is in `RtsSmoother`.
  final Float64List? stateMean;

  /// Its covariance, `stateDim x stateDim` per step, row-major.
  final Float64List? stateCovariance;

  /// The one-step-ahead moments, which stay predicted throughout.
  final Float64List? predictedMean;

  /// Their covariance, laid out as [stateCovariance].
  final Float64List? predictedCovariance;

  /// Sensitivity of the state to the flat directions, step-major and
  /// `stateDim x diffuseDim` per step. Null under an approximate prior.
  ///
  /// Smoothed alongside [stateMean], and consumed by
  /// `RtsSmoother.combineDiffuse`.
  ///
  /// Note what combining this with [diffuseMean] does and does not give. The
  /// estimate of the flat directions uses every observation, so
  /// `stateMean + stateSensitivity * diffuseMean` is conditioned on all
  /// the data in those directions and on the data so far in the others. That
  /// is the right combination after the backward pass, and at the last step,
  /// and a mixture of two conditionings anywhere else. A genuine filtered
  /// state under a flat prior needs the estimate rebuilt from the data up to
  /// that step, which nothing in this package currently asks for.
  final Float64List? stateSensitivity;

  /// The same, before each step's update.
  final Float64List? predictedSensitivity;

  /// Time of each standardised residual, populated only when the pass was
  /// asked for them.
  final Float64List? residualTimes;

  /// Standardised one-step-ahead prediction errors, `N - d` of them: see
  /// [recursiveResiduals] for what they are under a flat prior, which is not
  /// quite the obvious thing.
  final Float64List? standardisedResiduals;

  /// Whether anything is left to estimate a noise level from.
  ///
  /// False when every observation went on locating the flat directions —
  /// exactly `d` readings under a flat prior, or none at all — in which case
  /// both quantities below are undefined rather than merely imprecise.
  bool get hasResidualDegreesOfFreedom => usedObservations > 0;

  /// Restricted maximum-likelihood measurement variance given the *ratios* of
  /// all the other variances to it, or [double.nan] when
  /// [hasResidualDegreesOfFreedom] is false.
  ///
  /// Scaling every covariance in the model by a constant leaves the Kalman
  /// gains and every innovation `v_t` untouched and scales every `S_t` by that
  /// constant. So the measurement variance can be concentrated out of the
  /// likelihood analytically instead of being searched over — one dimension
  /// less for every fit, no matter how many components there are.
  ///
  /// Restricted rather than plain maximum likelihood: the divisor is
  /// [usedObservations], which under a flat prior is `N - d` rather than `N`.
  /// That is the right partner for a likelihood that has integrated `d`
  /// directions away, and it is a factor of `N / (N - d)` away from the plain
  /// estimate — twenty-five per cent on thirty readings with six flat
  /// directions.
  double get profileMeasurementVariance => hasResidualDegreesOfFreedom
      ? measurementVariance * sumWeightedSquares / usedObservations
      : double.nan;

  /// The likelihood with every variance in the model scaled so that the
  /// measurement variance is [variance], keeping the ratios this pass was run
  /// at, or [double.nan] when there is nothing left to profile over.
  ///
  /// At [profileMeasurementVariance] this is [profileLogLikelihood]. It is
  /// what the fit evaluates when the profiled variance falls below a floor.
  double logLikelihoodAtScale(double variance) {
    if (!hasResidualDegreesOfFreedom) return double.nan;
    final n = usedObservations;
    final weighted = sumWeightedSquares > 0 ? sumWeightedSquares : 0.0;
    return -0.5 *
        (n * _log2pi +
            n * math.log(variance / measurementVariance) +
            weighted * measurementVariance / variance +
            sumLogInnovationVariance +
            diffuseLogDeterminant);
  }

  /// The likelihood at [profileMeasurementVariance], as a function of the
  /// variance ratios alone, or [double.nan] when there is nothing left to
  /// profile over.
  ///
  /// Independent of the [measurementVariance] the pass happened to use: the
  /// scale cancels between `sum log S_t` and the fitted variance.
  double get profileLogLikelihood {
    if (!hasResidualDegreesOfFreedom) return double.nan;
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
  /// A filter for [components] at [measurementVariance] under
  /// [initialization], excluding the first [burnIn] observations from the
  /// likelihood.
  KalmanFilter(
    this.components, {
    required this.measurementVariance,
    required this.initialization,
    int? burnIn,
  }) : _offsets = blockOffsets(components),
       _diffuseStates = diffuseStateIndices(components),
       stateDim = components.fold(0, (n, c) => n + c.stateDim),
       diffuseDim = initialization is ExactDiffuse
           ? diffuseStateIndices(components).length
           : 0,
       burnIn =
           burnIn ??
           (initialization is ExactDiffuse
               ? 0
               : diffuseStateIndices(components).length) {
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

  /// The model's components, in state order.
  final List<Component> components;

  /// Noise variance of an observation of unit relative variance.
  final double measurementVariance;

  /// The prior on the first step's state.
  final Initialization initialization;

  /// Total states across [components].
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

  /// Runs the forward pass. With [keepHistory] the filtered and predicted
  /// moments are retained for the RTS backward pass.
  FilterResult run(
    Timeline timeline, {
    bool keepHistory = false,
    bool keepResiduals = false,
  }) {
    final n = stateDim;
    final d = diffuseDim;
    final steps = timeline.length;
    final pieces = keepResiduals
        ? ResidualPieces(timeline.observationCount, d)
        : null;

    final stateMean = keepHistory ? Float64List(steps * n) : null;
    final filteredCov = keepHistory ? Float64List(steps * n * n) : null;
    final predictedMean = keepHistory ? Float64List(steps * n) : null;
    final predictedCov = keepHistory ? Float64List(steps * n * n) : null;
    final stateSensitivity = keepHistory && d > 0
        ? Float64List(steps * n * d)
        : null;
    final predictedSensitivity = keepHistory && d > 0
        ? Float64List(steps * n * d)
        : null;

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
        pieces?.add(timeline.times[t], _innovation, _innovationVariance, _vb);
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
        stateMean!.setRange(t * n, (t + 1) * n, _x);
        filteredCov!.setRange(t * n * n, (t + 1) * n * n, _p);
        stateSensitivity?.setRange(t * n * d, (t + 1) * n * d, _xb);
      }
    }

    Float64List? diffuseMean;
    Float64List? diffuseCovariance;
    var diffuseLogDeterminant = 0.0;
    // With no steps there is nothing to report and nothing to estimate, so
    // the diffuse system is left unsolved rather than declared singular.
    if (d > 0 && steps > 0) {
      final solved = _solveDiffuseSystem(timeline);
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

    final residuals = pieces == null
        ? null
        : recursiveResiduals(pieces, burnIn: burnIn);

    return FilterResult(
      stateDim: n,
      stepCount: steps,
      logLikelihood:
          -0.5 *
          (used * _log2pi + sumLogS + sumWeighted + diffuseLogDeterminant),
      sumLogInnovationVariance: sumLogS,
      sumWeightedSquares: sumWeighted,
      usedObservations: used,
      measurementVariance: measurementVariance,
      diffuseDim: d,
      diffuseLogDeterminant: diffuseLogDeterminant,
      diffuseMean: diffuseMean,
      diffuseCovariance: diffuseCovariance,
      stateMean: stateMean,
      stateCovariance: filteredCov,
      predictedMean: predictedMean,
      predictedCovariance: predictedCov,
      stateSensitivity: stateSensitivity,
      predictedSensitivity: predictedSensitivity,
      residualTimes: residuals?.times,
      standardisedResiduals: residuals?.values,
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
    double from,
    Float64List horizon,
    FilterResult result,
  ) {
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
  _solveDiffuseSystem(Timeline timeline) {
    final d = diffuseDim;
    final factor = Float64List.fromList(_information);
    if (timeline.observationCount < d || !factorInformation(factor, d)) {
      throw UnderdeterminedModelException(
        singularDiffuseMessage(components, d, timeline),
      );
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
  /// semi-definite by construction for *any* gain, including one degraded by
  /// rounding, unlike the textbook `P = (I - KH) P-`.
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

    // Dense over both indices, unlike the reporting code, which skips zero
    // entries of H. This runs once per observation and is `O(n^2)` either way;
    // a branch inside it would cost more on the models where H is dense than
    // it saves on the ones where it is not.
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
      throw NumericalBreakdownException(innovationVarianceMessage(s, r, time));
    }

    for (var i = 0; i < n; i++) {
      gain[i] = ph[i] / s;
      x[i] = xPred[i] + gain[i] * v;
    }

    for (var i = 0; i < n; i++) {
      for (var j = i; j < n; j++) {
        final updated =
            pPred[i * n + j] -
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
String singularDiffuseMessage(
  List<Component> components,
  int diffuseDim,
  Timeline timeline,
) {
  final observationCount = timeline.observationCount;
  final times = [
    for (var t = 0; t < timeline.length; t++)
      if (timeline.hasObservation(t)) timeline.times[t],
  ];
  final from = times.isEmpty ? 0.0 : times.first;
  final to = times.isEmpty ? 0.0 : times.last;
  final resolution = typicalGap(times);
  final reasons = <String>[];
  if (observationCount < diffuseDim) {
    reasons.add(
      'there are only $observationCount observations for '
      '$diffuseDim flat directions, and each one costs a degree of freedom',
    );
  }
  for (final component in components) {
    final hint = component.identifiabilityHint(
      from,
      to,
      resolution: resolution,
    );
    if (hint != null) reasons.add(hint);
  }
  if (reasons.isEmpty && components.length > 1) {
    reasons.add(
      'two components can produce the same signal on this data — a '
      'trend and a level both supply a level, and two seasonals that share '
      'a harmonic are the same function of time',
    );
  }

  final buffer = StringBuffer(
    'the data does not determine the model\'s '
    '$diffuseDim diffuse states: $observationCount observations left the '
    'diffuse information matrix singular',
  );
  if (reasons.isEmpty) {
    buffer.write('. ');
  } else {
    buffer.write(', because ${reasons.join('; and ')}. ');
  }
  buffer.write(
    'Either supply more data, drop a component, or use '
    'ApproximateDiffuse, which returns a very wide posterior instead of '
    'refusing when time is in a unit that keeps rates of change near one, '
    'such as days.',
  );
  return buffer.toString();
}

/// Explains an innovation variance that is not a positive finite number.
///
/// Shared by both forward-pass implementations.
String innovationVarianceMessage(double s, double r, double time) {
  if (!r.isFinite) {
    return 'the measurement variance of the observation at time $time, its '
        'relativeVariance times the model\'s measurementVariance, is not a '
        'finite number';
  }
  if (r == 0) {
    return 'the observation at time $time has relativeVariance 0 and measures '
        'a direction the model has no uncertainty about, so it cannot be '
        'honoured exactly. Under ExactDiffuse the first observation always '
        'does: its value is not yet a constraint on anything. Give it a small '
        'positive relativeVariance instead';
  }
  return 'the innovation variance at time $time is $s, which is not a '
      'positive number: a component\'s process noise is not a valid '
      'covariance';
}
