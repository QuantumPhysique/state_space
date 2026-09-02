import 'dart:math' as math;
import 'dart:typed_data';

import '../component.dart';
import '../initialization.dart';
import 'cholesky.dart';
import 'kalman.dart';
import 'matrix_block.dart';
import 'recursive_residuals.dart';
import 'timeline.dart';

const double _log2pi = 1.8378770664093456;

/// Runs the forward pass on whichever engine suits the model.
///
/// The specialisation below is chosen when it applies and the generic engine
/// otherwise. Callers get the same [FilterResult] either way and never need to
/// know which ran; the equivalence test is what makes that claim safe.
FilterResult forwardPass(
  List<Component> components,
  Timeline timeline, {
  required double measurementVariance,
  required Initialization initialization,
  int? burnIn,
  bool keepHistory = false,
  bool keepResiduals = false,
}) {
  if (burnIn == null && FastPath2x2.handles(components, initialization)) {
    return FastPath2x2(
      components.single,
      measurementVariance: measurementVariance,
      initialization: initialization,
    ).run(timeline, keepHistory: keepHistory, keepResiduals: keepResiduals);
  }
  return KalmanFilter(
    components,
    measurementVariance: measurementVariance,
    initialization: initialization,
    burnIn: burnIn,
  ).run(timeline, keepHistory: keepHistory, keepResiduals: keepResiduals);
}

/// The forward pass for a model that is one two-state component, written out
/// in scalars.
///
/// Nothing here is new mathematics. It is the same predict, the same
/// Joseph-form update, the same augmentation for the flat directions, with the
/// loops unrolled and the state held in local doubles instead of typed arrays
/// — no bounds checks, no block offsets, no matrix views.
///
/// It exists because [fit] runs a forward pass per likelihood evaluation, some
/// fifty times per call, while the backward pass runs once. Optimising the
/// backward pass would be optimising the wrong half, so it is left generic.
///
/// **This is an optimisation, and it is never load-bearing.** The generic
/// engine came first, is what the golden and dense-Gaussian-process tests
/// validate, and remains the definition of what the answer is. This path is
/// held to it by `fast_path_equivalence_test.dart`, which asserts agreement to
/// 1e-12 across irregular gaps, repeated timestamps, missing observations and
/// both initialisations. If the two ever disagree, this one is wrong.
class FastPath2x2 {
  FastPath2x2(
    this.component, {
    required this.measurementVariance,
    required this.initialization,
  }) : _exact = initialization is ExactDiffuse;

  /// Whether the specialisation covers this model.
  ///
  /// One component, two states, and — under exact initialisation — both of
  /// them flat. A component with one flat state and one proper state is
  /// perfectly legal and simply goes down the generic path, because carrying a
  /// one-column sensitivity in unrolled scalars would double this file to
  /// serve a case nothing yet produces.
  static bool handles(
      List<Component> components, Initialization initialization) {
    if (components.length != 1) return false;
    final component = components.single;
    if (component.stateDim != 2) return false;
    if (initialization is! ExactDiffuse) return true;
    final diffuse = component.diffuseStates;
    return diffuse[0] && diffuse[1];
  }

  final Component component;
  final double measurementVariance;
  final Initialization initialization;
  final bool _exact;

  final MatrixBlock _transition = MatrixBlock.dense(2, 2);
  final MatrixBlock _noise = MatrixBlock.dense(2, 2);
  final Float64List _observation = Float64List(2);

  double _a00 = 0, _a01 = 0, _a10 = 0, _a11 = 0;
  double _q00 = 0, _q01 = 0, _q11 = 0;
  double _cachedGap = double.nan;

  void _buildStep(double dt) {
    if (dt == _cachedGap) return;
    // Zeroed first, as the generic engine zeroes its buffers. `Component` says
    // the block "may contain stale values", so a component that leaves an
    // entry unwritten is within its rights; without this it would work on the
    // generic path and be silently wrong here, which is the one way this
    // specialisation could become load-bearing.
    _transition.fill(0);
    _noise.fill(0);
    component.transition(dt, _transition);
    component.processNoise(dt, _noise);
    _a00 = _transition.at(0, 0);
    _a01 = _transition.at(0, 1);
    _a10 = _transition.at(1, 0);
    _a11 = _transition.at(1, 1);
    _q00 = _noise.at(0, 0);
    _q01 = _noise.at(0, 1);
    _q11 = _noise.at(1, 1);
    _cachedGap = dt;
  }

  /// The same contract as [KalmanFilter.run].
  FilterResult run(Timeline timeline,
      {bool keepHistory = false, bool keepResiduals = false}) {
    final steps = timeline.length;
    final dim = _exact ? 2 : 0;
    final pieces =
        keepResiduals ? ResidualPieces(timeline.observationCount, dim) : null;
    final loading = Float64List(dim);

    final filteredMean = keepHistory ? Float64List(steps * 2) : null;
    final filteredCov = keepHistory ? Float64List(steps * 4) : null;
    final predictedMean = keepHistory ? Float64List(steps * 2) : null;
    final predictedCov = keepHistory ? Float64List(steps * 4) : null;
    final filteredSensitivity =
        keepHistory && _exact ? Float64List(steps * 4) : null;
    final predictedSensitivity =
        keepHistory && _exact ? Float64List(steps * 4) : null;

    // Prior. Under exact initialisation the flat directions carry no variance
    // at all; under the approximate one they carry a wide proper prior.
    var x0 = 0.0, x1 = 0.0;
    var p00 = 0.0, p01 = 0.0, p11 = 0.0;
    if (!_exact) {
      final mean = Float64List(2);
      final covariance = MatrixBlock.dense(2, 2);
      component.properPrior(mean, covariance);
      x0 = mean[0];
      x1 = mean[1];
      p00 = covariance.at(0, 0);
      p01 = covariance.at(0, 1);
      p11 = covariance.at(1, 1);
      final kappa =
          (initialization as ApproximateDiffuse).variance * measurementVariance;
      final diffuse = component.diffuseStates;
      if (diffuse[0]) {
        x0 = 0;
        p00 = kappa;
        p01 = 0;
      }
      if (diffuse[1]) {
        x1 = 0;
        p11 = kappa;
        p01 = 0;
      }
    }

    // Sensitivity of the state to the flat directions, as an identity to
    // start with: at the first step the state *is* the flat directions.
    var b00 = 1.0, b01 = 0.0, b10 = 0.0, b11 = 1.0;
    var m00 = 0.0, m01 = 0.0, m11 = 0.0;
    var rhs0 = 0.0, rhs1 = 0.0;

    final burn = _exact
        ? 0
        : (component.diffuseStates[0] ? 1 : 0) +
            (component.diffuseStates[1] ? 1 : 0);
    var seen = 0;
    var used = 0;
    var sumLogS = 0.0;
    var sumWeighted = 0.0;

    for (var t = 0; t < steps; t++) {
      if (t > 0) {
        final dt = timeline.gaps[t];
        if (dt != 0) {
          _buildStep(dt);
          final nx0 = _a00 * x0 + _a01 * x1;
          final nx1 = _a10 * x0 + _a11 * x1;
          final t00 = _a00 * p00 + _a01 * p01;
          final t01 = _a00 * p01 + _a01 * p11;
          final t10 = _a10 * p00 + _a11 * p01;
          final t11 = _a10 * p01 + _a11 * p11;
          p00 = t00 * _a00 + t01 * _a01 + _q00;
          p01 = t00 * _a10 + t01 * _a11 + _q01;
          p11 = t10 * _a10 + t11 * _a11 + _q11;
          x0 = nx0;
          x1 = nx1;
          if (_exact) {
            final n00 = _a00 * b00 + _a01 * b10;
            final n01 = _a00 * b01 + _a01 * b11;
            final n10 = _a10 * b00 + _a11 * b10;
            final n11 = _a10 * b01 + _a11 * b11;
            b00 = n00;
            b01 = n01;
            b10 = n10;
            b11 = n11;
          }
        }
      }

      if (keepHistory) {
        predictedMean![t * 2] = x0;
        predictedMean[t * 2 + 1] = x1;
        predictedCov![t * 4] = p00;
        predictedCov[t * 4 + 1] = p01;
        predictedCov[t * 4 + 2] = p01;
        predictedCov[t * 4 + 3] = p11;
        if (_exact) {
          predictedSensitivity![t * 4] = b00;
          predictedSensitivity[t * 4 + 1] = b01;
          predictedSensitivity[t * 4 + 2] = b10;
          predictedSensitivity[t * 4 + 3] = b11;
        }
      }

      if (timeline.hasObservation(t)) {
        component.observationAt(timeline.times[t], _observation);
        final h0 = _observation[0];
        final h1 = _observation[1];
        final r = timeline.variances[t] * measurementVariance;

        final v = timeline.values[t] - (h0 * x0 + h1 * x1);
        final ph0 = p00 * h0 + p01 * h1;
        final ph1 = p01 * h0 + p11 * h1;
        final s = h0 * ph0 + h1 * ph1 + r;
        if (!(s > 0) || !s.isFinite) {
          throw StateError('Innovation variance $s at time ${timeline.times[t]}'
              ' is not positive. The model has become numerically degenerate.');
        }
        final k0 = ph0 / s;
        final k1 = ph1 / s;

        x0 += k0 * v;
        x1 += k1 * v;
        // Joseph form, expanded: P - (P H') K' - K (P H')' + S K K'.
        p00 += s * k0 * k0 - 2 * ph0 * k0;
        p01 += s * k0 * k1 - ph0 * k1 - k0 * ph1;
        p11 += s * k1 * k1 - 2 * ph1 * k1;

        var vb0 = 0.0, vb1 = 0.0;
        if (_exact) {
          vb0 = -(h0 * b00 + h1 * b10);
          vb1 = -(h0 * b01 + h1 * b11);
          b00 += k0 * vb0;
          b01 += k0 * vb1;
          b10 += k1 * vb0;
          b11 += k1 * vb1;
          final w0 = vb0 / s;
          final w1 = vb1 / s;
          m00 += w0 * vb0;
          m01 += w0 * vb1;
          m11 += w1 * vb1;
          rhs0 += w0 * v;
          rhs1 += w1 * v;
        }

        if (pieces != null) {
          if (_exact) {
            loading[0] = vb0;
            loading[1] = vb1;
          }
          pieces.add(timeline.times[t], v, s, loading);
        }

        seen++;
        if (seen > burn) {
          used++;
          sumLogS += math.log(s);
          sumWeighted += v * v / s;
        }
      }

      if (keepHistory) {
        filteredMean![t * 2] = x0;
        filteredMean[t * 2 + 1] = x1;
        filteredCov![t * 4] = p00;
        filteredCov[t * 4 + 1] = p01;
        filteredCov[t * 4 + 2] = p01;
        filteredCov[t * 4 + 3] = p11;
        if (_exact) {
          filteredSensitivity![t * 4] = b00;
          filteredSensitivity[t * 4 + 1] = b01;
          filteredSensitivity[t * 4 + 2] = b10;
          filteredSensitivity[t * 4 + 3] = b11;
        }
      }
    }

    Float64List? diffuseMean;
    Float64List? diffuseCovariance;
    var diffuseLogDeterminant = 0.0;
    if (_exact && steps > 0) {
      // Solved with the same Cholesky helper the generic engine uses, not by
      // the closed-form two-by-two inverse. This runs once per pass rather
      // than once per step, so it costs nothing measurable, and sharing it
      // means the two engines agree on the diffuse estimate to the last bit
      // instead of merely to twelve digits.
      final information = Float64List.fromList([m00, m01, m01, m11]);
      if (!choleskyFactor(information, 2)) {
        throw StateError(singularDiffuseMessage(
            [component],
            2,
            timeline.observationCount,
            timeline.times[0],
            timeline.times[steps - 1]));
      }
      diffuseLogDeterminant =
          2 * (math.log(information[0]) + math.log(information[3]));

      diffuseMean = Float64List.fromList([-rhs0, -rhs1]);
      choleskySolve(information, 2, diffuseMean, 0);

      diffuseCovariance = Float64List(4);
      diffuseCovariance[0] = 1;
      choleskySolve(information, 2, diffuseCovariance, 0);
      diffuseCovariance[3] = 1;
      choleskySolve(information, 2, diffuseCovariance, 2);

      sumWeighted += rhs0 * diffuseMean[0] + rhs1 * diffuseMean[1];
      used = timeline.observationCount - 2;
    }

    final residuals =
        pieces == null ? null : recursiveResiduals(pieces, burnIn: burn);

    return FilterResult(
      stateDim: 2,
      stepCount: steps,
      logLikelihood: -0.5 *
          (used * _log2pi + sumLogS + sumWeighted + diffuseLogDeterminant),
      sumLogInnovationVariance: sumLogS,
      sumWeightedSquares: sumWeighted,
      usedObservations: used,
      measurementVariance: measurementVariance,
      diffuseDim: dim,
      diffuseLogDeterminant: diffuseLogDeterminant,
      diffuseMean: diffuseMean,
      diffuseCovariance: diffuseCovariance,
      filteredMean: filteredMean,
      filteredCovariance: filteredCov,
      predictedMean: predictedMean,
      predictedCovariance: predictedCov,
      filteredSensitivity: filteredSensitivity,
      predictedSensitivity: predictedSensitivity,
      residualTimes: residuals?.times,
      standardisedResiduals: residuals?.values,
    );
  }
}
