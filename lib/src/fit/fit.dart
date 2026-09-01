import 'dart:math' as math;
import 'dart:typed_data';

import '../model.dart';
import '../observation.dart';
import '../result.dart';
import 'golden_section.dart';
import 'nelder_mead.dart';
import 'penalty.dart';
import 'profile_likelihood.dart';

const double _ln10 = 2.302585092994046;

/// Half a nat: the conventional "indistinguishable on this data" threshold,
/// and the drop a single extra parameter has to beat to be worth having.
const double _halfNat = 0.5;

/// Estimates a model's variances from [observations] by maximum marginal
/// likelihood.
///
/// The measurement variance is concentrated out analytically, so what is
/// searched is one log variance ratio per component rather than one variance
/// per component plus the noise. That saving of a dimension is exact and
/// applies at any number of components.
///
/// The search runs in three stages. A coordinate scan sweeps each ratio in
/// turn across the bracket, which finds the right basin — a likelihood that is
/// flat over decades will strand a local search wherever it started. Then
/// golden section for a one-parameter model, or Nelder-Mead with one restart
/// for several. Finally each parameter is probed along its own axis to see how
/// far it can move before the fit deteriorates by half a nat, which is what
/// [FitResult.plateauDecades] reports.
///
/// The bracket defaults are generous but still finite, and the right range
/// depends on the time unit: `q` is a variance per cubed time unit for a
/// trend, so measuring time in seconds rather than days moves the optimum by
/// about fifteen decades. Check [FitResult.atBracketEdge] before believing the
/// answer.
///
/// [penalty] defaults to none, including for several components. That was not
/// the plan — the expectation was that a penalty would be needed to stabilise
/// the trend/seasonal split on short histories — but the measurement says
/// otherwise, and [ComplexityPenalty] records what was measured. Pass one
/// explicitly when you have a reason to.
FitResult fit(
  StructuralModel initial,
  List<Observation> observations, {
  double lowerLogRatio = -20,
  double upperLogRatio = 10,
  int scanPoints = 25,
  double tolerance = 1e-4,
  Penalty? penalty,
}) {
  final k = initial.parameterCount;
  if (!(lowerLogRatio < upperLogRatio)) {
    throw ArgumentError('empty bracket [$lowerLogRatio, $upperLogRatio]');
  }
  if (scanPoints < 3) {
    throw ArgumentError.value(scanPoints, 'scanPoints', 'must be at least 3');
  }

  final chosen = penalty ?? const NoPenalty();
  final profile = ProfileLikelihood(initial, observations, penalty: chosen);

  final origin = Float64List(k)
    ..fillRange(0, k, (lowerLogRatio + upperLogRatio) / 2);
  if (profile.evaluate(origin).usedObservations < 1) {
    throw ArgumentError('too few observations to estimate anything: the model '
        'has ${initial.stateDim} states, and the first few observations are '
        'spent pinning them down.');
  }

  // Anything outside the bracket is refused without running a filter, which
  // both bounds the simplex and keeps a degenerate variance from reaching the
  // recursion at all.
  double objective(Float64List theta) {
    for (final value in theta) {
      if (value < lowerLogRatio || value > upperLogRatio) {
        return -double.maxFinite;
      }
    }
    return profile.at(theta);
  }

  final step = (upperLogRatio - lowerLogRatio) / (scanPoints - 1);
  final current = Float64List(k);
  var scanHitEnd = false;
  var bestScanIndex = 0;
  final scan = Float64List(scanPoints);

  // Coordinate scan: sweep each ratio across the whole bracket in turn,
  // holding the others where the previous sweeps left them. With one
  // parameter this is an exhaustive scan; with several it is one pass of
  // coordinate ascent, which is not an optimiser but is a much better place to
  // start one than wherever the caller's initial guess happened to be.
  for (var axis = 0; axis < k; axis++) {
    var best = 0;
    for (var i = 0; i < scanPoints; i++) {
      current[axis] = lowerLogRatio + i * step;
      scan[i] = objective(current);
      if (scan[i] > scan[best]) best = i;
    }
    current[axis] = lowerLogRatio + best * step;
    if (best == 0 || best == scanPoints - 1) scanHitEnd = true;
    bestScanIndex = best;
  }

  Float64List optimum;
  double peak;
  bool converged;

  if (k == 0) {
    // A model of nothing but regression components has no variance to search
    // over: the coefficients are states, and the measurement variance is
    // concentrated out in closed form. There is one evaluation to make and
    // then the answer is already exact.
    optimum = current;
    peak = objective(current);
    converged = true;
  } else if (k == 1) {
    final refined = maximise(
      (x) => objective(Float64List.fromList([x])),
      lowerLogRatio + math.max(0, bestScanIndex - 1) * step,
      lowerLogRatio + math.min(scanPoints - 1, bestScanIndex + 1) * step,
      tolerance: tolerance,
    );
    optimum = Float64List.fromList([refined.argument]);
    peak = refined.value;
    converged = refined.converged;
  } else {
    // The restart is the standard insurance against a simplex that has
    // collapsed along one direction and stopped making progress. A second run
    // that finds nothing new is decent evidence the first one finished.
    var simplex = maximiseSimplex(objective, current,
        step: 2 * step, tolerance: tolerance);
    simplex = maximiseSimplex(objective, simplex.argument,
        step: 0.5, tolerance: tolerance);
    optimum = simplex.argument;
    peak = simplex.value;
    converged = simplex.converged;
  }

  final widths = Float64List(k);
  for (var axis = 0; axis < k; axis++) {
    widths[axis] = _halfNatWidth(
            objective, optimum, axis, peak, lowerLogRatio, upperLogRatio) /
        _ln10;
  }

  final atOptimum = profile.evaluate(optimum);
  final measurementVariance = atOptimum.profileMeasurementVariance;
  final shift = math.log(measurementVariance);
  final absolute = Float64List(k);
  final ratios = Float64List(k);
  for (var i = 0; i < k; i++) {
    absolute[i] = optimum[i] + shift;
    ratios[i] = math.exp(optimum[i]);
  }

  var nearBound = false;
  for (final value in optimum) {
    if ((value - lowerLogRatio).abs() < 1e-6 ||
        (value - upperLogRatio).abs() < 1e-6) {
      nearBound = true;
    }
  }

  return FitResult(
    model: initial
        .withParameters(absolute)
        .withMeasurementVariance(measurementVariance),
    logMarginalLikelihood: profile.likelihoodAt(optimum),
    logPenalty: profile.penaltyAt(optimum),
    penalty: chosen,
    varianceRatios: ratios,
    evaluations: profile.evaluations,
    converged: converged,
    plateauDecadesByParameter: widths,
    atBracketEdge: scanHitEnd || nearBound,
  );
}

/// How far parameter [axis] can move on its own before the objective falls
/// half a nat below [peak].
///
/// Conditional on the others, which is the honest caveat: when two components
/// trade off against one another the joint region is wider than any of these
/// slices, and the number here understates how undetermined the fit really is.
/// It is still the question a reader asks first — "how well pinned down is
/// this one number" — and it costs about three dozen filter passes per
/// parameter to answer.
double _halfNatWidth(
  double Function(Float64List) objective,
  Float64List optimum,
  int axis,
  double peak,
  double lower,
  double upper,
) {
  final probe = Float64List.fromList(optimum);
  final floor = peak - _halfNat;

  double edge(int direction) {
    // Expand until the objective drops through the floor or the bracket runs
    // out, then bisect what is left.
    var inside = 0.0;
    var outside = double.nan;
    for (var reach = 0.25; reach <= 64; reach *= 2) {
      probe[axis] = optimum[axis] + direction * reach;
      if (probe[axis] < lower || probe[axis] > upper) {
        outside = direction > 0 ? upper - optimum[axis] : optimum[axis] - lower;
        break;
      }
      if (objective(probe) < floor) {
        outside = reach;
        break;
      }
      inside = reach;
    }
    if (outside.isNaN) return inside;

    for (var i = 0; i < 10; i++) {
      final middle = (inside + outside) / 2;
      probe[axis] = optimum[axis] + direction * middle;
      if (probe[axis] < lower || probe[axis] > upper) {
        outside = middle;
      } else if (objective(probe) < floor) {
        outside = middle;
      } else {
        inside = middle;
      }
    }
    return inside;
  }

  final up = edge(1);
  final down = edge(-1);
  probe[axis] = optimum[axis];
  return up + down;
}
