import 'dart:math' as math;
import 'dart:typed_data';

import '../model.dart';
import '../observation.dart';
import '../parameter_spec.dart';
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
/// searched is one log variance ratio per variance rather than every variance
/// plus the noise. That saving of a dimension is exact and applies at any
/// number of components.
///
/// The search runs in three stages. A coordinate scan sweeps each parameter in
/// turn across its bracket, which finds the right basin — a likelihood that is
/// flat over decades will strand a local search wherever it started, and one
/// that is multimodal will strand it in the wrong mode. Then golden section for
/// a one-parameter model, or Nelder-Mead with one restart for several. Finally
/// each parameter is probed along its own axis to see how far it can move
/// before the fit deteriorates by half a nat, which is what
/// [FitResult.plateauDecades] reports.
///
/// [lowerLogRatio] and [upperLogRatio] bracket the *variance* parameters. The
/// defaults are generous but still finite, and the right range depends on the
/// time unit: `q` is a variance per cubed time unit for a trend, so measuring
/// time in seconds rather than days moves the optimum by about fifteen decades.
/// Check [FitResult.atBracketEdge] before believing the answer. Shape
/// parameters — a Matérn length scale, a cycle's period — are bracketed by the
/// component that owns them instead, because a range that suits a variance
/// ratio suits nothing else; see [ParameterSpec].
///
/// [penalty] defaults to none, including for several components. That was not
/// the plan — the expectation was that a penalty would be needed to stabilise
/// the trend/seasonal split on short histories — but the measurement says
/// otherwise, and [ComplexityPenalty] records what was measured. Pass one
/// explicitly when you have a reason to.
///
/// ## Saying what the scale can measure
///
/// By default the noise level is whatever explains the data best, which on a
/// run of nearly identical readings can be an absurdly small number and a
/// credible band far narrower than any real instrument could justify. Two
/// arguments push back on that, and they are mutually exclusive:
///
/// - [minimumMeasurementVariance] is a floor. The fit runs normally, and only
///   if the estimate comes out below the floor is it redone with the noise
///   pinned there. This is the one to reach for: a kitchen scale reading to
///   100 g has a precision the data cannot argue with, and asserting it costs
///   nothing when the data agrees.
/// - [fixedMeasurementVariance] pins the noise level outright, which is what
///   you want when the smoothing is being chosen rather than estimated — trale
///   picking a curve stiffness from a user setting, for instance — and the
///   noise level still has to come from somewhere.
///
/// Either one costs a dimension: with the noise level asserted there is
/// nothing left to concentrate out, so the search has as many dimensions as the
/// model has parameters. Note also that a floor cannot simply be clamped onto
/// the estimate afterwards. Pinning one variance in absolute units breaks the
/// scale equivariance the concentrating step depends on, so the other variances
/// have to be re-estimated against it — which is exactly what the second fit
/// does.
FitResult fit(
  StructuralModel initial,
  List<Observation> observations, {
  double lowerLogRatio = -20,
  double upperLogRatio = 10,
  int scanPoints = 25,
  double tolerance = 1e-4,
  Penalty? penalty,
  double? fixedMeasurementVariance,
  double? minimumMeasurementVariance,
}) {
  if (fixedMeasurementVariance != null && minimumMeasurementVariance != null) {
    throw ArgumentError('pass fixedMeasurementVariance or '
        'minimumMeasurementVariance, not both: a noise level pinned to a value '
        'cannot also be given a floor to clear');
  }
  for (final (name, value) in [
    ('fixedMeasurementVariance', fixedMeasurementVariance),
    ('minimumMeasurementVariance', minimumMeasurementVariance),
  ]) {
    if (value != null && (!(value > 0) || !value.isFinite)) {
      throw ArgumentError.value(value, name, 'must be finite and positive');
    }
  }

  FitResult run(double? fixed) => _search(
        initial,
        observations,
        lowerLogRatio: lowerLogRatio,
        upperLogRatio: upperLogRatio,
        scanPoints: scanPoints,
        tolerance: tolerance,
        penalty: penalty ?? const NoPenalty(),
        fixedMeasurementVariance: fixed,
      );

  if (minimumMeasurementVariance != null) {
    final free = run(null);
    if (free.measurementVariance >= minimumMeasurementVariance) return free;
    return run(minimumMeasurementVariance);
  }
  return run(fixedMeasurementVariance);
}

FitResult _search(
  StructuralModel initial,
  List<Observation> observations, {
  required double lowerLogRatio,
  required double upperLogRatio,
  required int scanPoints,
  required double tolerance,
  required Penalty penalty,
  required double? fixedMeasurementVariance,
}) {
  final k = initial.parameterCount;
  if (!(lowerLogRatio < upperLogRatio)) {
    throw ArgumentError('empty bracket [$lowerLogRatio, $upperLogRatio]');
  }
  if (scanPoints < 3) {
    throw ArgumentError.value(scanPoints, 'scanPoints', 'must be at least 3');
  }

  // Each parameter gets its own bracket and its own scan resolution: the
  // caller's bracket for a variance ratio, and the owning component's for
  // anything else.
  final specs = initial.parameterSpecs;
  final lower = Float64List(k);
  final upper = Float64List(k);
  final points = List<int>.filled(k, scanPoints);
  for (var i = 0; i < k; i++) {
    switch (specs[i]) {
      case VarianceParameter():
        lower[i] = lowerLogRatio;
        upper[i] = upperLogRatio;
      case ShapeParameter(lower: final lo, upper: final hi, :final scanPoints):
        lower[i] = lo;
        upper[i] = hi;
        if (scanPoints != null) points[i] = scanPoints;
    }
  }
  final step = Float64List(k);
  for (var i = 0; i < k; i++) {
    step[i] = (upper[i] - lower[i]) / (points[i] - 1);
  }

  final profile = ProfileLikelihood(initial, observations,
      penalty: penalty, fixedMeasurementVariance: fixedMeasurementVariance);

  final origin = Float64List(k);
  for (var i = 0; i < k; i++) {
    origin[i] = (lower[i] + upper[i]) / 2;
  }
  if (profile.evaluate(origin).usedObservations < 1) {
    throw ArgumentError('too few observations to estimate anything: the model '
        'has ${initial.stateDim} states, and the first few observations are '
        'spent pinning them down.');
  }

  // Anything outside the bracket is refused without running a filter, which
  // both bounds the simplex and keeps a degenerate variance from reaching the
  // recursion at all.
  double objective(Float64List theta) {
    for (var i = 0; i < k; i++) {
      if (theta[i] < lower[i] || theta[i] > upper[i]) return -double.maxFinite;
    }
    return profile.at(theta);
  }

  final current = Float64List.fromList(origin);
  var bestScanIndex = 0;

  // Coordinate scan: sweep each parameter across the whole of its own bracket
  // in turn, holding the others where the previous sweeps left them. With one
  // parameter this is an exhaustive scan. With several it is coordinate
  // ascent, which is not an optimiser but is a much better place to start one
  // than wherever the caller's initial guess happened to be.
  //
  // Two sweeps rather than one, because the first sweeps a parameter against
  // arbitrary values of everything it has not reached yet. That is harmless
  // when the axes barely interact and badly wrong when they do: a cycle's
  // period scanned at the midpoint of its own variance bracket is scanned
  // against a cycle that is not there, and the scan reads flat.
  final sweeps = k > 1 ? 2 : 1;
  for (var sweep = 0; sweep < sweeps; sweep++) {
    for (var axis = 0; axis < k; axis++) {
      var bestValue = -double.infinity;
      var best = 0;
      for (var i = 0; i < points[axis]; i++) {
        current[axis] = lower[axis] + i * step[axis];
        final value = objective(current);
        if (value > bestValue) {
          bestValue = value;
          best = i;
        }
      }
      current[axis] = lower[axis] + best * step[axis];
      bestScanIndex = best;
    }
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
      lower[0] + math.max(0, bestScanIndex - 1) * step[0],
      lower[0] + math.min(points[0] - 1, bestScanIndex + 1) * step[0],
      tolerance: tolerance,
    );
    optimum = Float64List.fromList([refined.argument]);
    peak = refined.value;
    converged = refined.converged;
  } else {
    // The restart is the standard insurance against a simplex that has
    // collapsed along one direction and stopped making progress. A second run
    // that finds nothing new is decent evidence the first one finished.
    final wide = Float64List(k);
    final narrow = Float64List(k);
    for (var i = 0; i < k; i++) {
      wide[i] = 2 * step[i];
      narrow[i] = step[i] / 4;
    }
    var simplex =
        maximiseSimplex(objective, current, steps: wide, tolerance: tolerance);
    simplex = maximiseSimplex(objective, simplex.argument,
        steps: narrow, tolerance: tolerance);
    optimum = simplex.argument;
    peak = simplex.value;
    converged = simplex.converged;
  }

  final widths = Float64List(k);
  for (var axis = 0; axis < k; axis++) {
    widths[axis] = _halfNatWidth(
            objective, optimum, axis, peak, lower[axis], upper[axis]) /
        _ln10;
  }

  final atOptimum = profile.evaluate(optimum);
  final measurementVariance = profile.measurementVarianceOf(atOptimum);
  final shift = math.log(measurementVariance);
  final absolute = Float64List(k);
  final ratios = Float64List(k);
  for (var i = 0; i < k; i++) {
    final isVariance = specs[i] is VarianceParameter;
    absolute[i] = isVariance ? optimum[i] + shift : optimum[i];
    ratios[i] = isVariance ? math.exp(optimum[i]) : double.nan;
  }

  // A parameter counts as sitting on a bound when it finishes in the last cell
  // of the scan that found it. An exact comparison would miss the usual case:
  // golden section refines inside the outermost scan interval and stops a
  // little short of the bound, and the simplex is repelled from it by the
  // objective going to minus infinity just outside.
  final status = [
    for (var i = 0; i < k; i++)
      if (optimum[i] >= upper[i] - step[i])
        ParameterStatus.beyondBracket
      else if (optimum[i] <= lower[i] + step[i])
        // A variance at the bottom of its bracket has been shrunk out of the
        // model. A length scale at the bottom of its bracket has not been
        // shrunk out of anything; the bracket is simply in the wrong place,
        // which is what beyondBracket means.
        specs[i] is VarianceParameter
            ? ParameterStatus.shrunkToNothing
            : ParameterStatus.beyondBracket
      else
        ParameterStatus.determined
  ];

  return FitResult(
    model: initial
        .withParameters(absolute)
        .withMeasurementVariance(measurementVariance),
    logMarginalLikelihood: profile.likelihoodAt(optimum),
    logPenalty: profile.penaltyAt(optimum),
    penalty: penalty,
    varianceRatios: ratios,
    evaluations: profile.evaluations,
    converged: converged,
    plateauDecadesByParameter: widths,
    parameterStatus: status,
    parameterSpecs: specs,
    diffuseDimension: initial.diffuseDimension,
    measurementVariancePinned: fixedMeasurementVariance != null,
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
