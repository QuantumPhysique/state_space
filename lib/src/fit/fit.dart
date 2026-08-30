import 'dart:math' as math;
import 'dart:typed_data';

import '../model.dart';
import '../observation.dart';
import '../result.dart';
import 'golden_section.dart';
import 'profile_likelihood.dart';

const double _ln10 = 2.302585092994046;

/// Estimates a model's variances from [observations] by maximum marginal
/// likelihood.
///
/// The measurement variance is concentrated out analytically, so what is
/// actually searched is the single ratio `q = processVariance /
/// measurementVariance`. The search runs in `log q`: first a coarse scan of
/// [scanPoints] across the bracket to find the right basin, then golden
/// section inside it. The scan matters — a likelihood that is flat over
/// decades will happily strand a local search wherever it started.
///
/// The bracket defaults are generous but they are still finite, and the right
/// range depends on the time unit: `q` is a variance per cubed time unit for a
/// trend, so measuring time in seconds rather than days moves the optimum by
/// about fifteen decades. Check [FitResult.atBracketEdge] before believing the
/// answer.
///
/// This release fits models with exactly one free parameter, and assumes that
/// parameter is a log variance. Multi-component fitting needs a proper
/// multivariate optimiser and arrives with the seasonal components in 0.3.
FitResult fit(
  StructuralModel initial,
  List<Observation> observations, {
  double lowerLogRatio = -20,
  double upperLogRatio = 10,
  int scanPoints = 25,
  double tolerance = 1e-4,
}) {
  if (initial.parameterCount != 1) {
    throw UnsupportedError('fit() handles one free parameter; this model has '
        '${initial.parameterCount}. Multi-parameter fitting lands in 0.3.');
  }
  if (!(lowerLogRatio < upperLogRatio)) {
    throw ArgumentError('empty bracket [$lowerLogRatio, $upperLogRatio]');
  }
  if (scanPoints < 3) {
    throw ArgumentError.value(scanPoints, 'scanPoints', 'must be at least 3');
  }

  final profile = ProfileLikelihood(initial, observations);
  if (profile.evaluate(lowerLogRatio).usedObservations < 1) {
    throw ArgumentError('too few observations to estimate anything: the model '
        'has ${initial.stateDim} states, and the first few observations are '
        'spent pinning them down.');
  }

  final step = (upperLogRatio - lowerLogRatio) / (scanPoints - 1);
  final scan = Float64List(scanPoints);
  var best = 0;
  for (var i = 0; i < scanPoints; i++) {
    scan[i] = profile.at(lowerLogRatio + i * step);
    if (scan[i] > scan[best]) best = i;
  }

  final refined = maximise(
    profile.at,
    lowerLogRatio + math.max(0, best - 1) * step,
    lowerLogRatio + math.min(scanPoints - 1, best + 1) * step,
    tolerance: tolerance,
  );

  final atOptimum = profile.evaluate(refined.argument);
  final varianceRatio = math.exp(refined.argument);
  final measurementVariance = atOptimum.profileMeasurementVariance;

  final fitted = initial
      .withParameters(Float64List.fromList(
          [refined.argument + math.log(measurementVariance)]))
      .withMeasurementVariance(measurementVariance);

  return FitResult(
    model: fitted,
    logMarginalLikelihood: refined.value,
    varianceRatio: varianceRatio,
    evaluations: profile.evaluations,
    converged: refined.converged,
    plateauDecades: _plateauWidth(scan, refined.value, step) / _ln10,
    atBracketEdge: best == 0 || best == scanPoints - 1,
  );
}

/// Width in log units of the scanned region within half a nat of the peak.
///
/// Half a nat is the conventional "indistinguishable on this data" threshold;
/// it is the drop a single extra parameter has to beat to be worth having.
double _plateauWidth(Float64List scan, double peak, double step) {
  final floor = peak - 0.5;
  var lowest = -1;
  var highest = -1;
  for (var i = 0; i < scan.length; i++) {
    if (scan[i] < floor) continue;
    if (lowest < 0) lowest = i;
    highest = i;
  }
  if (lowest < 0) return 0;
  return (highest - lowest) * step;
}
