import 'dart:math' as math;
import 'dart:typed_data';

import '../arguments.dart';
import '../engine/layout.dart';
import '../engine/scale.dart';
import '../exceptions.dart';
import '../model.dart';
import '../observation.dart';
import '../parameter_spec.dart';
import '../result.dart';
import 'golden_section.dart';
import 'nelder_mead.dart';
import 'penalty.dart';
import 'prior.dart';
import 'profile_likelihood.dart';

const double _ln10 = 2.302585092994046;

/// Where [fit] begins its search.
enum SearchStart {
  /// Sweep every parameter across the whole of its bracket first, then refine.
  ///
  /// The default, and the only safe choice on a surface that may be flat over
  /// decades or multimodal, which includes every surface with a
  /// `StochasticCycle` in it.
  bracketScan,

  /// Take the parameters of the model passed to [fit] and refine from there,
  /// skipping the scan.
  ///
  /// For refitting as data arrives:
  ///
  /// ```dart
  /// var fitted = fit(model, data);                        // cold, once
  /// fitted = fit(fitted.model, longerData,                // warm, after
  ///     start: SearchStart.previousParameters);
  /// ```
  ///
  /// The variance parameters are read as ratios to the model's own
  /// measurement variance, so a [FitResult.model] round-trips exactly.
  /// Anything outside the bracket is clamped into it. The local search follows
  /// the likelihood as far as it rises, but it cannot jump to a different
  /// basin, so re-run a cold fit whenever the data changes character rather
  /// than merely grows.
  previousParameters,
}

/// Half a nat: the conventional "indistinguishable on this data" threshold.
const double _halfNat = 0.5;

/// Estimates a model's variances from [observations] by maximum marginal
/// likelihood.
///
/// The measurement variance is concentrated out analytically, so the search
/// runs over one log variance ratio per variance parameter, plus any shape
/// parameters, and not over the noise level itself, unless a held variance or
/// a prior ties the model to an absolute scale; see [heldParameters] and
/// [priors] below.
///
/// The search runs in three stages. A coordinate scan sweeps each parameter in
/// turn across its bracket to find the right basin. Then golden section for a
/// one-parameter model, or Nelder-Mead with one restart for several. Finally
/// each parameter is probed along its own axis to see how far it can move
/// before the fit deteriorates by half a nat, which is what
/// [FitResult.plateauDecades] reports. A fit costs a few dozen forward passes
/// for one parameter and a few hundred for three; see [FitResult.evaluations].
///
/// [lowerLogRatio] and [upperLogRatio] bracket the natural log of every
/// variance ratio, whatever the model passed in: the variances [initial]
/// carries are ignored under the default [start], unless they are held. The
/// default bracket,
/// `[-20, 10]`, is thirteen decades wide, from about `1e-8.7` to `1e4.3` times
/// the measurement variance. It suits time in days; a trend's variance is per
/// cubed time unit, so measuring time in seconds moves the optimum about
/// fifteen decades. Check [FitResult.atBracketEdge] before believing the
/// answer. Shape parameters (a Matérn length scale, a cycle's period) are
/// bracketed by the component that owns them; see [ParameterSpec].
///
/// A shape parameter measured in time units also gets a floor from the data:
/// the bottom of its bracket is raised to what the sampling can resolve. See
/// [Component.parameterSpecsAt] and [samplingResolution].
///
/// [start] decides whether the coordinate scan runs; see [SearchStart].
///
/// [penalty] defaults to none. See [ComplexityPenalty] for what one does.
///
/// ## Holding parameters
///
/// [heldParameters] are indices into `initial.parameters`, in the order of
/// [StructuralModel.parameterSpecs]. A held parameter is not searched: it
/// keeps the value [initial] gives it, a variance as an absolute value and a
/// shape parameter whether or not it lies inside its bracket.
/// [FitResult.parameterStatus] marks it [ParameterStatus.held], and it has no
/// width.
///
/// A held variance pins the model to an absolute scale, which concentrating
/// the noise level out would not keep. With one, unless
/// [fixedMeasurementVariance] is given, the log of the measurement variance
/// becomes one more search axis, over the same bracket of log ratios
/// centred on the noise level that would explain the spread of the readings
/// on its own. The free variances are still searched as ratios to it. With
/// every parameter held, the noise level is all that is left to search.
///
/// ## Priors
///
/// [priors] puts a [LogNormalPrior] on parameters by index, as
/// [heldParameters] counts them, and [measurementVariancePrior] one on the
/// measurement variance. The answer is then the posterior mode in the log of
/// every searched parameter: the maximum of the log-likelihood plus the log
/// prior density, which [FitResult.logPrior] reports. A prior on a variance is
/// on its absolute value, not its ratio, so like a held variance it makes the
/// noise level a search axis. The brackets still bound the search.
///
/// A prior needs a parameter searched as a log, so one on a held parameter or
/// on one searched as a logit, such as a cycle's damping, throws.
///
/// ## Saying what the scale can measure
///
/// By default the noise level is whatever explains the data best, which on a
/// run of nearly identical readings can be an absurdly small number. Two
/// arguments push back on that, and they are mutually exclusive:
///
/// - [minimumMeasurementVariance] is a floor. The fit runs normally, and only
///   if the estimate comes out below the floor is it redone with the noise
///   pinned there. Use it whenever the instrument's resolution is known: a
///   scale that rounds to 0.1 contributes a rounding error of standard
///   deviation `0.1 / sqrt(12)`, about 0.029, whatever the data says.
/// - [fixedMeasurementVariance] pins the noise level outright, for when it is
///   known, and estimates the smoothing given it. It excludes
///   [measurementVariancePrior] as well.
///
/// A floor cannot be clamped onto the estimate afterwards: pinning one
/// variance in absolute units breaks the scale equivariance the concentrating
/// step depends on, so the other variances are re-estimated against it.
///
/// To fix the smoothing and estimate the noise level instead, for instance
/// when the stiffness of a curve comes from a user setting, do not fit at all:
/// use [StructuralModel.withEstimatedScale].
///
/// ## Failures
///
/// Throws [UnderdeterminedModelException] when the data cannot support the
/// model: fewer observations than the model's flat directions plus one (three
/// for a trend, one more per diffuse state for each component added), two
/// components the data cannot tell apart, or a component that cannot be
/// resolved at this sampling, such as a seasonal harmonic past the Nyquist
/// frequency. Throws [ArgumentError] for invalid arguments, among them an index
/// the model has no parameter for.
///
/// {@category Choosing a model}
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
  SearchStart start = SearchStart.bracketScan,
  Set<int> heldParameters = const {},
  Map<int, LogNormalPrior> priors = const {},
  LogNormalPrior? measurementVariancePrior,
}) {
  if (fixedMeasurementVariance != null && minimumMeasurementVariance != null) {
    throw ArgumentError(
      'pass fixedMeasurementVariance or '
      'minimumMeasurementVariance, not both: a noise level pinned to a value '
      'cannot also be given a floor to clear',
    );
  }
  if (fixedMeasurementVariance != null && measurementVariancePrior != null) {
    throw ArgumentError(
      'pass fixedMeasurementVariance or measurementVariancePrior, not both: a '
      'noise level pinned to a value leaves a prior nothing to weigh',
    );
  }
  if (fixedMeasurementVariance != null) {
    checkPositive(fixedMeasurementVariance, 'fixedMeasurementVariance');
  }
  if (minimumMeasurementVariance != null) {
    checkPositive(minimumMeasurementVariance, 'minimumMeasurementVariance');
  }
  if (!(lowerLogRatio < upperLogRatio)) {
    throw ArgumentError('empty bracket [$lowerLogRatio, $upperLogRatio]');
  }
  if (scanPoints < 3) {
    throw ArgumentError.value(scanPoints, 'scanPoints', 'must be at least 3');
  }
  checkPositive(tolerance, 'tolerance');
  final specs = initial.parameterSpecs;
  void checkIndex(int index, String name) {
    if (index < 0 || index >= specs.length) {
      throw ArgumentError.value(
        index,
        name,
        'the model has ${specs.length} parameters',
      );
    }
  }

  for (final index in heldParameters) {
    checkIndex(index, 'heldParameters');
  }
  for (final index in priors.keys) {
    checkIndex(index, 'priors');
    if (heldParameters.contains(index)) {
      throw ArgumentError.value(index, 'priors', 'is a held parameter');
    }
    if (!specs[index].isLogarithmic) {
      throw ArgumentError.value(
        index,
        'priors',
        'is a parameter searched as a logit, not a log',
      );
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
    start: start,
    held: heldParameters,
    priors: priors,
    measurementVariancePrior: measurementVariancePrior,
  );

  final FitResult result;
  if (minimumMeasurementVariance != null) {
    final free = run(null);
    result = free.measurementVariance >= minimumMeasurementVariance
        ? free
        : run(minimumMeasurementVariance);
  } else {
    result = run(fixedMeasurementVariance);
  }
  return _withLargestResidual(result, observations);
}

/// Each parameter's search bracket, scan resolution and scan step.
typedef _Brackets = ({
  Float64List lower,
  Float64List upper,
  List<int> points,
  Float64List step,
});

/// Where a local search finished.
typedef _Optimum = ({Float64List argument, double value, bool converged});

FitResult _search(
  StructuralModel initial,
  List<Observation> observations, {
  required double lowerLogRatio,
  required double upperLogRatio,
  required int scanPoints,
  required double tolerance,
  required Penalty penalty,
  required double? fixedMeasurementVariance,
  required SearchStart start,
  required Set<int> held,
  required Map<int, LogNormalPrior> priors,
  required LogNormalPrior? measurementVariancePrior,
}) {
  final k = initial.parameterCount;
  final specs = initial.parameterSpecsAt(
    resolution: samplingResolution(observations),
  );
  final profile = ProfileLikelihood(
    initial,
    observations,
    penalty: penalty,
    fixedMeasurementVariance: fixedMeasurementVariance,
    held: held,
    priors: priors,
    measurementVariancePrior: measurementVariancePrior,
  );
  final free = profile.free;
  final scaleAxis = profile.searchesScale ? free.length : null;
  final axes = [
    for (final i in free) specs[i],
    if (scaleAxis != null)
      const VarianceParameter(label: 'measurement variance'),
  ];
  final m = axes.length;

  final brackets = _brackets(
    axes,
    lowerLogRatio,
    upperLogRatio,
    scanPoints: scanPoints,
  );
  final (:lower, :upper, :points, :step) = brackets;
  if (scaleAxis != null) {
    final centre = math.log(profile.spreadVariance);
    lower[scaleAxis] += centre;
    upper[scaleAxis] += centre;
  }

  final origin = Float64List(m);
  for (var i = 0; i < m; i++) {
    origin[i] = (lower[i] + upper[i]) / 2;
  }
  if (profile.evaluate(origin).usedObservations < 1) {
    final flat = diffuseStateIndices(initial.components).length;
    throw UnderdeterminedModelException(
      'too few observations to estimate '
      'anything: the model has $flat flat directions, which use up one '
      'observation each, and ${observations.length} observations leave none '
      'over to estimate a noise level from.',
    );
  }

  // Anything outside the bracket, or not a number, is refused without running
  // a filter. That bounds the simplex and keeps a degenerate variance from
  // reaching a component constructor.
  double objective(Float64List argument) {
    for (var i = 0; i < m; i++) {
      if (!(argument[i] >= lower[i] && argument[i] <= upper[i])) {
        return -double.maxFinite;
      }
    }
    final value = profile.at(argument);
    return value.isNaN ? -double.maxFinite : value;
  }

  final Float64List current;
  if (start == SearchStart.previousParameters) {
    current = _previous(initial, specs, free, scaleAxis, lower, upper);
  } else {
    current = _scan(objective, origin, brackets);
  }

  var best = _maximise(objective, current, axes, brackets, tolerance);
  final absorbed = _noiseAbsorbed(
    initial,
    specs,
    free,
    best.argument,
    brackets,
  );
  if (absorbed) {
    final alternative = _restartWithMoreNoise(
      objective,
      best,
      axes,
      brackets,
      tolerance,
      scaleAxis: scaleAxis,
    );
    if (alternative != null && alternative.value > best.value) {
      best = alternative;
    }
  }
  final optimum = best.argument;

  final widths = Float64List(k)..fillRange(0, k, double.nan);
  final decades = Float64List(k)..fillRange(0, k, double.nan);
  for (var axis = 0; axis < free.length; axis++) {
    final i = free[axis];
    widths[i] = _halfNatWidth(
      objective,
      optimum,
      axis,
      best.value,
      lower[axis],
      upper[axis],
    );
    // A width in a logit coordinate divided by ln 10 is not decades of
    // anything, so the decade view reports NaN there.
    decades[i] = specs[i].isLogarithmic ? widths[i] / _ln10 : double.nan;
  }

  final atOptimum = profile.evaluate(optimum);
  final measurementVariance = profile.measurementVarianceOf(atOptimum);
  final absolute = profile.parametersAt(optimum, measurementVariance);
  final shift = math.log(measurementVariance);
  final ratios = Float64List(k)..fillRange(0, k, double.nan);
  for (var i = 0; i < k; i++) {
    if (specs[i] is VarianceParameter) {
      ratios[i] = math.exp(absolute[i] - shift);
    }
  }

  final searched = _classify(objective, best, axes, brackets, free.length);
  final status = List.filled(k, ParameterStatus.held);
  for (var axis = 0; axis < free.length; axis++) {
    status[free[axis]] = searched[axis];
  }

  return newFitResult(
    model: initial
        .withParameters(absolute)
        .withMeasurementVariance(measurementVariance),
    logMarginalLikelihood: profile.likelihoodAt(optimum),
    logPenalty: profile.penaltyAt(optimum),
    logPrior: profile.priorAt(optimum),
    penalty: penalty,
    varianceRatios: ratios,
    evaluations: profile.evaluations,
    converged: best.converged,
    plateauDecadesByParameter: decades,
    plateauWidthByParameter: widths,
    parameterStatus: status,
    parameterSpecs: specs,
    diffuseDimension: initial.diffuseDimension,
    measurementVariancePinned: fixedMeasurementVariance != null,
  );
}

/// The caller's bracket for a variance ratio, and the owning component's for
/// a shape parameter.
_Brackets _brackets(
  List<ParameterSpec> specs,
  double lowerLogRatio,
  double upperLogRatio, {
  required int scanPoints,
}) {
  final k = specs.length;
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
  return (lower: lower, upper: upper, points: points, step: step);
}

/// The model's own free parameters in search coordinates, clamped into the
/// bracket: variances as ratios to the model's measurement variance, shape
/// parameters as they stand, and the log measurement variance on [scaleAxis]
/// if there is one.
Float64List _previous(
  StructuralModel initial,
  List<ParameterSpec> specs,
  List<int> free,
  int? scaleAxis,
  Float64List lower,
  Float64List upper,
) {
  final theta = initial.parameters;
  final shift = math.log(initial.measurementVariance);
  final start = [
    for (final i in free)
      specs[i] is VarianceParameter ? theta[i] - shift : theta[i],
    if (scaleAxis != null) shift,
  ];
  return Float64List.fromList([
    for (var axis = 0; axis < start.length; axis++)
      start[axis].clamp(lower[axis], upper[axis]),
  ]);
}

/// Coordinate scan: sweep each parameter across its own bracket in turn,
/// holding the others where the previous sweeps left them.
///
/// Two sweeps when there are several parameters, because the first sweeps each
/// parameter against arbitrary values of the ones not yet reached: a cycle's
/// period scanned against the midpoint of its own variance bracket is scanned
/// against a cycle that is not there.
Float64List _scan(
  double Function(Float64List) objective,
  Float64List origin,
  _Brackets brackets,
) {
  final (:lower, upper: _, :points, :step) = brackets;
  final k = origin.length;
  final current = Float64List.fromList(origin);
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
    }
  }
  return current;
}

/// The local search from [start]: nothing to do for no parameters, golden
/// section for one, Nelder-Mead with one restart for several.
_Optimum _maximise(
  double Function(Float64List) objective,
  Float64List start,
  List<ParameterSpec> specs,
  _Brackets brackets,
  double tolerance,
) {
  final k = start.length;
  if (k == 0) {
    // A model of nothing but regression components, whose coefficients are
    // states, or one held entirely under a fixed noise level: one evaluation
    // is the answer.
    return (argument: start, value: objective(start), converged: true);
  }
  if (k == 1) return _goldenSection(objective, start[0], brackets, tolerance);

  // The scan's resolution is the default displacement, except where a
  // component chose one; see ShapeParameter.searchStep. The restart is the
  // standard insurance against a simplex that has collapsed along one
  // direction.
  final wide = Float64List(k);
  final narrow = Float64List(k);
  for (var i = 0; i < k; i++) {
    final spec = specs[i];
    final chosen = spec is ShapeParameter ? spec.searchStep : null;
    wide[i] = chosen ?? 2 * brackets.step[i];
    narrow[i] = (chosen ?? brackets.step[i]) / 4;
  }
  var simplex = maximiseSimplex(
    objective,
    start,
    steps: wide,
    tolerance: tolerance,
  );
  simplex = maximiseSimplex(
    objective,
    simplex.argument,
    steps: narrow,
    tolerance: tolerance,
  );
  return (
    argument: simplex.argument,
    value: simplex.value,
    converged: simplex.converged,
  );
}

/// Golden section within one scan cell either side of [from], moving the
/// window along for as long as the optimum lands on its edge.
///
/// After a scan the neighbouring cells are already known to be worse, so the
/// window never moves. From a warm start it has to: a diary that grew by a
/// month can move its optimum several cells.
_Optimum _goldenSection(
  double Function(Float64List) objective,
  double from,
  _Brackets brackets,
  double tolerance,
) {
  final lower = brackets.lower[0];
  final step = brackets.step[0];
  final last = brackets.points[0] - 1;
  double at(double x) => objective(Float64List.fromList([x]));

  var cell = ((from - lower) / step).round();
  final visited = <int>{};
  while (true) {
    visited.add(cell);
    final left = math.max(0, cell - 1);
    final right = math.min(last, cell + 1);
    final refined = maximise(
      at,
      lower + left * step,
      lower + right * step,
      tolerance: tolerance,
    );
    final offset = (refined.argument - lower) / step;
    final int next;
    if (offset - left < 0.05 && left > 0) {
      next = left - 1;
    } else if (right - offset < 0.05 && right < last) {
      next = right + 1;
    } else {
      next = cell;
    }
    if (next == cell || visited.contains(next)) {
      return (
        argument: Float64List.fromList([refined.argument]),
        value: refined.value,
        converged: refined.converged,
      );
    }
    cell = next;
  }
}

/// Whether a free stationary component's variance ratio finished at the top
/// of its bracket: the signature of a component that has taken over the
/// measurement noise.
bool _noiseAbsorbed(
  StructuralModel model,
  List<ParameterSpec> specs,
  List<int> free,
  Float64List optimum,
  _Brackets brackets,
) {
  final stationary = [
    for (final component in model.components)
      for (var p = 0; p < component.parameterCount; p++)
        !component.diffuseStates.contains(true),
  ];
  for (var axis = 0; axis < free.length; axis++) {
    final i = free[axis];
    if (stationary[i] &&
        specs[i] is VarianceParameter &&
        optimum[axis] >= brackets.upper[axis] - brackets.step[axis]) {
      return true;
    }
  }
  return false;
}

/// Searches again from starts that give the measurement noise a larger share,
/// and returns the best optimum found there, or null if no start was worth
/// searching from.
///
/// Lowering every variance ratio by the same amount keeps the components'
/// shares of the signal and hands the rest to the noise. When a stationary
/// component has absorbed the noise, the better optimum usually lies in that
/// direction, past a stretch where the likelihood falls, which a local search
/// started at the collapsed point does not cross. Where the noise level is
/// searched on [scaleAxis] rather than concentrated out, it is raised by the
/// same amount, so the components keep their variances and the noise grows.
_Optimum? _restartWithMoreNoise(
  double Function(Float64List) objective,
  _Optimum collapsed,
  List<ParameterSpec> specs,
  _Brackets brackets,
  double tolerance, {
  required int? scaleAxis,
}) {
  final k = collapsed.argument.length;
  Float64List? bestStart;
  var bestValue = -double.infinity;
  for (final shift in const [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]) {
    final start = Float64List.fromList(collapsed.argument);
    for (var i = 0; i < k; i++) {
      if (i == scaleAxis) {
        start[i] = math.min(brackets.upper[i], start[i] + shift);
      } else if (specs[i] is VarianceParameter) {
        start[i] = math.max(brackets.lower[i], start[i] - shift);
      }
    }
    final value = objective(start);
    if (value > bestValue) {
      bestValue = value;
      bestStart = start;
    }
  }
  if (bestStart == null) return null;
  return _maximise(objective, bestStart, specs, brackets, tolerance);
}

/// A parameter counts as sitting on a bound when it finishes in the last cell
/// of the scan next to that bound, and the objective at the bound itself is
/// within half a nat of the optimum.
///
/// An exact comparison would miss the usual case: golden section stops a
/// little short of the bound, and the simplex is repelled from it by the
/// objective going to minus infinity just outside. The cell alone is too
/// coarse: on a log axis four decades wide it spans a factor of 1.5, so a
/// length scale of 1.4 over a floor of 1 would count as on the floor while
/// the floor is more than a nat worse.
///
/// A variance at the bottom of its bracket has been shrunk; a shape parameter
/// at the bottom has not been shrunk out of anything, and its bracket is in
/// the wrong place, which is what beyondBracket means.
///
/// Only the first [count] axes are classified, which leaves out the noise
/// level when it is searched.
List<ParameterStatus> _classify(
  double Function(Float64List) objective,
  _Optimum best,
  List<ParameterSpec> specs,
  _Brackets brackets,
  int count,
) {
  final (:lower, :upper, points: _, :step) = brackets;
  final optimum = best.argument;
  final probe = Float64List.fromList(optimum);
  bool reaches(int axis, double bound) {
    probe[axis] = bound;
    final value = objective(probe);
    probe[axis] = optimum[axis];
    return value >= best.value - _halfNat;
  }

  return [
    for (var i = 0; i < count; i++)
      if (optimum[i] >= upper[i] - step[i] && reaches(i, upper[i]))
        ParameterStatus.beyondBracket
      else if (optimum[i] <= lower[i] + step[i] && reaches(i, lower[i]))
        specs[i] is VarianceParameter
            ? ParameterStatus.shrunkToNothing
            : ParameterStatus.beyondBracket
      else
        ParameterStatus.determined,
  ];
}

/// [result] with [FitResult.largestResidual] filled in from one more forward
/// pass on the fitted model.
FitResult _withLargestResidual(
  FitResult result,
  List<Observation> observations,
) {
  // TODO(gwosd): iterate this into a robust fit that down-weights the worst
  // readings through relativeVariance and refits (see doc/roadmap.md).
  final diagnostics = result.model.diagnose(observations);
  final residuals = diagnostics.residuals;
  if (residuals.length < 8) return result;

  final sizes = [for (final r in residuals) r.abs()]..sort();
  final middle = sizes.length ~/ 2;
  final median = sizes.length.isOdd
      ? sizes[middle]
      : (sizes[middle - 1] + sizes[middle]) / 2;
  final scale = 1.4826 * median;
  if (!(scale > 0)) return result;

  var worst = 0;
  for (var i = 1; i < residuals.length; i++) {
    if (residuals[i].abs() > residuals[worst].abs()) worst = i;
  }
  return newFitResult(
    model: result.model,
    logMarginalLikelihood: result.logMarginalLikelihood,
    logPenalty: result.logPenalty,
    logPrior: result.logPrior,
    penalty: result.penalty,
    varianceRatios: result.varianceRatios,
    evaluations: result.evaluations,
    converged: result.converged,
    plateauDecadesByParameter: result.plateauDecadesByParameter,
    plateauWidthByParameter: result.plateauWidthByParameter,
    parameterStatus: result.parameterStatus,
    parameterSpecs: result.parameterSpecs,
    diffuseDimension: result.diffuseDimension,
    measurementVariancePinned: result.measurementVariancePinned,
    largestResidual: (
      time: diagnostics.times[worst],
      score: residuals[worst] / scale,
    ),
  );
}

/// The typical gap between readings, or zero when there are too few distinct
/// times for that to mean anything.
///
/// Readings much closer together than the typical gap (two weighings on one
/// morning) are counted as one visit, and the result is the median gap
/// between visits. The median rather than the mean, so that a diary with a
/// fortnight's holiday in it still counts as daily; visits rather than raw
/// gaps, so that weighing twice most mornings does not make it count as
/// sampled every five minutes.
///
/// This is what a shape parameter measured in time units has to clear to be a
/// different model rather than a second spelling of measurement noise; see
/// [Component.parameterSpecsAt].
double samplingResolution(List<Observation> observations) =>
    typicalGap([for (final o in observations) o.time]);

/// How far parameter [axis] can move on its own before the objective falls
/// half a nat below [peak].
///
/// Conditional on the others: when two components trade off against one
/// another the joint region is wider than any of these slices. It costs about
/// three dozen filter passes per parameter.
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
