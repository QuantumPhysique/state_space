import 'dart:math' as math;
import 'dart:typed_data';

import 'fit/fit.dart';
import 'fit/penalty.dart';
import 'model.dart';
import 'parameter_spec.dart';
import 'stats/normal.dart';

/// A central interval, in the units of the observations.
typedef Band = ({double lo, double hi});

/// A central interval at every output time, as two arrays of the same length.
typedef Bands = ({Float64List lo, Float64List hi});

/// The posterior of one regression coefficient.
///
/// A constant state under a flat prior, so the smoother returns the same
/// number at every step and there is one figure to report rather than a curve.
final class Coefficient {
  const Coefficient._({
    required this.name,
    required this.estimate,
    required this.variance,
  });

  /// The name given to the regressor this belongs to.
  final String name;

  /// Posterior mean, in signal units per unit of the column.
  final double estimate;

  /// Its posterior variance.
  final double variance;

  /// The posterior standard deviation, which is the `plus or minus` figure.
  double get standardError => math.sqrt(variance);

  /// A central credible interval for the coefficient.
  ///
  /// This is a posterior interval, not a confidence interval derived from
  /// asymptotics.
  Band interval({double coverage = 0.95}) {
    final half = twoSidedZ(coverage) * standardError;
    return (lo: estimate - half, hi: estimate + half);
  }

  @override
  String toString() =>
      '$name: ${estimate.toStringAsFixed(3)} '
      '+/- ${standardError.toStringAsFixed(3)}';
}

/// Builds a [Coefficient]. Internal to the package.
Coefficient newCoefficient({
  required String name,
  required double estimate,
  required double variance,
}) => Coefficient._(name: name, estimate: estimate, variance: variance);

Float64List _view(Float64List list) => list.asUnmodifiableView();

Bands _bands(
  Float64List mean,
  Float64List variance,
  double extra,
  double coverage,
) {
  final z = twoSidedZ(coverage);
  final lo = Float64List(mean.length);
  final hi = Float64List(mean.length);
  for (var i = 0; i < mean.length; i++) {
    final half = z * math.sqrt(variance[i] + extra);
    lo[i] = mean[i] - half;
    hi[i] = mean[i] + half;
  }
  return (lo: lo, hi: hi);
}

/// The posterior of a model, evaluated at each requested output time.
///
/// Every array has the same length and the same ordering: the observation
/// times, or the output grid if one was given. The arrays are read-only views.
///
/// {@category Getting started}
final class SmoothingResult {
  SmoothingResult._({
    required Float64List times,
    required Float64List mean,
    required Float64List variance,
    required List<Float64List> componentMeans,
    required List<Float64List> componentVariances,
    required List<Float64List?> componentSlopes,
    required List<Float64List?> componentSlopeVariances,
    required this.trendIndex,
    required this.logMarginalLikelihood,
    required this.measurementVariance,
    required List<Coefficient> coefficients,
  }) : times = _view(times),
       mean = _view(mean),
       variance = _view(variance),
       _componentMeans = [for (final m in componentMeans) _view(m)],
       _componentVariances = [for (final v in componentVariances) _view(v)],
       _componentSlopes = [
         for (final s in componentSlopes) s == null ? null : _view(s),
       ],
       _componentSlopeVariances = [
         for (final s in componentSlopeVariances) s == null ? null : _view(s),
       ],
       coefficients = List.unmodifiable(coefficients);

  /// Output times, ascending.
  final Float64List times;

  /// Posterior mean of the signal, `E[H(t) x(t) | y]`: the sum of every
  /// component's contribution. [componentMean] has each one on its own.
  final Float64List mean;

  /// Posterior variance of [mean]. This is uncertainty about the underlying
  /// signal and excludes measurement noise; see [predictiveInterval].
  final Float64List variance;

  /// Index of the component [trendSlope] is read from: the first
  /// non-stationary component with a rate state, or failing that the first
  /// stationary one, or null when none has one.
  final int? trendIndex;

  /// Posterior mean of the trend's rate of change, in signal units per time
  /// unit, or null when no component has a rate state.
  ///
  /// This is the rate of the trend component alone, [componentSlope] of
  /// [trendIndex]. With a seasonal or a stationary component in the model it
  /// is not the derivative of [mean]: [mean] includes the weekly wiggle, and
  /// the trend's slope does not.
  Float64List? get trendSlope =>
      trendIndex == null ? null : _componentSlopes[trendIndex!];

  /// Posterior variance of [trendSlope], or null alongside it.
  Float64List? get trendSlopeVariance =>
      trendIndex == null ? null : _componentSlopeVariances[trendIndex!];

  /// `log p(y | theta)` from the forward pass, with the diffuse burn-in
  /// excluded. The same number the textbook `O(N^3)` Gaussian process
  /// likelihood would give.
  ///
  /// Under exact diffuse initialisation it is the *restricted* likelihood, so
  /// it is comparable only across models that integrate out the same number of
  /// flat directions. See [FitResult.isComparableWith].
  final double logMarginalLikelihood;

  /// The model's measurement variance, needed for [predictiveInterval].
  final double measurementVariance;

  /// Every regression coefficient in the model, in the order the regressors
  /// appear across the model's components. Empty when there are none.
  final List<Coefficient> coefficients;

  final List<Float64List> _componentMeans;
  final List<Float64List> _componentVariances;
  final List<Float64List?> _componentSlopes;
  final List<Float64List?> _componentSlopeVariances;

  /// Number of output times.
  int get length => times.length;

  /// Number of components in the model this came from.
  int get componentCount => _componentMeans.length;

  /// Smoothed contribution of component [index] to the signal, where [index]
  /// is the component's position in [StructuralModel.components].
  Float64List componentMean(int index) => _componentMeans[index];

  /// Posterior variance of [componentMean].
  Float64List componentVariance(int index) => _componentVariances[index];

  /// Posterior mean of component [index]'s rate of change, or null when that
  /// component has no rate state. See [Component.rateStateIndex].
  Float64List? componentSlope(int index) => _componentSlopes[index];

  /// Posterior variance of [componentSlope].
  Float64List? componentSlopeVariance(int index) =>
      _componentSlopeVariances[index];

  /// Where the underlying signal is, at output index [i].
  ///
  /// This is the band to draw around a trend line. It is narrow, and most
  /// individual measurements fall outside it: it answers "where is the
  /// trend", not "where would the next reading land". For the latter, use
  /// [predictiveInterval].
  ///
  /// The measurement variance is treated as known. When it was estimated from
  /// a handful of readings the band is too narrow by the uncertainty in that
  /// estimate.
  Band credibleInterval(int i, {double coverage = 0.95}) {
    final half = twoSidedZ(coverage) * math.sqrt(variance[i]);
    return (lo: mean[i] - half, hi: mean[i] + half);
  }

  /// Where a new measurement would fall, at output index [i]: the credible
  /// interval widened by the measurement noise. Roughly 95% of the observed
  /// points should sit inside the 95% version of this band.
  Band predictiveInterval(int i, {double coverage = 0.95}) {
    final half =
        twoSidedZ(coverage) * math.sqrt(variance[i] + measurementVariance);
    return (lo: mean[i] - half, hi: mean[i] + half);
  }

  /// [credibleInterval] at every output time, as two arrays to draw.
  Bands credibleBand({double coverage = 0.95}) =>
      _bands(mean, variance, 0, coverage);

  /// [predictiveInterval] at every output time, as two arrays to draw.
  Bands predictiveBand({double coverage = 0.95}) =>
      _bands(mean, variance, measurementVariance, coverage);
}

/// Builds a [SmoothingResult]. Internal to the package.
SmoothingResult newSmoothingResult({
  required Float64List times,
  required Float64List mean,
  required Float64List variance,
  required double logMarginalLikelihood,
  required double measurementVariance,
  List<Float64List> componentMeans = const [],
  List<Float64List> componentVariances = const [],
  List<Float64List?> componentSlopes = const [],
  List<Float64List?> componentSlopeVariances = const [],
  int? trendIndex,
  List<Coefficient> coefficients = const [],
}) => SmoothingResult._(
  times: times,
  mean: mean,
  variance: variance,
  componentMeans: componentMeans,
  componentVariances: componentVariances,
  componentSlopes: componentSlopes,
  componentSlopeVariances: componentSlopeVariances,
  trendIndex: trendIndex,
  logMarginalLikelihood: logMarginalLikelihood,
  measurementVariance: measurementVariance,
  coefficients: coefficients,
);

/// The signal projected past the end of the data.
///
/// For a local linear trend the variance of the forecast grows like the cube
/// of the horizon, so the band widens like its three-halves power. That is the
/// model's statement that a trend whose slope is free to wander becomes
/// unknowable, not a defect.
///
/// {@category Getting started}
final class ForecastResult {
  ForecastResult._({
    required Float64List times,
    required Float64List mean,
    required Float64List variance,
    required this.measurementVariance,
  }) : times = _view(times),
       mean = _view(mean),
       variance = _view(variance);

  /// The requested horizon, ascending.
  final Float64List times;

  /// Posterior mean of the signal at each horizon time.
  final Float64List mean;

  /// Its variance, excluding measurement noise.
  final Float64List variance;

  /// The model's measurement variance, for [predictiveInterval].
  final double measurementVariance;

  /// Number of horizon points.
  int get length => times.length;

  /// Where the signal itself is heading.
  Band credibleInterval(int i, {double coverage = 0.95}) {
    final half = twoSidedZ(coverage) * math.sqrt(variance[i]);
    return (lo: mean[i] - half, hi: mean[i] + half);
  }

  /// Where an actual future reading would fall: the same band widened by the
  /// measurement noise. This is the one to use for a question like "when will
  /// the scale read 75?", because the scale reading includes the noise.
  Band predictiveInterval(int i, {double coverage = 0.95}) {
    final half =
        twoSidedZ(coverage) * math.sqrt(variance[i] + measurementVariance);
    return (lo: mean[i] - half, hi: mean[i] + half);
  }

  /// [credibleInterval] at every horizon time, as two arrays to draw.
  Bands credibleBand({double coverage = 0.95}) =>
      _bands(mean, variance, 0, coverage);

  /// [predictiveInterval] at every horizon time, as two arrays to draw.
  Bands predictiveBand({double coverage = 0.95}) =>
      _bands(mean, variance, measurementVariance, coverage);
}

/// Builds a [ForecastResult]. Internal to the package.
ForecastResult newForecastResult({
  required Float64List times,
  required Float64List mean,
  required Float64List variance,
  required double measurementVariance,
}) => ForecastResult._(
  times: times,
  mean: mean,
  variance: variance,
  measurementVariance: measurementVariance,
);

/// What became of one parameter during a fit.
enum ParameterStatus {
  /// The optimum is inside the search bracket, with the likelihood falling
  /// away on both sides of it. The estimate means what it says.
  determined,

  /// The optimum sits at the bottom of the bracket rather than being
  /// estimated.
  ///
  /// What that means depends on the parameter. For the variance driving a
  /// trend or a seasonal it means the component does not change over time: it
  /// is still in the model, fitted as a fixed line or a fixed pattern. For the
  /// variance of a stationary component it means the component contributes
  /// nothing.
  ///
  /// A variance of zero is the edge of the parameter space, not an interior
  /// point, and the usual asymptotics do not hold there. The width reported
  /// for such a parameter is one-sided, so it is not an error bar.
  shrunkToNothing,

  /// The optimum sits at the top of the bracket, so the real one may be
  /// outside it.
  ///
  /// For a trend's variance this is usually the time unit: the variance is per
  /// cubed time unit, so measuring time in seconds rather than days moves the
  /// optimum about fifteen decades. For a stationary component's variance it
  /// usually means the component has taken over the measurement noise; see
  /// [FitResult.warnings].
  beyondBracket,
}

/// What [StructuralModel] fitting returned, together with enough diagnostics
/// to tell a well-determined answer from a shrug.
///
/// {@category Choosing a model}
final class FitResult {
  FitResult._({
    required this.model,
    required this.logMarginalLikelihood,
    required this.logPenalty,
    required this.penalty,
    required this.evaluations,
    required this.converged,
    required List<ParameterStatus> parameterStatus,
    required List<ParameterSpec> parameterSpecs,
    required this.diffuseDimension,
    required this.measurementVariancePinned,
    required this.largestResidual,
    required Float64List varianceRatios,
    required Float64List plateauDecadesByParameter,
    required Float64List plateauWidthByParameter,
  }) : parameterStatus = List.unmodifiable(parameterStatus),
       parameterSpecs = List.unmodifiable(parameterSpecs),
       varianceRatios = _view(varianceRatios),
       plateauDecadesByParameter = _view(plateauDecadesByParameter),
       plateauWidthByParameter = _view(plateauWidthByParameter);

  /// The fitted model: components at their estimated variances, and the
  /// analytically concentrated measurement variance.
  final StructuralModel model;

  /// The maximised profile log-likelihood, *without* the penalty.
  ///
  /// The penalised objective is the right thing to maximise and the wrong
  /// thing to compare: two models penalised differently are not on the same
  /// scale. This is the unpenalised number, which removes that objection.
  ///
  /// **It is still comparable only across models of equal [diffuseDimension],**
  /// and [isComparableWith] is the check. Under exact diffuse initialisation
  /// this is a restricted likelihood: the flat directions have been integrated
  /// out against an improper prior of unit density, so the result carries the
  /// units of those directions. Two consequences, both measurable:
  ///
  /// * Writing a regression column in grams rather than kilograms shifts this
  ///   number by exactly `log 1000`, while the fit, the posterior and the
  ///   coefficient are unchanged.
  /// * Measuring [Observation.time] in half-days rather than days shifts it by
  ///   exactly `log 2` per diffuse direction that carries a time dimension.
  ///
  /// So a comparison between a trend and a trend-plus-seasonal, or between a
  /// model with a holiday indicator and one without, is decided by an
  /// arbitrary choice of units rather than by the data. To choose between
  /// models whose diffuse structure differs, use the fitted
  /// [measurementVariance], an out-of-sample error, or
  /// `StructuralModel.diagnose`.
  final double logMarginalLikelihood;

  /// What the penalty contributed at the optimum, so that the objective that
  /// was actually maximised is [logMarginalLikelihood] plus this. Zero under
  /// [NoPenalty].
  final double logPenalty;

  /// The penalty the fit used, which [fit] chooses for you unless you say
  /// otherwise.
  final Penalty penalty;

  /// The estimated ratio of each variance to the measurement variance, in
  /// parameter order.
  ///
  /// Not every parameter is a variance. Entries belonging to a shape parameter
  /// — a Matérn length scale, a cycle's period — are [double.nan], because a
  /// ratio of a period to a variance is not a quantity. [parameterSpecs] says
  /// which is which, and the fitted values themselves are on the components of
  /// [model], which is where a caller should read a period from anyway.
  final Float64List varianceRatios;

  /// Number of filter passes the search took, including the axis probes that
  /// produced [plateauDecadesByParameter].
  final int evaluations;

  /// Whether the search met its tolerance rather than running out of steps.
  final bool converged;

  /// Width, in decades, of the region around the optimum where the objective
  /// stays within half a nat of its maximum, one entry per parameter.
  ///
  /// Each is measured along its own axis with the others held at the optimum,
  /// so it answers "how well pinned down is this one number" and not "how well
  /// pinned down is the fit". When two components trade off against each other
  /// the joint region is wider than any of these slices, and these numbers
  /// understate how undetermined things are. Worse, when another parameter has
  /// finished on a bound the slice is taken at that bound, which can make a
  /// width look tiny for a reason that has nothing to do with the data; see
  /// [warnings].
  ///
  /// Entries whose coordinate is not a logarithm — a damping factor, which is
  /// searched as a logit — are [double.nan], because a width in logits divided
  /// by `ln 10` is not decades of anything. [plateauWidthByParameter] has the
  /// raw number for every parameter, and [ParameterSpec.isLogarithmic] says
  /// which is which. This mirrors what [varianceRatios] already does for a
  /// parameter that is not a variance.
  ///
  /// For a parameter whose [parameterStatus] is not
  /// [ParameterStatus.determined] the width is one-sided and is not an error
  /// bar; see that enum for why.
  final Float64List plateauDecadesByParameter;

  /// The same widths, in each parameter's own unconstrained coordinate, and
  /// finite for every parameter.
  ///
  /// This is [plateauDecadesByParameter] before the division by `ln 10`, and
  /// it is what to read for a parameter searched as a logit.
  final Float64List plateauWidthByParameter;

  /// What became of each parameter, in the same order as [varianceRatios].
  final List<ParameterStatus> parameterStatus;

  /// What each parameter is, in the same order again.
  final List<ParameterSpec> parameterSpecs;

  /// How many flat directions this fit integrated out.
  ///
  /// Zero under [ApproximateDiffuse]. Under [ExactDiffuse] it is the number of
  /// diffuse states, which is two for a trend, two per harmonic for a
  /// seasonal, one per regression column, and none for a stationary component.
  /// It is what [logMarginalLikelihood] has to match before two fits can be
  /// compared.
  final int diffuseDimension;

  /// Whether [logMarginalLikelihood] means the same thing for this fit and
  /// [other], so that the two numbers may be subtracted.
  ///
  /// True when both integrated out the same number of flat directions. False
  /// otherwise, and then the difference between the two likelihoods is not a
  /// statement about the data — see [logMarginalLikelihood] for why.
  ///
  /// It does not check that the fits are of the same observations, which no
  /// [FitResult] retains; that is the caller's to know.
  bool isComparableWith(FitResult other) =>
      diffuseDimension == other.diffuseDimension;

  /// Whether the measurement variance was asserted rather than estimated.
  ///
  /// True when `fixedMeasurementVariance` was passed to [fit], and also when
  /// `minimumMeasurementVariance` was passed and the free estimate came out
  /// below it, so the fit was redone with the noise held at the floor. False
  /// means [measurementVariance] is the maximum-likelihood estimate, floor or
  /// no floor.
  final bool measurementVariancePinned;

  /// The observation that sits furthest from what the rest of the data
  /// predicts for it, or null when there are too few residuals to judge.
  ///
  /// `score` is its one-step-ahead prediction error divided by a robust
  /// estimate of the typical error (1.4826 times the median absolute
  /// residual), so the bad reading does not hide itself by inflating the
  /// scale it is measured against. Gaussian noise almost never exceeds 6.
  final ({double time, double score})? largestResidual;

  /// Whether any parameter finished at an edge of the search bracket.
  ///
  /// Kept as a single flag because it is the first thing worth checking;
  /// [parameterStatus] says which parameter and which edge, which is what
  /// determines whether the answer is wrong or merely uninteresting.
  bool get atBracketEdge =>
      parameterStatus.any((s) => s != ParameterStatus.determined);

  /// The estimated ratio of process to measurement variance, for a model with
  /// exactly one of them. Its reciprocal is the smoothing-spline parameter
  /// `lambda`: small values give a stiff curve, large values one that chases
  /// the data.
  double get varianceRatio {
    final variances = [
      for (var i = 0; i < parameterSpecs.length; i++)
        if (parameterSpecs[i] is VarianceParameter) varianceRatios[i],
    ];
    if (variances.length != 1) {
      throw StateError(
        'this model has ${variances.length} variance ratios, so '
        'there is no single one to report. Use varianceRatios, which is in '
        'parameter order.',
      );
    }
    return variances.first;
  }

  /// The widest of [plateauDecadesByParameter]: how undetermined the least
  /// determined parameter is.
  ///
  /// A well-determined fit gives well under a decade. Several decades means
  /// the data does not distinguish a stiff curve from a flexible one, and the
  /// point estimate should be treated as a convention rather than a finding.
  ///
  /// Parameters that are not on a log scale are skipped rather than mixed in,
  /// since decades and logits do not compare.
  double get plateauDecades {
    var widest = 0.0;
    for (final width in plateauDecadesByParameter) {
      if (width.isNaN) continue;
      if (width > widest) widest = width;
    }
    return widest;
  }

  /// What is worth knowing about this fit before quoting anything from it, in
  /// plain sentences, empty when there is nothing to say.
  ///
  /// Five things are reported:
  ///
  /// 1. A reading far from what the rest of the data predicts for it (see
  ///    [largestResidual]). The fit is Gaussian, so one mistyped value
  ///    inflates the noise estimate and moves the whole curve; screen it out
  ///    or give it a large [Observation.relativeVariance].
  /// 2. A parameter that finished on a bound. Its estimate is a boundary, not
  ///    an interior optimum, and the width beside it is one-sided.
  /// 3. A stationary component's variance at the top of its bracket, which
  ///    means it has taken over the measurement noise.
  /// 4. A parameter the data barely constrains: a plateau over two decades
  ///    wide.
  /// 5. A parameter whose width was measured while a shape parameter of the
  ///    same component sat on a bound, which can make it look sharp for a
  ///    reason that is not about the data. See [StochasticCycle] for the
  ///    common case.
  List<String> get warnings {
    final found = <String>[];
    final outlier = largestResidual;
    if (outlier != null && outlier.score.abs() > outlierScore) {
      found.add(
        'the reading at time ${outlier.time} is '
        '${outlier.score.abs().toStringAsFixed(1)} typical errors from what '
        'the rest of the data predicts; if it is a mistake, remove it or give '
        'it a large relativeVariance, because one bad reading inflates the '
        'noise estimate and moves the whole curve',
      );
    }
    var at = 0;
    for (final component in model.components) {
      final count = component.parameterCount;
      final name = component.name;
      final stationary = !component.diffuseStates.contains(true);
      var pinnedShape = -1;
      for (var i = at; i < at + count; i++) {
        if (parameterSpecs[i] is ShapeParameter &&
            parameterStatus[i] != ParameterStatus.determined) {
          pinnedShape = i;
        }
      }
      for (var i = at; i < at + count; i++) {
        final label = '${parameterSpecs[i].label} of $name';
        final isVariance = parameterSpecs[i] is VarianceParameter;
        switch (parameterStatus[i]) {
          case ParameterStatus.shrunkToNothing:
            found.add(
              stationary
                  ? 'the $label was shrunk to the bottom of its bracket: the '
                        'component contributes nothing to this data, and the '
                        'width reported for it is one-sided'
                  : 'the $label was shrunk to the bottom of its bracket: the '
                        'data shows no sign of the component changing over time, '
                        'so it is fitted as fixed, and the width reported for it '
                        'is one-sided',
            );
          case ParameterStatus.beyondBracket:
            found.add(
              stationary && isVariance
                  ? 'the $label finished at the top of its bracket, which means '
                        'the component has taken over the measurement noise and '
                        'the fitted noise level is too small; pass '
                        'minimumMeasurementVariance at what the instrument can '
                        'resolve'
                  : 'the $label finished at the top of its bracket, so the real '
                        'optimum may be outside it',
            );
          case ParameterStatus.determined:
            if (plateauDecadesByParameter[i] > 2) {
              found.add(
                'the $label can move '
                '${plateauDecadesByParameter[i].toStringAsFixed(1)} decades '
                'without the fit getting half a nat worse, so its value is a '
                'convention rather than a finding',
              );
            } else if (pinnedShape >= 0 && pinnedShape != i) {
              found.add(
                'the width reported for the $label was measured with '
                'the ${parameterSpecs[pinnedShape].label} held on its bound, '
                'so it says how sharp the likelihood is there and not how '
                'well the data determines it',
              );
            }
        }
      }
      at += count;
    }
    return found;
  }

  /// The [largestResidual] score above which [warnings] reports a reading.
  static const double outlierScore = 6;

  /// True when the likelihood surface is too flat to support the estimate.
  bool get isFlat => plateauDecades > 2;

  /// The concentrated measurement variance, i.e. the fitted noise level.
  double get measurementVariance => model.measurementVariance;

  @override
  String toString() {
    final ratios = varianceRatios
        .map((r) => r.toStringAsPrecision(4))
        .join(', ');
    return 'FitResult(varianceRatios: [$ratios], measurementVariance: '
        '${measurementVariance.toStringAsPrecision(4)}, logLik: '
        '${logMarginalLikelihood.toStringAsFixed(3)}, evaluations: '
        '$evaluations, converged: $converged)';
  }
}

/// Builds a [FitResult]. Internal to the package.
FitResult newFitResult({
  required StructuralModel model,
  required double logMarginalLikelihood,
  required double logPenalty,
  required Penalty penalty,
  required Float64List varianceRatios,
  required int evaluations,
  required bool converged,
  required Float64List plateauDecadesByParameter,
  required Float64List plateauWidthByParameter,
  required List<ParameterStatus> parameterStatus,
  required List<ParameterSpec> parameterSpecs,
  required int diffuseDimension,
  bool measurementVariancePinned = false,
  ({double time, double score})? largestResidual,
}) => FitResult._(
  model: model,
  logMarginalLikelihood: logMarginalLikelihood,
  logPenalty: logPenalty,
  penalty: penalty,
  varianceRatios: varianceRatios,
  evaluations: evaluations,
  converged: converged,
  plateauDecadesByParameter: plateauDecadesByParameter,
  plateauWidthByParameter: plateauWidthByParameter,
  parameterStatus: parameterStatus,
  parameterSpecs: parameterSpecs,
  diffuseDimension: diffuseDimension,
  measurementVariancePinned: measurementVariancePinned,
  largestResidual: largestResidual,
);
