import 'dart:math' as math;
import 'dart:typed_data';

import 'fit/fit.dart';
import 'fit/penalty.dart';
import 'model.dart';
import 'parameter_spec.dart';
import 'stats/normal.dart';

/// A central interval, in the units of the observations.
typedef Interval = ({double lo, double hi});

/// The posterior of one regression coefficient.
///
/// A constant state under a flat prior, so the smoother returns the same
/// number at every step and there is one figure to report rather than a curve.
class Coefficient {
  const Coefficient({
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
  /// Note that this is a posterior interval and not a confidence interval
  /// derived from asymptotics. Nothing here is at a boundary — a coefficient
  /// is free to be any real number — so the usual warning about parameters
  /// pinned at zero does not apply to these, only to the variances.
  Interval interval({double coverage = 0.95}) {
    final half = twoSidedZ(coverage) * standardError;
    return (lo: estimate - half, hi: estimate + half);
  }

  @override
  String toString() => '$name: ${estimate.toStringAsFixed(3)} '
      '+/- ${standardError.toStringAsFixed(3)}';
}

/// The posterior of a fitted model, evaluated at each requested output time.
///
/// Every array has the same length and the same ordering: the observation
/// times, or the output grid if one was given.
class SmoothingResult {
  /// Built by [StructuralModel.smooth]; there is no reason to construct one
  /// by hand outside a test.
  SmoothingResult({
    required this.times,
    required this.level,
    required this.levelVariance,
    required this.slope,
    required this.slopeVariance,
    required this.logMarginalLikelihood,
    required this.measurementVariance,
    required List<Float64List> componentMeans,
    required List<Float64List> componentVariances,
    List<Coefficient> coefficients = const [],
  })  : _componentMeans = componentMeans,
        _componentVariances = componentVariances,
        coefficients = List.unmodifiable(coefficients);

  /// Output times, ascending.
  final Float64List times;

  /// Posterior mean of the signal, `E[H(t) x(t) | y]` — the sum of every
  /// component's contribution. With a single trend component this is the
  /// smoothed level.
  final Float64List level;

  /// Posterior variance of [level]. This is uncertainty about the underlying
  /// signal and excludes measurement noise; see [predictiveInterval].
  final Float64List levelVariance;

  /// Posterior mean of the rate of change, in signal units per time unit, or
  /// null when no component in the model has a rate state.
  final Float64List? slope;

  /// Posterior variance of [slope], or null alongside it.
  final Float64List? slopeVariance;

  /// `log p(y | theta)` from the forward pass, with the diffuse burn-in
  /// excluded. The same number the textbook `O(N^3)` Gaussian process
  /// likelihood would give.
  final double logMarginalLikelihood;

  /// The model's measurement variance, needed for [predictiveInterval].
  final double measurementVariance;

  /// Every regression coefficient in the model, in the order the regressors
  /// appear across the model's components. Empty when there are none.
  ///
  /// This is where "the fortnight over Christmas was worth 1.2 kg, give or
  /// take 0.3" comes from. The coefficients are states rather than parameters,
  /// so they cost the optimiser nothing and arrive with the rest of the
  /// posterior.
  final List<Coefficient> coefficients;

  final List<Float64List> _componentMeans;
  final List<Float64List> _componentVariances;

  /// Number of output times.
  int get length => times.length;

  /// Number of components in the model this came from.
  int get componentCount => _componentMeans.length;

  /// Smoothed contribution of component [index] to the signal.
  ///
  /// With one component this is just [level]; it exists from the first release
  /// so that adding seasonal and regression components later is not a breaking
  /// change for anyone who has already written a chart against it.
  Float64List componentMean(int index) => _componentMeans[index];

  /// Posterior variance of [componentMean].
  Float64List componentVariance(int index) => _componentVariances[index];

  /// Posterior standard deviation of the signal at output index [i].
  double levelStandardDeviation(int i) => math.sqrt(levelVariance[i]);

  /// Where the underlying signal is, at output index [i].
  ///
  /// This is the band to draw around a trend line. It is narrow, and most
  /// individual measurements will fall outside it — that is not a bug, it is
  /// the difference between "where is the trend" and "where would the next
  /// reading land". For the latter, use [predictiveInterval].
  Interval credibleInterval(int i, {double coverage = 0.95}) {
    final half = twoSidedZ(coverage) * math.sqrt(levelVariance[i]);
    return (lo: level[i] - half, hi: level[i] + half);
  }

  /// Where a new measurement would fall, at output index [i]: the credible
  /// interval widened by the measurement noise. Roughly 95% of the observed
  /// points should sit inside the 95% version of this band.
  Interval predictiveInterval(int i, {double coverage = 0.95}) {
    final half =
        twoSidedZ(coverage) * math.sqrt(levelVariance[i] + measurementVariance);
    return (lo: level[i] - half, hi: level[i] + half);
  }
}

/// The signal projected past the end of the data.
///
/// Forecast uncertainty grows quickly and does so at a rate the model
/// determines: for a local linear trend the variance of the level grows like
/// the cube of the horizon, so the band widens like its three-halves power.
/// That is a statement about the model, not a defect of it — a trend whose
/// slope is free to wander really does become unknowable, and a band that
/// stayed narrow would be lying.
class ForecastResult {
  /// Built by [StructuralModel.forecast].
  ForecastResult({
    required this.times,
    required this.mean,
    required this.variance,
    required this.measurementVariance,
  });

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
  Interval credibleInterval(int i, {double coverage = 0.95}) {
    final half = twoSidedZ(coverage) * math.sqrt(variance[i]);
    return (lo: mean[i] - half, hi: mean[i] + half);
  }

  /// Where an actual future reading would fall: the same band widened by the
  /// measurement noise. This is the one to use for a question like "when will
  /// the scale read 75?", because the scale reading includes the noise.
  Interval predictiveInterval(int i, {double coverage = 0.95}) {
    final half =
        twoSidedZ(coverage) * math.sqrt(variance[i] + measurementVariance);
    return (lo: mean[i] - half, hi: mean[i] + half);
  }
}

/// What became of one parameter during a fit.
enum ParameterStatus {
  /// The optimum is inside the search bracket, with the likelihood falling
  /// away on both sides of it. The estimate means what it says.
  determined,

  /// The optimum sits at the bottom of the bracket: the component has been
  /// shrunk out of the model rather than estimated.
  ///
  /// A variance of zero is the edge of the parameter space, not an interior
  /// point, and the usual asymptotics do not hold there. Whatever width is
  /// reported for such a parameter is one-sided — the likelihood cannot fall
  /// away below a boundary it cannot cross — so it is not an error bar and
  /// should not be quoted as one. The useful reading is qualitative: the data
  /// gives this component nothing to do.
  shrunkToNothing,

  /// The optimum sits at the top of the bracket, so the real one is probably
  /// outside it.
  ///
  /// Almost always a unit problem. A trend's variance is per cubed time unit,
  /// so measuring time in seconds rather than days moves the optimum about
  /// fifteen decades; widen the bracket or change the unit.
  beyondBracket,
}

/// What [StructuralModel] fitting returned, together with enough diagnostics
/// to tell a well-determined answer from a shrug.
class FitResult {
  /// Built by [fit].
  FitResult({
    required this.model,
    required this.logMarginalLikelihood,
    required this.logPenalty,
    required this.penalty,
    required this.varianceRatios,
    required this.evaluations,
    required this.converged,
    required this.plateauDecadesByParameter,
    required List<ParameterStatus> parameterStatus,
    required List<ParameterSpec> parameterSpecs,
    this.measurementVariancePinned = false,
  })  : parameterStatus = List.unmodifiable(parameterStatus),
        parameterSpecs = List.unmodifiable(parameterSpecs);

  /// The fitted model: components at their estimated variances, and the
  /// analytically concentrated measurement variance.
  final StructuralModel model;

  /// The maximised profile log-likelihood, *without* the penalty.
  ///
  /// This is the number to compare across models. The penalised objective is
  /// the right thing to maximise and the wrong thing to compare: two models
  /// penalised differently are not on the same scale.
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
  /// understate how undetermined things are.
  ///
  /// For a parameter whose [parameterStatus] is not
  /// [ParameterStatus.determined] the width is one-sided and is not an error
  /// bar; see that enum for why.
  final Float64List plateauDecadesByParameter;

  /// What became of each parameter, in the same order as [varianceRatios].
  final List<ParameterStatus> parameterStatus;

  /// What each parameter is, in the same order again.
  final List<ParameterSpec> parameterSpecs;

  /// Whether the measurement variance was asserted rather than estimated.
  ///
  /// True when `fixedMeasurementVariance` was passed to [fit], and also when
  /// `minimumMeasurementVariance` was passed and the free estimate came out
  /// below it, so the fit was redone with the noise held at the floor. False
  /// means [measurementVariance] is the maximum-likelihood estimate, floor or
  /// no floor.
  final bool measurementVariancePinned;

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
        if (parameterSpecs[i] is VarianceParameter) varianceRatios[i]
    ];
    if (variances.length != 1) {
      throw StateError('this model has ${variances.length} variance ratios, so '
          'there is no single one to report. Use varianceRatios, which is in '
          'parameter order.');
    }
    return variances.first;
  }

  /// The widest of [plateauDecadesByParameter]: how undetermined the least
  /// determined parameter is.
  ///
  /// A well-determined fit gives well under a decade. Several decades means
  /// the data does not distinguish a stiff curve from a flexible one, and the
  /// point estimate should be treated as a convention rather than a finding.
  double get plateauDecades {
    var widest = 0.0;
    for (final width in plateauDecadesByParameter) {
      if (width > widest) widest = width;
    }
    return widest;
  }

  /// True when the likelihood surface is too flat to support the estimate.
  bool get isFlat => plateauDecades > 2;

  /// The concentrated measurement variance, i.e. the fitted noise level.
  double get measurementVariance => model.measurementVariance;

  @override
  String toString() {
    final ratios =
        varianceRatios.map((r) => r.toStringAsPrecision(4)).join(', ');
    return 'FitResult(varianceRatios: [$ratios], measurementVariance: '
        '${measurementVariance.toStringAsPrecision(4)}, logLik: '
        '${logMarginalLikelihood.toStringAsFixed(3)}, evaluations: '
        '$evaluations, converged: $converged)';
  }
}
