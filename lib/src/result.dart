import 'dart:math' as math;
import 'dart:typed_data';

import 'fit/fit.dart';
import 'model.dart';
import 'stats/normal.dart';

/// A central interval, in the units of the observations.
typedef Interval = ({double lo, double hi});

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
  })  : _componentMeans = componentMeans,
        _componentVariances = componentVariances;

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

/// What [StructuralModel] fitting returned, together with enough diagnostics
/// to tell a well-determined answer from a shrug.
class FitResult {
  /// Built by [fit].
  const FitResult({
    required this.model,
    required this.logMarginalLikelihood,
    required this.varianceRatio,
    required this.evaluations,
    required this.converged,
    required this.plateauDecades,
    required this.atBracketEdge,
  });

  /// The fitted model: components at their estimated variances, and the
  /// analytically concentrated measurement variance.
  final StructuralModel model;

  /// The maximised profile log-likelihood.
  final double logMarginalLikelihood;

  /// The estimated ratio of process to measurement variance. Its reciprocal is
  /// the smoothing-spline parameter `lambda`: small values give a stiff curve,
  /// large values a curve that chases the data.
  final double varianceRatio;

  /// Number of filter passes the search took.
  final int evaluations;

  /// Whether the search met its tolerance rather than running out of steps.
  final bool converged;

  /// Width, in decades of the variance ratio, of the region where the profile
  /// likelihood is within half a nat of its maximum.
  ///
  /// A well-determined fit gives well under a decade. Several decades means
  /// the data does not distinguish a stiff curve from a flexible one, and the
  /// point estimate should be treated as a convention rather than a finding.
  final double plateauDecades;

  /// Whether the maximum sits at the edge of the search bracket, in which case
  /// the true optimum is probably outside it.
  final bool atBracketEdge;

  /// True when the likelihood surface is too flat to support the estimate.
  bool get isFlat => plateauDecades > 2;

  /// The concentrated measurement variance, i.e. the fitted noise level.
  double get measurementVariance => model.measurementVariance;

  @override
  String toString() => 'FitResult(varianceRatio: '
      '${varianceRatio.toStringAsPrecision(4)}, measurementVariance: '
      '${measurementVariance.toStringAsPrecision(4)}, logLik: '
      '${logMarginalLikelihood.toStringAsFixed(3)}, evaluations: '
      '$evaluations, converged: $converged)';
}
