import 'dart:math' as math;
import 'dart:typed_data';

import '../engine/fast_path_2x2.dart';
import '../engine/kalman.dart';
import '../engine/scale.dart';
import '../engine/timeline.dart';
import '../model.dart';
import '../observation.dart';
import '../parameter_spec.dart';
import 'penalty.dart';

/// The likelihood of a model as a function of its log variance ratios, with
/// the measurement variance concentrated out.
///
/// Rather than searching over `(sigma_1^2, ..., sigma_k^2, sigma_eps^2)`, the
/// filter is run at unit measurement variance and process variances
/// `exp(logRatios)`. The measurement variance that maximises the likelihood
/// for those ratios then has a closed form, so the search always has one
/// dimension fewer than the model has variances — a saving of exactly one, no
/// matter how many components there are. See
/// [FilterResult.profileMeasurementVariance] for why it works.
///
/// The timeline is built once and reused, so an evaluation is one forward pass
/// and nothing else.
///
/// With [fixedMeasurementVariance] set the concentrating step is skipped: the
/// filter runs at that noise level and the objective becomes the ordinary log
/// likelihood. The search still moves in log ratios, so the bracket means the
/// same thing either way and the search has the same number of dimensions.
class ProfileLikelihood {
  ProfileLikelihood(
    this.template,
    List<Observation> observations, {
    this.penalty = const NoPenalty(),
    this.fixedMeasurementVariance,
  })  : _timeline = Timeline.merge(observations, null),
        _specs = template.parameterSpecs,
        span = observations.isEmpty
            ? 0
            : observations.last.time - observations.first.time,
        _dataSpread = _standardDeviation(observations),
        _floor = scaleFloor(observations);

  /// The model whose parameters are being searched over. Its measurement
  /// variance is ignored.
  final StructuralModel template;

  /// What is added to the likelihood before [at] returns it.
  final Penalty penalty;

  /// The measurement variance to hold the filter at, or null to concentrate it
  /// out analytically.
  final double? fixedMeasurementVariance;

  /// Length of the series, which is what the penalty measures wandering over.
  final double span;

  final Timeline _timeline;

  /// Sample standard deviation of the observed values, which is the yardstick
  /// the penalty measures a component's wandering against.
  final double _dataSpread;

  /// The smallest measurement variance the concentrated likelihood is taken
  /// at. Data a model explains exactly would otherwise profile to zero and an
  /// unbounded likelihood; see [scaleFloor].
  final double _floor;

  /// Which entries of the parameter vector are log variances, and so have to
  /// be lifted from a ratio to an absolute value before the filter sees them.
  final List<ParameterSpec> _specs;

  int _evaluations = 0;
  Float64List? _cachedArgument;
  late FilterResult _cached;
  late StructuralModel _cachedModel;

  /// How many forward passes have been run.
  int get evaluations => _evaluations;

  /// The objective the search maximises: the profile log-likelihood plus the
  /// penalty.
  double at(Float64List logRatios) {
    evaluate(logRatios);
    return logLikelihoodOf(_cached) + _penaltyHere();
  }

  /// Whichever likelihood this instance is maximising: the profile one when
  /// the noise level is being concentrated out, the plain one when it is held.
  double logLikelihoodOf(FilterResult pass) {
    if (fixedMeasurementVariance != null) return pass.logLikelihood;
    final profiled = pass.profileMeasurementVariance;
    return profiled >= _floor
        ? pass.profileLogLikelihood
        : pass.logLikelihoodAtScale(_floor);
  }

  /// The measurement variance a pass implies, which is the fixed one if there
  /// is one and the concentrated estimate, no lower than a tiny floor,
  /// otherwise.
  double measurementVarianceOf(FilterResult pass) {
    final fixed = fixedMeasurementVariance;
    if (fixed != null) return fixed;
    final profiled = pass.profileMeasurementVariance;
    return profiled >= _floor ? profiled : _floor;
  }

  /// The profile log-likelihood alone, which is the number to report and to
  /// compare across models. A penalised objective is fine to maximise and
  /// meaningless to compare.
  double likelihoodAt(Float64List logRatios) =>
      logLikelihoodOf(evaluate(logRatios));

  /// The penalty alone at the same point.
  double penaltyAt(Float64List logRatios) {
    evaluate(logRatios);
    return _penaltyHere();
  }

  /// The penalty for the pass currently cached.
  ///
  /// The conversion factor is the measurement standard deviation this set of
  /// ratios implies, divided by the observed spread of the data. Both halves
  /// are needed: the filter reports wandering relative to the measurement
  /// noise, and the penalty is stated relative to the data.
  double _penaltyHere() {
    if (_dataSpread <= 0) return 0;
    final noise = math.sqrt(measurementVarianceOf(_cached));
    return penalty.at(_cachedModel.components, span, noise / _dataSpread);
  }

  /// The full forward pass at these ratios. Repeating the most recent
  /// argument is free, which is how the search can ask for the objective, the
  /// likelihood and the concentrated variance without paying three times.
  FilterResult evaluate(Float64List logRatios) {
    final cached = _cachedArgument;
    if (cached != null && _sameAs(cached, logRatios)) return _cached;

    final scale = fixedMeasurementVariance ?? 1.0;
    // The search coordinate is a ratio to the measurement variance in both
    // modes, so that the bracket a caller passes means one thing. Lifting the
    // ratios to absolute variances is what makes the filter agree: scaling
    // every variance in the model by the same constant is exactly the
    // equivariance the concentrating step relies on, and holding the noise
    // level fixed is the same model looked at from the other end.
    _cachedModel = template
        .withParameters(_lift(logRatios, scale))
        .withMeasurementVariance(scale);
    _cached = forwardPass(
      _cachedModel.components,
      _timeline,
      measurementVariance: scale,
      initialization: _cachedModel.initialization,
    );
    _cachedArgument = Float64List.fromList(logRatios);
    _evaluations++;
    return _cached;
  }

  /// Multiplies the variance entries of [logRatios] up by [scale], leaving the
  /// shape parameters — a length scale, a period — untouched.
  Float64List _lift(Float64List logRatios, double scale) {
    if (scale == 1) return logRatios;
    final shift = math.log(scale);
    final lifted = Float64List.fromList(logRatios);
    for (var i = 0; i < lifted.length; i++) {
      if (_specs[i] is VarianceParameter) lifted[i] += shift;
    }
    return lifted;
  }

  static bool _sameAs(Float64List a, Float64List b) {
    if (a.length != b.length) return false;
    for (var i = 0; i < a.length; i++) {
      if (a[i] != b[i]) return false;
    }
    return true;
  }
}

double _standardDeviation(List<Observation> observations) {
  if (observations.length < 2) return 0;
  var sum = 0.0;
  for (final o in observations) {
    sum += o.value;
  }
  final centre = sum / observations.length;
  var squares = 0.0;
  for (final o in observations) {
    final d = o.value - centre;
    squares += d * d;
  }
  return math.sqrt(squares / (observations.length - 1));
}
