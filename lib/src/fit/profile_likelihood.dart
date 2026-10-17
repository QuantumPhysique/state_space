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

/// The likelihood of a model as a function of the coordinates a fit searches,
/// with the measurement variance concentrated out where it can be.
///
/// Rather than searching over `(sigma_1^2, ..., sigma_k^2, sigma_eps^2)`, the
/// filter is run at unit measurement variance and process variances
/// `exp(logRatios)`. The measurement variance that maximises the likelihood
/// for those ratios then has a closed form, so the search has one dimension
/// fewer than the model has variances. See
/// [FilterResult.profileMeasurementVariance].
///
/// The coordinates are the parameters listed in [free], in that order; the
/// others keep the values [template] gives them. A held variance is an
/// absolute value, which the closed form cannot keep, so with one the log
/// measurement variance becomes one more coordinate, the last; see
/// [searchesScale]. The free variances are searched as log ratios to it all
/// the same, so that the bracket a caller passes means one thing.
///
/// With [fixedMeasurementVariance] set the concentrating step is skipped: the
/// filter runs at that noise level and the objective becomes the ordinary log
/// likelihood.
///
/// The timeline is built once and reused, so an evaluation is one forward pass
/// and nothing else.
class ProfileLikelihood {
  /// The likelihood of [template]'s parameters on [observations], over every
  /// parameter not in [held].
  ProfileLikelihood(
    this.template,
    List<Observation> observations, {
    this.penalty = const NoPenalty(),
    this.fixedMeasurementVariance,
    Set<int> held = const {},
  }) : _timeline = Timeline.merge(observations, null),
       _specs = template.parameterSpecs,
       _start = template.parameters,
       free = List.unmodifiable([
         for (var i = 0; i < template.parameterCount; i++)
           if (!held.contains(i)) i,
       ]),
       searchesScale =
           fixedMeasurementVariance == null &&
           _holdsVariance(template.parameterSpecs, held),
       span = observations.isEmpty
           ? 0
           : observations.last.time - observations.first.time,
       _dataSpread = _standardDeviation(observations),
       _meanRelativeVariance = _meanRelative(observations),
       _floor = scaleFloor(observations);

  /// The model whose parameters are being searched over. Its measurement
  /// variance is ignored.
  final StructuralModel template;

  /// What is added to the likelihood before [at] returns it.
  final Penalty penalty;

  /// The measurement variance to hold the filter at, or null to concentrate it
  /// out analytically or search it.
  final double? fixedMeasurementVariance;

  /// The parameters searched over, by their index in [template]'s parameter
  /// vector, in the order of the coordinates.
  final List<int> free;

  /// Whether the log measurement variance is a coordinate, after those of
  /// [free]: when a held variance ties the model to an absolute scale and no
  /// noise level is fixed.
  final bool searchesScale;

  /// Length of the series, which is what the penalty measures wandering over.
  final double span;

  final Timeline _timeline;

  /// Sample standard deviation of the observed values, which is the yardstick
  /// the penalty measures a component's wandering against.
  final double _dataSpread;

  final double _meanRelativeVariance;

  /// The smallest measurement variance the concentrated likelihood is taken
  /// at. Data a model explains exactly would otherwise profile to zero and an
  /// unbounded likelihood; see [scaleFloor].
  final double _floor;

  /// Which entries of the parameter vector are log variances, and so have to
  /// be lifted from a ratio to an absolute value before the filter sees them.
  final List<ParameterSpec> _specs;

  /// [template]'s parameters, which the held ones keep.
  final Float64List _start;

  int _evaluations = 0;
  Float64List? _cachedArgument;
  late FilterResult _cached;
  late StructuralModel _cachedModel;

  /// How many forward passes have been run.
  int get evaluations => _evaluations;

  /// How many coordinates the search has.
  int get dimension => free.length + (searchesScale ? 1 : 0);

  /// The noise level that would explain the spread of the readings on its
  /// own, or the tiny floor when they have none: where a search over the
  /// noise level is centred.
  double get spreadVariance {
    final variance = _dataSpread * _dataSpread / _meanRelativeVariance;
    return variance > _floor ? variance : _floor;
  }

  /// The objective the search maximises: the log-likelihood plus the
  /// penalty.
  double at(Float64List argument) {
    evaluate(argument);
    return logLikelihoodOf(_cached) + _penaltyHere();
  }

  /// Whichever likelihood this instance is maximising: the profile one when
  /// the noise level is being concentrated out, the plain one when it is held
  /// or searched.
  double logLikelihoodOf(FilterResult pass) {
    if (fixedMeasurementVariance != null || searchesScale) {
      return pass.logLikelihood;
    }
    final profiled = pass.profileMeasurementVariance;
    return profiled >= _floor
        ? pass.profileLogLikelihood
        : pass.logLikelihoodAtScale(_floor);
  }

  /// The measurement variance a pass implies: the fixed or searched one the
  /// filter ran at, or otherwise the concentrated estimate, no lower than a
  /// tiny floor.
  double measurementVarianceOf(FilterResult pass) {
    if (fixedMeasurementVariance != null || searchesScale) {
      return pass.measurementVariance;
    }
    final profiled = pass.profileMeasurementVariance;
    return profiled >= _floor ? profiled : _floor;
  }

  /// The log-likelihood alone, which is the number to report and to compare
  /// across models. A penalised objective is fine to maximise and meaningless
  /// to compare.
  double likelihoodAt(Float64List argument) =>
      logLikelihoodOf(evaluate(argument));

  /// The penalty alone at the same point.
  double penaltyAt(Float64List argument) {
    evaluate(argument);
    return _penaltyHere();
  }

  /// [template]'s parameter vector at [argument], with the free variances
  /// lifted from their ratios to a noise level of [measurementVariance].
  Float64List parametersAt(Float64List argument, double measurementVariance) {
    final shift = math.log(measurementVariance);
    final theta = Float64List.fromList(_start);
    for (var axis = 0; axis < free.length; axis++) {
      final i = free[axis];
      theta[i] = _specs[i] is VarianceParameter
          ? argument[axis] + shift
          : argument[axis];
    }
    return theta;
  }

  /// The penalty for the pass currently cached.
  ///
  /// The conversion factor is the measurement standard deviation this set of
  /// ratios implies, divided by the observed spread of the data. Both halves
  /// are needed: the filter reports wandering relative to the noise level the
  /// pass ran at, and the penalty is stated relative to the data.
  double _penaltyHere() {
    if (_dataSpread <= 0) return 0;
    final noise = math.sqrt(
      measurementVarianceOf(_cached) / _cachedModel.measurementVariance,
    );
    return penalty.at(_cachedModel.components, span, noise / _dataSpread);
  }

  /// The full forward pass at these coordinates. Repeating the most recent
  /// argument is free, which is how the search can ask for the objective, the
  /// likelihood and the concentrated variance without paying three times.
  FilterResult evaluate(Float64List argument) {
    final cached = _cachedArgument;
    if (cached != null && _sameAs(cached, argument)) return _cached;

    // Concentrated, the filter runs at unit noise: scaling every variance in
    // the model by the same constant is exactly the equivariance the
    // concentrating step relies on.
    final scale =
        fixedMeasurementVariance ??
        (searchesScale ? math.exp(argument[free.length]) : 1.0);
    _cachedModel = template
        .withParameters(parametersAt(argument, scale))
        .withMeasurementVariance(scale);
    _cached = forwardPass(
      _cachedModel.components,
      _timeline,
      measurementVariance: scale,
      initialization: _cachedModel.initialization,
    );
    _cachedArgument = Float64List.fromList(argument);
    _evaluations++;
    return _cached;
  }

  static bool _holdsVariance(List<ParameterSpec> specs, Set<int> held) =>
      held.any((i) => specs[i] is VarianceParameter);

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

double _meanRelative(List<Observation> observations) {
  if (observations.isEmpty) return 1;
  var sum = 0.0;
  for (final o in observations) {
    sum += o.relativeVariance;
  }
  return sum > 0 ? sum / observations.length : 1;
}
