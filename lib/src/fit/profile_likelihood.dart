import 'dart:typed_data';

import '../engine/kalman.dart';
import '../engine/timeline.dart';
import '../model.dart';
import '../observation.dart';

/// The likelihood of a one-parameter model as a function of the log variance
/// ratio, with the measurement variance concentrated out.
///
/// Rather than searching over `(sigma^2, sigma_eps^2)`, the filter is run at
/// unit measurement variance and process variance `q = exp(logRatio)`. The
/// measurement variance that maximises the likelihood for that `q` then has a
/// closed form, so the search is one-dimensional. See
/// [FilterResult.profileMeasurementVariance] for why that works.
///
/// The timeline is built once and reused, so an evaluation is one forward pass
/// and nothing else.
class ProfileLikelihood {
  ProfileLikelihood(this.template, List<Observation> observations)
      : _timeline = Timeline.merge(observations, null);

  /// The model whose parameters are being searched over. Its measurement
  /// variance is ignored.
  final StructuralModel template;

  final Timeline _timeline;

  int _evaluations = 0;
  double? _cachedArgument;
  late FilterResult _cached;

  /// How many forward passes have been run.
  int get evaluations => _evaluations;

  /// The profile log-likelihood at `q = exp(logRatio)`.
  double at(double logRatio) => evaluate(logRatio).profileLogLikelihood;

  /// The full forward pass at `q = exp(logRatio)`. Repeating the most recent
  /// argument is free, which is how the search can ask for the likelihood and
  /// the concentrated variance separately without paying twice.
  FilterResult evaluate(double logRatio) {
    if (_cachedArgument == logRatio) return _cached;
    final model = template
        .withParameters(Float64List.fromList([logRatio]))
        .withMeasurementVariance(1);
    final filter = KalmanFilter(
      model.components,
      measurementVariance: 1,
      diffuseVariance: model.diffuseVariance,
    );
    _cached = filter.run(_timeline);
    _cachedArgument = logRatio;
    _evaluations++;
    return _cached;
  }
}
