import 'dart:typed_data';

import 'component.dart';
import 'components/local_level.dart';
import 'components/local_linear_trend.dart';
import 'components/regression.dart';
import 'diagnostics.dart';
import 'engine/fast_path_2x2.dart';
import 'engine/kalman.dart';
import 'engine/rts.dart';
import 'engine/timeline.dart';
import 'initialization.dart';
import 'observation.dart';
import 'parameter_spec.dart';
import 'result.dart';

/// An additive structural time-series model: a list of components plus
/// observation noise.
///
/// ```text
/// y(t) = sum_i H_i(t) x_i(t) + eps(t),   eps ~ N(0, measurementVariance)
/// ```
///
/// The model is immutable and cheap to copy, so exploring a likelihood surface
/// means building models rather than mutating one.
class StructuralModel {
  StructuralModel(
    List<Component> components, {
    this.measurementVariance = 1.0,
    this.initialization = const ExactDiffuse(),
  }) : components = List.unmodifiable(components) {
    if (components.isEmpty) {
      throw ArgumentError.value(
          components, 'components', 'a model needs at least one component');
    }
    if (!(measurementVariance > 0) || !measurementVariance.isFinite) {
      throw ArgumentError.value(measurementVariance, 'measurementVariance',
          'must be finite and positive');
    }
  }

  /// A single [LocalLinearTrend]: a smooth curve whose slope wanders. The
  /// default choice for a trend through noisy readings.
  factory StructuralModel.localLinearTrend({
    required double processVariance,
    double measurementVariance = 1.0,
    Initialization initialization = const ExactDiffuse(),
  }) =>
      StructuralModel(
        [LocalLinearTrend(processVariance: processVariance)],
        measurementVariance: measurementVariance,
        initialization: initialization,
      );

  /// A single [LocalLevel]: a level with no persistent direction.
  factory StructuralModel.localLevel({
    required double processVariance,
    double measurementVariance = 1.0,
    Initialization initialization = const ExactDiffuse(),
  }) =>
      StructuralModel(
        [LocalLevel(processVariance: processVariance)],
        measurementVariance: measurementVariance,
        initialization: initialization,
      );

  /// The additive blocks of the model, in state order.
  final List<Component> components;

  /// Variance of the measurement noise for an observation of unit
  /// [Observation.relativeVariance].
  final double measurementVariance;

  /// How the prior on the first step's state is specified.
  ///
  /// Exact by default. [ApproximateDiffuse] remains available, and is the
  /// thing to reach for when the data cannot determine the flat directions
  /// and a very large number is more useful than an exception — a single
  /// observation under a two-state trend, for instance.
  final Initialization initialization;

  /// Total number of states across all components.
  int get stateDim => components.fold(0, (n, c) => n + c.stateDim);

  /// Total number of free parameters across all components.
  int get parameterCount => components.fold(0, (n, c) => n + c.parameterCount);

  /// What each entry of [parameters] is, concatenated in the same order.
  List<ParameterSpec> get parameterSpecs =>
      [for (final component in components) ...component.parameterSpecs];

  /// The concatenated unconstrained parameter vectors of every component.
  Float64List get parameters {
    final theta = Float64List(parameterCount);
    var at = 0;
    for (final component in components) {
      theta.setAll(at, component.parameters);
      at += component.parameterCount;
    }
    return theta;
  }

  /// A copy with [theta] distributed over the components in order.
  StructuralModel withParameters(Float64List theta) {
    if (theta.length != parameterCount) {
      throw ArgumentError.value(theta, 'theta',
          'expected $parameterCount parameters, got ${theta.length}');
    }
    final rebuilt = <Component>[];
    var at = 0;
    for (final component in components) {
      final slice =
          Float64List.sublistView(theta, at, at + component.parameterCount);
      rebuilt.add(component.withParameters(slice));
      at += component.parameterCount;
    }
    return StructuralModel(
      rebuilt,
      measurementVariance: measurementVariance,
      initialization: initialization,
    );
  }

  /// A copy with a different [measurementVariance].
  StructuralModel withMeasurementVariance(double variance) => StructuralModel(
        components,
        measurementVariance: variance,
        initialization: initialization,
      );

  /// Log marginal likelihood of [observations] under this model.
  ///
  /// One forward pass, no smoothing, nothing retained: `O(N)` time and `O(1)`
  /// memory beyond the input.
  double logLikelihood(List<Observation> observations) =>
      _filter(Timeline.merge(observations, null), keepHistory: false)
          .logLikelihood;

  /// What the one-step-ahead prediction errors say about this model on
  /// [observations].
  ///
  /// One forward pass, and it is a separate one: [smooth] does not compute
  /// residuals, because reconstructing them costs about as much again as the
  /// backward pass and most callers plotting a trend never look at them. Ask
  /// for them when you want to know whether the model deserves to be
  /// believed, which is usually once per model rather than once per redraw.
  InnovationDiagnostics diagnose(List<Observation> observations) {
    final result = forwardPass(
      components,
      Timeline.merge(observations, null),
      measurementVariance: measurementVariance,
      initialization: initialization,
      keepResiduals: true,
    );
    return InnovationDiagnostics(
      times: result.residualTimes!,
      residuals: result.standardisedResiduals!,
    );
  }

  /// Posterior of the states given [observations], reported at the observation
  /// times or, if [grid] is given, at the grid times.
  ///
  /// Grid points are simply time steps with no observation attached, so asking
  /// for output between measurements costs one more step in the same
  /// recursion — there is no interpolation anywhere, and no gap filling.
  ///
  /// [grid] must be sorted ascending. Where a grid time coincides exactly with
  /// an observation time, the reported state is the one that includes that
  /// observation.
  ///
  /// A grid may extend past the data at either end, and the posterior variance
  /// widens accordingly. Note that the prior is stated at the first step,
  /// whichever it turns out to be, so a grid point before the first
  /// observation moves where the diffuse prior sits — immaterial in the
  /// diffuse limit, and worth knowing when comparing runs digit by digit.
  SmoothingResult smooth(List<Observation> observations, {Float64List? grid}) {
    final timeline = Timeline.merge(observations, grid);
    final filtered = _filter(timeline, keepHistory: true);
    RtsSmoother(components)
      ..smoothInPlace(timeline, filtered)
      ..combineDiffuse(filtered);
    return _report(timeline, filtered);
  }

  /// Projects the signal past the last observation, at each time in
  /// [horizon].
  ///
  /// One forward pass and then a walk along the horizon, so `O(N + H)` time
  /// and no history kept. [horizon] must be sorted ascending and must not
  /// start before the last observation; for output *within* the data, ask
  /// [smooth] for a grid instead, which conditions on the whole series rather
  /// than only on the past.
  ForecastResult forecast(List<Observation> observations, Float64List horizon) {
    if (observations.isEmpty) {
      throw ArgumentError('nothing to forecast from: no observations');
    }
    final last = observations.last.time;
    for (var i = 0; i < horizon.length; i++) {
      if (!horizon[i].isFinite) {
        throw ArgumentError.value(horizon[i], 'horizon[$i]', 'not finite');
      }
      if (i > 0 && horizon[i] < horizon[i - 1]) {
        throw ArgumentError('horizon must be sorted ascending, but horizon[$i] '
            '(${horizon[i]}) precedes horizon[${i - 1}] (${horizon[i - 1]})');
      }
      if (horizon[i] < last) {
        throw ArgumentError('horizon[$i] (${horizon[i]}) is before the last '
            'observation at $last. Use smooth(observations, grid: ...) for '
            'times inside the data; it conditions on everything, not just on '
            'what came before.');
      }
    }

    final timeline = Timeline.merge(observations, null);
    final filter = KalmanFilter(
      components,
      measurementVariance: measurementVariance,
      initialization: initialization,
    );
    final result = filter.run(timeline);
    final projected = filter.project(last, horizon, result);

    return ForecastResult(
      times: Float64List.fromList(horizon),
      mean: projected.mean,
      variance: projected.variance,
      measurementVariance: measurementVariance,
    );
  }

  FilterResult _filter(Timeline timeline, {required bool keepHistory}) =>
      forwardPass(
        components,
        timeline,
        measurementVariance: measurementVariance,
        initialization: initialization,
        keepHistory: keepHistory,
      );

  /// Collapses the smoothed states down to the quantities callers actually
  /// plot: the signal, its variance, each component's share, and a slope if
  /// the model has one.
  SmoothingResult _report(Timeline timeline, FilterResult filtered) {
    final n = stateDim;
    final indices = timeline.outputIndices;
    final count = indices.length;
    final mean = filtered.filteredMean!;
    final covariance = filtered.filteredCovariance!;

    final offsets = <int>[];
    var next = 0;
    for (final component in components) {
      offsets.add(next);
      next += component.stateDim;
    }

    final h = Float64List(n);
    final slices = [
      for (var b = 0; b < components.length; b++)
        Float64List.sublistView(
            h, offsets[b], offsets[b] + components[b].stateDim)
    ];

    final times = Float64List(count);
    final level = Float64List(count);
    final levelVariance = Float64List(count);
    final componentMeans = [
      for (var b = 0; b < components.length; b++) Float64List(count)
    ];
    final componentVariances = [
      for (var b = 0; b < components.length; b++) Float64List(count)
    ];

    final rate = _rateStateIndex(offsets);
    final slope = rate == null ? null : Float64List(count);
    final slopeVariance = rate == null ? null : Float64List(count);

    for (var k = 0; k < count; k++) {
      final t = indices[k];
      times[k] = timeline.times[t];
      for (var b = 0; b < components.length; b++) {
        components[b].observationAt(times[k], slices[b]);
      }

      var signal = 0.0;
      for (var i = 0; i < n; i++) {
        signal += h[i] * mean[t * n + i];
      }
      level[k] = signal;

      var variance = 0.0;
      for (var i = 0; i < n; i++) {
        if (h[i] == 0) continue;
        var row = 0.0;
        for (var j = 0; j < n; j++) {
          row += covariance[t * n * n + i * n + j] * h[j];
        }
        variance += h[i] * row;
      }
      levelVariance[k] = variance;

      for (var b = 0; b < components.length; b++) {
        final start = offsets[b];
        final dim = components[b].stateDim;
        var contribution = 0.0;
        var spread = 0.0;
        for (var i = 0; i < dim; i++) {
          contribution += h[start + i] * mean[t * n + start + i];
          for (var j = 0; j < dim; j++) {
            spread += h[start + i] *
                covariance[t * n * n + (start + i) * n + start + j] *
                h[start + j];
          }
        }
        componentMeans[b][k] = contribution;
        componentVariances[b][k] = spread;
      }

      if (rate != null) {
        slope![k] = mean[t * n + rate];
        slopeVariance![k] = covariance[t * n * n + rate * n + rate];
      }
    }

    return SmoothingResult(
      coefficients: _coefficients(timeline, filtered, offsets),
      times: times,
      level: level,
      levelVariance: levelVariance,
      slope: slope,
      slopeVariance: slopeVariance,
      logMarginalLikelihood: filtered.logLikelihood,
      measurementVariance: measurementVariance,
      componentMeans: componentMeans,
      componentVariances: componentVariances,
    );
  }

  /// Reads the regression coefficients off the smoothed states.
  ///
  /// They are taken at the last step rather than the last output time, because
  /// a caller may ask for an empty grid and there would then be no output to
  /// read. It makes no difference otherwise: a state with `A = I` and `Q = 0`
  /// has the same full-data posterior at every step, which is the whole reason
  /// a single number is the right thing to report.
  List<Coefficient> _coefficients(
      Timeline timeline, FilterResult filtered, List<int> offsets) {
    final found = <Coefficient>[];
    final n = stateDim;
    final last = timeline.length - 1;
    if (last < 0) return const [];
    final mean = filtered.filteredMean!;
    final covariance = filtered.filteredCovariance!;

    for (var b = 0; b < components.length; b++) {
      final component = components[b];
      if (component is! RegressionComponent) continue;
      for (var i = 0; i < component.regressors.length; i++) {
        final at = offsets[b] + i;
        found.add(Coefficient(
          name: component.regressors[i].name,
          estimate: mean[last * n + at],
          variance: covariance[last * n * n + at * n + at],
        ));
      }
    }
    return found;
  }

  /// The single global state index holding a rate of change, if the model has
  /// exactly one. With two trends there is no unambiguous slope to report, so
  /// callers are sent to [SmoothingResult.componentMean] instead.
  int? _rateStateIndex(List<int> offsets) {
    int? found;
    for (var b = 0; b < components.length; b++) {
      final local = components[b].rateStateIndex;
      if (local == null) continue;
      if (found != null) return null;
      found = offsets[b] + local;
    }
    return found;
  }

  @override
  String toString() => 'StructuralModel($components, measurementVariance: '
      '$measurementVariance)';
}
