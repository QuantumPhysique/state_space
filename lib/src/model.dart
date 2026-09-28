import 'dart:math' as math;
import 'dart:typed_data';

import 'arguments.dart';
import 'component.dart';
import 'components/local_level.dart';
import 'components/local_linear_trend.dart';
import 'components/regression.dart';
import 'diagnostics.dart';
import 'engine/fast_path_2x2.dart';
import 'engine/kalman.dart';
import 'engine/layout.dart';
import 'engine/rts.dart';
import 'engine/scale.dart';
import 'engine/timeline.dart';
import 'exceptions.dart';
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
///
/// {@category Getting started}
/// {@category Choosing a model}
final class StructuralModel {
  /// A model whose signal is the sum of [components], in state order, observed
  /// with noise of variance [measurementVariance] per unit
  /// [Observation.relativeVariance].
  ///
  /// [components] must not be empty. Each component's [Component.parameters]
  /// must have [Component.parameterCount] entries and its
  /// [Component.diffuseStates] [Component.stateDim] entries; both are checked
  /// here, once, rather than failing inside the recursion.
  StructuralModel(
    List<Component> components, {
    this.measurementVariance = 1.0,
    this.initialization = const ExactDiffuse(),
  }) : components = List.unmodifiable(components) {
    if (components.isEmpty) {
      throw ArgumentError.value(
        components,
        'components',
        'a model needs at least one component',
      );
    }
    checkPositive(measurementVariance, 'measurementVariance');
    for (var i = 0; i < components.length; i++) {
      final c = components[i];
      if (c.parameters.length != c.parameterCount) {
        throw ArgumentError.value(
          c,
          'components[$i]',
          'parameters has ${c.parameters.length} entries but parameterCount '
              'is ${c.parameterCount}',
        );
      }
      if (c.diffuseStates.length != c.stateDim) {
        throw ArgumentError.value(
          c,
          'components[$i]',
          'diffuseStates has ${c.diffuseStates.length} entries but stateDim '
              'is ${c.stateDim}',
        );
      }
    }
  }

  /// A single [LocalLinearTrend]: a smooth curve whose slope wanders. The
  /// default choice for a trend through noisy readings.
  factory StructuralModel.localLinearTrend({
    required double processVariance,
    double measurementVariance = 1.0,
    Initialization initialization = const ExactDiffuse(),
  }) => StructuralModel(
    [LocalLinearTrend(processVariance: processVariance)],
    measurementVariance: measurementVariance,
    initialization: initialization,
  );

  /// A single [LocalLevel]: a level with no persistent direction.
  factory StructuralModel.localLevel({
    required double processVariance,
    double measurementVariance = 1.0,
    Initialization initialization = const ExactDiffuse(),
  }) => StructuralModel(
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
  /// Exact by default, in which case [smooth], [forecast], [logLikelihood]
  /// and [diagnose] throw [UnderdeterminedModelException] when the data cannot
  /// determine the flat directions. [ApproximateDiffuse] returns a very wide
  /// posterior instead; see its documentation for when that is sound.
  final Initialization initialization;

  /// Total number of states across all components.
  int get stateDim => components.fold(0, (n, c) => n + c.stateDim);

  /// How many flat directions the model integrates out, which is zero under
  /// [ApproximateDiffuse] and the number of diffuse states under
  /// [ExactDiffuse].
  ///
  /// Worth knowing because it is what decides whether two models'
  /// [SmoothingResult.logMarginalLikelihood] values are on the same scale. Two
  /// models with the same diffuse dimension have integrated the same thing
  /// away and can be compared; two with different dimensions cannot. See
  /// [logLikelihood].
  int get diffuseDimension => initialization is ExactDiffuse
      ? diffuseStateIndices(components).length
      : 0;

  /// Total number of free parameters across all components.
  int get parameterCount => components.fold(0, (n, c) => n + c.parameterCount);

  /// What each entry of [parameters] is, concatenated in the same order.
  List<ParameterSpec> get parameterSpecs => [
    for (final component in components) ...component.parameterSpecs,
  ];

  /// The same, narrowed by what a series sampled every [resolution] time units
  /// can resolve. See [Component.parameterSpecsAt].
  List<ParameterSpec> parameterSpecsAt({required double resolution}) => [
    for (final component in components)
      ...component.parameterSpecsAt(resolution: resolution),
  ];

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
      throw ArgumentError.value(
        theta,
        'theta',
        'expected $parameterCount parameters, got ${theta.length}',
      );
    }
    final rebuilt = <Component>[];
    var at = 0;
    for (final component in components) {
      final slice = Float64List.sublistView(
        theta,
        at,
        at + component.parameterCount,
      );
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

  /// A copy whose ratios of every variance to the measurement variance are
  /// this model's, and whose scale is estimated from [observations].
  ///
  /// This is the model to use when the smoothing is chosen rather than
  /// estimated, for instance from a user setting: the stiffness of the curve
  /// is fixed by the ratios, and the noise level is the restricted maximum
  /// likelihood estimate given them, in closed form from one forward pass.
  /// [fit] with the same model and a bracket of zero width would return the
  /// same number, after a search.
  ///
  /// [minimumMeasurementVariance], if given, is a floor on the estimate. On a
  /// short series the estimate comes from a handful of residuals and can be
  /// far too small; a floor at a realistic spread keeps the band honest.
  ///
  /// Only the variances are rescaled; a shape parameter such as a length
  /// scale or a period is kept as it is.
  ///
  /// Throws [UnderdeterminedModelException] when nothing is left to estimate
  /// a scale from: fewer observations than [diffuseDimension] plus one.
  StructuralModel withEstimatedScale(
    List<Observation> observations, {
    double? minimumMeasurementVariance,
  }) {
    final floor = minimumMeasurementVariance;
    if (floor != null) checkPositive(floor, 'minimumMeasurementVariance');
    final pass = _filter(
      Timeline.merge(observations, null),
      keepHistory: false,
    );
    if (!pass.hasResidualDegreesOfFreedom) {
      throw UnderdeterminedModelException(
        '${observations.length} '
        'observations leave nothing to estimate a noise level from once the '
        'model\'s flat directions are located',
      );
    }
    var variance = pass.profileMeasurementVariance;
    final tiny = scaleFloor(observations);
    if (!(variance > tiny)) variance = tiny;
    if (floor != null && variance < floor) variance = floor;
    final shift = math.log(variance / measurementVariance);
    final theta = parameters;
    final specs = parameterSpecs;
    for (var i = 0; i < theta.length; i++) {
      if (specs[i] is VarianceParameter) theta[i] += shift;
    }
    return withParameters(theta).withMeasurementVariance(variance);
  }

  /// Log marginal likelihood of [observations] under this model.
  ///
  /// One forward pass, no smoothing, nothing retained: `O(N)` time and `O(1)`
  /// memory beyond the input.
  ///
  /// Under [ExactDiffuse] this is the *restricted* likelihood: the flat
  /// directions have been integrated out against an improper prior. That makes
  /// it the right thing to maximise, and it makes it comparable only with
  /// models of the same [diffuseDimension]. Across different diffuse
  /// dimensions it is not on a common scale — and not merely by an unknown
  /// constant, but by one the caller controls without meaning to: writing a
  /// regression column in grams rather than kilograms shifts this number by
  /// `log 1000`, and so does changing the unit of [Observation.time]. Use the
  /// fitted noise level, an out-of-sample error, or [diagnose] to choose
  /// between models whose diffuse structure differs.
  ///
  /// Throws [UnderdeterminedModelException] when the data cannot determine the
  /// model's flat directions (see [initialization]), [ArgumentError] for
  /// unsorted or non-finite input.
  double logLikelihood(List<Observation> observations) => _filter(
    Timeline.merge(observations, null),
    keepHistory: false,
  ).logLikelihood;

  /// What the one-step-ahead prediction errors say about this model on
  /// [observations].
  ///
  /// One forward pass, separate from [smooth], which does not compute
  /// residuals.
  ///
  /// Throws [UnderdeterminedModelException] when the data cannot determine the
  /// model's flat directions (see [initialization]), [ArgumentError] for
  /// unsorted or non-finite input.
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
  /// Grid points are time steps with no observation attached, so output
  /// between measurements costs one more step in the same recursion; there is
  /// no interpolation and no gap filling.
  ///
  /// [grid] must be sorted ascending. Where a grid time coincides exactly with
  /// an observation time, the reported state is the one that includes that
  /// observation. A grid may extend past the data at either end, and the
  /// posterior variance widens accordingly.
  ///
  /// Throws [UnderdeterminedModelException] when the data cannot determine the
  /// model's flat directions (see [initialization]), [ArgumentError] for
  /// unsorted or non-finite input.
  SmoothingResult smooth(List<Observation> observations, {List<double>? grid}) {
    final timeline = Timeline.merge(observations, grid);
    final filtered = _filter(timeline, keepHistory: true);
    RtsSmoother(components, initialization: initialization)
      ..smoothInPlace(timeline, filtered)
      ..combineDiffuse(filtered, steps: _stepsWorthFolding(timeline));
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
  ///
  /// A horizon whose first entry *is* the last observation time reports the
  /// filtered state there: the estimate conditioned on everything up to and
  /// including that reading. That is the figure to show when it must not move
  /// once shown.
  ///
  /// Throws [UnderdeterminedModelException] when the data cannot determine the
  /// model's flat directions (see [initialization]), [ArgumentError] for
  /// unsorted or non-finite input.
  ForecastResult forecast(
    List<Observation> observations,
    List<double> horizon,
  ) {
    if (observations.isEmpty) {
      throw ArgumentError('nothing to forecast from: no observations');
    }
    final last = observations.last.time;
    for (var i = 0; i < horizon.length; i++) {
      if (!horizon[i].isFinite) {
        throw ArgumentError.value(horizon[i], 'horizon[$i]', 'not finite');
      }
      if (i > 0 && horizon[i] < horizon[i - 1]) {
        throw ArgumentError(
          'horizon must be sorted ascending, but horizon[$i] '
          '(${horizon[i]}) precedes horizon[${i - 1}] (${horizon[i - 1]})',
        );
      }
      if (horizon[i] < last) {
        throw ArgumentError(
          'horizon[$i] (${horizon[i]}) is before the last '
          'observation at $last. Use smooth(observations, grid: ...) for '
          'times inside the data; it conditions on everything, not just on '
          'what came before.',
        );
      }
    }

    final timeline = Timeline.merge(observations, null);
    final filter = KalmanFilter(
      components,
      measurementVariance: measurementVariance,
      initialization: initialization,
    );
    final result = filter.run(timeline);
    final times = Float64List.fromList(horizon);
    final projected = filter.project(last, times, result);

    return newForecastResult(
      times: times,
      mean: projected.mean,
      variance: projected.variance,
      measurementVariance: measurementVariance,
    );
  }

  /// The steps [_report] and [_coefficients] will read: the output grid, plus
  /// the last step, where the regression coefficients are taken from.
  ///
  /// Folding the flat directions into a step costs `O(stateDim^2 * d)`, so on
  /// a model with many regression columns it is worth doing only where the
  /// answer is wanted.
  static Int32List _stepsWorthFolding(Timeline timeline) {
    final outputs = timeline.outputIndices;
    final last = timeline.length - 1;
    if (last < 0) return outputs;
    if (outputs.isNotEmpty && outputs.last == last) return outputs;
    final wanted = Int32List(outputs.length + 1);
    wanted.setAll(0, outputs);
    wanted[outputs.length] = last;
    return wanted;
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
  /// plot: the signal, its variance, each component's share, and each
  /// component's rate where it has one.
  SmoothingResult _report(Timeline timeline, FilterResult filtered) {
    final n = stateDim;
    final indices = timeline.outputIndices;
    final count = indices.length;
    final mean = filtered.stateMean!;
    final covariance = filtered.stateCovariance!;

    final offsets = blockOffsets(components);

    final h = Float64List(n);
    final slices = [
      for (var b = 0; b < components.length; b++)
        Float64List.sublistView(
          h,
          offsets[b],
          offsets[b] + components[b].stateDim,
        ),
    ];

    final times = Float64List(count);
    final signal = Float64List(count);
    final signalVariance = Float64List(count);
    final componentMeans = [
      for (var b = 0; b < components.length; b++) Float64List(count),
    ];
    final componentVariances = [
      for (var b = 0; b < components.length; b++) Float64List(count),
    ];
    final rates = [
      for (final component in components) component.rateStateIndex,
    ];
    final slopes = [
      for (final rate in rates) rate == null ? null : Float64List(count),
    ];
    final slopeVariances = [
      for (final rate in rates) rate == null ? null : Float64List(count),
    ];

    for (var k = 0; k < count; k++) {
      final t = indices[k];
      final row = t * n;
      final block = t * n * n;
      times[k] = timeline.times[t];
      for (var b = 0; b < components.length; b++) {
        components[b].observationAt(times[k], slices[b]);
      }

      var total = 0.0;
      for (var i = 0; i < n; i++) {
        total += h[i] * mean[row + i];
      }
      signal[k] = total;

      var spread = 0.0;
      for (var i = 0; i < n; i++) {
        if (h[i] == 0) continue;
        var sum = 0.0;
        for (var j = 0; j < n; j++) {
          sum += covariance[block + i * n + j] * h[j];
        }
        spread += h[i] * sum;
      }
      signalVariance[k] = spread;

      for (var b = 0; b < components.length; b++) {
        final start = offsets[b];
        final dim = components[b].stateDim;
        var contribution = 0.0;
        var own = 0.0;
        for (var i = 0; i < dim; i++) {
          contribution += h[start + i] * mean[row + start + i];
          for (var j = 0; j < dim; j++) {
            own +=
                h[start + i] *
                covariance[block + (start + i) * n + start + j] *
                h[start + j];
          }
        }
        componentMeans[b][k] = contribution;
        componentVariances[b][k] = own;

        final rate = rates[b];
        if (rate != null) {
          final at = start + rate;
          slopes[b]![k] = mean[row + at];
          slopeVariances[b]![k] = covariance[block + at * n + at];
        }
      }
    }

    return newSmoothingResult(
      coefficients: _coefficients(timeline, filtered, offsets),
      times: times,
      mean: signal,
      variance: signalVariance,
      trendIndex: _trendIndex(),
      logMarginalLikelihood: filtered.logLikelihood,
      measurementVariance: measurementVariance,
      componentMeans: componentMeans,
      componentVariances: componentVariances,
      componentSlopes: slopes,
      componentSlopeVariances: slopeVariances,
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
    Timeline timeline,
    FilterResult filtered,
    List<int> offsets,
  ) {
    final found = <Coefficient>[];
    final n = stateDim;
    final last = timeline.length - 1;
    if (last < 0) return const [];
    final mean = filtered.stateMean!;
    final covariance = filtered.stateCovariance!;

    for (var b = 0; b < components.length; b++) {
      final component = components[b];
      if (component is! RegressionComponent) continue;
      for (var i = 0; i < component.regressors.length; i++) {
        final at = offsets[b] + i;
        found.add(
          newCoefficient(
            name: component.regressors[i].name,
            estimate: mean[last * n + at],
            variance: covariance[last * n * n + at * n + at],
          ),
        );
      }
    }
    return found;
  }

  /// The component [SmoothingResult.trendSlope] reads: the first
  /// non-stationary component with a rate state, or failing that the first
  /// with one at all.
  int? _trendIndex() {
    int? stationary;
    for (var b = 0; b < components.length; b++) {
      final component = components[b];
      if (component.rateStateIndex == null) continue;
      if (component.diffuseStates.contains(true)) return b;
      stationary ??= b;
    }
    return stationary;
  }

  @override
  bool operator ==(Object other) {
    if (other is! StructuralModel ||
        other.measurementVariance != measurementVariance ||
        other.initialization != initialization ||
        other.components.length != components.length) {
      return false;
    }
    for (var i = 0; i < components.length; i++) {
      if (other.components[i] != components[i]) return false;
    }
    return true;
  }

  @override
  int get hashCode => Object.hash(
    measurementVariance,
    initialization,
    Object.hashAll(components),
  );

  @override
  String toString() =>
      'StructuralModel($components, measurementVariance: '
      '$measurementVariance)';
}
