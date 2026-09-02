import 'dart:math' as math;
import 'dart:typed_data';

import 'package:state_space/src/engine/kalman.dart';
import 'package:state_space/src/engine/rts.dart';
import 'package:state_space/src/engine/timeline.dart';
import 'package:state_space/state_space.dart';
import 'package:test/test.dart';

/// A regression component that declines to admit it is static.
///
/// Identical in every other respect, so smoothing a model built from it runs
/// the full `n x n` backward pass over states the reduced path skips. That
/// makes it the reference: if the two disagree, the reduction is wrong.
class _Opaque extends Component {
  _Opaque(this.inner);

  final RegressionComponent inner;

  @override
  bool get isStatic => false;

  @override
  int get stateDim => inner.stateDim;
  @override
  int get parameterCount => inner.parameterCount;
  @override
  void transition(double dt, MatrixBlock out) => inner.transition(dt, out);
  @override
  void processNoise(double dt, MatrixBlock out) => inner.processNoise(dt, out);
  @override
  void observationAt(double time, Float64List out) =>
      inner.observationAt(time, out);
  @override
  List<bool> get diffuseStates => inner.diffuseStates;
  @override
  void properPrior(Float64List mean, MatrixBlock covariance) =>
      inner.properPrior(mean, covariance);
  @override
  Float64List get parameters => inner.parameters;
  @override
  Component withParameters(Float64List theta) => this;
  @override
  double wanderOver(double span) => 0;
}

void main() {
  List<Observation> diary(int n, {int seed = 5}) {
    final random = math.Random(seed);
    double gaussian() =>
        math.sqrt(-2 * math.log(random.nextDouble())) *
        math.cos(2 * math.pi * random.nextDouble());
    return [
      for (var i = 0; i < n; i++)
        if (i % 9 != 4) // a missing morning here and there
          Observation(
              i.toDouble(),
              80 -
                  0.004 * i +
                  (i >= 40 && i < 54 ? 1.3 : 0.0) +
                  (i >= 120 && i < 127 ? -0.5 : 0.0) +
                  0.3 * gaussian(),
              relativeVariance: i % 17 == 0 ? 0.5 : 1.0)
    ];
  }

  RegressionComponent events() => RegressionComponent([
        IndicatorRegressor('christmas', [(from: 40.0, to: 54.0)]),
        IndicatorRegressor('conference', [(from: 120.0, to: 127.0)]),
        StepRegressor('dose', Float64List.fromList([30, 90]),
            Float64List.fromList([1, 0.25])),
      ]);

  /// Smooths with the backward pass forced to run over every state.
  SmoothingResult unreduced(
      List<Component> components, List<Observation> data, Float64List? grid) {
    final timeline = Timeline.merge(data, grid);
    final filter = KalmanFilter(components,
        measurementVariance: 0.09, initialization: const ExactDiffuse());
    final result = filter.run(timeline, keepHistory: true);
    RtsSmoother(components)
      ..smoothInPlace(timeline, result)
      ..combineDiffuse(result);
    // Read the same quantities StructuralModel._report would.
    final n = components.fold(0, (int a, c) => a + c.stateDim);
    final h = Float64List(n);
    final times = Float64List(timeline.outputIndices.length);
    final level = Float64List(timeline.outputIndices.length);
    final variance = Float64List(timeline.outputIndices.length);
    for (var k = 0; k < timeline.outputIndices.length; k++) {
      final t = timeline.outputIndices[k];
      times[k] = timeline.times[t];
      var at = 0;
      for (final c in components) {
        c.observationAt(timeline.times[t],
            Float64List.sublistView(h, at, at + c.stateDim));
        at += c.stateDim;
      }
      var signal = 0.0, spread = 0.0;
      for (var i = 0; i < n; i++) {
        signal += h[i] * result.filteredMean![t * n + i];
        for (var j = 0; j < n; j++) {
          spread += h[i] * result.filteredCovariance![t * n * n + i * n + j] *
              h[j];
        }
      }
      level[k] = signal;
      variance[k] = spread;
    }
    return SmoothingResult(
      times: times,
      level: level,
      levelVariance: variance,
      slope: null,
      slopeVariance: null,
      logMarginalLikelihood: result.logLikelihood,
      measurementVariance: 0.09,
      componentMeans: const [],
      componentVariances: const [],
    );
  }

  group('skipping static states changes nothing about the answer', () {
    final data = diary(200);

    for (final (name, grid) in [
      ('at the observation times', null),
      (
        'on a coarse grid',
        Float64List.fromList([for (var d = 0; d <= 210; d += 7) d.toDouble()])
      ),
      (
        'on a grid running past both ends',
        Float64List.fromList([for (var d = -10; d <= 220; d += 3) d.toDouble()])
      ),
    ]) {
      test(name, () {
        final reduced = StructuralModel(
          [const LocalLinearTrend(processVariance: 1e-4), events()],
          measurementVariance: 0.09,
        ).smooth(data, grid: grid);
        final reference = unreduced(
            [const LocalLinearTrend(processVariance: 1e-4), _Opaque(events())],
            data,
            grid);

        // Relative, because the reference is the *less* accurate of the two:
        // its predicted covariance is structurally singular, so every one of
        // its steps goes through the jittered factorisation the reduced path
        // never reaches.
        // Nine significant figures, with a floor for quantities near zero.
        double tolerance(double value) => 1e-9 * value.abs() + 1e-12;

        expect(reduced.length, reference.length);
        for (var i = 0; i < reduced.length; i++) {
          expect(reduced.level[i],
              closeTo(reference.level[i], tolerance(reference.level[i])),
              reason: 'level at output $i');
          expect(
              reduced.levelVariance[i],
              closeTo(reference.levelVariance[i],
                  tolerance(reference.levelVariance[i])),
              reason: 'variance at output $i');
        }
        expect(
            reduced.logMarginalLikelihood,
            closeTo(reference.logMarginalLikelihood,
                tolerance(reference.logMarginalLikelihood)));
      });
    }

    test('including the coefficients and their standard errors', () {
      final model = StructuralModel(
        [const LocalLinearTrend(processVariance: 1e-4), events()],
        measurementVariance: 0.09,
      );
      final coefficients = model.smooth(data).coefficients;
      expect(coefficients, hasLength(3));
      expect(coefficients[0].estimate, closeTo(1.3, 0.2));
      expect(coefficients[1].estimate, closeTo(-0.5, 0.3));
      for (final c in coefficients) {
        expect(c.standardError, greaterThan(0));
        expect(c.standardError.isFinite, isTrue);
      }
    });

    test('a model of nothing but regression columns still works', () {
      final flat = [
        for (var i = 0; i < 60; i++)
          Observation(i.toDouble(), i >= 20 && i < 40 ? 3.0 : 1.0)
      ];
      final posterior = StructuralModel(
        [
          RegressionComponent([
            IndicatorRegressor('always', [(from: -1.0, to: 100.0)]),
            IndicatorRegressor('middle', [(from: 20.0, to: 40.0)]),
          ])
        ],
        measurementVariance: 1e-6,
      ).smooth(flat);
      expect(posterior.coefficients[0].estimate, closeTo(1.0, 1e-6));
      expect(posterior.coefficients[1].estimate, closeTo(2.0, 1e-6));
      for (var i = 0; i < posterior.length; i++) {
        expect(posterior.level[i], closeTo(flat[i].value, 1e-6));
      }
    });

    test('an approximate prior keeps every state in the backward pass', () {
      // Nothing is dropped there, because a wide proper prior gives a
      // coefficient real variance and a real gain.
      final model = StructuralModel(
        [const LocalLinearTrend(processVariance: 1e-4), events()],
        measurementVariance: 0.09,
        initialization: const ApproximateDiffuse(variance: 1e8),
      );
      final posterior = model.smooth(data);
      final exact = StructuralModel(
        [const LocalLinearTrend(processVariance: 1e-4), events()],
        measurementVariance: 0.09,
      ).smooth(data);
      for (var i = 0; i < posterior.length; i++) {
        expect(posterior.level[i], closeTo(exact.level[i], 1e-4));
      }
    });
  });
}
