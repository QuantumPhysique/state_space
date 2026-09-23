import 'dart:math' as math;
import 'dart:typed_data';

import 'package:state_space/state_space.dart';
import 'package:test/test.dart';

/// A slowly bending series, at integer days or at a random time of day.
List<Observation> _series(int n, int seed, {bool jitter = false}) {
  final random = math.Random(seed);
  var level = 80.0, slope = 0.0;
  return [
    for (var i = 0; i < n; i++)
      () {
        slope += 0.01 * (random.nextDouble() - 0.5);
        level += slope + 0.05 * (random.nextDouble() - 0.5);
        final time = i + (jitter ? random.nextDouble() : 0.0);
        return Observation(time, level + 0.1 * (random.nextDouble() - 0.5));
      }()
  ];
}

final _refused = throwsA(isA<UnderdeterminedModelException>());

void main() {
  group('a model the data cannot determine is refused', () {
    test('trend plus level, at integer days and at random times of day', () {
      final model = StructuralModel([
        LocalLinearTrend(processVariance: 1e-4),
        LocalLevel(processVariance: 1e-3),
      ], measurementVariance: 0.01);
      for (var seed = 0; seed < 20; seed++) {
        for (final jitter in [false, true]) {
          final data = _series(90, seed, jitter: jitter);
          expect(() => model.smooth(data), _refused);
          expect(() => model.logLikelihood(data), _refused);
        }
      }
    });

    test('the same indicator entered twice', () {
      final model = StructuralModel([
        LocalLinearTrend(processVariance: 1e-4),
        RegressionComponent([
          IndicatorRegressor('trip', [(from: 10.0, to: 20.0)]),
          IndicatorRegressor('holiday', [(from: 10.0, to: 20.0)]),
        ]),
      ], measurementVariance: 0.01);
      expect(() => model.smooth(_series(90, 1)), _refused);
    });

    test('a step that switches on before the first reading', () {
      final model = StructuralModel([
        LocalLinearTrend(processVariance: 1e-4),
        RegressionComponent([
          StepRegressor(
              'dose', Float64List.fromList([-5]), Float64List.fromList([1])),
        ]),
      ], measurementVariance: 0.01);
      for (var seed = 0; seed < 10; seed++) {
        expect(() => model.smooth(_series(90, seed, jitter: true)), _refused);
      }
    });

    test('one reading off the output grid, whatever the noise level', () {
      final grid = Float64List.fromList([0, 1, 2, 3, 4]);
      for (final noise in [0.04, 0.25, 1.0, 4.0]) {
        final model = StructuralModel.localLinearTrend(
            processVariance: 1, measurementVariance: noise);
        for (final time in [0.0, 0.3, 1.3, 2.7, 4.0]) {
          expect(() => model.smooth([Observation(time, 80)], grid: grid),
              _refused);
        }
      }
    });
  });

  group('an ill-conditioned but determined model is not refused', () {
    test('an annual seasonal on a fraction of a year', () {
      final model = StructuralModel([
        LocalLinearTrend(processVariance: 1e-4),
        TrigonometricSeasonal(
            period: 365.25, harmonics: 2, processVariance: 1e-9),
      ]);
      expect(model.smooth(_series(60, 3)).mean, hasLength(60));
    });

    test('time in milliseconds since the epoch', () {
      const day = 86400000.0;
      final data = [
        for (final o in _series(365, 3))
          Observation(1.7e12 + o.time * day, o.value)
      ];
      final model = StructuralModel.localLinearTrend(
          processVariance: 1e-4 / (day * day * day), measurementVariance: 0.01);
      expect(model.smooth(data).variance.every((v) => v > 0), isTrue);
    });
  });

  group('an exact reading', () {
    test('first under a flat prior is refused with the reason', () {
      final data = [
        const Observation(0, 80, relativeVariance: 0),
        const Observation(1, 80.2),
        const Observation(2, 80.1),
      ];
      expect(
          () => StructuralModel.localLevel(processVariance: 0.01).smooth(data),
          throwsA(isA<NumericalBreakdownException>().having(
              (e) => e.message, 'message', contains('relativeVariance 0'))));
    });

    test('whose variance overflows says so', () {
      final data = [
        const Observation(0, 80),
        const Observation(1, 80.2, relativeVariance: 1e308),
        const Observation(2, 80.1),
      ];
      expect(
          () => StructuralModel.localLevel(
                  processVariance: 0.01, measurementVariance: 10)
              .smooth(data),
          throwsA(isA<NumericalBreakdownException>().having(
              (e) => e.message, 'message', contains('not a finite number'))));
    });
  });
}
