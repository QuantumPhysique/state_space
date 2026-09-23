import 'dart:math' as math;

import 'package:state_space/state_space.dart';
import 'package:test/test.dart';

import '../tool/calibration/synthetic.dart';

final _today = DateTime(2026, 9, 17);

double _gaussian(math.Random random) =>
    math.sqrt(-2 * math.log(1 - random.nextDouble())) *
    math.cos(2 * math.pi * random.nextDouble());

StructuralModel _trend() =>
    StructuralModel.localLinearTrend(processVariance: 1e-3);

StructuralModel _trendWeekly() => StructuralModel([
      LocalLinearTrend(processVariance: 1e-3),
      TrigonometricSeasonal(period: 7, harmonics: 2, processVariance: 1e-5),
    ]);

void main() {
  group('a Matern beside the trend', () {
    test('does not take over the noise on the calibration diaries', () {
      for (final series in syntheticSeries(today: _today)) {
        for (final weekly in [false, true]) {
          final model = StructuralModel([
            LocalLinearTrend(processVariance: 1e-3),
            if (weekly)
              TrigonometricSeasonal(
                  period: 7, harmonics: 2, processVariance: 1e-5),
            Matern.oneHalf(variance: 0.05, lengthScale: 3),
          ]);
          final free = fit(model, series.observations);
          // Starting from a realistic noise level and letting it go.
          final pinned =
              fit(model, series.observations, fixedMeasurementVariance: 0.01);
          final released = fit(pinned.model, series.observations,
              start: SearchStart.previousParameters);
          expect(free.logMarginalLikelihood,
              greaterThanOrEqualTo(released.logMarginalLikelihood - 0.01),
              reason:
                  '${series.name}, weekly $weekly: a better optimum exists');
          expect(math.sqrt(free.measurementVariance), greaterThan(0.02),
              reason: '${series.name}, weekly $weekly');
        }
      }
    });
  });

  group('a mistyped reading', () {
    List<Observation> diary(int seed) {
      final random = math.Random(seed);
      var level = 80.0;
      return [
        for (var d = 0; d < 200; d++)
          () {
            level += 0.02 * _gaussian(random);
            return Observation(d.toDouble(),
                ((level + 0.25 * _gaussian(random)) * 10).round() / 10);
          }()
      ];
    }

    test('is named in the warnings, ten times too large or too small', () {
      for (final seed in [1, 2, 3]) {
        final clean = diary(seed);
        for (final factor in [10.0, 0.1]) {
          final typo = [
            for (final o in clean)
              o.time == 150 ? Observation(150, o.value * factor) : o
          ];
          for (final model in [_trend(), _trendWeekly()]) {
            final fitted = fit(model, typo);
            expect(fitted.largestResidual!.time, 150);
            expect(fitted.warnings,
                contains(contains('the reading at time 150.0')));
          }
        }
      }
    });

    test('is not invented on clean data', () {
      for (var seed = 1; seed <= 20; seed++) {
        final fitted = fit(_trendWeekly(), diary(seed));
        expect(fitted.largestResidual!.score.abs(),
            lessThan(FitResult.outlierScore),
            reason: 'seed $seed');
      }
    });
  });

  group('data a model explains exactly', () {
    test('fits instead of throwing, floor or no floor', () {
      for (final value in [80.0, 81.3, 72.45]) {
        for (final offset in [0.0, 0.3]) {
          for (final n in [5, 10, 20]) {
            final flat = [
              for (var i = 0; i < n; i++) Observation(i + offset, value)
            ];
            for (final model in [_trend(), if (n >= 10) _trendWeekly()]) {
              expect(fit(model, flat).measurementVariance, greaterThan(0));
              expect(
                  fit(model, flat, minimumMeasurementVariance: 0.01)
                      .measurementVariance,
                  0.01);
            }
          }
        }
      }
      final line = [
        const Observation(0, 80.0),
        const Observation(1, 80.1),
        const Observation(2, 80.2),
      ];
      expect(fit(_trend(), line).measurementVariance, greaterThan(0));
    });
  });

  test('a short series with a rich seasonal never blames a caller argument',
      () {
    // Nine rounded readings and three weekly harmonics: the search can wander
    // into a corner of the surface where the likelihood is not a number.
    for (var seed = 0; seed < 40; seed++) {
      final random = math.Random(seed);
      final data = [
        for (var i = 0; i < 9; i++)
          Observation(
              i.toDouble(), ((80 + 0.3 * _gaussian(random)) * 10).round() / 10)
      ];
      final model = StructuralModel([
        LocalLinearTrend(processVariance: 1e-2),
        TrigonometricSeasonal(period: 7, harmonics: 3, processVariance: 1e-3),
      ]);
      try {
        fit(model, data);
      } on UnderdeterminedModelException {
        // An honest refusal is fine; an ArgumentError about processVariance
        // is not.
      }
    }
  });

  test('too few observations is one exception, whatever the count', () {
    for (final n in [0, 1, 2]) {
      expect(
          () => fit(_trend(),
              [for (var i = 0; i < n; i++) Observation(i.toDouble(), 80)]),
          throwsA(isA<UnderdeterminedModelException>()),
          reason: '$n observations');
    }
  });

  test('a warm start follows the optimum as far as the data moved it', () {
    for (final series in syntheticSeries(today: _today)) {
      for (final (from, to) in [(90, 97), (90, 120), (120, 186), (60, 150)]) {
        final before = series.firstDays(from).observations;
        final after = series.firstDays(to).observations;
        final cold = fit(_trend(), after);
        final warm = fit(fit(_trend(), before).model, after,
            start: SearchStart.previousParameters);
        expect(warm.logMarginalLikelihood,
            closeTo(cold.logMarginalLikelihood, 1e-3),
            reason: '${series.name}, $from to $to days');
      }
    }
  });

  test('a seasonal fitted as fixed is not reported as doing nothing', () {
    var shrunk = 0;
    for (var seed = 1; seed < 10; seed++) {
      final random = math.Random(seed);
      var level = 80.0;
      final data = [
        for (var d = 0; d < 2 * 365; d++)
          () {
            level += 0.02 * _gaussian(random);
            return Observation(
                d.toDouble(),
                level +
                    0.2 * math.sin(2 * math.pi * d / 7) +
                    0.2 * _gaussian(random));
          }()
      ];
      final fitted = fit(_trendWeekly(), data);
      if (fitted.parameterStatus[1] != ParameterStatus.shrunkToNothing) {
        continue;
      }
      shrunk++;
      expect(fitted.warnings, contains(contains('fitted as fixed')));
      expect(fitted.warnings, isNot(contains(contains('nothing to'))));
    }
    expect(shrunk, greaterThan(0), reason: 'the case this test is about');
  });
}
