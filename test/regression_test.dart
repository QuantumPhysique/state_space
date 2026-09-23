import 'dart:math' as math;
import 'dart:typed_data';

import 'package:state_space/authoring.dart';
import 'package:test/test.dart';

double _gaussian(math.Random random) =>
    math.sqrt(-2 * math.log(1 - random.nextDouble())) *
    math.cos(2 * math.pi * random.nextDouble());

/// Daily readings with a slow drift, plus two events that shift the level
/// while they are running: a fortnight in December and a week in March.
///
/// The indicator is written out here by hand rather than read off the
/// regressor, so that the test compares the model against the data-generating
/// process instead of against itself.
List<Observation> _withEvents({
  required double holidayEffect,
  required double conferenceEffect,
  int seed = 3,
  int days = 400,
  double noise = 0.25,
}) {
  final random = math.Random(seed);
  return [
    for (var day = 0; day < days; day++)
      Observation(
        day.toDouble(),
        70 -
            0.002 * day +
            (day >= 80 && day < 87 ? conferenceEffect : 0) +
            (day >= 350 && day < 364 ? holidayEffect : 0) +
            noise * _gaussian(random),
      ),
  ];
}

RegressionComponent _events() => RegressionComponent([
  IndicatorRegressor('conference', [(from: 80, to: 87)]),
  IndicatorRegressor('holiday', [(from: 350, to: 364)]),
]);

void main() {
  group('an indicator regressor', () {
    final regressor = IndicatorRegressor('holiday', [
      (from: 10, to: 14),
      (from: 100, to: 107),
    ]);

    test('is one inside a span and zero outside it', () {
      expect(regressor.at(9.99), 0);
      expect(regressor.at(10), 1);
      expect(regressor.at(13.99), 1);
      // Half open, so the closing instant is already outside.
      expect(regressor.at(14), 0);
      expect(regressor.at(50), 0);
      expect(regressor.at(103), 1);
      expect(regressor.at(107), 0);
      expect(regressor.at(1e6), 0);
      expect(regressor.at(-1e6), 0);
    });

    test('refuses spans that are out of order or overlapping', () {
      expect(
        () => IndicatorRegressor('x', [(from: 5, to: 9), (from: 3, to: 4)]),
        throwsArgumentError,
      );
      expect(
        () => IndicatorRegressor('x', [(from: 0, to: 9), (from: 3, to: 12)]),
        throwsArgumentError,
      );
      expect(
        () => IndicatorRegressor('x', [(from: 5, to: 5)]),
        throwsArgumentError,
      );
    });

    test('knows when it has nothing to say about a stretch of time', () {
      expect(regressor.isSilentOver(0, 5), isTrue);
      expect(regressor.isSilentOver(20, 90), isTrue);
      expect(regressor.isSilentOver(0, 11), isFalse);
      expect(regressor.isSilentOver(106, 200), isFalse);
    });
  });

  group('a step regressor', () {
    final regressor = StepRegressor(
      'dose',
      Float64List.fromList([0, 30, 60]),
      Float64List.fromList([1, 2.5, 0]),
      before: -1,
    );

    test('holds each value forward until the next knot', () {
      expect(regressor.at(-5), -1);
      expect(regressor.at(0), 1);
      expect(regressor.at(29.9), 1);
      expect(regressor.at(30), 2.5);
      expect(regressor.at(59.9), 2.5);
      expect(regressor.at(60), 0);
      expect(regressor.at(1e6), 0);
    });

    test('refuses mismatched or unsorted knots', () {
      expect(
        () => StepRegressor(
          'x',
          Float64List.fromList([0, 1]),
          Float64List.fromList([1]),
        ),
        throwsArgumentError,
      );
      expect(
        () => StepRegressor(
          'x',
          Float64List.fromList([5, 1]),
          Float64List.fromList([1, 2]),
        ),
        throwsArgumentError,
      );
    });
  });

  group('the regression component', () {
    final component = RegressionComponent([
      IndicatorRegressor('holiday', [(from: 10, to: 14)]),
      IndicatorRegressor('trip', [(from: 40, to: 50)]),
    ]);

    test('holds its coefficients still', () {
      final a = MatrixBlock.dense(2, 2);
      final q = MatrixBlock.dense(2, 2);
      component
        ..transition(7.5, a)
        ..processNoise(7.5, q);
      expect([a.at(0, 0), a.at(0, 1), a.at(1, 0), a.at(1, 1)], [1, 0, 0, 1]);
      expect([q.at(0, 0), q.at(0, 1), q.at(1, 0), q.at(1, 1)], [0, 0, 0, 0]);
      expect(component.wanderOver(1000), 0);
    });

    test('costs the optimiser nothing', () {
      // The whole point: coefficients are states, not parameters. Twenty
      // holiday indicators still leave a one-dimensional search.
      expect(component.parameterCount, 0);
      expect(component.parameters, isEmpty);
      expect(
        StructuralModel([
          LocalLinearTrend(processVariance: 1e-3),
          component,
        ]).parameterCount,
        1,
      );
    });

    test('reads the design at the time it is asked about', () {
      final row = Float64List(2);
      component.observationAt(12, row);
      expect(row, [1, 0]);
      component.observationAt(45, row);
      expect(row, [0, 1]);
      component.observationAt(0, row);
      expect(row, [0, 0]);
    });

    test('needs at least one column', () {
      expect(() => RegressionComponent([]), throwsArgumentError);
    });
  });

  group('recovering what an event cost', () {
    test('gets the effect and an honest error bar', () {
      const holiday = 1.2;
      const conference = -0.4;
      final data = _withEvents(
        holidayEffect: holiday,
        conferenceEffect: conference,
      );

      final fitted = fit(
        StructuralModel([LocalLinearTrend(processVariance: 1e-4), _events()]),
        data,
      );
      // One free variance, not three: the two coefficients are states.
      expect(fitted.varianceRatios.length, 1);
      expect(fitted.varianceRatio, isPositive);

      final posterior = fitted.model.smooth(data);
      expect(posterior.coefficients.map((c) => c.name), [
        'conference',
        'holiday',
      ]);

      final estimates = {for (final c in posterior.coefficients) c.name: c};
      for (final (name, truth) in [
        ('holiday', holiday),
        ('conference', conference),
      ]) {
        final coefficient = estimates[name]!;
        expect(
          (coefficient.estimate - truth).abs(),
          lessThan(2.5 * coefficient.standardError),
          reason: '$name: $coefficient against a true $truth',
        );
        final interval = coefficient.interval();
        expect(interval.lo, lessThan(truth));
        expect(interval.hi, greaterThan(truth));
      }

      // The error bar should be about the noise divided by the root of the
      // number of days the event was running, give or take what the trend
      // costs: 0.25 / sqrt(14) is 0.067 for the fortnight, 0.25 / sqrt(7) is
      // 0.094 for the week.
      expect(estimates['holiday']!.standardError, lessThan(0.2));
      expect(
        estimates['conference']!.standardError,
        greaterThan(estimates['holiday']!.standardError),
      );
    });

    test('and covers the truth about as often as it claims to', () {
      // Twenty replications of a nominal 95% interval. Seeing it miss five
      // times would be a bad sign; seeing it miss none would suggest the
      // intervals are too wide to mean anything.
      const holiday = 1.2;
      var covered = 0;
      for (var seed = 1; seed <= 20; seed++) {
        final data = _withEvents(
          holidayEffect: holiday,
          conferenceEffect: -0.4,
          seed: seed,
        );
        final fitted = fit(
          StructuralModel([LocalLinearTrend(processVariance: 1e-4), _events()]),
          data,
        );
        final coefficient = fitted.model
            .smooth(data)
            .coefficients
            .firstWhere((c) => c.name == 'holiday');
        final interval = coefficient.interval();
        if (interval.lo <= holiday && holiday <= interval.hi) covered++;
      }
      expect(covered, greaterThanOrEqualTo(16));
    });

    test('says which regressor never fired, rather than going singular '
        'quietly', () {
      final data = _withEvents(
        holidayEffect: 1.2,
        conferenceEffect: -0.4,
        days: 200,
      );
      expect(
        () => StructuralModel([
          LocalLinearTrend(processVariance: 1e-4),
          _events(),
        ]).smooth(data),
        throwsA(
          isA<UnderdeterminedModelException>()
              .having((e) => e.message, 'message', contains('"holiday"'))
              .having((e) => e.message, 'message', contains('zero everywhere')),
        ),
      );
    });

    test('a model of nothing but regressors has no variance to search', () {
      // No trend and no seasonal: every state is a constant coefficient, so
      // there is nothing for the optimiser to do. The measurement variance
      // still comes out in closed form.
      final data = _withEvents(holidayEffect: 1.2, conferenceEffect: -0.4);
      final model = StructuralModel([
        RegressionComponent([
          IndicatorRegressor('level', [(from: -1, to: 1e9)]),
          IndicatorRegressor('conference', [(from: 80, to: 87)]),
          IndicatorRegressor('holiday', [(from: 350, to: 364)]),
        ]),
      ]);

      final fitted = fit(model, data);
      expect(fitted.converged, isTrue);
      expect(fitted.varianceRatios, isEmpty);
      expect(fitted.plateauDecadesByParameter, isEmpty);
      expect(fitted.plateauDecades, 0);
      expect(fitted.evaluations, 1, reason: 'nothing was searched');

      final estimates = {
        for (final c in fitted.model.smooth(data).coefficients) c.name: c,
      };
      // An always-on column is the level, and the series drifts down by 0.002
      // a day over four hundred days, so it lands on the average of 69.6.
      expect(estimates['level']!.estimate, closeTo(69.6, 0.1));
    });

    test('and dropping the trend biases every event by where the trend was', () {
      // Worth pinning, because the fit still looks fine. Without somewhere for
      // the drift to go, each event coefficient absorbs the difference between
      // the level when it happened and the level on average -- the December
      // fortnight is late, where the trend is low, so its effect comes back
      // understated; the March week is early and high, so its effect is
      // understated the other way. Neither is a small error next to the
      // standard errors being reported alongside them.
      final data = _withEvents(holidayEffect: 1.2, conferenceEffect: -0.4);

      final withTrend = fit(
        StructuralModel([LocalLinearTrend(processVariance: 1e-4), _events()]),
        data,
      );
      final without = fit(
        StructuralModel([
          RegressionComponent([
            IndicatorRegressor('level', [(from: -1, to: 1e9)]),
            ..._events().regressors,
          ]),
        ]),
        data,
      );

      Map<String, Coefficient> read(FitResult f) => {
        for (final c in f.model.smooth(data).coefficients) c.name: c,
      };
      final good = read(withTrend);
      final bad = read(without);

      // With the trend: 1.239 +/- 0.070 and -0.296 +/- 0.095 against a true
      // 1.2 and -0.4.
      expect(good['holiday']!.estimate, closeTo(1.2, 0.15));
      expect(good['conference']!.estimate, closeTo(-0.4, 0.2));

      // Without it: 0.952 and -0.088. Both pulled towards zero, by three and
      // four standard errors respectively, with no widening of the error bars
      // to warn anybody.
      expect(
        bad['holiday']!.estimate,
        lessThan(good['holiday']!.estimate - 0.2),
      );
      expect(
        bad['conference']!.estimate,
        greaterThan(good['conference']!.estimate + 0.15),
      );

      // The one thing that does give it away is the noise level: the drift
      // has nowhere to go, so it is reported as measurement error. 0.33
      // against 0.25.
      expect(
        math.sqrt(without.measurementVariance),
        greaterThan(1.25 * math.sqrt(withTrend.measurementVariance)),
      );
    });
  });
}
