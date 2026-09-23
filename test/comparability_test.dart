import 'dart:math' as math;
import 'dart:typed_data';

import 'package:state_space/state_space.dart';
import 'package:test/test.dart';

/// The diffuse likelihood is the *restricted* likelihood, and these tests pin
/// down exactly how far that lets it be compared.
///
/// `log p(y)` here is `integral p(y | d) dd` against an improper flat prior of
/// unit density on the flat directions. `d` carries units, so the integral does
/// too, and anything that rescales a diffuse direction shifts the answer by the
/// log of the scale — without touching the model, the data or the posterior.
/// That is not a defect of the implementation; it is what a restricted
/// likelihood is. It is tested because the package used to claim the opposite,
/// and because the shift is large enough to reverse a model comparison.
void main() {
  List<Observation> series(int n, {double slope = 0.005, int seed = 11}) {
    final random = math.Random(seed);
    double gaussian() =>
        math.sqrt(-2 * math.log(random.nextDouble())) *
        math.cos(2 * math.pi * random.nextDouble());
    return [
      for (var i = 0; i < n; i++)
        Observation(i.toDouble(), 80 + slope * i + 0.3 * gaussian()),
    ];
  }

  group('the restricted likelihood carries the units of the flat directions', () {
    test(
      'rescaling a regression column shifts it by exactly minus log scale',
      () {
        final data = series(120);
        double likelihoodAtColumnScale(double scale) {
          final model = StructuralModel([
            LocalLinearTrend(processVariance: 1e-4),
            RegressionComponent([
              StepRegressor(
                'dose',
                Float64List.fromList([50, 60]),
                Float64List.fromList([scale, 0]),
              ),
            ]),
          ], measurementVariance: 0.09);
          return model.logLikelihood(data);
        }

        final unit = likelihoodAtColumnScale(1);
        for (final scale in [1000.0, 1e-3, 7.5]) {
          expect(
            likelihoodAtColumnScale(scale) - unit,
            closeTo(-math.log(scale), 1e-9),
            reason: 'a column written in different units is the same model',
          );
        }
      },
    );

    test('the same rescaling leaves the posterior alone', () {
      final data = series(120);
      SmoothingResult posteriorAtColumnScale(double scale) {
        final model = StructuralModel([
          LocalLinearTrend(processVariance: 1e-4),
          RegressionComponent([
            StepRegressor(
              'dose',
              Float64List.fromList([50, 60]),
              Float64List.fromList([scale, 0]),
            ),
          ]),
        ], measurementVariance: 0.09);
        return model.smooth(data);
      }

      final unit = posteriorAtColumnScale(1);
      final thousandfold = posteriorAtColumnScale(1000);
      for (var i = 0; i < unit.length; i++) {
        expect(thousandfold.mean[i], closeTo(unit.mean[i], 1e-9));
      }
      // The coefficient absorbs the scale exactly, which is what makes the
      // likelihood shift a statement about units rather than about fit.
      expect(
        thousandfold.coefficients.single.estimate * 1000,
        closeTo(unit.coefficients.single.estimate, 1e-9),
      );
    });

    test('changing the time unit shifts it by log of the factor', () {
      final data = series(120);
      const q = 1e-4, r = 0.09;
      final atUnitTime = StructuralModel.localLinearTrend(
        processVariance: q,
        measurementVariance: r,
      ).logLikelihood(data);

      for (final factor in [2.0, 10.0]) {
        // The same stochastic process written on a rescaled axis: t -> t / c
        // sends a trend's variance per cubed time unit to q * c^3. The density
        // of y is unchanged, so a likelihood that measured only the data would
        // not move.
        final rescaled = [
          for (final o in data) Observation(o.time / factor, o.value),
        ];
        final shifted = StructuralModel.localLinearTrend(
          processVariance: q * factor * factor * factor,
          measurementVariance: r,
        ).logLikelihood(rescaled);
        expect(
          shifted - atUnitTime,
          closeTo(math.log(factor), 1e-9),
          reason: 'the slope direction of B is scaled by 1 / factor',
        );
      }
    });

    test('a proper prior has no such freedom, and does not move', () {
      final data = series(120);
      const q = 1e-4, r = 0.09;
      final wide = ApproximateDiffuse(variance: 1e8);
      final atUnitTime = StructuralModel.localLinearTrend(
        processVariance: q,
        measurementVariance: r,
        initialization: wide,
      ).logLikelihood(data);

      for (final factor in [2.0, 10.0]) {
        final rescaled = [
          for (final o in data) Observation(o.time / factor, o.value),
        ];
        final shifted = StructuralModel.localLinearTrend(
          processVariance: q * factor * factor * factor,
          measurementVariance: r,
          initialization: wide,
        ).logLikelihood(rescaled);
        expect(
          shifted - atUnitTime,
          closeTo(0, 1e-4),
          reason: 'a proper prior makes this an ordinary density in y',
        );
      }
    });
  });

  group('diffuseDimension says when two fits may be subtracted', () {
    test('counts the flat directions of each component', () {
      expect(
        StructuralModel([
          LocalLinearTrend(processVariance: 1e-4),
        ]).diffuseDimension,
        2,
      );
      expect(
        StructuralModel([
          LocalLinearTrend(processVariance: 1e-4),
          TrigonometricSeasonal(period: 7, harmonics: 2, processVariance: 1e-4),
        ]).diffuseDimension,
        6,
      );
      expect(
        StructuralModel([
          LocalLinearTrend(processVariance: 1e-4),
          RegressionComponent([
            IndicatorRegressor('a', [(from: 1.0, to: 2.0)]),
            IndicatorRegressor('b', [(from: 3.0, to: 4.0)]),
          ]),
        ]).diffuseDimension,
        4,
      );
    });

    test('a stationary component adds none', () {
      expect(
        StructuralModel([
          LocalLinearTrend(processVariance: 1e-4),
          Matern.oneHalf(variance: 0.05, lengthScale: 5),
        ]).diffuseDimension,
        2,
      );
    });

    test('an approximate prior integrates nothing out', () {
      expect(
        StructuralModel([
          LocalLinearTrend(processVariance: 1e-4),
        ], initialization: ApproximateDiffuse()).diffuseDimension,
        0,
      );
    });

    test('isComparableWith is false exactly when the dimensions differ', () {
      final data = series(160);
      final trend = fit(
        StructuralModel([LocalLinearTrend(processVariance: 1)]),
        data,
      );
      final trendAgain = fit(
        StructuralModel([LocalLinearTrend(processVariance: 1e-6)]),
        data,
      );
      final withMatern = fit(
        StructuralModel([
          LocalLinearTrend(processVariance: 1),
          Matern.oneHalf(
            variance: 0.05,
            lengthScale: 5,
            lengthScaleBounds: (lower: 1, upper: 1e4),
          ),
        ]),
        data,
      );
      final withSeasonal = fit(
        StructuralModel([
          LocalLinearTrend(processVariance: 1),
          TrigonometricSeasonal(period: 7, harmonics: 2, processVariance: 1),
        ]),
        data,
      );

      expect(trend.isComparableWith(trendAgain), isTrue);
      expect(
        trend.isComparableWith(withMatern),
        isTrue,
        reason: 'a stationary component adds no flat direction',
      );
      expect(trend.isComparableWith(withSeasonal), isFalse);
      expect(withSeasonal.isComparableWith(trend), isFalse);
    });
  });
}
