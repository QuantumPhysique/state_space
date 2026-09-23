import 'dart:math' as math;
import 'dart:typed_data';

import 'package:state_space/state_space.dart';
import 'package:test/test.dart';

double _gaussian(math.Random random) =>
    math.sqrt(-2 * math.log(1 - random.nextDouble())) *
    math.cos(2 * math.pi * random.nextDouble());

/// Daily readings carrying a slow trend, a fixed weekly pattern and noise.
/// The weekly part is deliberately large enough to be obvious and small
/// enough that a trend-only model still looks like a decent fit by eye.
List<Observation> _weeklySeries({int n = 400, int seed = 8}) {
  final random = math.Random(seed);
  return [
    for (var i = 0; i < n; i++)
      Observation(
        i.toDouble(),
        80 -
            0.004 * i +
            0.45 * math.cos(2 * math.pi * i / 7) +
            0.2 * math.sin(4 * math.pi * i / 7) +
            0.25 * _gaussian(random),
      ),
  ];
}

void main() {
  group('autocorrelation', () {
    test('is minus one for a sequence that alternates', () {
      final diagnostics = InnovationDiagnostics(
        times: Float64List.fromList([for (var i = 0; i < 100; i++) i * 1.0]),
        residuals: Float64List.fromList([
          for (var i = 0; i < 100; i++) i.isEven ? 1.0 : -1.0,
        ]),
      );
      // Not exactly minus one: the sum in the numerator runs over 99 pairs
      // and the one in the denominator over 100 terms.
      expect(diagnostics.autocorrelation(1), closeTo(-0.99, 1e-12));
      expect(diagnostics.autocorrelation(2), closeTo(0.98, 1e-12));
      expect(diagnostics.mean, closeTo(0, 1e-15));
      expect(diagnostics.variance, closeTo(1, 1e-15));
    });

    test('refuses a lag it cannot compute', () {
      final diagnostics = InnovationDiagnostics(
        times: Float64List.fromList([0, 1, 2]),
        residuals: Float64List.fromList([0.1, -0.2, 0.3]),
      );
      expect(() => diagnostics.autocorrelation(0), throwsArgumentError);
      expect(() => diagnostics.autocorrelation(3), throwsArgumentError);
      expect(() => diagnostics.ljungBox(lags: 3), throwsArgumentError);
      expect(
        () => diagnostics.ljungBox(lags: 2, fittedParameters: 2),
        throwsArgumentError,
      );
    });
  });

  group('a model that has left something out', () {
    final data = _weeklySeries();

    test('is caught by the portmanteau test', () {
      // A trend alone fits this series to about a quarter of a unit and looks
      // entirely reasonable plotted. What it cannot do is make the errors
      // independent: it is missing a weekly cycle, so every seventh residual
      // agrees with the last one.
      final diagnostics = StructuralModel.localLinearTrend(
        processVariance: 1e-8,
        measurementVariance: 0.0625,
      ).diagnose(data);

      final test14 = diagnostics.ljungBox(lags: 14, fittedParameters: 1);
      expect(test14.degreesOfFreedom, 13);
      expect(test14.pValue, lessThan(1e-12)); // measured 1.2e-225

      // And it says where: a positive spike at the period and at twice it, a
      // trough at half -- the signature of a cycle rather than of drift.
      expect(diagnostics.autocorrelation(7), greaterThan(0.5)); // 0.697
      expect(diagnostics.autocorrelation(14), greaterThan(0.5)); // 0.679
      expect(diagnostics.autocorrelation(3), lessThan(-0.3)); // -0.500

      // The model also has to inflate its own noise threefold to cover the
      // pattern, which is the other half of the tell: too big, and correlated.
      expect(diagnostics.variance, greaterThan(2.5)); // 3.047
    });

    test('and putting it back makes the evidence go away', () {
      final diagnostics = StructuralModel([
        LocalLinearTrend(processVariance: 1e-8),
        TrigonometricSeasonal(period: 7, harmonics: 2, processVariance: 1e-6),
      ], measurementVariance: 0.0625).diagnose(data);

      final test14 = diagnostics.ljungBox(lags: 14, fittedParameters: 2);
      expect(test14.pValue, greaterThan(0.05)); // measured 0.194
      expect(diagnostics.autocorrelation(7).abs(), lessThan(0.1)); // 0.017

      // The residuals are also the right size now, which the trend-only model
      // could not manage either: it had to inflate them to cover the pattern.
      expect(
        diagnostics.mean.abs(),
        lessThan(3 / math.sqrt(diagnostics.count)),
      );
      expect(
        (diagnostics.variance - 1).abs(),
        lessThan(3 * math.sqrt(2 / diagnostics.count)),
      );
    });

    test('the residual count is the observation count less the flat '
        'directions', () {
      final diagnostics = StructuralModel([
        LocalLinearTrend(processVariance: 1e-8),
        TrigonometricSeasonal(period: 7, harmonics: 2, processVariance: 1e-6),
      ], measurementVariance: 0.0625).diagnose(data);
      expect(diagnostics.count, data.length - 6);
      expect(diagnostics.times.first, data[6].time);
    });
  });

  group('a model telling the truth', () {
    test('is not rejected', () {
      // Simulated from exactly the model that reads it, so there is nothing
      // for the test to find and it should not find anything.
      final random = math.Random(91);
      const processVariance = 1e-4;
      const measurementVariance = 0.02;
      final data = <Observation>[];
      var level = 50.0;
      var slope = 0.0;
      var time = 0.0;
      for (var i = 0; i < 900; i++) {
        final dt = 0.4 + 1.2 * random.nextDouble();
        final sd = math.sqrt(processVariance);
        final a = _gaussian(random), b = _gaussian(random);
        final root = dt * math.sqrt(dt);
        level += slope * dt + sd * (root / math.sqrt(3) * a + root / 2 * b);
        slope += sd * math.sqrt(dt) * b;
        time += dt;
        data.add(
          Observation(
            time,
            level + math.sqrt(measurementVariance) * _gaussian(random),
          ),
        );
      }

      final diagnostics = StructuralModel.localLinearTrend(
        processVariance: processVariance,
        measurementVariance: measurementVariance,
      ).diagnose(data);

      expect(diagnostics.ljungBox(lags: 12).pValue, greaterThan(0.05));
      expect(
        diagnostics.autocorrelation(1).abs(),
        lessThan(3 / math.sqrt(diagnostics.count)),
      );
    });
  });
}
