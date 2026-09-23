import 'dart:math' as math;

import 'package:state_space/authoring.dart';
import 'package:test/test.dart';

/// A shape parameter measured in time units has a floor below which it stops
/// being a different model.
///
/// The case that matters is a Matern with `nu = 1/2` and a length scale far
/// under the sampling interval: to the data that is white noise, so it competes
/// with the measurement error rather than with the trend — and wins, because
/// taking the noise for itself buys a little likelihood. Everything downstream
/// is then wrong together: the reported noise level, the credible band, and the
/// trend curve, which is now interpolating the noise.
void main() {
  List<Observation> weekly(int days, {int seed = 3}) {
    final random = math.Random(seed);
    double gaussian() =>
        math.sqrt(-2 * math.log(random.nextDouble())) *
        math.cos(2 * math.pi * random.nextDouble());
    return [
      for (var i = 0; i < days; i++)
        Observation(
            i.toDouble(),
            80 +
                0.002 * i +
                0.5 * math.sin(2 * math.pi * i / 7) +
                0.3 * gaussian())
    ];
  }

  group('samplingResolution', () {
    test('is the median gap, and ignores repeated timestamps', () {
      expect(
          samplingResolution([
            Observation(0, 1),
            Observation(0, 1.1),
            Observation(1, 2),
            Observation(2, 3),
            Observation(20, 4),
          ]),
          1);
    });

    test('is zero when there is nothing to measure', () {
      expect(samplingResolution([]), 0);
      expect(samplingResolution([Observation(0, 1)]), 0);
      expect(samplingResolution([Observation(4, 1), Observation(4, 2)]), 0);
    });

    test('counts two readings on one morning as one visit', () {
      // Every day weighed twice, five minutes apart: the readings are a day
      // apart, not five minutes.
      final doubled = [
        for (var i = 0; i < 100; i++) ...[
          Observation(i.toDouble(), 80),
          Observation(i + 5 / 1440, 80.1),
        ]
      ];
      expect(samplingResolution(doubled), closeTo(1, 0.01));
      final someDoubled = [
        for (var i = 0; i < 100; i++) ...[
          Observation(i.toDouble(), 80),
          if (i % 5 < 2) Observation(i + 5 / 1440, 80.1),
        ]
      ];
      expect(samplingResolution(someDoubled), closeTo(1, 0.01));
    });

    test('survives a diary with holidays in it', () {
      expect(
          samplingResolution([
            for (var i = 0; i < 30; i++) Observation(i.toDouble(), 80),
            Observation(60, 80),
            Observation(61, 80),
          ]),
          1);
    });
  });

  group('the floor keeps a Matern from impersonating measurement noise', () {
    test('daily data recovers the true noise level', () {
      final data = weekly(730);
      const trueNoise = 0.3;

      final withoutMatern = fit(
          StructuralModel([
            LocalLinearTrend(processVariance: 1e-4),
            TrigonometricSeasonal(period: 7, harmonics: 2, processVariance: 1),
          ]),
          data);
      final withMatern = fit(
          StructuralModel([
            LocalLinearTrend(processVariance: 1e-4),
            TrigonometricSeasonal(period: 7, harmonics: 2, processVariance: 1),
            Matern.oneHalf(variance: 0.05, lengthScale: 5),
          ]),
          data);

      // Before the floor this came back at 0.002, a hundred and fifty times
      // too small, because the Matern had taken the measurement error.
      expect(math.sqrt(withMatern.measurementVariance),
          closeTo(trueNoise, 0.1 * trueNoise));
      expect(math.sqrt(withMatern.measurementVariance),
          closeTo(math.sqrt(withoutMatern.measurementVariance), 0.05));
    });

    test('and a predictive band that covers about 95 per cent', () {
      final data = weekly(730);
      final fitted = fit(
          StructuralModel([
            LocalLinearTrend(processVariance: 1e-4),
            TrigonometricSeasonal(period: 7, harmonics: 2, processVariance: 1),
            Matern.oneHalf(variance: 0.05, lengthScale: 5),
          ]),
          data);
      final posterior = fitted.model.smooth(data);

      var inside = 0;
      for (var i = 0; i < data.length; i++) {
        final band = posterior.predictiveInterval(i);
        if (data[i].value >= band.lo && data[i].value <= band.hi) inside++;
      }
      // Before the floor this was 100 per cent: the band had swallowed the
      // noise, so nothing could fall outside it.
      expect(inside / data.length, closeTo(0.95, 0.03));
    });

    test('the length scale never lands below one sampling interval', () {
      final data = weekly(730);
      final fitted = fit(
          StructuralModel([
            LocalLinearTrend(processVariance: 1e-4),
            Matern.oneHalf(variance: 0.05, lengthScale: 5),
          ]),
          data);
      final matern = fitted.model.components[1] as Matern;
      expect(matern.lengthScale, greaterThanOrEqualTo(1.0));
    });

    test('the floor scales with the sampling, not with the calendar', () {
      // The same series read every ten time units instead of every one. The
      // floor has to move with it.
      final sparse = [
        for (var i = 0; i < 200; i++)
          Observation(i * 10.0, 80 + 0.02 * i + 0.3 * math.sin(i / 3.0))
      ];
      expect(samplingResolution(sparse), 10);
      final fitted = fit(
          StructuralModel([
            LocalLinearTrend(processVariance: 1e-4),
            Matern.oneHalf(variance: 0.05, lengthScale: 50),
          ]),
          sparse);
      expect((fitted.model.components[1] as Matern).lengthScale,
          greaterThanOrEqualTo(10.0));
    });

    test('a caller who asks for a higher floor still gets it', () {
      final data = weekly(365);
      final fitted = fit(
          StructuralModel([
            LocalLinearTrend(processVariance: 1e-4),
            Matern.oneHalf(
                variance: 0.05,
                lengthScale: 30,
                lengthScaleBounds: (lower: 20, upper: 1e4)),
          ]),
          data);
      expect((fitted.model.components[1] as Matern).lengthScale,
          greaterThanOrEqualTo(20.0));
    });

    test('and is told plainly when no bracket survives the sampling', () {
      final sparse = [
        for (var i = 0; i < 40; i++) Observation(i * 1000.0, 80 + 0.001 * i)
      ];
      expect(
          () => fit(
              StructuralModel([
                LocalLinearTrend(processVariance: 1e-4),
                Matern.oneHalf(
                    variance: 0.05,
                    lengthScale: 5,
                    lengthScaleBounds: (lower: 0.1, upper: 10)),
              ]),
              sparse),
          throwsA(isA<UnderdeterminedModelException>().having(
              (e) => e.toString(),
              'message',
              contains('is measurement noise rather than a separate'))));
    });
  });

  group('a cycle is held above the Nyquist limit of the actual sampling', () {
    test('the fitted period is at least twice the sampling interval', () {
      final sparse = [
        for (var i = 0; i < 300; i++)
          Observation(i * 5.0, 80 + math.sin(2 * math.pi * i * 5 / 90))
      ];
      final fitted = fit(
          StructuralModel([
            LocalLevel(processVariance: 1e-4),
            StochasticCycle(period: 90, damping: 0.99, stationaryVariance: 0.5),
          ]),
          sparse);
      expect((fitted.model.components[1] as StochasticCycle).period,
          greaterThanOrEqualTo(10.0));
    });

    test('and says so when the whole bracket is below Nyquist', () {
      final sparse = [
        for (var i = 0; i < 60; i++) Observation(i * 500.0, 80 + 0.001 * i)
      ];
      expect(
          () => fit(
              StructuralModel([
                LocalLevel(processVariance: 1e-4),
                StochasticCycle(
                    period: 90,
                    damping: 0.99,
                    stationaryVariance: 0.5,
                    periodBounds: (lower: 2, upper: 400)),
              ]),
              sparse),
          throwsA(isA<UnderdeterminedModelException>()
              .having((e) => e.toString(), 'message', contains('Nyquist'))));
    });
  });
}
