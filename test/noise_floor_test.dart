import 'dart:math' as math;

import 'package:state_space/state_space.dart';
import 'package:test/test.dart';

/// A smooth curve read off an implausibly good instrument. The point is that
/// the readings are far more consistent than any real scale, so the fit will
/// happily report a precision no device could deliver.
List<Observation> _tooGoodToBeTrue({double noise = 0.002, int seed = 4}) {
  final random = math.Random(seed);
  return [
    for (var day = 0; day < 120; day++)
      Observation(
          day.toDouble(),
          80 -
              0.02 * day +
              0.4 * math.sin(day / 25) +
              noise * (random.nextDouble() - 0.5))
  ];
}

/// The same shape with an honest amount of scatter on it.
List<Observation> _ordinary({int seed = 9}) {
  final random = math.Random(seed);
  return [
    for (var day = 0; day < 120; day++)
      Observation(
          day.toDouble(),
          80 -
              0.02 * day +
              0.4 * math.sin(day / 25) +
              0.3 * (random.nextDouble() - 0.5))
  ];
}

void main() {
  StructuralModel trend() =>
      StructuralModel.localLinearTrend(processVariance: 1e-4);

  group('a floor on the measurement variance', () {
    test('does nothing when the data is noisier than the floor', () {
      final data = _ordinary();
      final free = fit(trend(), data);
      final floored = fit(trend(), data, minimumMeasurementVariance: 1e-6);

      expect(free.measurementVariance, greaterThan(1e-6));
      expect(floored.measurementVariance, free.measurementVariance);
      expect(floored.measurementVariancePinned, isFalse);
      expect(floored.varianceRatio, free.varianceRatio);
    });

    test('holds the noise level up when the data claims to be cleaner', () {
      // 0.05 kg is about what a domestic scale reading in 100 g steps can
      // actually deliver. The series below is a hundred times more consistent
      // than that, and without a floor the fit believes it.
      const floor = 0.05 * 0.05;
      final data = _tooGoodToBeTrue();
      final free = fit(trend(), data);
      final floored = fit(trend(), data, minimumMeasurementVariance: floor);

      expect(free.measurementVariance, lessThan(floor));
      expect(free.measurementVariancePinned, isFalse);
      expect(floored.measurementVariance, closeTo(floor, 1e-15));
      expect(floored.measurementVariancePinned, isTrue);
    });

    test('and the band widens to match, which is the point of asking', () {
      const floor = 0.05 * 0.05;
      final data = _tooGoodToBeTrue();
      final free = fit(trend(), data).model.smooth(data);
      final floored = fit(trend(), data, minimumMeasurementVariance: floor)
          .model
          .smooth(data);

      final narrow = free.credibleInterval(60);
      final honest = floored.credibleInterval(60);
      expect(honest.hi - honest.lo, greaterThan(4 * (narrow.hi - narrow.lo)));
    });

    test('the process variance is re-estimated against the floor, not kept',
        () {
      // The reason a floor cannot be clamped on afterwards. Pinning the noise
      // level in absolute units breaks the scale equivariance that lets the
      // fit search ratios, so everything else has to be found again against
      // the new noise level. Here that moves the absolute process variance by
      // 0.60 decades, a factor of four: a curve that no longer has to explain
      // every reading exactly is allowed to be markedly stiffer. The threshold
      // below is set under that measurement rather than at it, since the exact
      // figure is a property of this simulation.
      const floor = 0.05 * 0.05;
      final data = _tooGoodToBeTrue();
      final free = fit(trend(), data);
      final floored = fit(trend(), data, minimumMeasurementVariance: floor);

      final freeAbsolute = free.varianceRatio * free.measurementVariance;
      final flooredAbsolute =
          floored.varianceRatio * floored.measurementVariance;
      expect((math.log(flooredAbsolute / freeAbsolute) / math.ln10).abs(),
          greaterThan(0.4));
    });
  });

  group('a fixed measurement variance', () {
    test('reproduces the free fit when set to what the free fit found', () {
      // The invariant that says the two code paths are the same likelihood
      // seen from two ends. At the joint maximum the derivative in every
      // direction is zero, so holding the noise level at its own estimate and
      // maximising over what is left has to land in the same place.
      final data = _ordinary();
      final free = fit(trend(), data);
      final pinned = fit(trend(), data,
          fixedMeasurementVariance: free.measurementVariance);

      expect(pinned.measurementVariance, free.measurementVariance);
      expect(pinned.measurementVariancePinned, isTrue);
      expect((math.log(pinned.varianceRatio / free.varianceRatio)).abs(),
          lessThan(1e-3),
          reason: 'free ${free.varianceRatio}, pinned ${pinned.varianceRatio}');
      expect(pinned.logMarginalLikelihood,
          closeTo(free.logMarginalLikelihood, 1e-6));
    });

    test('recovers the process variance when the noise level is known', () {
      final data = _ordinary();
      // The simulated noise is uniform on a range of 0.3, whose variance is
      // 0.3^2 / 12.
      const truth = 0.3 * 0.3 / 12;
      final pinned = fit(trend(), data, fixedMeasurementVariance: truth);
      expect(math.sqrt(pinned.measurementVariance),
          closeTo(math.sqrt(truth), 1e-15));
      expect(pinned.parameterStatus.first, ParameterStatus.determined);
    });
  });

  group('arguments', () {
    test('a floor and a fixed value together are refused', () {
      expect(
          () => fit(trend(), _ordinary(),
              fixedMeasurementVariance: 1, minimumMeasurementVariance: 1),
          throwsArgumentError);
    });

    test('a non-positive noise level is refused', () {
      expect(() => fit(trend(), _ordinary(), fixedMeasurementVariance: 0),
          throwsArgumentError);
      expect(() => fit(trend(), _ordinary(), minimumMeasurementVariance: -1),
          throwsArgumentError);
    });
  });
}
