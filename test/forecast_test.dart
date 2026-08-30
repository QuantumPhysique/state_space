import 'dart:math' as math;
import 'dart:typed_data';

import 'package:state_space/state_space.dart';
import 'package:test/test.dart';

List<Observation> _series({int n = 50, int seed = 8, double slope = 0.04}) {
  final random = math.Random(seed);
  final data = <Observation>[];
  var time = 0.0;
  for (var i = 0; i < n; i++) {
    time += 0.6 + 1.8 * random.nextDouble();
    data.add(Observation(time, 78 + slope * time + random.nextDouble() - 0.5));
  }
  return data;
}

StructuralModel _model() => StructuralModel.localLinearTrend(
      processVariance: 6e-4,
      measurementVariance: 0.1,
    );

void main() {
  test('a forecast is what smoothing on a trailing grid already said', () {
    // Beyond the last observation there is no future data to smooth against,
    // so the backward pass has nothing to add and the two routes must agree.
    // This is the check that keeps forecast() honest: it is an optimisation of
    // a path that is already tested, not a second implementation.
    final data = _series();
    final last = data.last.time;
    final horizon =
        Float64List.fromList([for (var k = 1; k <= 20; k++) last + k * 2.0]);

    final forecast = _model().forecast(data, horizon);
    final smoothed = _model().smooth(data, grid: horizon);

    expect(forecast.length, horizon.length);
    for (var i = 0; i < horizon.length; i++) {
      expect(forecast.times[i], horizon[i]);
      expect(forecast.mean[i], closeTo(smoothed.level[i], 1e-9));
      expect(forecast.variance[i], closeTo(smoothed.levelVariance[i], 1e-10));
    }
  });

  test('starting the horizon at the last observation reproduces it', () {
    final data = _series();
    final last = data.last.time;
    final forecast = _model().forecast(data, Float64List.fromList([last]));
    final smoothed = _model().smooth(data);

    expect(forecast.mean.single, closeTo(smoothed.level.last, 1e-9));
  });

  test('the band widens like the three-halves power of the horizon', () {
    final data = _series();
    final last = data.last.time;
    final forecast =
        _model().forecast(data, Float64List.fromList([last + 50, last + 400]));

    final growth = math.sqrt(forecast.variance[1] / forecast.variance[0]);
    expect(growth, greaterThan(15));
    expect(growth, lessThan(math.pow(8, 1.5)));
  });

  test('the forecast continues the fitted slope', () {
    // A clean linear series with little noise: a year out, the trend should
    // still be going where it was going, and the credible band should contain
    // the straight-line continuation.
    final data = _series(n: 120, slope: 0.05);
    final fitted = fit(
      StructuralModel.localLinearTrend(processVariance: 1),
      data,
    );
    final last = data.last.time;
    final forecast = fitted.model
        .forecast(data, Float64List.fromList([last + 10, last + 30]));

    for (var i = 0; i < forecast.length; i++) {
      final straight = 78 + 0.05 * forecast.times[i];
      final band = forecast.credibleInterval(i);
      expect(band.lo, lessThan(straight));
      expect(band.hi, greaterThan(straight));
    }
  });

  test('the predictive band is wider than the credible one, by the noise', () {
    final data = _series();
    final last = data.last.time;
    final forecast = _model().forecast(data, Float64List.fromList([last + 5]));

    final credible = forecast.credibleInterval(0);
    final predictive = forecast.predictiveInterval(0);
    expect(
        predictive.hi - predictive.lo, greaterThan(credible.hi - credible.lo));
    // 1.95996... is the 97.5% normal quantile; the tolerance is set by how
    // many of its digits are written here, not by the package.
    expect(
      predictive.hi - predictive.lo,
      closeTo(2 * 1.959963985 * math.sqrt(forecast.variance[0] + 0.1), 1e-7),
    );
  });

  group('input validation', () {
    test('rejects a horizon that reaches back into the data', () {
      final data = _series();
      expect(
        () => _model().forecast(
            data, Float64List.fromList([data[10].time, data.last.time + 1])),
        throwsA(isA<ArgumentError>().having((e) => e.message.toString(),
            'message', contains('smooth(observations, grid:'))),
      );
    });

    test('rejects an unsorted horizon and a missing series', () {
      final data = _series();
      final last = data.last.time;
      expect(
          () => _model()
              .forecast(data, Float64List.fromList([last + 5, last + 1])),
          throwsArgumentError);
      expect(() => _model().forecast(const [], Float64List.fromList([1])),
          throwsArgumentError);
    });
  });
}
