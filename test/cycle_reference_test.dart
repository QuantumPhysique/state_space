import 'dart:math' as math;
import 'dart:typed_data';

import 'package:state_space/state_space.dart';
import 'package:test/test.dart';

import 'support/dense_reference.dart';

List<Observation> _series(int n, {int seed = 23}) {
  final random = math.Random(seed);
  final data = <Observation>[];
  var time = 0.0;
  for (var i = 0; i < n; i++) {
    time += 0.5 + 1.1 * random.nextDouble();
    data.add(Observation(time, 0.7 * math.cos(2 * math.pi * time / 9 + 0.4),
        relativeVariance: 0.7 + random.nextDouble()));
  }
  return data;
}

void main() {
  const period = 9.0;
  const damping = 0.93;
  const stationaryVariance = 0.5;
  const measurementVariance = 0.03;

  group('a damped cycle against its kernel', () {
    // The quasi-periodic kernel `sigma^2 rho^|tau| cos(2 pi tau / p)` is what
    // the rotation-with-damping actually implements, and this is the check
    // that it does. As with the Matern component there are no flat directions
    // here, so the comparison is against the plain textbook likelihood with no
    // restricted-likelihood correction to reason about.
    final data = _series(140);

    test('likelihood, posterior mean and variance', () {
      final model = StructuralModel([
        StochasticCycle(
            period: period,
            damping: damping,
            stationaryVariance: stationaryVariance)
      ], measurementVariance: measurementVariance);
      final posterior = model.smooth(data);

      final dense = densePosterior(
        data,
        [for (final o in data) o.time],
        [cycleKernel(period, damping, stationaryVariance)],
        measurementVariance: measurementVariance,
      );

      expect(
          posterior.logMarginalLikelihood, closeTo(dense.logLikelihood, 1e-9));
      for (var i = 0; i < data.length; i++) {
        expect(posterior.mean[i], closeTo(dense.mean[0][i], 1e-10),
            reason: 'mean at $i');
        expect(posterior.variance[i], closeTo(dense.variance[0][i], 1e-11),
            reason: 'variance at $i');
      }
    });

    test('and between the readings, where the phase has to be carried', () {
      final grid = [for (var i = 0; i <= 60; i++) 2.4 + i * 1.37];
      final posterior = StructuralModel([
        StochasticCycle(
            period: period,
            damping: damping,
            stationaryVariance: stationaryVariance)
      ], measurementVariance: measurementVariance)
          .smooth(data, grid: Float64List.fromList(grid));
      final dense = densePosterior(
        data,
        grid,
        [cycleKernel(period, damping, stationaryVariance)],
        measurementVariance: measurementVariance,
      );
      for (var i = 0; i < grid.length; i++) {
        expect(posterior.mean[i], closeTo(dense.mean[0][i], 1e-10),
            reason: 'mean at ${grid[i]}');
        expect(posterior.variance[i], closeTo(dense.variance[0][i], 1e-11),
            reason: 'variance at ${grid[i]}');
      }
    });
  });

  group('a cycle underneath a trend', () {
    // The realistic arrangement, and the one where a decomposition can go
    // wrong: the trend carries the flat directions and the cycle carries none,
    // so the two are told apart by the shape of their covariance alone.
    test('the restricted likelihood matches', () {
      final data = _series(180, seed: 31);
      const processVariance = 2e-5;
      final model = StructuralModel([
        LocalLinearTrend(processVariance: processVariance),
        StochasticCycle(
            period: period,
            damping: damping,
            stationaryVariance: stationaryVariance),
      ], measurementVariance: measurementVariance);

      final dense = restrictedLikelihood(
        data,
        sumOf([
          splineKernel(processVariance),
          cycleKernel(period, damping, stationaryVariance),
        ]),
        trendBasis(),
        measurementVariance: measurementVariance,
      );

      expect(model.smooth(data).logMarginalLikelihood,
          closeTo(dense.logLikelihood, 1e-9));
    });
  });
}
