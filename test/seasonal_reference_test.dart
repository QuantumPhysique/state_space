import 'dart:math' as math;
import 'dart:typed_data';

import 'package:state_space/src/engine/kalman.dart';
import 'package:state_space/src/engine/timeline.dart';
import 'package:state_space/state_space.dart';
import 'package:test/test.dart';

import 'support/dense_reference.dart';

/// Irregularly sampled days carrying a trend, a weekly pattern and a longer
/// one, which is the shape of the problem the package was written for.
List<Observation> _series(int n, {int seed = 11}) {
  final random = math.Random(seed);
  final data = <Observation>[];
  var time = 0.0;
  for (var i = 0; i < n; i++) {
    time += 0.4 + 1.6 * random.nextDouble();
    final signal = 80 +
        0.01 * time -
        0.9 * math.cos(2 * math.pi * time / 7) +
        0.3 * math.sin(4 * math.pi * time / 7) +
        0.5 * math.sin(2 * math.pi * time / 29);
    data.add(Observation(time, signal + 0.3 * (random.nextDouble() - 0.5),
        relativeVariance: 0.6 + random.nextDouble()));
  }
  return data;
}

void main() {
  const measurementVariance = 0.04;
  const seasonalVariance = 2e-4;
  const trendVariance = 3e-5;
  const cycleVariance = 5e-5;
  const diffuseVariance = 50.0;
  const kappa = diffuseVariance * measurementVariance;

  group('a seasonal component on its own', () {
    // Detrended data: the model here has no level, so neither should the
    // series it is asked to explain.
    List<Observation> data() {
      final random = math.Random(5);
      final out = <Observation>[];
      var time = 0.0;
      for (var i = 0; i < 110; i++) {
        time += 0.4 + 1.6 * random.nextDouble();
        final signal = -0.9 * math.cos(2 * math.pi * time / 7) +
            0.3 * math.sin(4 * math.pi * time / 7);
        out.add(Observation(time, signal + 0.3 * (random.nextDouble() - 0.5)));
      }
      return out;
    }

    test('has the kernel min(s, t) times a comb of cosines', () {
      final observations = data();
      final grid = Float64List.fromList([
        for (var i = 0; i < 45; i++)
          observations.first.time +
              (observations.last.time - observations.first.time) * i / 44,
      ]);

      final fast = StructuralModel(
        [
          TrigonometricSeasonal(
              period: 7, harmonics: 2, processVariance: seasonalVariance)
        ],
        measurementVariance: measurementVariance,
        initialization: const ApproximateDiffuse(variance: diffuseVariance),
      ).smooth(observations, grid: grid);

      final slow = densePosterior(
        observations,
        grid.toList(),
        [
          withDiffusePrior(seasonalKernel(7, 2, seasonalVariance),
              seasonalBasis(7, 2), kappa)
        ],
        measurementVariance: measurementVariance,
      );

      for (var i = 0; i < grid.length; i++) {
        expect(fast.level[i], closeTo(slow.mean[0][i], 1e-9),
            reason: 'mean at grid point $i');
        expect(fast.levelVariance[i], closeTo(slow.variance[0][i], 1e-9),
            reason: 'variance at grid point $i');
      }
    });

    test('six flat directions give the dense restricted likelihood', () {
      final observations = data();
      final forward = KalmanFilter(
        [
          TrigonometricSeasonal(
              period: 7, harmonics: 3, processVariance: seasonalVariance)
        ],
        measurementVariance: measurementVariance,
        initialization: const ExactDiffuse(),
      ).run(Timeline.merge(observations, null));

      final dense = restrictedLikelihood(
        observations,
        seasonalKernel(7, 3, seasonalVariance),
        seasonalBasis(7, 3),
        measurementVariance: measurementVariance,
      );

      expect(forward.diffuseDim, 6);
      expect(forward.usedObservations, observations.length - 6);
      expect(
          forward.diffuseLogDeterminant, closeTo(dense.logDeterminant, 1e-8));
      expect(forward.logLikelihood, closeTo(dense.logLikelihood, 1e-9));
    });
  });

  group('a trend and two seasonals composed', () {
    List<Component> components() => [
          const LocalLinearTrend(processVariance: trendVariance),
          TrigonometricSeasonal(
              period: 7, harmonics: 2, processVariance: seasonalVariance),
          TrigonometricSeasonal(
              period: 29, harmonics: 1, processVariance: cycleVariance),
        ];

    List<Kernel> kernels() => [
          withDiffusePrior(splineKernel(trendVariance), trendBasis(), kappa),
          withDiffusePrior(seasonalKernel(7, 2, seasonalVariance),
              seasonalBasis(7, 2), kappa),
          withDiffusePrior(seasonalKernel(29, 1, cycleVariance),
              seasonalBasis(29, 1), kappa),
        ];

    test('each component gets its own share, not just the total', () {
      // Three blocks of different sizes -- two, four and two states -- summed
      // by the engine and never inspected by it. A confounding bug between
      // the trend and the weekly pattern would leave the total fit intact and
      // show up only here.
      final observations = _series(130);
      final grid = Float64List.fromList([
        for (var i = 0; i < 35; i++)
          observations.first.time +
              (observations.last.time - observations.first.time) * i / 34,
      ]);

      final fast = StructuralModel(
        components(),
        measurementVariance: measurementVariance,
        initialization: const ApproximateDiffuse(variance: diffuseVariance),
      ).smooth(observations, grid: grid);
      final slow = densePosterior(observations, grid.toList(), kernels(),
          measurementVariance: measurementVariance);

      expect(fast.componentCount, 3);
      for (var b = 0; b < 3; b++) {
        for (var i = 0; i < grid.length; i++) {
          expect(fast.componentMean(b)[i], closeTo(slow.mean[b][i], 1e-9),
              reason: 'component $b mean at grid point $i');
          expect(
              fast.componentVariance(b)[i], closeTo(slow.variance[b][i], 1e-9),
              reason: 'component $b variance at grid point $i');
        }
      }

      for (var i = 0; i < grid.length; i++) {
        expect(fast.level[i], closeTo(slow.mean[3][i], 1e-9));
        expect(fast.levelVariance[i], closeTo(slow.variance[3][i], 1e-9));
      }
    });

    test('eight flat directions give the dense restricted likelihood', () {
      final observations = _series(150, seed: 23);
      final forward = KalmanFilter(
        components(),
        measurementVariance: measurementVariance,
        initialization: const ExactDiffuse(),
      ).run(Timeline.merge(observations, null));

      final trend = trendBasis();
      final weekly = seasonalBasis(7, 2);
      final monthly = seasonalBasis(29, 1);
      final dense = restrictedLikelihood(
        observations,
        sumOf([
          splineKernel(trendVariance),
          seasonalKernel(7, 2, seasonalVariance),
          seasonalKernel(29, 1, cycleVariance),
        ]),
        (s) => [...trend(s), ...weekly(s), ...monthly(s)],
        measurementVariance: measurementVariance,
      );

      expect(forward.stateDim, 8);
      expect(forward.diffuseDim, 8);
      expect(forward.usedObservations, observations.length - 8);
      expect(forward.logLikelihood, closeTo(dense.logLikelihood, 1e-8));
    });
  });
}
