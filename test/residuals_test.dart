import 'dart:math' as math;

import 'package:state_space/src/engine/fast_path_2x2.dart';
import 'package:state_space/src/engine/kalman.dart';
import 'package:state_space/src/engine/timeline.dart';
import 'package:state_space/state_space.dart';
import 'package:test/test.dart';

List<Observation> _series(int n, {int seed = 3}) {
  final random = math.Random(seed);
  final data = <Observation>[];
  var time = 0.0;
  for (var i = 0; i < n; i++) {
    time += 0.3 + 1.4 * random.nextDouble();
    final signal = 70 + 0.03 * time - 0.8 * math.cos(2 * math.pi * time / 7);
    data.add(Observation(time, signal + 0.2 * (random.nextDouble() - 0.5)));
  }
  return data;
}

double _sumOfSquares(List<double> values) {
  var total = 0.0;
  for (final v in values) {
    total += v * v;
  }
  return total;
}

void main() {
  group('recursive residuals under a flat prior', () {
    final data = _series(140);
    final components = [
      const LocalLinearTrend(processVariance: 4e-5),
      TrigonometricSeasonal(period: 7, harmonics: 2, processVariance: 3e-4),
    ];

    FilterResult run({bool exact = true}) => KalmanFilter(
          components,
          measurementVariance: 0.01,
          initialization:
              exact ? const ExactDiffuse() : const ApproximateDiffuse(),
        ).run(Timeline.merge(data, null), keepResiduals: true);

    test('cost exactly one observation per flat direction', () {
      final forward = run();
      expect(forward.diffuseDim, 6);
      expect(forward.standardisedResiduals!.length, data.length - 6);
      expect(forward.residualTimes!.first, data[6].time);
      expect(forward.residualTimes!.last, data.last.time);
    });

    test('add up to the weighted residual sum in the likelihood', () {
      // The restricted likelihood factorises as a product of one-step-ahead
      // predictive densities, so the recursive residuals must reproduce its
      // quadratic form exactly. Anything less than that and they are not the
      // residuals of the model whose likelihood was reported -- which is the
      // failure mode a plausible-looking but wrongly conditioned residual
      // would have.
      final forward = run();
      final sum = _sumOfSquares(forward.standardisedResiduals!);
      // They agree to 6.3e-11 relative rather than to the last bit, and the
      // reason is worth knowing: the batch form accumulates one information
      // matrix and solves once, the sequential form solves a differently
      // conditioned system at every observation. Same quantity, 140 times as
      // much arithmetic, and about five digits of headroom left over.
      expect((sum - forward.sumWeightedSquares).abs() / sum, lessThan(1e-9));
    });

    test('are not the in-sample residuals, which are shrunk', () {
      // The tempting shortcut is to subtract the smoothed fit from the data.
      // Those residuals are conditioned on everything, including what came
      // after, so they are pulled towards zero and under-report the noise. A
      // whiteness test reading them would be measuring the smoother rather
      // than the model.
      final forward = run();
      final smoothed =
          StructuralModel(components, measurementVariance: 0.01).smooth(data);
      var inSample = 0.0;
      for (var i = 0; i < data.length; i++) {
        final r = data[i].value - smoothed.level[i];
        inSample += r * r / 0.01;
      }
      expect(inSample,
          lessThan(0.9 * _sumOfSquares(forward.standardisedResiduals!)));
    });

    test('are the limit of a wide proper prior, at the rate 1/kappa', () {
      // The flat prior is the limit of a wide one, so the two constructions
      // have to meet -- and the *rate* is the sharper test, because an
      // implementation that is merely nearly right converges to the wrong
      // place rather than converging slowly. Measured worst gap over all 134
      // residuals:
      //
      //   kappa 1e4  7.232e-1     kappa 1e8   7.367e-5
      //   kappa 1e5  7.353e-2     kappa 1e9   7.377e-6
      //   kappa 1e6  7.365e-3     kappa 1e10  7.755e-7
      //   kappa 1e7  7.366e-4
      //
      // A clean decade per decade until 1e10, where rounding in a prior of
      // 1e8 against data of order 70 starts to show. The constant is the
      // signal scale divided by the prior, which is what it should be.
      final exact = run().standardisedResiduals!;
      var previous = double.infinity;
      for (final kappa in [1e4, 1e6, 1e8]) {
        final approximate = KalmanFilter(
          components,
          measurementVariance: 0.01,
          initialization: ApproximateDiffuse(variance: kappa),
        ).run(Timeline.merge(data, null), keepResiduals: true);

        var worst = 0.0;
        for (var i = 0; i < exact.length; i++) {
          final gap = (exact[i] - approximate.standardisedResiduals![i]).abs();
          if (gap > worst) worst = gap;
        }
        expect(worst, lessThan(previous / 90),
            reason:
                'kappa \$kappa should be a hundredfold closer than the last');
        previous = worst;
      }
      expect(previous, lessThan(1e-4));
    });
  });

  group('residuals of a model that is telling the truth', () {
    test('have mean zero and unit variance, near enough', () {
      // Simulated from exactly the model that is then fitted, so the only
      // thing left in the residuals is the noise that was put there.
      final random = math.Random(17);
      const processVariance = 2e-4;
      const measurementVariance = 0.04;
      final data = <Observation>[];
      var time = 0.0;
      var level = 60.0;
      var slope = 0.0;
      for (var i = 0; i < 1500; i++) {
        final dt = 0.5 + random.nextDouble();
        // Exact discretisation of the trend: slope diffuses, level integrates
        // it, and the two are correlated over the step.
        final sd = math.sqrt(processVariance);
        final a = _gaussian(random), b = _gaussian(random);
        level += slope * dt +
            sd *
                (dt * math.sqrt(dt) / math.sqrt(3) * a +
                    dt * math.sqrt(dt) / 2 * b);
        slope += sd * math.sqrt(dt) * b;
        time += dt;
        data.add(Observation(
            time, level + math.sqrt(measurementVariance) * _gaussian(random)));
      }

      final forward = forwardPass(
        [const LocalLinearTrend(processVariance: processVariance)],
        Timeline.merge(data, null),
        measurementVariance: measurementVariance,
        initialization: const ExactDiffuse(),
        keepResiduals: true,
      );

      final e = forward.standardisedResiduals!;
      var mean = 0.0;
      for (final r in e) {
        mean += r;
      }
      mean /= e.length;
      final variance = _sumOfSquares(e) / e.length - mean * mean;

      // Three standard errors on each: 1/sqrt(N) for the mean, sqrt(2/N) for
      // the variance, at N just under 1500.
      expect(mean.abs(), lessThan(3 / math.sqrt(e.length)));
      expect((variance - 1).abs(), lessThan(3 * math.sqrt(2 / e.length)));
    });
  });

  group('both engines', () {
    test('produce the same residuals', () {
      final data = _series(90, seed: 44);
      final component = const LocalLinearTrend(processVariance: 5e-4);
      final fast = forwardPass([component], Timeline.merge(data, null),
          measurementVariance: 0.02,
          initialization: const ExactDiffuse(),
          keepResiduals: true);
      final generic = KalmanFilter([component],
              measurementVariance: 0.02, initialization: const ExactDiffuse())
          .run(Timeline.merge(data, null), keepResiduals: true);

      expect(fast.standardisedResiduals!.length,
          generic.standardisedResiduals!.length);
      for (var i = 0; i < fast.standardisedResiduals!.length; i++) {
        expect(fast.standardisedResiduals![i],
            closeTo(generic.standardisedResiduals![i], 1e-12));
      }
    });
  });
}

/// Box-Muller, one draw at a time. Only a test needs this, so the wasted
/// second normal costs nothing worth caring about.
double _gaussian(math.Random random) =>
    math.sqrt(-2 * math.log(1 - random.nextDouble())) *
    math.cos(2 * math.pi * random.nextDouble());
