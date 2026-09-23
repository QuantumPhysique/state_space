import 'dart:math' as math;
import 'dart:typed_data';

import 'package:state_space/state_space.dart';
import 'package:test/test.dart';

import 'support/dense_reference.dart';

/// Irregularly spaced readings with a slow wobble in them, which is the shape
/// a Matérn component is for.
List<Observation> _series(int n, {int seed = 41}) {
  final random = math.Random(seed);
  final data = <Observation>[];
  var time = 0.0;
  for (var i = 0; i < n; i++) {
    time += 0.4 + 1.2 * random.nextDouble();
    data.add(Observation(time, 0.9 * math.sin(time / 6) + 0.3 * math.cos(time),
        relativeVariance: 0.6 + random.nextDouble()));
  }
  return data;
}

void main() {
  const variance = 0.8;
  const lengthScale = 5.0;
  const measurementVariance = 0.05;

  group('a stationary component against the dense Gaussian process', () {
    // The crown-jewel test, in its cleanest form. A model of nothing but a
    // Matérn component has no flat directions at all, so there is no diffuse
    // burn-in, no restricted likelihood and no least-squares solve to reason
    // about: the filter's log likelihood is the plain textbook
    // `-0.5 (y' C^-1 y + log|C| + N log 2pi)` and has to match it exactly.
    final data = _series(120);

    for (final order in MaternOrder.values) {
      test('${order.name}: likelihood, posterior mean and variance', () {
        final component =
            Matern(order: order, variance: variance, lengthScale: lengthScale);
        final model = StructuralModel([component],
            measurementVariance: measurementVariance);
        final posterior = model.smooth(data);

        final dense = densePosterior(
          data,
          [for (final o in data) o.time],
          [maternKernel(order, variance, lengthScale)],
          measurementVariance: measurementVariance,
        );

        expect(posterior.logMarginalLikelihood,
            closeTo(dense.logLikelihood, 1e-9));
        for (var i = 0; i < data.length; i++) {
          expect(posterior.mean[i], closeTo(dense.mean[0][i], 1e-10),
              reason: 'mean at $i');
          expect(posterior.variance[i], closeTo(dense.variance[0][i], 1e-11),
              reason: 'variance at $i');
        }
      });
    }

    test('and off the observation times, where interpolation actually happens',
        () {
      // A grid point between two readings is where a wrong stationary prior
      // would show up first: the filter has to fill the gap from the kernel
      // rather than from an observation.
      final component =
          Matern.fiveHalves(variance: variance, lengthScale: lengthScale);
      final grid =
          Float64List.fromList([for (var i = 0; i <= 40; i++) 3.17 + i * 1.83]);
      final posterior =
          StructuralModel([component], measurementVariance: measurementVariance)
              .smooth(data, grid: grid);
      final dense = densePosterior(
        data,
        grid,
        [maternKernel(MaternOrder.fiveHalves, variance, lengthScale)],
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

  group('a trend with a Matérn deviation on top of it', () {
    // The combination that motivated the component: a trend that should be
    // smooth and a correlated wobble that should not be part of it. The
    // decomposition is the thing being checked, because summing to the right
    // total is easy and splitting correctly is not.
    final data = _series(160, seed: 7);
    const processVariance = 3e-5;

    test('the restricted likelihood matches, with two flat directions', () {
      final model = StructuralModel([
        LocalLinearTrend(processVariance: processVariance),
        Matern.threeHalves(variance: variance, lengthScale: lengthScale),
      ], measurementVariance: measurementVariance);

      final dense = restrictedLikelihood(
        data,
        sumOf([
          splineKernel(processVariance),
          maternKernel(MaternOrder.threeHalves, variance, lengthScale),
        ]),
        trendBasis(),
        measurementVariance: measurementVariance,
      );

      expect(model.smooth(data).logMarginalLikelihood,
          closeTo(dense.logLikelihood, 1e-9));
    });

    test('and each component gets its own share', () {
      // A dense computation has no exact diffuse limit to take, so both sides
      // are run with the same wide proper prior instead of comparing an exact
      // one against an approximation of it. That is the established pattern
      // here: it tests the decomposition, which is what can go wrong, rather
      // than re-testing how well 1 / kappa approximates zero.
      const diffuseVariance = 50.0;
      const kappa = diffuseVariance * measurementVariance;
      final model = StructuralModel(
        [
          LocalLinearTrend(processVariance: processVariance),
          Matern.threeHalves(variance: variance, lengthScale: lengthScale),
        ],
        measurementVariance: measurementVariance,
        initialization: ApproximateDiffuse(variance: diffuseVariance),
      );
      final posterior = model.smooth(data);

      final dense = densePosterior(
        data,
        [for (final o in data) o.time],
        [
          withDiffusePrior(splineKernel(processVariance), trendBasis(), kappa),
          maternKernel(MaternOrder.threeHalves, variance, lengthScale),
        ],
        measurementVariance: measurementVariance,
      );

      for (var i = 0; i < data.length; i++) {
        expect(posterior.componentMean(0)[i], closeTo(dense.mean[0][i], 1e-9),
            reason: 'trend at $i');
        expect(posterior.componentMean(1)[i], closeTo(dense.mean[1][i], 1e-9),
            reason: 'wobble at $i');
        expect(posterior.componentVariance(1)[i],
            closeTo(dense.variance[1][i], 1e-10),
            reason: 'wobble variance at $i');
      }
    });
  });
}
