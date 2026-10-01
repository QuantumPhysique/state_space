import 'dart:math' as math;

import 'package:state_space/authoring.dart';
import 'package:test/test.dart';

import 'support/dense_reference.dart';

/// Irregular readings of a course that drifts, turns, and has a hole in it:
/// the shape the time scale is for.
List<Observation> _series(int n, {int seed = 11}) {
  final random = math.Random(seed);
  final data = <Observation>[];
  var time = 0.0;
  for (var i = 0; i < n; i++) {
    time += i == n ~/ 2 ? 40 : 0.4 + 1.6 * random.nextDouble();
    data.add(
      Observation(
        time,
        80 -
            0.05 * time +
            1.5 * math.sin(time / 25) +
            0.6 * random.nextDouble(),
        relativeVariance: i % 7 == 0 ? 2.0 : 1.0,
      ),
    );
  }
  return data;
}

/// A path drawn from the model itself, one reading per time unit.
List<Observation> _simulated(
  int n, {
  required double processVariance,
  required double timeScale,
  required double measurementVariance,
  required int seed,
}) {
  final random = math.Random(seed);
  double gaussian() =>
      math.sqrt(-2 * math.log(1 - random.nextDouble())) *
      math.cos(2 * math.pi * random.nextDouble());
  final component = DampedLinearTrend(
    processVariance: processVariance,
    timeScale: timeScale,
  );
  final a = MatrixBlock.dense(2, 2);
  final q = MatrixBlock.dense(2, 2);
  component.transition(1, a);
  component.processNoise(1, q);
  final l00 = math.sqrt(q.at(0, 0));
  final l10 = q.at(1, 0) / l00;
  final l11 = math.sqrt(q.at(1, 1) - l10 * l10);
  var level = 0.0;
  var slope = gaussian() * math.sqrt(component.stationarySlopeVariance);
  return [
    for (var i = 0; i < n; i++)
      () {
        final reading = Observation(
          i.toDouble(),
          level + math.sqrt(measurementVariance) * gaussian(),
        );
        final z0 = gaussian(), z1 = gaussian();
        final nextLevel = level + a.at(0, 1) * slope + l00 * z0;
        slope = a.at(1, 1) * slope + l10 * z0 + l11 * z1;
        level = nextLevel;
        return reading;
      }(),
  ];
}

void main() {
  const processVariance = 4e-3;
  const timeScale = 12.0;
  const measurementVariance = 0.09;
  final component = DampedLinearTrend(
    processVariance: processVariance,
    timeScale: timeScale,
  );

  group('against the dense Gaussian process', () {
    final data = _series(90);
    final last = data.last.time;
    // Readings, points inside the hole, and a forecast well past the end.
    final queries = [
      for (final o in data) o.time,
      for (var k = 1; k < 8; k++) data[44].time + 5.0 * k,
      for (var k = 1; k <= 6; k++) last + 7.0 * k,
    ]..sort();
    final kernel = dampedTrendKernel(processVariance, timeScale);

    test('the restricted likelihood matches, with one flat direction', () {
      final model = StructuralModel([
        component,
      ], measurementVariance: measurementVariance);
      final dense = restrictedLikelihood(
        data,
        kernel,
        levelBasis(),
        measurementVariance: measurementVariance,
      );
      final posterior = model.smooth(data);
      expect(
        posterior.logMarginalLikelihood,
        closeTo(dense.logLikelihood, 1e-9),
      );
      expect(fit(model, data).diffuseDimension, 1);
    });

    test(
      'so do the posterior mean and variance, under exact initialisation',
      () {
        final posterior = StructuralModel(
          [component],
          measurementVariance: measurementVariance,
        ).smooth(data, grid: queries);
        final dense = denseExactPosterior(
          data,
          queries,
          kernel,
          levelBasis(),
          measurementVariance: measurementVariance,
        );
        for (var i = 0; i < queries.length; i++) {
          expect(
            posterior.mean[i],
            closeTo(dense.mean[i], 1e-9),
            reason: 'mean at ${queries[i]}',
          );
          expect(
            posterior.variance[i],
            closeTo(dense.variance[i], 1e-9 * math.max(1, dense.variance[i])),
            reason: 'variance at ${queries[i]}',
          );
        }
      },
    );

    test('and under an approximate prior, which takes the fast path', () {
      const diffuseVariance = 50.0;
      final posterior = StructuralModel(
        [component],
        measurementVariance: measurementVariance,
        initialization: ApproximateDiffuse(variance: diffuseVariance),
      ).smooth(data, grid: queries);
      final dense = densePosterior(data, queries, [
        withDiffusePrior(
          kernel,
          levelBasis(),
          diffuseVariance * measurementVariance,
        ),
      ], measurementVariance: measurementVariance);
      for (var i = 0; i < queries.length; i++) {
        expect(
          posterior.mean[i],
          closeTo(dense.mean[0][i], 1e-9),
          reason: 'mean at ${queries[i]}',
        );
        expect(
          posterior.variance[i],
          closeTo(
            dense.variance[0][i],
            1e-9 * math.max(1, dense.variance[0][i]),
          ),
          reason: 'variance at ${queries[i]}',
        );
      }
    });
  });

  group('limits', () {
    final data = _series(60);

    double worstDifference(List<double> a, List<double> b) {
      var worst = 0.0;
      for (var i = 0; i < a.length; i++) {
        worst = math.max(worst, (a[i] - b[i]).abs());
      }
      return worst;
    }

    test('a long time scale is a LocalLinearTrend', () {
      final reference = StructuralModel.localLinearTrend(
        processVariance: processVariance,
        measurementVariance: measurementVariance,
      ).smooth(data);

      ({double mean, double slope}) errorAt(double longTimeScale) {
        final damped = StructuralModel.dampedLinearTrend(
          processVariance: processVariance,
          timeScale: longTimeScale,
          measurementVariance: measurementVariance,
        ).smooth(data);
        return (
          mean: worstDifference(damped.mean, reference.mean),
          slope: worstDifference(damped.trendSlope!, reference.trendSlope!),
        );
      }

      final near = errorAt(1e5);
      final nearer = errorAt(1e7);
      expect(near.mean, lessThan(1e-3));
      expect(near.slope, lessThan(1e-4));
      expect(nearer.mean, lessThan(near.mean / 30));
      expect(nearer.slope, lessThan(near.slope / 30));
    });

    test('a short one is a LocalLevel of variance q tau^2, likelihood and '
        'all', () {
      const levelVariance = 0.02;
      final reference = StructuralModel.localLevel(
        processVariance: levelVariance,
        measurementVariance: measurementVariance,
      ).smooth(data);

      ({double mean, double likelihood}) errorAt(double shortTimeScale) {
        final damped = StructuralModel.dampedLinearTrend(
          processVariance: levelVariance / (shortTimeScale * shortTimeScale),
          timeScale: shortTimeScale,
          measurementVariance: measurementVariance,
        ).smooth(data);
        return (
          mean: worstDifference(damped.mean, reference.mean),
          likelihood:
              (damped.logMarginalLikelihood - reference.logMarginalLikelihood)
                  .abs(),
        );
      }

      final near = errorAt(1e-3);
      final nearer = errorAt(1e-5);
      expect(near.mean, lessThan(1e-2));
      expect(near.likelihood, lessThan(1e-1));
      expect(nearer.mean, lessThan(near.mean / 30));
      expect(nearer.likelihood, lessThan(near.likelihood / 30));
    });
  });

  group('one flat direction', () {
    test('a single reading gives a flat line at it', () {
      final model = StructuralModel.dampedLinearTrend(
        processVariance: processVariance,
        timeScale: timeScale,
        measurementVariance: measurementVariance,
      );
      final posterior = model.smooth(
        const [Observation(3, 80)],
        grid: [3, 10, 100],
      );
      for (var i = 0; i < 3; i++) {
        expect(posterior.mean[i], closeTo(80, 1e-12));
        expect(posterior.trendSlope![i], closeTo(0, 1e-15));
      }
      expect(
        posterior.trendSlopeVariance![0],
        closeTo(component.stationarySlopeVariance, 1e-15),
      );
      expect(
        () => StructuralModel.localLinearTrend(
          processVariance: processVariance,
        ).smooth(const [Observation(3, 80)]),
        throwsA(isA<UnderdeterminedModelException>()),
      );
    });

    test('a forecast levels off at level + tau * slope, its variance growing '
        'linearly', () {
      final data = _series(60);
      final last = data.last.time;
      const far = 100 * timeScale;
      final posterior = StructuralModel(
        [component],
        measurementVariance: measurementVariance,
      ).smooth(data, grid: [last, last + far, last + 2 * far]);
      final settled = posterior.mean[0] + timeScale * posterior.trendSlope![0];
      expect(posterior.mean[1], closeTo(settled, 1e-9));
      expect(posterior.mean[2], closeTo(settled, 1e-9));
      // Var(level(T)) = q tau^2 T + const once T is many time scales.
      expect(
        posterior.variance[2] - posterior.variance[1],
        closeTo(processVariance * timeScale * timeScale * far, 1e-6),
      );
    });

    test('a grid point before the first reading changes nothing', () {
      // The slope starts from its stationary distribution and the level is
      // flat, so where the prior is stated does not matter.
      final data = _series(40);
      final times = [for (final o in data) o.time];
      final model = StructuralModel([
        component,
      ], measurementVariance: measurementVariance);
      final plain = model.smooth(data, grid: times);
      final early = model.smooth(data, grid: [data.first.time - 50, ...times]);
      for (var i = 0; i < times.length; i++) {
        expect(early.mean[i + 1], closeTo(plain.mean[i], 1e-9));
        expect(early.variance[i + 1], closeTo(plain.variance[i], 1e-9));
      }
      expect(
        early.logMarginalLikelihood,
        closeTo(plain.logMarginalLikelihood, 1e-9),
      );
    });
  });

  group('invariances', () {
    final data = _series(50);
    final model = StructuralModel([
      component,
    ], measurementVariance: measurementVariance);

    test('reversing time reverses the curve and flips the slope', () {
      final forward = model.smooth(data);
      final backward = model.smooth([
        for (final o in data.reversed)
          Observation(-o.time, o.value, relativeVariance: o.relativeVariance),
      ]);
      final n = data.length;
      for (var i = 0; i < n; i++) {
        expect(backward.mean[n - 1 - i], closeTo(forward.mean[i], 1e-9));
        expect(
          backward.trendSlope![n - 1 - i],
          closeTo(-forward.trendSlope![i], 1e-9),
        );
        expect(
          backward.variance[n - 1 - i],
          closeTo(forward.variance[i], 1e-9),
        );
      }
    });

    test('scaling every variance scales the posterior variance only, so the '
        'scale can be estimated', () {
      const c = 7.0;
      final scaled = StructuralModel([
        DampedLinearTrend(
          processVariance: c * processVariance,
          timeScale: timeScale,
        ),
      ], measurementVariance: c * measurementVariance);
      final a = model.smooth(data);
      final b = scaled.smooth(data);
      for (var i = 0; i < data.length; i++) {
        expect(b.mean[i], closeTo(a.mean[i], 1e-9));
        expect(b.variance[i], closeTo(c * a.variance[i], 1e-9));
      }

      final estimated = model.withEstimatedScale(data);
      final found = estimated.components.single as DampedLinearTrend;
      expect(found.timeScale, timeScale);
      expect(
        found.processVariance / estimated.measurementVariance,
        closeTo(processVariance / measurementVariance, 1e-12),
      );
    });
  });

  test('a steady slope is recovered between readings and pulled towards '
      'zero at the last one, as documented', () {
    // Noise-free, so the answer is the expected answer on noisy readings.
    final line = [
      for (var d = 0; d < 120; d++) Observation(d.toDouble(), 80 - d / 7),
    ];
    for (final bandwidth in [4.0, 7.0]) {
      for (final (bandwidths, documented) in [
        (5.0, 0.77),
        (10.0, 0.88),
        (15.0, 0.92),
        (30.0, 0.96),
      ]) {
        final posterior = StructuralModel([
          DampedLinearTrend(
            processVariance: math.pow(bandwidth, -4).toDouble(),
            timeScale: bandwidths * bandwidth,
          ),
        ]).smooth(line, grid: [60, 119]);
        final reason = 'bandwidth $bandwidth, time scale $bandwidths of them';
        expect(-7 * posterior.trendSlope![0], closeTo(1, 1e-3), reason: reason);
        expect(
          -7 * posterior.trendSlope![1],
          closeTo(documented, 0.01),
          reason: reason,
        );
      }
    }
  });

  test('the process noise keeps its digits for any gap against the time '
      'scale', () {
    // Q(0,0) / (sigma^2 tau^3) is integral_0^u (1 - e^-s)^2 ds; summed by
    // Simpson's rule here, it has no cancellation in it at any u.
    double integral(double u) {
      const n = 20000;
      final h = u / n;
      double f(double s) => math.pow(1 - math.exp(-s), 2).toDouble();
      var sum = f(0) + f(u);
      for (var i = 1; i < n; i++) {
        sum += (i.isOdd ? 4 : 2) * f(i * h);
      }
      return sum * h / 3;
    }

    final unit = DampedLinearTrend(processVariance: 1, timeScale: 1);
    final q = MatrixBlock.dense(2, 2);
    for (final u in [1e-3, 0.1, 0.4999, 0.5, 0.5001, 2.0, 30.0]) {
      unit.processNoise(u, q);
      expect(
        q.at(0, 0),
        closeTo(integral(u), 1e-12 * integral(u)),
        reason: '$u',
      );
    }
    // Far below the series threshold the leading term is the spline's, and
    // the matrix stays a covariance.
    for (final dt in [1e-12, 1e-9, 1e-6]) {
      component.processNoise(dt, q);
      expect(
        q.at(0, 0),
        closeTo(processVariance * dt * dt * dt / 3, 1e-6 * q.at(0, 0)),
        reason: '$dt',
      );
      expect(q.at(0, 0) * q.at(1, 1) - q.at(0, 1) * q.at(1, 0), greaterThan(0));
    }
    // And a time scale far beyond every gap is the LocalLinearTrend, A and Q.
    final spline = MatrixBlock.dense(2, 2);
    final a = MatrixBlock.dense(2, 2);
    final long = DampedLinearTrend(processVariance: 0.3, timeScale: 1e12);
    for (final dt in [1e-6, 1.0, 100.0]) {
      long.processNoise(dt, q);
      long.transition(dt, a);
      LocalLinearTrend(processVariance: 0.3).processNoise(dt, spline);
      expect(a.at(0, 1), closeTo(dt, 1e-9 * dt));
      for (final (i, j) in [(0, 0), (0, 1), (1, 1)]) {
        expect(
          q.at(i, j),
          closeTo(spline.at(i, j), 1e-9 * spline.at(i, j)),
          reason: 'Q($i, $j) at $dt',
        );
      }
    }
  });

  test('wanderOver is the path spread of its own kernel', () {
    // (1/T) integral k(t, t) dt - (1/T^2) double integral k(s, t) ds dt, by
    // Simpson's rule on the dense kernel.
    final kernel = dampedTrendKernel(processVariance, timeScale);
    double simpsonWeight(int i, int n) =>
        i == 0 || i == n ? 1 : (i.isOdd ? 4 : 2);
    for (final span in [0.1 * timeScale, timeScale, 10 * timeScale]) {
      const n = 200;
      final h = span / n;
      var diagonal = 0.0;
      var square = 0.0;
      for (var i = 0; i <= n; i++) {
        final wi = simpsonWeight(i, n);
        diagonal += wi * kernel(i * h, i * h);
        for (var j = 0; j <= n; j++) {
          square += wi * simpsonWeight(j, n) * kernel(i * h, j * h);
        }
      }
      diagonal *= h / 3;
      square *= h * h / 9;
      final expected = math.sqrt(diagonal / span - square / (span * span));
      expect(
        component.wanderOver(span),
        closeTo(expected, 1e-6 * expected),
        reason: 'span $span',
      );
    }
    expect(component.wanderOver(0), 0);
  });

  test('fit recovers both parameters, the time scale as a shape parameter', () {
    expect(component.parameterSpecs, [
      const VarianceParameter(),
      isA<ShapeParameter>().having((s) => s.label, 'label', 'time scale'),
    ]);
    for (var seed = 0; seed < 3; seed++) {
      final data = _simulated(
        1000,
        processVariance: 1e-3,
        timeScale: 10,
        measurementVariance: 0.04,
        seed: seed,
      );
      final fitted = fit(
        StructuralModel.dampedLinearTrend(processVariance: 1e-2, timeScale: 5),
        data,
      );
      final found = fitted.model.components.single as DampedLinearTrend;
      expect(fitted.warnings, isEmpty, reason: 'seed $seed');
      expect(fitted.parameterStatus, [
        ParameterStatus.determined,
        ParameterStatus.determined,
      ]);
      expect(found.timeScale, inInclusiveRange(10 / 1.5, 10 * 1.5));
      expect(found.processVariance, inInclusiveRange(1e-3 / 1.5, 1e-3 * 1.5));
    }
  });

  test('a random walk drives the time scale to the sampling floor, and the fit '
      'says so', () {
    final random = math.Random(5);
    var level = 70.0;
    final walk = [
      for (var d = 0; d < 400; d++)
        () {
          level += 0.2 * (random.nextDouble() - 0.5);
          return Observation(d.toDouble(), level + 0.3 * random.nextDouble());
        }(),
    ];
    final fitted = fit(
      StructuralModel.dampedLinearTrend(processVariance: 1e-2, timeScale: 5),
      walk,
    );
    final found = fitted.model.components.single as DampedLinearTrend;
    expect(found.timeScale, closeTo(1, 1e-3));
    expect(fitted.parameterStatus[1], ParameterStatus.beyondBracket);
    expect(
      fitted.warnings,
      contains(
        contains('determine only a combination of it with the variance'),
      ),
    );
    expect(
      () => DampedLinearTrend(
        processVariance: 1,
        timeScale: 0.1,
        timeScaleBounds: (lower: 0.01, upper: 0.5),
      ).parameterSpecsAt(resolution: 1),
      throwsA(isA<UnderdeterminedModelException>()),
    );
  });

  test('arguments are checked', () {
    for (final bad in [0.0, -1.0, double.nan, double.infinity]) {
      expect(
        () => DampedLinearTrend(processVariance: 1, timeScale: bad),
        throwsArgumentError,
      );
    }
    expect(
      () => DampedLinearTrend(
        processVariance: 1,
        timeScale: 1,
        timeScaleBounds: (lower: 2, upper: 1),
      ),
      throwsArgumentError,
    );
  });
}
