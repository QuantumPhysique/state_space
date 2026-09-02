import 'dart:math' as math;
import 'dart:typed_data';

import 'package:matrices/matrices.dart';
import 'package:state_space/src/engine/kalman.dart';
import 'package:state_space/src/engine/rts.dart';
import 'package:state_space/src/engine/timeline.dart';
import 'package:state_space/state_space.dart';
import 'package:test/test.dart';

/// The Gaussian process the state-space model of a [LocalLinearTrend] implies.
///
/// Two pieces. The diffuse prior on the initial state contributes
/// `kappa (1 + s s')`, since `H A(s) = (1, s)` carries the initial level and
/// slope forward. The driving noise contributes the cubic spline kernel
///
/// ```text
/// sigma^2 (m^3 / 3 + m^2 |s - s'| / 2),   m = min(s, s')
/// ```
///
/// with `s` measured from the first time point, where the prior is stated.
double _kernel(double s, double t,
    {required double processVariance, required double kappa}) {
  final m = math.min(s, t);
  final spline = m * m * m / 3 + m * m * (s - t).abs() / 2;
  return kappa * (1 + s * t) + processVariance * spline;
}

/// Posterior mean and variance of the signal at [queries], the textbook way.
///
/// `E[f_* | y] = k_* C^-1 y` and `Var[f_* | y] = k_** - k_*' C^-1 k_*`, with
/// `C = K + R`. Cubic in the number of observations, quadratic in memory, and
/// completely uninterested in whether the sampling is regular.
({List<double> mean, List<double> variance, double logLikelihood})
    _densePosterior(
  List<Observation> data,
  List<double> queries, {
  required double processVariance,
  required double measurementVariance,
  required double diffuseVariance,
}) {
  final n = data.length;
  final origin = data.first.time;
  final kappa = diffuseVariance * measurementVariance;
  double k(double a, double b) =>
      _kernel(a, b, processVariance: processVariance, kappa: kappa);

  final c = <List<double>>[
    for (var i = 0; i < n; i++)
      [
        for (var j = 0; j < n; j++)
          k(data[i].time - origin, data[j].time - origin) +
              (i == j ? data[i].relativeVariance * measurementVariance : 0.0)
      ]
  ];

  // One right-hand side per thing we want: the data, then each query's
  // covariance with the data.
  final rhs = <List<double>>[
    for (var i = 0; i < n; i++)
      [
        data[i].value,
        for (final q in queries) k(q - origin, data[i].time - origin),
      ]
  ];

  final factor = Matrix64.fromRows(c).cholesky();
  final solved = factor.solve(Matrix64.fromRows(rhs));

  var quadratic = 0.0;
  for (var i = 0; i < n; i++) {
    quadratic += data[i].value * solved(i, 0);
  }
  // The package's own logAbsDeterminant takes the determinant first, which
  // overflows long before N = 100 here; the Cholesky diagonal does not.
  var logDeterminant = 0.0;
  for (var i = 0; i < n; i++) {
    logDeterminant += 2 * math.log(factor.lower(i, i));
  }

  final mean = <double>[];
  final variance = <double>[];
  for (var q = 0; q < queries.length; q++) {
    var m = 0.0;
    var explained = 0.0;
    for (var i = 0; i < n; i++) {
      final kStar = k(queries[q] - origin, data[i].time - origin);
      m += kStar * solved(i, 0);
      explained += kStar * solved(i, q + 1);
    }
    mean.add(m);
    variance.add(k(queries[q] - origin, queries[q] - origin) - explained);
  }

  return (
    mean: mean,
    variance: variance,
    logLikelihood:
        -0.5 * (quadratic + logDeterminant + n * math.log(2 * math.pi)),
  );
}

/// The restricted likelihood of the same model, computed densely.
///
/// Exact diffuse initialisation says the trend starts from an unknown point
/// with a flat prior. Densely, that is the linear model `y = B d + f + e` with
/// `f` a spline-kernel Gaussian process, `B = [1, s]` carrying the unknown
/// level and slope, and `d` integrated out under a flat prior. What survives
/// the integration is the restricted likelihood
///
/// ```text
/// -2 log L = (N - d) log 2pi + log|C| + log|B' C^-1 B| + y' P y
/// ```
///
/// with `C = K + R` and `P = C^-1 - C^-1 B (B' C^-1 B)^-1 B' C^-1`. The middle
/// determinant is the term the augmented filter accumulates as `log|M|`, and
/// it is what makes the diffuse likelihood the restricted likelihood rather
/// than something a constant away from it: getting it wrong shifts every
/// likelihood by an amount nothing else in the package would notice. What it
/// does *not* do is make likelihoods comparable across different `B`; see
/// `comparability_test.dart`.
({
  double logLikelihood,
  double logDeterminant,
  List<double> estimate,
  List<List<double>> spread,
}) _restrictedLikelihood(
  List<Observation> data, {
  required double processVariance,
  required double measurementVariance,
}) {
  final n = data.length;
  final origin = data.first.time;
  double spline(double a, double b) {
    final m = math.min(a, b);
    return processVariance * (m * m * m / 3 + m * m * (a - b).abs() / 2);
  }

  final times = [for (final o in data) o.time - origin];
  final covariance = <List<double>>[
    for (var i = 0; i < n; i++)
      [
        for (var j = 0; j < n; j++)
          spline(times[i], times[j]) +
              (i == j ? data[i].relativeVariance * measurementVariance : 0.0)
      ]
  ];

  // Solve against the data and against both columns of the design at once.
  final rhs = <List<double>>[
    for (var i = 0; i < n; i++) [data[i].value, 1.0, times[i]]
  ];
  final factor = Matrix64.fromRows(covariance).cholesky();
  final solved = factor.solve(Matrix64.fromRows(rhs));

  var logDeterminant = 0.0;
  for (var i = 0; i < n; i++) {
    logDeterminant += 2 * math.log(factor.lower(i, i));
  }

  final design = [
    for (var i = 0; i < n; i++) [1.0, times[i]]
  ];
  var quadratic = 0.0;
  final projected = [0.0, 0.0];
  final information = [
    [0.0, 0.0],
    [0.0, 0.0]
  ];
  for (var i = 0; i < n; i++) {
    quadratic += data[i].value * solved(i, 0);
    for (var k = 0; k < 2; k++) {
      projected[k] += design[i][k] * solved(i, 0);
      for (var l = 0; l < 2; l++) {
        information[k][l] += design[i][k] * solved(i, l + 1);
      }
    }
  }

  final determinant = information[0][0] * information[1][1] -
      information[0][1] * information[1][0];
  final inverse = [
    [information[1][1] / determinant, -information[0][1] / determinant],
    [-information[1][0] / determinant, information[0][0] / determinant],
  ];
  var explained = 0.0;
  final estimate = [0.0, 0.0];
  for (var k = 0; k < 2; k++) {
    for (var l = 0; l < 2; l++) {
      explained += projected[k] * inverse[k][l] * projected[l];
      estimate[k] += inverse[k][l] * projected[l];
    }
  }

  return (
    logLikelihood: -0.5 *
        ((n - 2) * math.log(2 * math.pi) +
            logDeterminant +
            math.log(determinant) +
            quadratic -
            explained),
    logDeterminant: math.log(determinant),
    estimate: estimate,
    spread: inverse,
  );
}

List<Observation> _irregularSeries(int n, {int seed = 7}) {
  final random = math.Random(seed);
  final data = <Observation>[];
  var time = 0.0;
  for (var i = 0; i < n; i++) {
    time += 0.2 + 3 * random.nextDouble();
    final trend = 80 + 0.02 * time - 3 * math.sin(time / 25);
    data.add(Observation(time, trend + 0.4 * (random.nextDouble() - 0.5),
        relativeVariance: 0.5 + random.nextDouble()));
  }
  return data;
}

void main() {
  group('exact diffuse initialisation against the dense restricted likelihood',
      () {
    test('the augmented filter computes the restricted likelihood', () {
      const processVariance = 3e-4;
      const measurementVariance = 0.04;
      final data = _irregularSeries(100);

      final forward = KalmanFilter(
        [const LocalLinearTrend(processVariance: processVariance)],
        measurementVariance: measurementVariance,
        initialization: const ExactDiffuse(),
      ).run(Timeline.merge(data, null));

      final dense = _restrictedLikelihood(data,
          processVariance: processVariance,
          measurementVariance: measurementVariance);

      expect(forward.logLikelihood, closeTo(dense.logLikelihood, 1e-9));
      expect(forward.usedObservations, data.length - 2);
    });

    test('the smoothed initial state is the least-squares estimate', () {
      // At the first time point the state *is* the pair of flat directions,
      // so the smoothed level and slope there must equal the generalised
      // least-squares estimate the dense form computes directly. This is the
      // sharpest single check on the augmentation: it involves the whole
      // series through C, and it is exactly where an implementation that
      // fumbles the flat directions goes wrong first.
      const processVariance = 5e-4;
      const measurementVariance = 0.05;
      final data = _irregularSeries(90, seed: 31);

      final dense = _restrictedLikelihood(data,
          processVariance: processVariance,
          measurementVariance: measurementVariance);
      final smoothed = StructuralModel.localLinearTrend(
        processVariance: processVariance,
        measurementVariance: measurementVariance,
      ).smooth(data);

      expect(smoothed.level[0], closeTo(dense.estimate[0], 1e-9));
      expect(smoothed.slope![0], closeTo(dense.estimate[1], 1e-11));

      // And its covariance is the estimate's covariance, all four entries.
      final timeline = Timeline.merge(data, null);
      final components = [
        const LocalLinearTrend(processVariance: processVariance)
      ];
      final forward = KalmanFilter(components,
              measurementVariance: measurementVariance,
              initialization: const ExactDiffuse())
          .run(timeline, keepHistory: true);
      RtsSmoother(components)
        ..smoothInPlace(timeline, forward)
        ..combineDiffuse(forward);

      for (var i = 0; i < 2; i++) {
        for (var j = 0; j < 2; j++) {
          expect(forward.stateCovariance![i * 2 + j],
              closeTo(dense.spread[i][j], 1e-12),
              reason: 'P^s[0][$i][$j]');
        }
      }
    });

    test('and the log determinant it subtracts is the one REML subtracts', () {
      // Pinning the pieces separately, not only the total. A sign error in
      // log|M| and a compensating one in the residual sum would cancel in the
      // likelihood and survive the test above; they cannot survive this one.
      const processVariance = 2e-3;
      const measurementVariance = 0.09;
      final data = _irregularSeries(60, seed: 12);

      final dense = _restrictedLikelihood(data,
          processVariance: processVariance,
          measurementVariance: measurementVariance);
      final forward = KalmanFilter(
        [const LocalLinearTrend(processVariance: processVariance)],
        measurementVariance: measurementVariance,
        initialization: const ExactDiffuse(),
      ).run(Timeline.merge(data, null));

      expect(forward.diffuseLogDeterminant,
          closeTo(dense.logDeterminant, 1e-9 * dense.logDeterminant.abs()));
      expect(forward.logLikelihood, closeTo(dense.logLikelihood, 1e-9));
    });
  });

  group('the linear-time recursion computes the cubic-time posterior', () {
    // A modest diffuse variance keeps the dense covariance well conditioned,
    // so any disagreement is the algorithm's fault rather than the reference
    // implementation's. The default 1e6 is checked separately, loosely.
    //
    // Means are compared to 1e-9 and variances to 1e-9 absolute. The looser
    // figure for the variances is the reference's limitation, not the
    // filter's: `k_** - k_*' C^-1 k_*` subtracts two numbers of order 1e6 to
    // get one of order 1e-3, which throws away nine digits before the two
    // implementations are even compared.
    const processVariance = 3e-4;
    const measurementVariance = 0.04;
    const diffuseVariance = 50.0;

    test('smoothed mean and variance at the observation times', () {
      final data = _irregularSeries(120);
      final model = StructuralModel.localLinearTrend(
        processVariance: processVariance,
        measurementVariance: measurementVariance,
        initialization: ApproximateDiffuse(variance: diffuseVariance),
      );

      final fast = model.smooth(data);
      final slow = _densePosterior(
        data,
        [for (final o in data) o.time],
        processVariance: processVariance,
        measurementVariance: measurementVariance,
        diffuseVariance: diffuseVariance,
      );

      for (var i = 0; i < data.length; i++) {
        expect(fast.level[i], closeTo(slow.mean[i], 1e-9),
            reason: 'mean at index $i');
        expect(fast.levelVariance[i], closeTo(slow.variance[i], 1e-9),
            reason: 'variance at index $i');
      }
    });

    test('and at grid points that are nowhere near an observation', () {
      final data = _irregularSeries(80);
      final span = data.last.time - data.first.time;
      final grid = Float64List.fromList([
        for (var i = 0; i < 40; i++) data.first.time + span * i / 39,
      ]);

      final model = StructuralModel.localLinearTrend(
        processVariance: processVariance,
        measurementVariance: measurementVariance,
        initialization: ApproximateDiffuse(variance: diffuseVariance),
      );

      final fast = model.smooth(data, grid: grid);
      final slow = _densePosterior(
        data,
        grid.toList(),
        processVariance: processVariance,
        measurementVariance: measurementVariance,
        diffuseVariance: diffuseVariance,
      );

      for (var i = 0; i < grid.length; i++) {
        expect(fast.level[i], closeTo(slow.mean[i], 1e-9),
            reason: 'mean at grid point $i');
        expect(fast.levelVariance[i], closeTo(slow.variance[i], 1e-9),
            reason: 'variance at grid point $i');
      }
    });

    test('log marginal likelihood', () {
      final data = _irregularSeries(120);
      final model = StructuralModel.localLinearTrend(
        processVariance: processVariance,
        measurementVariance: measurementVariance,
        initialization: ApproximateDiffuse(variance: diffuseVariance),
      );

      // The prior here is proper, so every innovation carries information and
      // none of them should be burned.
      final forward = KalmanFilter(
        model.components,
        measurementVariance: measurementVariance,
        initialization: ApproximateDiffuse(variance: diffuseVariance),
        burnIn: 0,
      ).run(Timeline.merge(data, null));

      final slow = _densePosterior(
        data,
        const [],
        processVariance: processVariance,
        measurementVariance: measurementVariance,
        diffuseVariance: diffuseVariance,
      );

      expect(forward.logLikelihood, closeTo(slow.logLikelihood, 1e-9));
    });

    test('still agrees under the default, nearly diffuse prior', () {
      final data = _irregularSeries(60);
      const defaultPrior = ApproximateDiffuse();
      final model = StructuralModel.localLinearTrend(
        processVariance: processVariance,
        measurementVariance: measurementVariance,
        initialization: defaultPrior,
      );

      final fast = model.smooth(data);
      final slow = _densePosterior(
        data,
        [for (final o in data) o.time],
        processVariance: processVariance,
        measurementVariance: measurementVariance,
        diffuseVariance: defaultPrior.variance,
      );

      // A prior variance of 1e6 times the noise pushes the dense covariance to
      // a condition number around 1e10, so the reference itself is only good
      // to a few digits here. The filter is the better-conditioned of the two.
      for (var i = 0; i < data.length; i++) {
        expect(fast.level[i], closeTo(slow.mean[i], 1e-6));
      }
    });
  });
}
