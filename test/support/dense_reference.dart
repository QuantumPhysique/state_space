/// Dense `O(N^3)` references for the models this package computes in `O(N)`.
///
/// Shared by the reference tests rather than duplicated into each, because the
/// value of these is that they are written from the equations in the README
/// and not from the recursion: two copies drifting apart would quietly weaken
/// both.
library;

import 'dart:math' as math;

import 'package:matrices/matrices.dart';
import 'package:state_space/state_space.dart';

/// The covariance function of a component, with time measured from the point
/// where the prior is stated.
typedef Kernel = double Function(double s, double t);

/// The functions of time the flat directions of a component load onto:
/// `H A(s) e_k` for each diffuse state `k`.
typedef Basis = List<double> Function(double s);

/// The cubic spline kernel of a [LocalLinearTrend], driving noise only.
Kernel splineKernel(double processVariance) => (s, t) {
  final m = math.min(s, t);
  return processVariance * (m * m * m / 3 + m * m * (s - t).abs() / 2);
};

/// The kernel of a [TrigonometricSeasonal], driving noise only.
///
/// The rotation is orthogonal and the noise isotropic, so `A(t-u) Q A(t'-u)'`
/// does not depend on `u`: the integral over the driving noise collapses to
/// Brownian motion times a comb of cosines. This closed form is not an
/// approximation of the state-space model, it is the same object written the
/// other way round, which is what makes it worth testing against.
Kernel seasonalKernel(double period, int harmonics, double processVariance) =>
    (s, t) {
      var comb = 0.0;
      for (var j = 1; j <= harmonics; j++) {
        comb += math.cos(2 * math.pi * j / period * (s - t));
      }
      return processVariance * math.min(s, t) * comb;
    };

/// The Matérn covariance function, written from the textbook formula.
///
/// Deliberately not `Matern.covariance`, and deliberately not derived from the
/// state-space matrices. The point of a reference is that it comes from a
/// different place than the thing it checks: this is the kernel as a Gaussian
/// process textbook states it, and the component has to agree with it after
/// going all the way round through a transition matrix, a stationary prior and
/// a recursion.
Kernel maternKernel(MaternOrder order, double variance, double lengthScale) {
  final nu = switch (order) {
    MaternOrder.oneHalf => 0.5,
    MaternOrder.threeHalves => 1.5,
    MaternOrder.fiveHalves => 2.5,
  };
  final scale = math.sqrt(2 * nu) / lengthScale;
  return (s, t) {
    final a = scale * (s - t).abs();
    final polynomial = switch (order) {
      MaternOrder.oneHalf => 1.0,
      MaternOrder.threeHalves => 1 + a,
      MaternOrder.fiveHalves => 1 + a + a * a / 3,
    };
    return variance * polynomial * math.exp(-a);
  };
}

/// The quasi-periodic kernel of a [StochasticCycle]: a cosine at the cycle
/// frequency, damped by `rho` per time unit.
Kernel cycleKernel(double period, double damping, double stationaryVariance) =>
    (s, t) {
      final lag = (s - t).abs();
      return stationaryVariance *
          math.pow(damping, lag) *
          math.cos(2 * math.pi * lag / period);
    };

Basis trendBasis() =>
    (s) => [1, s];

Basis seasonalBasis(double period, int harmonics) =>
    (s) => [
      for (var j = 1; j <= harmonics; j++) ...[
        math.cos(2 * math.pi * j / period * s),
        math.sin(2 * math.pi * j / period * s),
      ],
    ];

/// Adds the contribution of a wide proper prior on the flat directions, which
/// is what [ApproximateDiffuse] actually puts there.
Kernel withDiffusePrior(Kernel driving, Basis basis, double kappa) => (s, t) {
  final bs = basis(s);
  final bt = basis(t);
  var prior = 0.0;
  for (var k = 0; k < bs.length; k++) {
    prior += bs[k] * bt[k];
  }
  return driving(s, t) + kappa * prior;
};

Kernel sumOf(List<Kernel> parts) => (s, t) {
  var total = 0.0;
  for (final part in parts) {
    total += part(s, t);
  }
  return total;
};

/// Posterior of every component at [queries], the textbook `O(N^3)` way.
///
/// One Cholesky of `C = K + R` serves everything: the posterior mean of a
/// component is `k_i* C^-1 y` and its variance `k_i(q,q) - k_i*' C^-1 k_i*`,
/// with `k_i` that component's own kernel and `C` built from the sum. That
/// decomposition is the definition the filter's per-component output has to
/// match, and it is the part a trend/seasonal confounding bug hides behind.
({List<List<double>> mean, List<List<double>> variance, double logLikelihood})
densePosterior(
  List<Observation> data,
  List<double> queries,
  List<Kernel> kernels, {
  required double measurementVariance,
}) {
  final n = data.length;
  final origin = data.first.time;
  final times = [for (final o in data) o.time - origin];
  final shifted = [for (final q in queries) q - origin];
  final total = sumOf(kernels);
  // The sum is reported alongside the parts, so the caller can check that the
  // decomposition adds up as well as that each piece is right.
  final all = [...kernels, total];

  final c = <List<double>>[
    for (var i = 0; i < n; i++)
      [
        for (var j = 0; j < n; j++)
          total(times[i], times[j]) +
              (i == j ? data[i].relativeVariance * measurementVariance : 0.0),
      ],
  ];

  final rhs = <List<double>>[
    for (var i = 0; i < n; i++)
      [
        data[i].value,
        for (final kernel in all)
          for (final q in shifted) kernel(q, times[i]),
      ],
  ];

  final factor = Matrix64.fromRows(c).cholesky();
  final solved = factor.solve(Matrix64.fromRows(rhs));

  var quadratic = 0.0;
  var logDeterminant = 0.0;
  for (var i = 0; i < n; i++) {
    quadratic += data[i].value * solved(i, 0);
    logDeterminant += 2 * math.log(factor.lower(i, i));
  }

  final mean = <List<double>>[];
  final variance = <List<double>>[];
  for (var k = 0; k < all.length; k++) {
    final m = <double>[];
    final v = <double>[];
    for (var q = 0; q < queries.length; q++) {
      final column = 1 + k * queries.length + q;
      var centre = 0.0;
      var explained = 0.0;
      for (var i = 0; i < n; i++) {
        final cross = all[k](shifted[q], times[i]);
        centre += cross * solved(i, 0);
        explained += cross * solved(i, column);
      }
      m.add(centre);
      v.add(all[k](shifted[q], shifted[q]) - explained);
    }
    mean.add(m);
    variance.add(v);
  }
  return (
    mean: mean,
    variance: variance,
    logLikelihood:
        -0.5 * (quadratic + logDeterminant + n * math.log(2 * math.pi)),
  );
}

/// The restricted likelihood of a model whose flat directions load onto
/// [basis], for an arbitrary number of them.
///
/// ```text
/// -2 log L = (N - d) log 2pi + log|C| + log|B' C^-1 B| + y' P y
/// ```
///
/// The generalisation over the two-column version in
/// `dense_gp_reference_test.dart` is only that `d` is no longer two: a weekly
/// seasonal alone contributes six flat directions, and a trend with two
/// seasonals twelve.
({
  double logLikelihood,
  double logDeterminant,
  List<double> estimate,
  List<double> variance,
})
restrictedLikelihood(
  List<Observation> data,
  Kernel kernel,
  Basis basis, {
  required double measurementVariance,
}) {
  final n = data.length;
  final origin = data.first.time;
  final times = [for (final o in data) o.time - origin];
  final design = [for (final s in times) basis(s)];
  final d = design.first.length;

  final c = <List<double>>[
    for (var i = 0; i < n; i++)
      [
        for (var j = 0; j < n; j++)
          kernel(times[i], times[j]) +
              (i == j ? data[i].relativeVariance * measurementVariance : 0.0),
      ],
  ];
  final rhs = <List<double>>[
    for (var i = 0; i < n; i++) [data[i].value, ...design[i]],
  ];

  final factor = Matrix64.fromRows(c).cholesky();
  final solved = factor.solve(Matrix64.fromRows(rhs));

  var quadratic = 0.0;
  var logDeterminant = 0.0;
  final projected = List<double>.filled(d, 0);
  final information = [for (var k = 0; k < d; k++) List<double>.filled(d, 0)];
  for (var i = 0; i < n; i++) {
    quadratic += data[i].value * solved(i, 0);
    logDeterminant += 2 * math.log(factor.lower(i, i));
    for (var k = 0; k < d; k++) {
      projected[k] += design[i][k] * solved(i, 0);
      for (var l = 0; l < d; l++) {
        information[k][l] += design[i][k] * solved(i, l + 1);
      }
    }
  }

  final m = Matrix64.fromRows(information).cholesky();
  var informationLogDeterminant = 0.0;
  for (var k = 0; k < d; k++) {
    informationLogDeterminant += 2 * math.log(m.lower(k, k));
  }
  // One right-hand side for the estimate, then the identity, so that the
  // diagonal of `M^-1` comes back alongside it -- that is the posterior
  // variance of each flat direction under the flat prior, and it is what a
  // reported standard error has to match.
  final solvedSystem = m.solve(
    Matrix64.fromRows([
      for (var k = 0; k < d; k++)
        [projected[k], for (var l = 0; l < d; l++) k == l ? 1.0 : 0.0],
    ]),
  );
  var explained = 0.0;
  for (var k = 0; k < d; k++) {
    explained += projected[k] * solvedSystem(k, 0);
  }

  return (
    logLikelihood:
        -0.5 *
        ((n - d) * math.log(2 * math.pi) +
            logDeterminant +
            informationLogDeterminant +
            quadratic -
            explained),
    logDeterminant: informationLogDeterminant,
    estimate: [for (var k = 0; k < d; k++) solvedSystem(k, 0)],
    variance: [for (var k = 0; k < d; k++) solvedSystem(k, k + 1)],
  );
}
