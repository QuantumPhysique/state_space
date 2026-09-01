import 'dart:math' as math;
import 'dart:typed_data';

import 'package:state_space/state_space.dart';
import 'package:test/test.dart';

/// Reads a component's block out into nested lists, which is the only shape
/// the little dense helpers below want to work in.
List<List<double>> _matrix(void Function(MatrixBlock) fill, int n) {
  final block = MatrixBlock.dense(n, n);
  block.fill(double.nan);
  fill(block);
  return [
    for (var i = 0; i < n; i++) [for (var j = 0; j < n; j++) block.at(i, j)]
  ];
}

List<List<double>> _product(List<List<double>> a, List<List<double>> b) => [
      for (var i = 0; i < a.length; i++)
        [
          for (var j = 0; j < b[0].length; j++)
            [for (var k = 0; k < b.length; k++) a[i][k] * b[k][j]]
                .reduce((x, y) => x + y)
        ]
    ];

List<List<double>> _transpose(List<List<double>> a) => [
      for (var j = 0; j < a[0].length; j++)
        [for (var i = 0; i < a.length; i++) a[i][j]]
    ];

void _expectClose(List<List<double>> actual, List<List<double>> expected,
    double tolerance, String reason) {
  for (var i = 0; i < expected.length; i++) {
    for (var j = 0; j < expected[i].length; j++) {
      expect(actual[i][j], closeTo(expected[i][j], tolerance),
          reason: '$reason at ($i, $j)');
    }
  }
}

void main() {
  const orders = MaternOrder.values;
  const variance = 2.3;
  const lengthScale = 4.5;

  Matern build(MaternOrder order) =>
      Matern(order: order, variance: variance, lengthScale: lengthScale);

  group('the state-space form reproduces the kernel', () {
    // The whole claim of the component: propagating the stationary covariance
    // over a gap and reading the observed entry gives back the Matern
    // covariance function, which is the object anyone coming from the Gaussian
    // process literature actually recognises.
    for (final order in orders) {
      test('${order.name}: H A(tau) P_inf H\' equals k(tau)', () {
        final component = build(order);
        final n = component.stateDim;
        final prior = _matrix((block) {
          block.fill(0);
          component.properPrior(Float64List(n), block);
        }, n);
        for (final lag in [0.0, 0.3, 1.0, 4.5, 12.0, 40.0]) {
          final a = _matrix((block) => component.transition(lag, block), n);
          final propagated = _product(a, prior);
          expect(propagated[0][0], closeTo(component.covariance(lag), 1e-13),
              reason: 'lag $lag');
        }
      });
    }
  });

  group('the stationary covariance is stationary', () {
    // A P A' + Q = P is the defining property, and it is worth checking
    // separately from the kernel: the kernel test only reads one entry, and
    // this one fails if any entry of either matrix is wrong.
    for (final order in orders) {
      test('${order.name}: A P A\' + Q returns P', () {
        final component = build(order);
        final n = component.stateDim;
        final prior = _matrix((block) {
          block.fill(0);
          component.properPrior(Float64List(n), block);
        }, n);
        for (final gap in [0.05, 0.7, 3.0, 11.0]) {
          final a = _matrix((block) => component.transition(gap, block), n);
          final q = _matrix((block) => component.processNoise(gap, block), n);
          final propagated = _product(_product(a, prior), _transpose(a));
          final restored = [
            for (var i = 0; i < n; i++)
              [for (var j = 0; j < n; j++) propagated[i][j] + q[i][j]]
          ];
          _expectClose(restored, prior, 1e-13 * variance, 'gap $gap');
        }
      });
    }
  });

  group('the transition is a semigroup', () {
    // A(s + t) = A(s) A(t) is what makes an irregular series a non-issue: the
    // answer must not depend on whether a gap is crossed in one step or two.
    for (final order in orders) {
      test('${order.name}: A(s + t) equals A(s) A(t)', () {
        final component = build(order);
        final n = component.stateDim;
        const s = 1.7, t = 2.9;
        final combined =
            _matrix((block) => component.transition(s + t, block), n);
        final stepwise = _product(
          _matrix((block) => component.transition(s, block), n),
          _matrix((block) => component.transition(t, block), n),
        );
        _expectClose(stepwise, combined, 1e-13, 'split gap');
      });
    }
  });

  group('the Ornstein-Uhlenbeck case against its closed form', () {
    // One state, one line of algebra, no room for a sign error to hide: the
    // other two orders are checked against structural identities, so it is
    // worth having one checked against the textbook formula directly.
    test('A and Q are the AR(1) coefficients', () {
      final component = Matern.oneHalf(variance: 1.5, lengthScale: 3.0);
      for (final gap in [0.25, 1.0, 6.0]) {
        final rho = math.exp(-gap / 3.0);
        expect(_matrix((b) => component.transition(gap, b), 1)[0][0],
            closeTo(rho, 1e-15));
        expect(_matrix((b) => component.processNoise(gap, b), 1)[0][0],
            closeTo(1.5 * (1 - rho * rho), 1e-15));
      }
    });
  });

  group('degenerate gaps', () {
    for (final order in orders) {
      test('${order.name}: a zero gap moves nothing', () {
        final component = build(order);
        final n = component.stateDim;
        final a = _matrix((block) => component.transition(0, block), n);
        final q = _matrix((block) => component.processNoise(0, block), n);
        for (var i = 0; i < n; i++) {
          for (var j = 0; j < n; j++) {
            expect(a[i][j], i == j ? 1.0 : 0.0, reason: 'A($i, $j)');
            expect(q[i][j], 0.0, reason: 'Q($i, $j)');
          }
        }
      });
    }
  });

  group('parameters', () {
    test('a round trip through the parameter vector changes nothing', () {
      for (final order in orders) {
        final component = build(order);
        final copy = component.withParameters(component.parameters) as Matern;
        expect(copy.variance, closeTo(variance, 1e-14));
        expect(copy.lengthScale, closeTo(lengthScale, 1e-14));
        expect(copy.order, order);
      }
    });

    test('the length scale is a shape parameter, not a variance', () {
      final specs = build(MaternOrder.threeHalves).parameterSpecs;
      expect(specs[0], isA<VarianceParameter>());
      expect(specs[1], isA<ShapeParameter>());
    });

    test('a non-positive length scale is refused', () {
      expect(() => Matern.threeHalves(variance: 1, lengthScale: 0),
          throwsArgumentError);
      expect(() => Matern.threeHalves(variance: 0, lengthScale: 1),
          throwsArgumentError);
    });
  });

  group('fitting', () {
    // Box-Muller: a Matern process simulated with uniform kicks is not the
    // process whose length scale is being recovered.
    double Function() normals(int seed) {
      final random = math.Random(seed);
      double? spare;
      return () {
        final held = spare;
        if (held != null) {
          spare = null;
          return held;
        }
        final radius = math.sqrt(-2 * math.log(1 - random.nextDouble()));
        final angle = 2 * math.pi * random.nextDouble();
        spare = radius * math.sin(angle);
        return radius * math.cos(angle);
      };
    }

    /// An Ornstein-Uhlenbeck path sampled daily and read with white noise,
    /// started from its own stationary distribution.
    List<Observation> ornsteinUhlenbeck({
      required int days,
      required double sd,
      required double lengthScale,
      required double noise,
      int seed = 2,
    }) {
      final next = normals(seed);
      final persistence = math.exp(-1 / lengthScale);
      final kick = sd * math.sqrt(1 - persistence * persistence);
      var state = sd * next();
      final data = <Observation>[];
      for (var day = 0; day < days; day++) {
        data.add(Observation(day.toDouble(), state + noise * next()));
        state = persistence * state + kick * next();
      }
      return data;
    }

    test('recovers the marginal variance and the length scale', () {
      final data =
          ornsteinUhlenbeck(days: 900, sd: 0.6, lengthScale: 8, noise: 0.2);
      final fitted = fit(
          StructuralModel([Matern.oneHalf(variance: 1, lengthScale: 1)]), data);
      final component = fitted.model.components.first as Matern;

      expect(math.sqrt(component.variance), closeTo(0.6, 0.15));
      expect(component.lengthScale, closeTo(8, 2.5));
      expect(math.sqrt(fitted.measurementVariance), closeTo(0.2, 0.05));
      expect(fitted.parameterStatus, everyElement(ParameterStatus.determined));
    });

    test('and separates a trend from the correlated wobble on top of it', () {
      // The arrangement the component exists for. A trend-only model has
      // nowhere to put autocorrelated deviation, so it either chases it -- and
      // reports a trend that is not a trend -- or absorbs it into the noise
      // and reports readings four times worse than they are. Giving the
      // wobble its own home is what lets both come out right.
      final next = normals(17);
      final persistence = math.exp(-1 / 5);
      final kick = 0.35 * math.sqrt(1 - persistence * persistence);
      var wobble = 0.35 * next();
      final data = <Observation>[];
      for (var day = 0; day < 500; day++) {
        final trend = 80 - 0.012 * day;
        data.add(Observation(day.toDouble(), trend + wobble + 0.1 * next()));
        wobble = persistence * wobble + kick * next();
      }

      final fitted = fit(
        StructuralModel([
          const LocalLinearTrend(processVariance: 1e-4),
          Matern.oneHalf(variance: 0.1, lengthScale: 3),
        ]),
        data,
      );
      final wobbleComponent = fitted.model.components[1] as Matern;
      final posterior = fitted.model.smooth(data);

      // The trend is the straight line it was built from, to within a tenth
      // of a kilogram over five hundred days.
      expect(posterior.componentMean(0).first, closeTo(80, 0.15));
      expect(posterior.componentMean(0).last, closeTo(80 - 0.012 * 499, 0.15));
      expect(math.sqrt(wobbleComponent.variance), closeTo(0.35, 0.12));

      // And the reading error comes back as the reading error rather than as
      // the reading error plus everything the trend could not explain.
      expect(math.sqrt(fitted.measurementVariance), closeTo(0.1, 0.03));
    });

    test('a trend alone reports the same readings as twice as noisy', () {
      final next = normals(17);
      final persistence = math.exp(-1 / 5);
      final kick = 0.35 * math.sqrt(1 - persistence * persistence);
      var wobble = 0.35 * next();
      final data = <Observation>[];
      for (var day = 0; day < 500; day++) {
        final trend = 80 - 0.012 * day;
        data.add(Observation(day.toDouble(), trend + wobble + 0.1 * next()));
        wobble = persistence * wobble + kick * next();
      }

      final trendOnly =
          fit(StructuralModel.localLinearTrend(processVariance: 1e-4), data);
      // 0.194 kg against a true 0.1. Not the whole of the wobble, because a
      // spline flexible enough to chase a five-day length scale does chase
      // part of it -- which is the other half of the damage, and shows up as
      // a trend that wiggles rather than as noise.
      expect(math.sqrt(trendOnly.measurementVariance), greaterThan(0.17));

      // And says so: the residuals of the misspecified model are visibly
      // autocorrelated, which is the diagnostic that points at the missing
      // component in the first place.
      final plain = trendOnly.model.diagnose(data);
      expect(
          plain.ljungBox(lags: 10, fittedParameters: 1).pValue, lessThan(0.01));
      final better = fit(
        StructuralModel([
          const LocalLinearTrend(processVariance: 1e-4),
          Matern.oneHalf(variance: 0.1, lengthScale: 3),
        ]),
        data,
      ).model.diagnose(data);
      expect(better.ljungBox(lags: 10, fittedParameters: 3).pValue,
          greaterThan(0.05));
    });
  });

  group('wandering', () {
    test('over a long window it approaches the marginal standard deviation',
        () {
      // The process forgets where it has been, so over many length scales its
      // spread about its own average is the whole of it.
      final component = Matern.threeHalves(variance: 4, lengthScale: 1);
      expect(component.wanderOver(400), closeTo(2.0, 0.02));
    });

    test('over a short window it barely moves', () {
      final component = Matern.threeHalves(variance: 4, lengthScale: 100);
      expect(component.wanderOver(1), lessThan(0.05));
    });
  });
}
