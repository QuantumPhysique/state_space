import 'dart:math' as math;

import 'package:state_space/state_space.dart';
import 'package:test/test.dart';

/// Box-Muller, because a cycle simulated with uniform noise is not the process
/// whose period is being recovered and the difference matters here: the
/// likelihood in the period is sharp, and feeding it the wrong noise would
/// make a failure ambiguous.
class _Normal {
  _Normal(int seed) : _random = math.Random(seed);

  final math.Random _random;
  double? _spare;

  double next() {
    final held = _spare;
    if (held != null) {
      _spare = null;
      return held;
    }
    final radius = math.sqrt(-2 * math.log(1 - _random.nextDouble()));
    final angle = 2 * math.pi * _random.nextDouble();
    _spare = radius * math.sin(angle);
    return radius * math.cos(angle);
  }
}

/// Daily readings from a damped cycle observed with white noise, started from
/// its own stationary distribution so that the series is what the model says
/// it is from the first reading rather than after a burn-in.
List<Observation> _cycle({
  required int days,
  required double period,
  required double damping,
  required double amplitudeVariance,
  required double noise,
  int seed = 3,
}) {
  final normal = _Normal(seed);
  final scale = math.sqrt(amplitudeVariance);
  var a = scale * normal.next();
  var b = scale * normal.next();
  final decay = damping;
  final angle = 2 * math.pi / period;
  final kick = math.sqrt(amplitudeVariance * (1 - damping * damping));
  final data = <Observation>[];
  for (var day = 0; day < days; day++) {
    data.add(Observation(day.toDouble(), a + noise * normal.next()));
    final nextA = decay * (a * math.cos(angle) + b * math.sin(angle)) +
        kick * normal.next();
    final nextB = decay * (-a * math.sin(angle) + b * math.cos(angle)) +
        kick * normal.next();
    a = nextA;
    b = nextB;
  }
  return data;
}

void main() {
  group('recovering a period', () {
    // The starting values in the model handed to `fit` are deliberately wrong:
    // the coordinate scan sweeps the whole period bracket before anything
    // local runs, so where the caller started should make no difference at
    // all. That is the entire reason the scan exists.
    test('from a long series, against a bad starting guess', () {
      final data = _cycle(
        days: 500,
        period: 14,
        damping: 0.97,
        amplitudeVariance: 0.25,
        noise: 0.15,
      );

      final fitted = fit(
        StructuralModel([
          StochasticCycle(
              period: 3,
              damping: 0.5,
              stationaryVariance: 1.0,
              periodBounds: (lower: 3, upper: 120)),
        ]),
        data,
      );
      final cycle = fitted.model.components.first as StochasticCycle;

      expect(math.sqrt(cycle.stationaryVariance), closeTo(0.5, 0.2));
      expect(math.sqrt(fitted.measurementVariance), closeTo(0.15, 0.05));

      // A tolerance of one day, measured rather than guessed. Over sixteen
      // simulations of this process -- eight seeds at each of two damping
      // levels -- the estimate stayed between 13.72 and 14.72 days, with no
      // sign of bias either way. That spread is the process rather than the
      // fit: a cycle damped at 0.97 stays coherent for about two and a half
      // turns, so a single realisation of it genuinely does not repeat every
      // fourteen days.
      expect(cycle.period, closeTo(14, 1.0));

      // The same sixteen runs put four estimates outside the half-nat
      // interval, which is what a quarter of them landing beyond one standard
      // deviation looks like: the reported width is calibrated rather than
      // decorative. It is still a conditional slice, taken with the damping
      // held at its optimum and the two trading off against one another, so
      // what is checked here is that it comes out small -- not that it
      // covers.
      expect(fitted.plateauDecadesByParameter[2] / 2, lessThan(0.02),
          reason: 'five hundred days should pin the period to a few per cent');
    });

    test('and it does not settle for half or twice the truth', () {
      // The failure this guards against is specific. A cycle fitted at half
      // the period explains every second peak and sits on its own local
      // maximum, so a local search started at the wrong scale converges
      // confidently to the wrong answer. Checking the period is right is only
      // half the test; checking that the aliases are genuinely worse is what
      // says the scan found the global mode rather than got lucky.
      final data = _cycle(
        days: 500,
        period: 14,
        damping: 0.97,
        amplitudeVariance: 0.25,
        noise: 0.15,
      );
      final fitted = fit(
        StructuralModel([
          StochasticCycle(
              period: 3,
              damping: 0.5,
              stationaryVariance: 1.0,
              periodBounds: (lower: 3, upper: 120)),
        ]),
        data,
      );

      double atPeriod(double period) {
        final cycle = fitted.model.components.first as StochasticCycle;
        return StructuralModel([
          StochasticCycle(
              period: period,
              damping: cycle.damping,
              stationaryVariance: cycle.stationaryVariance),
        ], measurementVariance: fitted.measurementVariance)
            .smooth(data)
            .logMarginalLikelihood;
      }

      final best =
          atPeriod((fitted.model.components.first as StochasticCycle).period);
      expect(best, greaterThan(atPeriod(7) + 10));
      expect(best, greaterThan(atPeriod(28) + 10));
    });
  });

  group('how well the period is pinned down depends on how much there is', () {
    // The claim worth testing is monotone rather than absolute. How many turns
    // it takes before a period is worth quoting depends on how noisy the cycle
    // is, so a threshold would be a statement about this simulation and not
    // about the package. That more data narrows the reported interval is a
    // statement about the package, and it is the one a caller relies on when
    // reading `plateauDecadesByParameter` to decide whether to show a number.
    test('the reported interval narrows as the history grows', () {
      final full = _cycle(
        days: 400,
        period: 14,
        damping: 0.97,
        amplitudeVariance: 0.25,
        noise: 0.3,
        seed: 11,
      );

      double widthOver(int days) => fit(
            StructuralModel([
              StochasticCycle(
                  period: 14,
                  damping: 0.9,
                  stationaryVariance: 0.25,
                  periodBounds: (lower: 3, upper: 120)),
            ]),
            full.take(days).toList(),
          ).plateauDecadesByParameter[2];

      final short = widthOver(28);
      final long = widthOver(400);
      expect(long, lessThan(short / 2),
          reason: 'a month gave $short decades and a year gave $long');
    });
  });

  group('the shape parameters stay out of the variance machinery', () {
    test('a fitted period is not rescaled by the noise level', () {
      // The bug this guards against would be invisible: `fit` multiplies every
      // variance ratio back up by the fitted noise level at the end, and a
      // period that went through that step would come back plausible and
      // wrong. Data whose noise is nowhere near one makes the difference
      // large enough to see.
      final data = _cycle(
        days: 400,
        period: 20,
        damping: 0.98,
        amplitudeVariance: 100,
        noise: 6,
        seed: 5,
      );
      final fitted = fit(
        StructuralModel([
          StochasticCycle(
              period: 5,
              damping: 0.5,
              stationaryVariance: 1.0,
              periodBounds: (lower: 3, upper: 120)),
        ]),
        data,
      );
      final cycle = fitted.model.components.first as StochasticCycle;
      expect(fitted.measurementVariance, greaterThan(10));
      expect(cycle.period, closeTo(20, 1.0));
      expect(cycle.damping, closeTo(0.98, 0.03));
    });

    test('and the ratios report NaN where there is no ratio to report', () {
      final data = _cycle(
        days: 200,
        period: 12,
        damping: 0.95,
        amplitudeVariance: 0.3,
        noise: 0.2,
      );
      final fitted = fit(
        StructuralModel([
          StochasticCycle(period: 12, damping: 0.95, stationaryVariance: 0.3),
        ]),
        data,
      );
      expect(fitted.varianceRatios[0].isFinite, isTrue);
      expect(fitted.varianceRatios[1].isNaN, isTrue);
      expect(fitted.varianceRatios[2].isNaN, isTrue);
      expect(fitted.varianceRatio, fitted.varianceRatios[0]);
    });
  });

  group('validation', () {
    test('damping outside (0, 1) is refused', () {
      expect(
          () => StochasticCycle(period: 7, damping: 1, stationaryVariance: 1),
          throwsArgumentError);
      expect(
          () => StochasticCycle(period: 7, damping: 0, stationaryVariance: 1),
          throwsArgumentError);
    });

    test('a round trip through the parameter vector changes nothing', () {
      final cycle =
          StochasticCycle(period: 28, damping: 0.94, stationaryVariance: 1.7);
      final copy = cycle.withParameters(cycle.parameters) as StochasticCycle;
      expect(copy.period, closeTo(28, 1e-12));
      expect(copy.damping, closeTo(0.94, 1e-12));
      expect(copy.stationaryVariance, closeTo(1.7, 1e-12));
    });
  });
}
