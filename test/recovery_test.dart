import 'dart:math' as math;
import 'dart:typed_data';

import 'package:state_space/state_space.dart';
import 'package:test/test.dart';

/// Box-Muller, because `dart:math` only offers uniforms.
class Gaussian {
  Gaussian(int seed) : _uniform = math.Random(seed);
  final math.Random _uniform;
  double? _spare;

  double next() {
    final spare = _spare;
    if (spare != null) {
      _spare = null;
      return spare;
    }
    final radius = math.sqrt(-2 * math.log(1 - _uniform.nextDouble()));
    final angle = 2 * math.pi * _uniform.nextDouble();
    _spare = radius * math.sin(angle);
    return radius * math.cos(angle);
  }
}

/// Draws a path of the integrated Wiener process and observes it with noise.
///
/// The exact discretisation has a closed-form Cholesky factor,
///
/// ```text
/// L(dt) = sigma [[dt^1.5 / sqrt(3),        0     ],
///                [sqrt(3 dt) / 2,   sqrt(dt) / 2 ]]
/// ```
///
/// so the simulation is as exact as the filter it is testing — no
/// fine-grained Euler stepping, and irregular gaps for free.
List<Observation> simulate({
  required int count,
  required double processVariance,
  required double measurementVariance,
  required int seed,
  double meanGap = 1.0,
  bool irregular = true,
}) {
  final noise = Gaussian(seed);
  final gaps = math.Random(seed * 31 + 7);
  final sigma = math.sqrt(processVariance);
  final observationSd = math.sqrt(measurementVariance);

  var level = 80.0;
  var rate = 0.0;
  var time = 0.0;
  final data = <Observation>[];

  for (var i = 0; i < count; i++) {
    if (i > 0) {
      final dt =
          irregular ? meanGap * (0.25 + 1.5 * gaps.nextDouble()) : meanGap;
      final z0 = noise.next();
      final z1 = noise.next();
      level += rate * dt + sigma * (dt * math.sqrt(dt) / math.sqrt(3) * z0);
      rate += sigma * (math.sqrt(3 * dt) / 2 * z0 + math.sqrt(dt) / 2 * z1);
      time += dt;
    }
    data.add(Observation(time, level + observationSd * noise.next()));
  }
  return data;
}

void main() {
  group('parameter recovery', () {
    const processVariance = 4e-4;
    const measurementVariance = 0.09;
    const trueRatio = processVariance / measurementVariance;

    test('a long series recovers the variance ratio and the noise level', () {
      for (final seed in [1, 2, 3]) {
        final data = simulate(
          count: 3000,
          processVariance: processVariance,
          measurementVariance: measurementVariance,
          seed: seed,
        );
        final result = fit(
          StructuralModel.localLinearTrend(processVariance: 1),
          data,
        );

        expect(result.converged, isTrue);
        expect(result.atBracketEdge, isFalse);
        expect(result.isFlat, isFalse);
        expect(math.log(result.varianceRatio / trueRatio).abs(), lessThan(0.7),
            reason: 'seed $seed recovered ${result.varianceRatio}');
        expect(result.measurementVariance,
            closeTo(measurementVariance, 0.15 * measurementVariance),
            reason: 'seed $seed');
      }
    });

    test('and recovers it better than a short one does', () {
      // The estimator is consistent, so the median error over a handful of
      // seeds has to fall as the series grows. Medians rather than means
      // because a single unlucky seed at N = 200 can be off by a factor of
      // ten and would drown any signal.
      double medianError(int count) {
        final errors = <double>[
          for (var seed = 1; seed <= 7; seed++)
            math
                .log(fit(
                      StructuralModel.localLinearTrend(processVariance: 1),
                      simulate(
                        count: count,
                        processVariance: processVariance,
                        measurementVariance: measurementVariance,
                        seed: seed * 17,
                      ),
                    ).varianceRatio /
                    trueRatio)
                .abs()
        ]..sort();
        return errors[errors.length ~/ 2];
      }

      final short = medianError(150);
      final long = medianError(2400);
      expect(long, lessThan(short));
    });

    test('the fitted model reproduces the likelihood it reports', () {
      final data = simulate(
        count: 400,
        processVariance: processVariance,
        measurementVariance: measurementVariance,
        seed: 5,
      );
      final result = fit(
        StructuralModel.localLinearTrend(processVariance: 1),
        data,
      );

      // The profile likelihood is the real likelihood at the concentrated
      // variance, so refitting nothing and simply running the forward pass on
      // the returned model has to give the same number back.
      expect(result.model.logLikelihood(data),
          closeTo(result.logMarginalLikelihood, 1e-9));
    });
  });

  group('diagnostics', () {
    test(
        'white noise drives the trend to the floor and says the profile is '
        'flat', () {
      // There is no trend to find, so every sufficiently small process
      // variance fits equally well and the likelihood has nothing to say.
      final noise = Gaussian(42);
      final data = [
        for (var i = 0; i < 300; i++)
          Observation(i.toDouble(), 50 + 0.5 * noise.next())
      ];
      final result = fit(
        StructuralModel.localLinearTrend(processVariance: 1),
        data,
      );
      expect(result.isFlat, isTrue);
      expect(result.plateauDecades, greaterThan(2));
      expect(result.measurementVariance, closeTo(0.25, 0.05));
    });

    test('refuses a model whose components cannot be told apart', () {
      // A trend and a level both supply a level, so no amount of data
      // separates them. Until 0.3 this was refused by fit() itself, which
      // handled only one parameter; now the search runs and the engine
      // refuses it for the real reason, on the first evaluation.
      expect(
        () => fit(
            StructuralModel([
              const LocalLinearTrend(processVariance: 1e-3),
              const LocalLevel(processVariance: 1e-3),
            ]),
            simulate(
              count: 50,
              processVariance: 1e-3,
              measurementVariance: 0.1,
              seed: 1,
            )),
        throwsA(isA<StateError>()
            .having((e) => e.message, 'message', contains('the same signal'))),
      );
    });

    test('refuses to fit fewer observations than there are diffuse states', () {
      expect(
        () => fit(StructuralModel.localLinearTrend(processVariance: 1),
            [const Observation(0, 1), const Observation(1, 2)]),
        throwsArgumentError,
      );
    });
  });

  test('smoothing the fitted model tracks the simulated truth', () {
    final data = simulate(
      count: 600,
      processVariance: 1e-3,
      measurementVariance: 0.04,
      seed: 9,
    );
    final result = fit(
      StructuralModel.localLinearTrend(processVariance: 1),
      data,
    );
    final grid = Float64List.fromList([for (final o in data) o.time]);
    final posterior = result.model.smooth(data, grid: grid);

    // The smoothed curve should sit inside its own predictive band for
    // roughly the nominal fraction of the readings. Anything far off means
    // the variances are wrong even though the curve looks fine.
    var inside = 0;
    for (var i = 0; i < data.length; i++) {
      final band = posterior.predictiveInterval(i);
      if (data[i].value >= band.lo && data[i].value <= band.hi) inside++;
    }
    expect(inside / data.length, closeTo(0.95, 0.04));
  });
}
