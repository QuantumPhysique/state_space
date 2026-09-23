// What a model costs as components are added: smoothing, fitting, and a warm
// refit. Produces the tables under "Models with many components" and "What a
// fit costs" in doc/validation.md.
//
//   dart compile exe benchmark/components_benchmark.dart -o /tmp/components
//   /tmp/components
//
// Run it AOT-compiled, as for scaling_benchmark.dart.

// ignore_for_file: avoid_print

import 'dart:math' as math;

import 'package:state_space/state_space.dart';

/// Daily readings of a slowly bending trend with a weekly pattern.
List<Observation> diary(int days, {int seed = 7}) {
  final random = math.Random(seed);
  double gaussian() =>
      math.sqrt(-2 * math.log(1 - random.nextDouble())) *
      math.cos(2 * math.pi * random.nextDouble());
  var level = 80.0, slope = -0.01;
  var water = 0.0;
  return [
    for (var d = 0; d < days; d++)
      () {
        slope += 0.002 * gaussian();
        level += slope;
        water = 0.8 * water + 0.2 * gaussian();
        return Observation(
          d.toDouble(),
          level +
              0.2 * math.sin(2 * math.pi * d / 7) +
              water +
              0.1 * gaussian(),
        );
      }(),
  ];
}

/// Median of [samples] timed runs after a warmup, in milliseconds.
double medianMs(void Function() body, {int warmup = 2, int samples = 5}) {
  for (var i = 0; i < warmup; i++) {
    body();
  }
  final times = <double>[];
  for (var s = 0; s < samples; s++) {
    final watch = Stopwatch()..start();
    body();
    times.add(watch.elapsedMicroseconds / 1000);
  }
  times.sort();
  return times[times.length ~/ 2];
}

LocalLinearTrend trend() => LocalLinearTrend(processVariance: 1e-4);
TrigonometricSeasonal weekly() =>
    TrigonometricSeasonal(period: 7, harmonics: 2, processVariance: 1e-6);
Matern wobble() => Matern.oneHalf(variance: 0.04, lengthScale: 4);

RegressionComponent indicators(int count) => RegressionComponent([
  for (var k = 0; k < count; k++)
    IndicatorRegressor('event $k', [(from: 30.0 * k + 5, to: 30.0 * k + 12)]),
]);

void main() {
  final long = diary(20000);
  print('smooth, 20 000 daily points');
  for (final (name, model) in [
    ('LocalLevel', StructuralModel([LocalLevel(processVariance: 1e-2)])),
    ('LocalLinearTrend', StructuralModel([trend()])),
    (
      'trend + Matern 5/2',
      StructuralModel([
        trend(),
        Matern.fiveHalves(variance: 0.04, lengthScale: 4),
      ]),
    ),
    ('trend + weekly seasonal', StructuralModel([trend(), weekly()])),
    ('trend + 1 indicator', StructuralModel([trend(), indicators(1)])),
    ('trend + 20 indicators', StructuralModel([trend(), indicators(20)])),
  ]) {
    final ms = medianMs(() => model.smooth(long), samples: 3);
    print(
      '  ${name.padRight(26)}${model.stateDim.toString().padLeft(3)} states'
      '${ms.toStringAsFixed(1).padLeft(10)} ms',
    );
  }

  print('');
  print('logLikelihood, 730 daily points');
  final twoYears = diary(730);
  for (final count in [0, 1, 20]) {
    final model = StructuralModel([
      trend(),
      weekly(),
      if (count > 0) indicators(count),
    ]);
    final ms = medianMs(() => model.logLikelihood(twoYears), samples: 7);
    print(
      '  trend + weekly + ${count.toString().padLeft(2)} indicators'
      '${ms.toStringAsFixed(3).padLeft(12)} ms',
    );
  }

  print('');
  print('fit: forward passes and time');
  for (final years in [1, 3, 5]) {
    final data = diary(365 * years);
    for (final (name, build) in [
      ('trend', () => StructuralModel([trend()])),
      ('trend + weekly', () => StructuralModel([trend(), weekly()])),
      (
        'trend + weekly + Matern',
        () => StructuralModel([trend(), weekly(), wobble()]),
      ),
    ]) {
      late FitResult fitted;
      final ms = medianMs(
        () => fitted = fit(build(), data),
        warmup: 1,
        samples: 3,
      );
      print(
        '  ${years}y ${name.padRight(26)}${fitted.evaluations.toString().padLeft(5)} passes'
        '${ms.toStringAsFixed(0).padLeft(8)} ms',
      );
    }
  }

  print('');
  print('warm refit after one more reading, trend + weekly + Matern, 730 days');
  final longer = [...twoYears, Observation(730, twoYears.last.value)];
  final first = fit(StructuralModel([trend(), weekly(), wobble()]), twoYears);
  late FitResult cold, warm;
  final coldMs = medianMs(
    () => cold = fit(StructuralModel([trend(), weekly(), wobble()]), longer),
    warmup: 1,
    samples: 3,
  );
  final warmMs = medianMs(
    () =>
        warm = fit(first.model, longer, start: SearchStart.previousParameters),
    warmup: 1,
    samples: 3,
  );
  print(
    '  cold ${cold.evaluations} passes ${coldMs.toStringAsFixed(0)} ms, '
    'log likelihood ${cold.logMarginalLikelihood.toStringAsFixed(4)}',
  );
  print(
    '  warm ${warm.evaluations} passes ${warmMs.toStringAsFixed(0)} ms, '
    'log likelihood ${warm.logMarginalLikelihood.toStringAsFixed(4)}',
  );
}
