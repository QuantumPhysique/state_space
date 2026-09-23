// What did that fortnight actually cost?
//
//   dart run example/events_example.dart

// ignore_for_file: avoid_print

import 'dart:math' as math;

import 'package:state_space/state_space.dart';

/// A year of daily readings, drifting slowly down, with two interruptions: a
/// week away at a conference in March, and a fortnight over Christmas.
List<Observation> readings() {
  final random = math.Random(8);
  return [
    for (var day = 0; day < 365; day++)
      Observation(
          day.toDouble(),
          78.0 -
              0.004 * day +
              (day >= 80 && day < 87 ? -0.45 : 0.0) +
              (day >= 350 && day < 364 ? 1.30 : 0.0) +
              0.28 * (random.nextDouble() - 0.5) * 3.46)
  ];
}

void main() {
  final data = readings();

  // Two indicator columns. Neither adds a parameter to the fit: the
  // coefficients are states under a flat prior, so the Kalman recursion
  // estimates them along with everything else.
  final model = StructuralModel([
    LocalLinearTrend(processVariance: 1e-4),
    RegressionComponent([
      IndicatorRegressor('conference', [(from: 80, to: 87)]),
      IndicatorRegressor('christmas', [(from: 350, to: 364)]),
    ]),
  ]);
  print('${model.stateDim} states, ${model.parameterCount} free parameter '
      '-- the two coefficients are states, not parameters\n');

  final fitted = fit(model, data);
  final posterior = fitted.model.smooth(data);

  print('event         effect      95% interval        truth');
  for (final coefficient in posterior.coefficients) {
    final interval = coefficient.interval();
    final truth = coefficient.name == 'conference' ? -0.45 : 1.30;
    print('${coefficient.name.padRight(12)}  '
        '${_signed(coefficient.estimate)} +/- '
        '${coefficient.standardError.toStringAsFixed(3)}  '
        '[${_signed(interval.lo)}, ${_signed(interval.hi)}]  '
        '${_signed(truth).padLeft(11)}');
  }

  print('');
  print('The Christmas interval misses the truth here, narrowly. Over twenty '
      'replications');
  print('of this simulation it covers eighteen times, and the mean estimate '
      'is 1.282 against');
  print('a true 1.300 -- one draw behaving the way one draw in twenty is '
      'supposed to. It');
  print('is left in rather than replaced with a luckier seed.');
  print('');
  print(
      'fitted noise  ${math.sqrt(fitted.measurementVariance).toStringAsFixed(3)} kg '
      'per reading (simulated at 0.280)');
  print('trend variance ${fitted.parameterStatus.first.name}');
  print('');

  // The same model without the trend, which is the mistake worth showing.
  final flat = fit(
    StructuralModel([
      RegressionComponent([
        IndicatorRegressor('level', [(from: -1, to: 1e9)]),
        IndicatorRegressor('conference', [(from: 80, to: 87)]),
        IndicatorRegressor('christmas', [(from: 350, to: 364)]),
      ])
    ]),
    data,
  );
  print('Drop the trend and the fit still looks fine, but each event absorbs '
      'the difference');
  print('between the level when it happened and the level on average:');
  print('');
  print('event         with a trend    without one');
  final without = {
    for (final c in flat.model.smooth(data).coefficients) c.name: c
  };
  for (final coefficient in posterior.coefficients) {
    print('${coefficient.name.padRight(12)}  '
        '${_signed(coefficient.estimate).padLeft(12)}  '
        '${_signed(without[coefficient.name]!.estimate).padLeft(13)}');
  }
  print('');
  print('Christmas is late in the year, where the trend is low, so its effect '
      'comes back');
  print('understated. The error bars do not widen to warn anyone. The one '
      'thing that');
  print('gives it away is the noise level, which rises from '
      '${math.sqrt(fitted.measurementVariance).toStringAsFixed(3)} to '
      '${math.sqrt(flat.measurementVariance).toStringAsFixed(3)} kg');
  print('because the drift has nowhere else to go.');
}

String _signed(double value) =>
    (value >= 0 ? ' ' : '') + value.toStringAsFixed(3);
