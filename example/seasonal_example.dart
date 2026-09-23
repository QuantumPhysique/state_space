// Separating a weekly pattern from a trend, and checking that the model
// deserves to be believed.
//
//   dart run example/seasonal_example.dart

// ignore_for_file: avoid_print

import 'dart:math' as math;

import 'package:state_space/state_space.dart';

/// Twelve weeks of readings, most mornings but not all, carrying a slow
/// downward trend and a weekend that undoes some of the week.
List<Observation> readings() {
  final random = math.Random(12);
  final data = <Observation>[];
  for (var day = 0; day < 84; day++) {
    // Roughly one morning in six is missed, which is what actually happens.
    if (random.nextDouble() < 0.17) continue;
    final trend = 82.0 - 0.035 * day;
    final weekly = 0.45 * math.cos(2 * math.pi * day / 7) +
        0.18 * math.sin(4 * math.pi * day / 7);
    data.add(Observation(
        day.toDouble(), trend + weekly + 0.3 * (random.nextDouble() - 0.5)));
  }
  return data;
}

void main() {
  final data = readings();
  print('${data.length} readings over 84 days\n');

  // A trend alone. It will fit the series perfectly well and be wrong about
  // what the series is doing.
  final trendOnly = fit(
    StructuralModel.localLinearTrend(processVariance: 1),
    data,
  );
  final plain = trendOnly.model.diagnose(data);
  final plainTest = plain.ljungBox(lags: 14, fittedParameters: 1);

  // The same data with somewhere for the weekly pattern to go.
  final withSeasonal = fit(
    StructuralModel([
      LocalLinearTrend(processVariance: 1e-3),
      TrigonometricSeasonal(period: 7, harmonics: 2, processVariance: 1e-3),
    ]),
    data,
  );
  final better = withSeasonal.model.diagnose(data);
  final betterTest = better.ljungBox(lags: 14, fittedParameters: 2);

  print('                          trend only    trend + weekly');
  print('log likelihood         ${_pad(trendOnly.logMarginalLikelihood, 2)}'
      '${_pad(withSeasonal.logMarginalLikelihood, 2)}');
  print('fitted noise, kg       '
      '${_pad(math.sqrt(trendOnly.measurementVariance), 3)}'
      '${_pad(math.sqrt(withSeasonal.measurementVariance), 3)}');
  print('autocorrelation, lag 3 '
      '${_pad(plain.autocorrelation(3), 3)}'
      '${_pad(better.autocorrelation(3), 3)}');
  print('Ljung-Box p            ${_pad(plainTest.pValue, 4)}'
      '${_pad(betterTest.pValue, 4)}');
  print('');
  print('The readings were generated with a noise of 0.087 kg. The trend-only '
      'model has');
  print('nowhere to put the weekly pattern, so it puts it in the noise and '
      'reports a scale');
  print('four times worse than it is -- and still leaves residuals that '
      'oscillate with the');
  print('week: up at lag 1, down at lag 3, up again at lag 6. That is what the '
      'portmanteau');
  print('test is reading.');
  print('');
  print('Note what is not diagnostic here. The standardised residual variance '
      'is 0.98 and');
  print('1.00. Fitting chooses the noise level that makes it one, so it always '
      'will be; the');
  print('number that moves is the noise level itself.');
  print('');

  // What the better model thinks the two pieces are.
  final posterior = withSeasonal.model.smooth(data);
  print(' day    reading    trend    weekly');
  for (var i = 0; i < posterior.length; i += 7) {
    print('${posterior.times[i].toStringAsFixed(0).padLeft(4)}  '
        '${data[i].value.toStringAsFixed(2).padLeft(9)}  '
        '${posterior.componentMean(0)[i].toStringAsFixed(2).padLeft(7)}  '
        '${posterior.componentMean(1)[i].toStringAsFixed(2).padLeft(8)}');
  }

  print('');
  print('weekly amplitude   ${_amplitude(posterior).toStringAsFixed(3)} kg '
      'peak to trough');
  print('trend over 12 wks  '
      '${(posterior.componentMean(0).last - posterior.componentMean(0).first).toStringAsFixed(2)} kg');
  print('fitted noise       '
      '${math.sqrt(withSeasonal.measurementVariance).toStringAsFixed(3)} kg per reading');
}

String _pad(double value, int digits) =>
    value.toStringAsFixed(digits).padLeft(14);

double _amplitude(SmoothingResult posterior) {
  final weekly = posterior.componentMean(1);
  var lowest = weekly.first;
  var highest = weekly.first;
  for (final value in weekly) {
    if (value < lowest) lowest = value;
    if (value > highest) highest = value;
  }
  return highest - lowest;
}
