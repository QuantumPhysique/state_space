// Writes the data behind doc/images/trend.png: noisy readings with a gap, the
// smoothed trend with its credible and predictive bands, and a forecast.
//
//   dart run tool/figure/figure.dart > /tmp/trend.csv
//   uv run --with matplotlib tool/figure/plot.py /tmp/trend.csv doc/images/trend.png

// ignore_for_file: avoid_print

import 'dart:math' as math;

import 'package:state_space/state_space.dart';

void main() {
  final random = math.Random(4);
  double gaussian() =>
      math.sqrt(-2 * math.log(1 - random.nextDouble())) *
      math.cos(2 * math.pi * random.nextDouble());

  final data = <Observation>[];
  var water = 0.0;
  for (var day = 0; day < 100; day++) {
    water = 0.7 * water + 0.15 * gaussian();
    final away = day >= 45 && day < 62;
    if (away || random.nextDouble() < 0.2) continue;
    final course =
        84 - 4 * (1 - math.exp(-day / 40)) + 0.6 * math.sin(day / 14);
    final reading = course + water + 0.15 * gaussian();
    data.add(Observation(day + 0.3, (reading * 10).round() / 10));
  }

  final fitted = fit(
    StructuralModel.localLinearTrend(processVariance: 1e-3),
    data,
    minimumMeasurementVariance: 0.029 * 0.029,
  );
  final grid = [for (var t = 0.0; t <= 100; t += 0.25) t];
  final posterior = fitted.model.smooth(data, grid: grid);
  final band = posterior.credibleBand();
  final spread = posterior.predictiveBand();
  final horizon = [for (var t = 100.0; t <= 130; t += 0.25) t];
  final ahead = fitted.model.forecast(data, horizon);
  final aheadBand = ahead.credibleBand();
  final aheadSpread = ahead.predictiveBand();

  print('kind,time,value,lo,hi,plo,phi');
  for (final o in data) {
    print('reading,${o.time},${o.value},,,,');
  }
  for (var i = 0; i < grid.length; i++) {
    print(
      'trend,${grid[i]},${posterior.mean[i]},'
      '${band.lo[i]},${band.hi[i]},${spread.lo[i]},${spread.hi[i]}',
    );
  }
  for (var i = 0; i < horizon.length; i++) {
    print(
      'forecast,${horizon[i]},${ahead.mean[i]},'
      '${aheadBand.lo[i]},${aheadBand.hi[i]},'
      '${aheadSpread.lo[i]},${aheadSpread.hi[i]}',
    );
  }
}
