// Writes the data behind the package icon and social preview: a handful of
// readings with a gap, the smoothed trend with its credible band, and a
// forecast. The parameters are fixed rather than fitted, to keep the shape.
//
//   dart run tool/figure/icon.dart > /tmp/icon.csv
//   uv run --with matplotlib tool/figure/icon.py /tmp/icon.csv /tmp/trend.csv doc/images

// ignore_for_file: avoid_print

import 'package:state_space/state_space.dart';

void main() {
  const data = [
    Observation(0, 3.2),
    Observation(1, 2.5),
    Observation(2, 2.1),
    Observation(6, 2.4),
    Observation(7, 3.0),
  ];

  final model = StructuralModel.localLinearTrend(
    processVariance: 0.15,
    measurementVariance: 0.004,
  );
  final grid = [for (var t = 0.0; t <= 7; t += 0.05) t];
  final posterior = model.smooth(data, grid: grid);
  final band = posterior.credibleBand();
  final horizon = [for (var t = 7.0; t <= 10; t += 0.05) t];
  final ahead = model.forecast(data, horizon);
  final aheadBand = ahead.credibleBand();

  print('kind,time,value,lo,hi');
  for (final o in data) {
    print('reading,${o.time},${o.value},,');
  }
  for (var i = 0; i < grid.length; i++) {
    print(
      'trend,${grid[i]},${posterior.mean[i]},'
      '${band.lo[i]},${band.hi[i]}',
    );
  }
  for (var i = 0; i < horizon.length; i++) {
    print(
      'forecast,${horizon[i]},${ahead.mean[i]},'
      '${aheadBand.lo[i]},${aheadBand.hi[i]}',
    );
  }
}
