// A worked example: thirty noisy readings with a three-week hole in the
// middle, an estimated smoothing level, and a trend with an honest band.
//
//   dart run example/example.dart

// ignore_for_file: avoid_print

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:state_space/state_space.dart';

/// Readings on days 0-14 and 35-49, with nothing in between.
List<Observation> readings() {
  final random = math.Random(4);
  final data = <Observation>[];
  for (final day in [
    ...List.generate(15, (i) => i),
    ...List.generate(15, (i) => 35 + i),
  ]) {
    final truth = 82.0 - 0.04 * day + 0.6 * math.sin(day / 7);
    data.add(Observation(
        day.toDouble(), truth + 0.35 * (random.nextDouble() - 0.5)));
  }
  return data;
}

void main() {
  final data = readings();

  // The starting process variance is irrelevant -- fit() searches the ratio of
  // process to measurement variance over thirteen decades and concentrates the
  // measurement variance out analytically.
  final fitted = fit(
    StructuralModel.localLinearTrend(processVariance: 1),
    data,
  );

  print('variance ratio     ${fitted.varianceRatio.toStringAsPrecision(3)}  '
      '(smoothing parameter lambda = '
      '${(1 / fitted.varianceRatio).toStringAsPrecision(3)})');
  print('measurement noise  '
      '${math.sqrt(fitted.measurementVariance).toStringAsFixed(3)} kg');
  print('log likelihood     ${fitted.logMarginalLikelihood.toStringAsFixed(2)} '
      'in ${fitted.evaluations} filter passes');
  print('profile plateau    ${fitted.plateauDecades.toStringAsFixed(2)} decades'
      '${fitted.isFlat ? "  -- too flat to trust" : ""}');
  print('');

  // Ask for a value every day, including the ones nobody stepped on a scale.
  // Grid points are steps with no observation attached; nothing is filled in.
  final daily = Float64List.fromList(
      [for (var day = 0; day <= 49; day++) day.toDouble()]);
  final posterior = fitted.model.smooth(data, grid: daily);

  print(' day   trend    slope/day   95% band          measured');
  for (var i = 0; i < posterior.length; i += 3) {
    final band = posterior.credibleInterval(i);
    final measured = data.where((o) => o.time == posterior.times[i]);
    print('${posterior.times[i].toStringAsFixed(0).padLeft(4)}  '
        '${posterior.level[i].toStringAsFixed(3)}  '
        '${posterior.slope![i].toStringAsFixed(4).padLeft(9)}   '
        '[${band.lo.toStringAsFixed(2)}, ${band.hi.toStringAsFixed(2)}]'
        '${measured.isEmpty ? "" : "     ${measured.first.value.toStringAsFixed(2)}"}');
  }

  print('');
  final widest = _argmax(posterior.levelVariance);
  print(
      'The band is widest on day ${posterior.times[widest].toStringAsFixed(0)}, '
      'in the middle of the gap: '
      '${(2 * 1.96 * math.sqrt(posterior.levelVariance[widest])).toStringAsFixed(2)} kg '
      'across, against '
      '${(2 * 1.96 * math.sqrt(posterior.levelVariance[7])).toStringAsFixed(2)} kg '
      'where the readings are dense. Nothing was interpolated to get there --');
  print(
      'the recursion simply had no observation to update on for three weeks.');
}

int _argmax(Float64List values) {
  var best = 0;
  for (var i = 1; i < values.length; i++) {
    if (values[i] > values[best]) best = i;
  }
  return best;
}
