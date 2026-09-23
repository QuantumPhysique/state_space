// What to do about wobble that is not noise, and what it costs.
//
//   dart run example/stationary_example.dart

// ignore_for_file: avoid_print

import 'dart:math' as math;

import 'package:state_space/state_space.dart';

/// A year of daily readings: a steady decline, hydration that persists across
/// days rather than scattering, and a scale that reads to a tenth of a kilo.
List<Observation> readings() {
  final random = math.Random(31);
  double gauss() =>
      math.sqrt(-2 * math.log(1 - random.nextDouble())) *
      math.cos(2 * math.pi * random.nextDouble());

  // An AR(1) with a memory of about four days, which is roughly how long
  // glycogen and the water bound to it take to turn over.
  const persistence = 0.78;
  final kick = 0.35 * math.sqrt(1 - persistence * persistence);
  var water = 0.35 * gauss();

  return [
    for (var day = 0; day < 365; day++)
      () {
        final value = 84.0 - 0.011 * day + water + 0.09 * gauss();
        water = persistence * water + kick * gauss();
        // The display rounds to 100 g, which is a second and larger source of
        // error than the reading itself.
        return Observation(day.toDouble(), (value * 10).roundToDouble() / 10);
      }(),
  ];
}

void main() {
  final data = readings();
  print(
    '${data.length} daily readings, simulated with 0.09 kg of reading '
    'error',
  );
  print('and 0.35 kg of hydration wobble that persists for about four days.\n');

  // A trend on its own. It has nowhere to put the wobble.
  final trendOnly = fit(
    StructuralModel.localLinearTrend(processVariance: 1e-3),
    data,
  );
  final plain = trendOnly.model.diagnose(data);

  // The same data with somewhere for the wobble to go.
  final withWobble = fit(
    StructuralModel([
      LocalLinearTrend(processVariance: 1e-3),
      Matern.oneHalf(variance: 0.1, lengthScale: 3),
    ]),
    data,
  );
  final better = withWobble.model.diagnose(data);
  final wobble = withWobble.model.components[1] as Matern;

  print('                        trend only    trend + Matern');
  print(
    'fitted noise, kg     ${_pad(math.sqrt(trendOnly.measurementVariance), 3)}'
    '${_pad(math.sqrt(withWobble.measurementVariance), 3)}',
  );
  print(
    'Ljung-Box p          '
    '${_pad(plain.ljungBox(lags: 14, fittedParameters: 1).pValue, 4)}'
    '${_pad(better.ljungBox(lags: 14, fittedParameters: 3).pValue, 4)}',
  );
  print(
    'trend plateau, dec   ${_pad(trendOnly.plateauDecadesByParameter[0], 2)}'
    '${_pad(withWobble.plateauDecadesByParameter[0], 2)}',
  );
  print('');
  print(
    'The trend-only model puts the wobble in the noise and reports the '
    'readings as',
  );
  print('about twice as bad as they are, and leaves residuals a portmanteau');
  print(
    'test rejects outright. The Matern component fixes both: it recovers '
    'the',
  );
  print(
    'hydration at ${math.sqrt(wobble.variance).toStringAsFixed(2)} kg with '
    'a length scale of '
    '${wobble.lengthScale.toStringAsFixed(1)} days, against a',
  );
  print('simulated 0.35 kg and about four.');
  print('');
  print(
    'And it costs something, which is the part worth knowing before '
    'reaching for it.',
  );
  print(
    'The trend and the Matern component both describe slow variation, so '
    'they',
  );
  print(
    'compete for it, and the trend\'s own variance goes from '
    '${trendOnly.plateauDecadesByParameter[0].toStringAsFixed(1)} decades of',
  );
  print(
    'plateau to '
    '${withWobble.plateauDecadesByParameter[0].toStringAsFixed(1)}. If what '
    'you want is an honest noise level and a residual',
  );
  print(
    'test that passes, this is the model. If what you want is a stiffness '
    'you can',
  );
  print(
    'report or hand to a user, it is not -- fix one of the two rather '
    'than fitting both.',
  );
  print('');

  // Saying what the instrument can do, which is the other half of the same
  // problem.
  const floor = 0.029 * 0.029; // rounding to 100 g: 0.1 / sqrt(12)
  final floored = fit(
    StructuralModel([
      LocalLinearTrend(processVariance: 1e-3),
      Matern.oneHalf(variance: 0.1, lengthScale: 3),
    ]),
    data,
    minimumMeasurementVariance: floor,
  );
  print(
    'A noise floor of ${math.sqrt(floor).toStringAsFixed(3)} kg -- the '
    'rounding to 100 g steps, which no amount',
  );
  print(
    'of data can see through -- '
    '${floored.measurementVariancePinned ? 'binds here' : 'does not bind here'}: '
    'the fit found '
    '${math.sqrt(withWobble.measurementVariance).toStringAsFixed(3)} kg on '
    'its own.',
  );
  print(
    'A floor that does not bind costs nothing; one that binds costs one '
    'more fit.',
  );
}

String _pad(double value, int digits) =>
    value.toStringAsFixed(digits).padLeft(14);
