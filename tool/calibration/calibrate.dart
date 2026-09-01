// What does this package actually do to a body-weight diary?
//
//   dart run tool/calibration/calibrate.dart                    # demo diaries
//   dart run tool/calibration/calibrate.dart export.txt         # your own
//   dart run tool/calibration/calibrate.dart --stability export.txt
//
// The package is tested against dense Gaussian process references, which says
// it computes the right thing. It says nothing about whether the right thing
// is any use on the data it will meet. This is the other half: run the models
// on realistic diaries and print what comes back, so that a decision about
// them -- how much to smooth, whether to let a user choose, whether the fit
// can be trusted on a short history -- is made against numbers.
//
// Everything here is read-only and offline. A file passed on the command line
// is parsed and printed and nothing else.

// ignore_for_file: avoid_print

import 'dart:io';
import 'dart:math' as math;

import 'package:state_space/state_space.dart';

import 'series.dart';
import 'synthetic.dart';
import 'trale_export.dart';

/// A domestic scale reading in 100 g steps rounds with a standard deviation of
/// 0.1 / sqrt(12) kg, and that error is in the data whatever else is. Used as
/// the noise floor: it is the one number about the measurement process that
/// the readings genuinely cannot argue with.
double roundingFloor = math.pow(0.1 / math.sqrt(12), 2).toDouble();

/// The candidate models, cheapest first.
///
/// The trend alone is what trale computes today. Each of the others adds one
/// thing the demo data is known to contain, so the table below reads as a
/// series of questions: does the weekly pattern matter, does the water
/// retention matter, do they matter together.
final Map<String, StructuralModel Function()> models = {
  'trend': () => StructuralModel.localLinearTrend(processVariance: 1e-3),
  'trend + weekly': () => StructuralModel([
        const LocalLinearTrend(processVariance: 1e-3),
        TrigonometricSeasonal(period: 7, harmonics: 2, processVariance: 1e-3),
      ]),
  'trend + water': () => StructuralModel([
        const LocalLinearTrend(processVariance: 1e-3),
        Matern.oneHalf(variance: 0.05, lengthScale: 3),
      ]),
  'trend + both': () => StructuralModel([
        const LocalLinearTrend(processVariance: 1e-3),
        TrigonometricSeasonal(period: 7, harmonics: 2, processVariance: 1e-3),
        Matern.oneHalf(variance: 0.05, lengthScale: 3),
      ]),
};

void main(List<String> arguments) {
  final files = <String>[];
  var stability = false;
  var floor = false;
  DateTime? today;

  for (final argument in arguments) {
    if (argument == '--stability') {
      stability = true;
    } else if (argument == '--floor') {
      floor = true;
    } else if (argument.startsWith('--floor=')) {
      floor = true;
      final sd = double.parse(argument.substring('--floor='.length));
      roundingFloor = sd * sd;
    } else if (argument == '--all') {
      stability = true;
      floor = true;
    } else if (argument.startsWith('--today=')) {
      today = DateTime.parse(argument.substring('--today='.length));
    } else if (argument == '-h' || argument == '--help') {
      _usage();
      return;
    } else if (argument.startsWith('-')) {
      stderr.writeln('unknown option: $argument');
      _usage();
      exitCode = 2;
      return;
    } else {
      files.add(argument);
    }
  }

  final List<Series> series;
  if (files.isEmpty) {
    series = syntheticSeries(today: today);
    print('Four synthetic diaries, built against '
        '${_day(today ?? DateTime.now())}.');
    print('These are a stand-in. Pass a trale export on the command line to '
        'use real data,');
    print('which is the only thing that settles anything.');
  } else {
    series = [for (final path in files) readExport(path)];
  }
  print('');

  for (final one in series) {
    _describe(one);
  }
  if (stability) _stability(series);
  if (floor) _floor(series);

  if (!stability && !floor) {
    print('');
    print('Run again with --stability to see how the fit behaves as a diary '
        'grows,');
    print('or --floor to see when a noise floor would change the answer. '
        '--all does both.');
  }
}

void _usage() {
  print('usage: dart run tool/calibration/calibrate.dart [options] [files]');
  print('');
  print('  files            trale exports, or any file of "<date> <weight>"');
  print('                   lines. With none, four demo diaries are used.');
  print('  --stability      refit at growing history lengths');
  print('  --floor[=SD]     show where a noise floor would bind, for a scale');
  print('                   of the given precision in kg (default 0.029, the');
  print('                   rounding to 100 g steps)');
  print('  --all            both of the above');
  print('  --today=DATE     anchor the demo calendar, for reproducibility');
}

/// Fits every candidate model to one diary and prints what each says.
void _describe(Series series) {
  print('${series.name} -- ${series.length} readings over '
      '${series.span.toStringAsFixed(0)} days');
  print('  model            par  diff   log L    noise    bandwidth  plateau  '
      'Ljung-Box');
  for (final entry in models.entries) {
    final model = entry.value();
    final fitted = fit(model, series.observations);
    final diagnostics = fitted.model.diagnose(series.observations);
    final portmanteau = diagnostics.ljungBox(
        lags: 14, fittedParameters: model.parameterCount + 1);
    print('  ${entry.key.padRight(15)}'
        '  ${model.parameterCount.toString().padLeft(2)}'
        '  ${_diffuseDim(model).toString().padLeft(4)}'
        '  ${fitted.logMarginalLikelihood.toStringAsFixed(1).padLeft(7)}'
        '  ${math.sqrt(fitted.measurementVariance).toStringAsFixed(3)} kg'
        '  ${_bandwidth(fitted).padLeft(7)} d'
        '  ${fitted.plateauDecadesByParameter[0].toStringAsFixed(2).padLeft(6)}'
        '  ${_pValue(portmanteau.pValue)}');
  }
  print('');
  print('  Log likelihoods compare only down a run of equal "diff": a model '
      'with more');
  print('  flat directions has integrated more of them away, and the two '
      'numbers are');
  print('  not on the same scale. The noise level and the Ljung-Box p compare '
      'across');
  print('  everything, and are the honest way to choose here.');
  print('');
}

/// How the fitted smoothing would have looked at each point in a diary's life.
///
/// The question behind the table: is the fitted bandwidth something a user
/// could be shown, or does it lurch about as the diary grows? A chart whose
/// character changes when one more reading arrives is a bug report waiting to
/// be filed, whatever the likelihood says.
void _stability(List<Series> series) {
  const lengths = [14, 21, 30, 45, 60, 90, 120, 186];
  for (final name in ['trend', 'trend + weekly', 'trend + both']) {
    print('BANDWIDTH AND NOISE OVER A GROWING DIARY -- $name');
    print('  days  ${[for (final s in series) _short(s).padLeft(18)].join()}');
    for (final days in lengths) {
      final cells = <String>[];
      for (final one in series) {
        final cut = one.firstDays(days);
        if (cut.length < 8) {
          cells.add('--'.padLeft(18));
          continue;
        }
        try {
          final fitted = fit(models[name]!(), cut.observations);
          // The star has to come from the trend's own plateau and not from
          // FitResult.isFlat, which is the widest plateau over every
          // parameter. A weekly component whose drift rate is undetermined
          // makes isFlat true while saying nothing about the bandwidth, and
          // starring every row on that account would hide the thing this
          // table is for.
          final plateau = fitted.plateauDecadesByParameter[0];
          cells.add('${_bandwidth(fitted)}${plateau > 2 ? '*' : ' '} '
                  '(${plateau.toStringAsFixed(2)})'
              .padLeft(18));
        } on ArgumentError {
          cells.add('--'.padLeft(18));
        }
      }
      print('  ${days.toString().padLeft(4)}  ${cells.join()}');
    }
    print('');
    print('  Bandwidth in days, and in brackets how many decades the trend\'s '
        'own');
    print('  variance can move before the fit is half a nat worse. A star '
        'marks more');
    print('  than two decades, where the bandwidth printed is a convention '
        'rather');
    print('  than an estimate.');
    print('');
  }
}

/// Where asserting the scale's own precision changes the answer.
void _floor(List<Series> series) {
  const lengths = [14, 21, 30, 45, 60, 90, 120, 186];
  for (final name in ['trend + weekly', 'trend + both']) {
    print('WHERE A NOISE FLOOR BINDS -- '
        '${math.sqrt(roundingFloor).toStringAsFixed(3)} kg, $name');
    print('  days  ${[for (final s in series) _short(s).padLeft(20)].join()}');
    for (final days in lengths) {
      final cells = <String>[];
      for (final one in series) {
        final cut = one.firstDays(days);
        if (cut.length < 8) {
          cells.add('--'.padLeft(20));
          continue;
        }
        try {
          final free = fit(models[name]!(), cut.observations);
          final floored = fit(models[name]!(), cut.observations,
              minimumMeasurementVariance: roundingFloor);
          cells.add((floored.measurementVariancePinned
                  ? '${_bandwidth(free)} -> ${_bandwidth(floored)}'
                  : '.  '
                      '(${math.sqrt(free.measurementVariance).toStringAsFixed(3)})')
              .padLeft(20));
        } on ArgumentError {
          cells.add('--'.padLeft(20));
        }
      }
      print('  ${days.toString().padLeft(4)}  ${cells.join()}');
    }
    print('');
    print('  A dot means the data was noisier than the floor and the fit was '
        'left');
    print('  alone, with the estimated noise in brackets. Otherwise the two '
        'figures');
    print('  are the bandwidth before and after the floor was applied. A '
        'floor that');
    print('  never binds costs one extra fit and nothing else, which is the '
        'argument');
    print('  for asking for one even when you expect it not to.');
    print('');
  }
}

/// Silverman\'s equivalent-kernel bandwidth for the trend, in days.
///
/// A smoothing spline with parameter `lambda` on data sampled at density `f`
/// behaves like a kernel smoother of bandwidth `(lambda / f)^(1/4)`
/// (Silverman 1984). Here `lambda` is the reciprocal of the fitted variance
/// ratio and `f` is one reading per day, so the bandwidth is the ratio to the
/// power of minus a quarter -- which is the number to compare against a
/// setting expressed in days, and the reason a factor of sixteen in the
/// variance ratio is only a factor of two on the chart.
String _bandwidth(FitResult fitted) {
  final ratio = fitted.varianceRatios[0];
  if (!ratio.isFinite || ratio <= 0) return '?';
  final days = math.pow(ratio, -0.25).toDouble();
  return days >= 100 ? days.toStringAsFixed(0) : days.toStringAsFixed(2);
}

int _diffuseDim(StructuralModel model) {
  var count = 0;
  for (final component in model.components) {
    for (final flag in component.diffuseStates) {
      if (flag) count++;
    }
  }
  return count;
}

String _pValue(double p) =>
    p < 1e-4 ? '  <0.0001' : p.toStringAsFixed(4).padLeft(8);

String _short(Series series) {
  final comma = series.name.indexOf(',');
  return comma > 0 ? series.name.substring(0, comma) : series.name;
}

String _day(DateTime when) => '${when.year}-'
    '${when.month.toString().padLeft(2, '0')}-'
    '${when.day.toString().padLeft(2, '0')}';
