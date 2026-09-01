import 'dart:io';

import 'package:state_space/state_space.dart';

import 'series.dart';

/// Reads a weight diary off disk.
///
/// The native format is trale's own export — a couple of `#` comment lines and
/// then one reading per line as an ISO 8601 timestamp, a space, and a weight
/// in kilograms:
///
/// ```text
/// # This file was created with trale.
/// #Date weight[kg]
/// 2026-02-14T06:51:00.000 81.4000000000
/// ```
///
/// Commas and semicolons are accepted as separators too, so an ordinary
/// two-column CSV of date and weight works without being converted first.
/// Anything after the first two fields is ignored, which is what makes an
/// export from some other app usually just work.
///
/// Nothing here is uploaded, cached or written back. The file is read, turned
/// into numbers, and the numbers are printed.
Series readExport(String path) {
  final file = File(path);
  if (!file.existsSync()) {
    throw ArgumentError.value(path, 'path', 'no such file');
  }

  final readings = <({DateTime at, double weight})>[];
  var lineNumber = 0;
  var skipped = 0;
  for (final raw in file.readAsLinesSync()) {
    lineNumber++;
    final line = raw.trim();
    if (line.isEmpty || line.startsWith('#')) continue;

    final fields = line.split(RegExp(r'[\s,;]+'));
    if (fields.length < 2) {
      skipped++;
      continue;
    }
    final at = DateTime.tryParse(fields[0]);
    final weight = double.tryParse(fields[1]);
    if (at == null || weight == null || !weight.isFinite || weight <= 0) {
      // A header row that is not commented out lands here, which is the usual
      // reason for one skipped line and no cause for alarm.
      skipped++;
      continue;
    }
    readings.add((at: at, weight: weight));
  }

  if (readings.length < 2) {
    throw FormatException(
        'read $lineNumber lines from $path and found ${readings.length} '
        'usable readings. Expected "<ISO 8601 date> <weight>" per line.');
  }
  if (skipped > 0) {
    stderr.writeln('note: skipped $skipped unparseable line'
        '${skipped == 1 ? '' : 's'} in $path');
  }

  readings.sort((a, b) => a.at.compareTo(b.at));
  final origin = readings.first.at;
  return Series(
    _nameFor(path),
    [
      for (final r in readings)
        Observation(r.at.difference(origin).inMinutes / (60 * 24), r.weight)
    ],
  );
}

String _nameFor(String path) {
  final base = path.split(Platform.pathSeparator).last;
  final dot = base.lastIndexOf('.');
  return dot > 0 ? base.substring(0, dot) : base;
}
