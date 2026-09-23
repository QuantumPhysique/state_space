import '../observation.dart';

/// The smallest measurement variance a fit will report for [observations].
///
/// A series that a model explains exactly — every reading identical, or on a
/// straight line under a trend — has a restricted maximum likelihood noise
/// level of zero, or a rounding error either side of it, and no model can be
/// built with that. This floor is far below anything an instrument resolves:
/// a standard deviation of one part in a billion of the largest reading.
double scaleFloor(List<Observation> observations) {
  var largest = 0.0;
  for (final o in observations) {
    final size = o.value.abs();
    if (size > largest) largest = size;
  }
  final sd = 1e-9 * largest;
  final variance = sd * sd;
  return variance > 1e-300 ? variance : 1e-300;
}

/// The median gap between visits in [times], which must be ascending, or
/// zero when there are too few distinct times for that to mean anything.
///
/// Times closer together than a tenth of the typical long gap count as one
/// visit, where the typical long gap is the median of the gaps at or above the
/// median gap. That picks out the spacing between visits as long as at least
/// one gap in three is between visits, which is to say up to three readings
/// per visit. See `samplingResolution`.
double typicalGap(List<double> times) {
  final gaps = <double>[];
  for (var i = 1; i < times.length; i++) {
    final gap = times[i] - times[i - 1];
    if (gap > 0) gaps.add(gap);
  }
  if (gaps.isEmpty) return 0;
  gaps.sort();
  final long = gaps.sublist(gaps.length ~/ 2);
  final together = _median(long) / 10;
  final visits = <double>[];
  var visitStart = times.first;
  for (var i = 1; i < times.length; i++) {
    if (times[i] - times[i - 1] <= together) continue;
    visits.add(times[i] - visitStart);
    visitStart = times[i];
  }
  if (visits.isEmpty) return 0;
  visits.sort();
  return _median(visits);
}

/// The median of an ascending, non-empty list.
double _median(List<double> sorted) {
  final middle = sorted.length ~/ 2;
  return sorted.length.isOdd
      ? sorted[middle]
      : (sorted[middle - 1] + sorted[middle]) / 2;
}
