import 'package:state_space/state_space.dart';

/// One weight history, ready to hand to the package.
///
/// Time is measured in **days since the first reading**, including the
/// fraction of a day the reading was taken at. That matters more than it
/// looks: a diary weighed at 06:40 on weekdays and 08:30 at weekends is
/// irregularly sampled, and the whole reason for using this package rather
/// than a fixed-lag filter is that it does not have to be told twice.
class Series {
  const Series(this.name, this.observations);

  /// Something to print at the head of a table.
  final String name;

  /// Readings in ascending time order.
  final List<Observation> observations;

  int get length => observations.length;

  double get span => observations.last.time - observations.first.time;

  /// The first [days] days of the history, counted from the first reading:
  /// what a fit would have had to work with at that point in the diary's
  /// life.
  Series firstDays(int days) {
    final cutoff = observations.first.time + days;
    final kept = observations.where((o) => o.time <= cutoff).toList();
    return Series('$name (${days}d)', kept);
  }

  @override
  String toString() =>
      '$name: $length readings over '
      '${span.toStringAsFixed(0)} days';
}
