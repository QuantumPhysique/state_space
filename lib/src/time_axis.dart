import 'observation.dart';

const int _microsecondsPerDay = Duration.microsecondsPerDay;

/// Converts between calendar dates and times in days since an origin, which
/// is the time unit the components' defaults are chosen for.
///
/// A calendar day always counts as exactly one, including the 23- and 25-hour
/// days at a daylight-saving change, and the time of day is read off the wall
/// clock. So a reading at 07:00 every morning lands on whole numbers of days
/// all year round, and [dateAt] inverts [timeOf] exactly. Subtracting two
/// [DateTime]s and dividing by 24 hours does neither across a change of
/// clocks.
///
/// Dates are compared in the time zone of [origin]: local if it is local, UTC
/// if it is UTC.
///
/// ```dart
/// final axis = TimeAxis.days(readings.first.date);
/// final data = [for (final r in readings) axis.observation(r.date, r.kg)];
/// final grid = [for (var d = 0; d <= axis.timeOf(today); d++) d.toDouble()];
/// final posterior = model.smooth(data, grid: grid);
/// final firstDay = axis.dateAt(posterior.times.first);
/// ```
///
/// {@category Getting started}
final class TimeAxis {
  /// Days since [origin], which is time zero.
  TimeAxis.days(this.origin);

  /// The date and time that is time zero.
  final DateTime origin;

  /// Days from [origin] to [date], counting calendar days as one each and
  /// adding the difference in wall-clock time of day.
  double timeOf(DateTime date) {
    final at = origin.isUtc ? date.toUtc() : date.toLocal();
    final days = DateTime.utc(
      at.year,
      at.month,
      at.day,
    ).difference(DateTime.utc(origin.year, origin.month, origin.day)).inDays;
    return days + (_clock(at) - _clock(origin)) / _microsecondsPerDay;
  }

  /// The date and wall-clock time [time] days after [origin]; the inverse of
  /// [timeOf].
  DateTime dateAt(double time) {
    final total = time * _microsecondsPerDay + _clock(origin);
    final days = (total / _microsecondsPerDay).floor();
    final clock = (total - days * _microsecondsPerDay).round();
    return origin.isUtc
        ? DateTime.utc(
            origin.year,
            origin.month,
            origin.day + days,
            0,
            0,
            0,
            0,
            clock,
          )
        : DateTime(
            origin.year,
            origin.month,
            origin.day + days,
            0,
            0,
            0,
            0,
            clock,
          );
  }

  /// An [Observation] of [value] at [date].
  Observation observation(
    DateTime date,
    double value, {
    double relativeVariance = 1,
  }) => Observation(timeOf(date), value, relativeVariance: relativeVariance);

  static int _clock(DateTime t) =>
      ((t.hour * 60 + t.minute) * 60 + t.second) * 1000000 +
      t.millisecond * 1000 +
      t.microsecond;

  @override
  bool operator ==(Object other) => other is TimeAxis && other.origin == origin;

  @override
  int get hashCode => Object.hash(TimeAxis, origin);

  @override
  String toString() => 'TimeAxis.days($origin)';
}
