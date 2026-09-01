// Synthetic weight diaries, for when there is no real one to hand.
//
// Deliberately written from scratch rather than ported from trale's screenshot
// generator, which does the same job for the same reasons. Two of them.
//
// The first is licensing. trale is AGPLv3+ and has many contributors; that
// file is one person's work and not ours to relicense. This package is MIT and
// this repository will be public, so code copied in here would be
// redistributed under the wrong licence. The phenomena below are not anyone's
// property -- that hydration swings persist for days, that the weekend shows
// on the scale on Monday, that people miss mornings in clumps -- but an
// implementation of them is.
//
// The second is that a synthetic diary is the fallback and not the point. What
// this tool is for is pointing at a real export. Everything here exists so
// that the tables have something to print before you have one, and so that a
// change to the package can be checked against a fixed series that does not
// leave the machine it was generated on.
//
// The constants are set out with where they came from. Where a number is a
// guess it says so.

import 'dart:math';

import 'package:state_space/state_space.dart';

import 'series.dart';

/// Days of history the synthetic diaries span.
const int historyDays = 186;

/// How much of yesterday's hydration swing carries into today.
///
/// Glycogen and the water bound to it turn over across a few days rather than
/// overnight, so an independent daily wobble is the wrong shape: it would give
/// a series that scatters evenly about the trend, where a real one wanders in
/// clumps. A persistence of 0.8 per day gives a memory of about four and a
/// half days, which is the right order for what a diary looks like.
const double _waterPersistence = 0.8;

/// Standard deviation of the hydration swing once it has settled, at the
/// reference weight.
///
/// Popular advice about daily weight fluctuation puts it at one to two
/// kilograms peak to peak. This is the quiet end of that: someone weighing in
/// fasted every morning, which is what a diary app encourages.
const double _waterSpread = 0.35;

/// Standard deviation of the scale's own reading error, at the reference
/// weight, before the display rounds.
///
/// A guess, and on the optimistic side: repeatability figures for domestic
/// scales are hard to come by and depend mostly on where the thing is
/// standing.
const double _readingError = 0.10;

/// The step the display rounds to.
///
/// This is a second and larger source of measurement error than
/// [_readingError], and the one a noise floor is really about: rounding to
/// 100 g contributes a uniform error of standard deviation 0.029 kg that no
/// amount of data can see through.
const double _displayStep = 0.1;

/// Weekday offset in kg at the reference weight.
///
/// A lookup rather than a sinusoid, because the shape is not sinusoidal: a
/// weekend of larger meals and more salt shows up on the Sunday and Monday
/// scale and has drained away by Friday. The shape is what matters here more
/// than the size, since it is what a weekly seasonal component has to be able
/// to represent.
const Map<int, double> _weekdayOffset = {
  DateTime.saturday: 0.05,
  DateTime.sunday: 0.20,
  DateTime.monday: 0.30,
  DateTime.tuesday: 0.15,
  DateTime.wednesday: 0.05,
  DateTime.thursday: -0.05,
  DateTime.friday: -0.10,
};

/// Body weight the amplitudes above were chosen at. Each diary scales them by
/// its own weight over this, since a 63 kg maintainer does not swing as much
/// as a 90 kg dieter.
const double _referenceWeight = 80;

/// A two-state Markov chain over whether a morning gets weighed: the chance of
/// carrying on after a day that was recorded, and of picking the habit back up
/// after one that was not.
///
/// The chain matters more than the marginal rate it implies. Independent
/// per-day sampling drops isolated days; a real diary has isolated days *and*
/// the occasional three-day lapse, and a gap is where a smoother has to do
/// something rather than nothing.
const double _keepGoing = 0.90;
const double _pickUpAgain = 0.45;

/// A trip: a stretch with no readings at all, and a bump that comes off over
/// the following week.
const int _tripStartsDaysIn = 78;
const int _tripDays = 9;
const double _tripGain = 1.2;
const double _tripDecayDays = 6;

/// Standard normal, by Box-Muller. `1 - nextDouble()` lands in `(0, 1]`, which
/// keeps the logarithm away from zero.
double _gauss(Random rng) =>
    sqrt(-2 * log(1 - rng.nextDouble())) * cos(2 * pi * rng.nextDouble());

/// One synthetic diary: a course, a seed, and the noise that goes on top.
class SyntheticDiary {
  const SyntheticDiary({
    required this.name,
    required this.seed,
    required this.corners,
    required this.story,
  });

  final String name;
  final int seed;

  /// Corner points of the underlying course, oldest first.
  final List<({int day, double weight})> corners;

  /// What this diary is doing, for the table heading.
  final String story;

  double get _scale => corners.last.weight / _referenceWeight;

  /// The course as drawn: straight lines between the corners, extrapolated
  /// along the first and last segment past either end.
  double _cornerAt(int day) {
    for (var i = 1; i < corners.length; i++) {
      final a = corners[i - 1];
      final b = corners[i];
      if (day < b.day || i == corners.length - 1) {
        return a.weight +
            (b.weight - a.weight) * (day - a.day) / (b.day - a.day);
      }
    }
    return corners.last.weight;
  }

  /// The course with its corners rounded off.
  ///
  /// Straight segments meeting at a point are not what a weight course looks
  /// like, and the kink is exactly the feature a smoother would be judged on.
  /// A centred moving average over a fortnight turns the corners into changes
  /// of pace, which is what a real one has. The window is a fortnight because
  /// that is roughly how long a change of habit takes to show.
  double _courseAt(int day) {
    const halfWidth = 7;
    var sum = 0.0;
    for (var d = day - halfWidth; d <= day + halfWidth; d++) {
      sum += _cornerAt(d);
    }
    return sum / (2 * halfWidth + 1);
  }

  /// Builds the diary as this package wants it: time in days from the first
  /// reading, weight in kilograms.
  ///
  /// [today] anchors the calendar, which matters because the weekday offsets
  /// are read off real weekdays: the same seed on a Tuesday and on a Wednesday
  /// gives different series. Pin it to compare two runs.
  Series build({DateTime? today}) {
    final now = today ?? DateTime.now();
    final rng = Random(seed);
    final scale = _scale;
    final readings = <({DateTime at, double weight})>[];

    var water = 0.0;
    var weighedYesterday = true;

    for (var day = 0; day <= historyDays; day++) {
      // Advanced on every day including the unrecorded ones, so that the swing
      // stays continuous across a gap rather than restarting after it.
      water = _waterPersistence * water +
          _waterSpread *
              sqrt(1 - _waterPersistence * _waterPersistence) *
              scale *
              _gauss(rng);

      final away =
          day >= _tripStartsDaysIn && day < _tripStartsDaysIn + _tripDays;
      final firstDayBack = day == _tripStartsDaysIn + _tripDays;
      final weighed = away
          ? false
          : firstDayBack ||
              rng.nextDouble() < (weighedYesterday ? _keepGoing : _pickUpAgain);
      weighedYesterday = weighed;
      if (!weighed) continue;

      // Built from calendar components rather than by subtracting a Duration,
      // which would shift the wall-clock hour across a daylight-saving change
      // and could move a reading into the neighbouring day.
      final date = DateTime(now.year, now.month, now.day - historyDays + day);
      final lateStart =
          date.weekday == DateTime.saturday || date.weekday == DateTime.sunday;
      final at = DateTime(date.year, date.month, date.day, lateStart ? 8 : 6,
          (lateStart ? 5 : 30) + rng.nextInt(60));

      var weight = _courseAt(day) +
          _weekdayOffset[date.weekday]! * scale +
          water +
          _readingError * scale * _gauss(rng);

      final sinceBack = day - (_tripStartsDaysIn + _tripDays);
      if (sinceBack >= 0) {
        weight += _tripGain * scale * exp(-sinceBack / _tripDecayDays);
      }

      readings.add((
        at: at,
        weight: (weight / _displayStep).roundToDouble() * _displayStep,
      ));
    }

    final origin = readings.first.at;
    return Series(
      '$name, ${corners.last.weight.toStringAsFixed(0)} kg ($story)',
      [
        for (final r in readings)
          Observation(r.at.difference(origin).inMinutes / (60 * 24), r.weight)
      ],
    );
  }
}

/// Four diaries: two losing weight at different scales, one holding it, one
/// building up. Four rather than one because the interesting question is
/// whether a fit behaves the same way for all of them, and a single series
/// cannot answer that.
const List<SyntheticDiary> diaries = [
  SyntheticDiary(
    name: 'faster loss',
    seed: 29,
    story: 'losing, 10 kg',
    corners: [
      (day: 0, weight: 91.2),
      (day: 17, weight: 89.2),
      (day: 70, weight: 85.6),
      (day: 100, weight: 85.0),
      (day: 186, weight: 80.8),
    ],
  ),
  SyntheticDiary(
    name: 'slower loss',
    seed: 9,
    story: 'losing, 8 kg',
    corners: [
      (day: 0, weight: 82.4),
      (day: 17, weight: 80.9),
      (day: 70, weight: 77.8),
      (day: 100, weight: 77.3),
      (day: 186, weight: 73.9),
    ],
  ),
  SyntheticDiary(
    name: 'maintaining',
    seed: 7,
    story: 'holding, no trend',
    corners: [
      (day: 0, weight: 63.9),
      (day: 45, weight: 63.2),
      (day: 95, weight: 63.7),
      (day: 140, weight: 63.1),
      (day: 186, weight: 63.4),
    ],
  ),
  SyntheticDiary(
    name: 'lean bulk',
    seed: 11,
    story: 'gaining, 4 kg',
    corners: [
      (day: 0, weight: 61.8),
      (day: 20, weight: 62.4),
      (day: 70, weight: 63.8),
      (day: 100, weight: 64.1),
      (day: 186, weight: 66.1),
    ],
  ),
];

/// Every diary, built against the same calendar.
List<Series> syntheticSeries({DateTime? today}) =>
    [for (final d in diaries) d.build(today: today)];
