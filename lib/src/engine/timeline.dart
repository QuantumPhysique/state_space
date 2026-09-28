import 'dart:typed_data';

import '../observation.dart';

/// The ordered set of time points the recursion visits.
///
/// Observations and requested output times are merged into one non-decreasing
/// sequence. A step may carry an observation, be marked for reporting, or
/// both; a step with no observation is simply a predict with the update
/// skipped, which is how the filter handles gaps, output grids and missing
/// data with the same code path.
class Timeline {
  Timeline._(
    this.times,
    this.values,
    this.variances,
    this.gaps,
    this.outputIndices,
    this.observationCount,
  );

  /// Time of each step, non-decreasing.
  final Float64List times;

  /// Observed value per step, or NaN where there is no observation.
  final Float64List values;

  /// Observation variance relative to the model's measurement variance, per
  /// step. Undefined where [values] is NaN.
  final Float64List variances;

  /// `times[i] - times[i - 1]`, with `gaps[0] == 0`.
  final Float64List gaps;

  /// Steps to report, in output order.
  final Int32List outputIndices;

  /// How many steps carry an observation.
  final int observationCount;

  /// Number of steps.
  int get length => times.length;

  /// Whether [step] carries an observation rather than only an output time.
  bool hasObservation(int step) => !values[step].isNaN;

  /// Merges [observations] with an optional output [grid].
  ///
  /// Both must be sorted by time and free of NaN and infinities. Duplicate
  /// times are fine: a zero gap gives `A = I`, `Q = 0`, so a second reading at
  /// the same instant is just a second update. When a grid point coincides
  /// exactly with an observation the two share a step rather than producing a
  /// redundant one.
  ///
  /// With no grid, every observation is reported, in input order.
  ///
  /// When several observations share a time and a grid point lands on it, the
  /// grid takes the first of them. That is not the loss it looks like: the gap
  /// between two such steps is zero, so `A = I` and `Q = 0`, the smoother gain
  /// between them is the identity, and their smoothed moments are equal to the
  /// last bit.
  factory Timeline.merge(List<Observation> observations, List<double>? grid) {
    _checkObservations(observations);
    if (grid != null) {
      _checkGrid(grid);
    }

    final gridLength = grid?.length ?? 0;
    final capacity = observations.length + gridLength;
    final times = Float64List(capacity);
    final values = Float64List(capacity);
    final variances = Float64List(capacity);
    final outputs = Int32List(grid == null ? observations.length : gridLength);

    var step = 0;
    var next = 0; // next observation
    var nextGrid = 0;
    var outputCount = 0;
    var observed = 0;

    void pushObservation(Observation o) {
      times[step] = o.time;
      values[step] = o.value;
      variances[step] = o.relativeVariance;
      observed++;
    }

    while (next < observations.length || nextGrid < gridLength) {
      final haveObs = next < observations.length;
      final haveGrid = nextGrid < gridLength;
      final obsTime = haveObs ? observations[next].time : double.infinity;
      final gridTime = haveGrid ? grid![nextGrid] : double.infinity;

      if (obsTime <= gridTime) {
        pushObservation(observations[next++]);
        if (grid == null) {
          outputs[outputCount++] = step;
        } else if (haveGrid && gridTime == obsTime) {
          // Report the state after the update at this instant, not before it.
          outputs[outputCount++] = step;
          nextGrid++;
        }
      } else {
        times[step] = gridTime;
        values[step] = double.nan;
        outputs[outputCount++] = step;
        nextGrid++;
      }
      step++;
    }

    final gaps = Float64List(step);
    for (var i = 1; i < step; i++) {
      gaps[i] = times[i] - times[i - 1];
    }

    return Timeline._(
      Float64List.sublistView(times, 0, step),
      Float64List.sublistView(values, 0, step),
      Float64List.sublistView(variances, 0, step),
      gaps,
      Int32List.sublistView(outputs, 0, outputCount),
      observed,
    );
  }

  static void _checkObservations(List<Observation> observations) {
    for (var i = 0; i < observations.length; i++) {
      final o = observations[i];
      if (!o.time.isFinite) {
        throw ArgumentError.value(
          o.time,
          'observations[$i].time',
          'not finite',
        );
      }
      if (!o.value.isFinite) {
        throw ArgumentError.value(
          o.value,
          'observations[$i].value',
          'not finite',
        );
      }
      if (!(o.relativeVariance >= 0) || !o.relativeVariance.isFinite) {
        throw ArgumentError.value(
          o.relativeVariance,
          'observations[$i].relativeVariance',
          'must be finite and >= 0',
        );
      }
      if (i > 0 && o.time < observations[i - 1].time) {
        throw ArgumentError(
          'observations must be sorted by time, but observations[$i] at '
          '${o.time} precedes observations[${i - 1}] at '
          '${observations[i - 1].time}. Sort the list before calling; '
          'results are reported in input order.',
        );
      }
    }
  }

  static void _checkGrid(List<double> grid) {
    for (var i = 0; i < grid.length; i++) {
      if (!grid[i].isFinite) {
        throw ArgumentError.value(grid[i], 'grid[$i]', 'not finite');
      }
      if (i > 0 && grid[i] < grid[i - 1]) {
        throw ArgumentError(
          'grid must be sorted by time, but grid[$i] '
          '(${grid[i]}) precedes grid[${i - 1}] (${grid[i - 1]})',
        );
      }
    }
  }
}
