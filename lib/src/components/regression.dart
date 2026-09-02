import 'dart:typed_data';

import '../component.dart';
import '../engine/matrix_block.dart';

/// A half-open interval `[from, to)` on the time axis.
typedef Span = ({double from, double to});

/// One column of a regression design: a known function of time whose
/// coefficient is unknown.
///
/// Deliberately data rather than a closure, though not for the reason one
/// might assume: a plain closure crosses an isolate boundary perfectly well on
/// the Dart VM. The problem is what it can carry. A closure captures whatever
/// happens to be in scope where it was written, and if any of that turns out
/// to be unsendable — a port, a file handle, a native resource — the send
/// fails at runtime, in the caller's code, over something the type system
/// never showed them. A model is expected to survive being handed to another
/// isolate, which is how the application this package was written for keeps
/// smoothing off the interface thread, and that promise is only as good as the
/// least inspectable thing in the model.
///
/// Data has the other advantages too. An [IndicatorRegressor] can say when it
/// fires, print itself, and be compared with another; a closure can do none of
/// those, so neither could the model holding it.
///
/// A regressor must be evaluable at *any* time, not only at the observation
/// times, because an output grid asks for the signal between readings and
/// past the end of them. That rules out "one value per observation" as a
/// representation and is why both implementations here are defined by knots
/// rather than by samples.
sealed class Regressor {
  const Regressor();

  /// What this column represents, used when reporting its coefficient.
  String get name;

  /// The value of the column at [time].
  double at(double time);

  /// Whether the column is identically zero over `[from, to)`, in which case
  /// its coefficient is not determined by anything.
  bool isSilentOver(double from, double to);
}

/// One while something is happening, zero otherwise.
///
/// This is the shape most real schedules have. A fortnight over Christmas, a
/// conference, a holiday, a course of medication: none of them are periodic,
/// and asking a sum of sinusoids to represent a two-week rectangle costs a
/// dozen harmonics and rings on both sides of it. An indicator represents it
/// exactly with one state, and the answer comes back as a number a person can
/// read — how much that fortnight was worth, and how sure the model is.
final class IndicatorRegressor extends Regressor {
  /// [spans] must be sorted by start time and must not overlap. Both are
  /// checked, because an overlap would silently make the column count some
  /// stretch twice and the coefficient would quietly mean something else.
  IndicatorRegressor(this.name, List<Span> spans)
      : spans = List.unmodifiable(spans) {
    for (var i = 0; i < spans.length; i++) {
      final span = spans[i];
      if (!span.from.isFinite || !span.to.isFinite) {
        throw ArgumentError.value(
            span, 'spans[$i]', 'endpoints must be finite');
      }
      if (!(span.to > span.from)) {
        throw ArgumentError.value(span, 'spans[$i]',
            'must end after it starts; an empty span contributes nothing');
      }
      if (i > 0 && span.from < spans[i - 1].to) {
        throw ArgumentError('spans must be sorted and disjoint, but '
            'spans[$i] starts at ${span.from}, before spans[${i - 1}] ends at '
            '${spans[i - 1].to}. Merge them rather than letting the column '
            'count that stretch twice.');
      }
    }
  }

  @override
  final String name;

  /// When the indicator is on, sorted and disjoint.
  final List<Span> spans;

  @override
  double at(double time) {
    // Binary search for the last span that starts at or before `time`.
    var low = 0;
    var high = spans.length - 1;
    var found = -1;
    while (low <= high) {
      final middle = (low + high) >> 1;
      if (spans[middle].from <= time) {
        found = middle;
        low = middle + 1;
      } else {
        high = middle - 1;
      }
    }
    if (found < 0) return 0;
    return time < spans[found].to ? 1 : 0;
  }

  @override
  bool isSilentOver(double from, double to) {
    for (final span in spans) {
      if (span.from < to && span.to > from) return false;
    }
    return true;
  }

  @override
  String toString() => 'IndicatorRegressor($name, ${spans.length} spans)';
}

/// A covariate that changes at known instants and holds its value in between.
///
/// The escape hatch for anything that is neither periodic nor a simple on and
/// off — a dose, an altitude, a training load. Holding the last value forward
/// is the only choice that can answer at an arbitrary time without looking
/// into the future, which an output grid between two knots would otherwise
/// require.
final class StepRegressor extends Regressor {
  /// [knots] must be sorted ascending and the same length as [values]. Before
  /// the first knot the column is [before], which defaults to zero.
  StepRegressor(this.name, Float64List knots, Float64List values,
      {this.before = 0})
      : knots = Float64List.fromList(knots),
        values = Float64List.fromList(values) {
    if (knots.length != values.length) {
      throw ArgumentError('there are ${knots.length} knots and '
          '${values.length} values; they must correspond one to one');
    }
    for (var i = 0; i < knots.length; i++) {
      if (!knots[i].isFinite) {
        throw ArgumentError.value(knots[i], 'knots[$i]', 'not finite');
      }
      if (!values[i].isFinite) {
        throw ArgumentError.value(values[i], 'values[$i]', 'not finite');
      }
      if (i > 0 && knots[i] < knots[i - 1]) {
        throw ArgumentError('knots must be sorted ascending, but knots[$i] '
            '(${knots[i]}) precedes knots[${i - 1}] (${knots[i - 1]})');
      }
    }
  }

  @override
  final String name;

  /// Instants at which the column takes a new value.
  final Float64List knots;

  /// The value taken from each knot until the next.
  final Float64List values;

  /// The value before the first knot.
  final double before;

  @override
  double at(double time) {
    var low = 0;
    var high = knots.length - 1;
    var found = -1;
    while (low <= high) {
      final middle = (low + high) >> 1;
      if (knots[middle] <= time) {
        found = middle;
        low = middle + 1;
      } else {
        high = middle - 1;
      }
    }
    return found < 0 ? before : values[found];
  }

  @override
  bool isSilentOver(double from, double to) {
    if (before != 0 && knots.isNotEmpty && knots.first > from) return false;
    if (knots.isEmpty) return before == 0;
    for (var i = 0; i < knots.length; i++) {
      final until = i + 1 < knots.length ? knots[i + 1] : double.infinity;
      if (values[i] != 0 && knots[i] < to && until > from) return false;
    }
    return true;
  }

  @override
  String toString() => 'StepRegressor($name, ${knots.length} knots)';
}

/// Coefficients on known columns, estimated as part of the state.
///
/// ```text
/// y(t) = ... + sum_j beta_j z_j(t) + eps(t)
/// ```
///
/// Each coefficient is one state with `A = I` and `Q = 0` — constant forever —
/// under a flat prior. The Kalman recursion then estimates it for nothing: the
/// coefficients are exactly the sort of flat direction exact diffuse
/// initialisation already integrates out, so they cost `d` extra mean
/// propagations in the forward pass and no covariance work at all.
///
/// The consequence worth noticing is that [parameterCount] is zero. A
/// regression component adds no dimension to the fitting problem. A model of a
/// trend plus twenty holiday indicators is still a one-dimensional search, and
/// the twenty coefficients fall out of the same recursion that produces the
/// trend, with posterior standard errors, at no cost to the optimiser.
///
/// The coefficients do not wander, so [wanderOver] is zero and no penalty on
/// the variances ever touches them.
///
/// {@category Components}
class RegressionComponent extends Component {
  RegressionComponent(List<Regressor> regressors)
      : regressors = List.unmodifiable(regressors) {
    if (regressors.isEmpty) {
      throw ArgumentError.value(regressors, 'regressors',
          'a regression component needs at least one column');
    }
  }

  /// The design columns, in state order.
  final List<Regressor> regressors;

  @override
  int get stateDim => regressors.length;

  @override
  int get parameterCount => 0;

  @override
  void transition(double dt, MatrixBlock out) => out.setIdentity();

  @override
  void processNoise(double dt, MatrixBlock out) => out.fill(0);

  @override
  void observationAt(double time, Float64List out) {
    for (var i = 0; i < regressors.length; i++) {
      out[i] = regressors[i].at(time);
    }
  }

  @override
  double wanderOver(double span) => 0;

  /// `A = I`, `Q = 0`, every state diffuse: a coefficient is a number, not a
  /// process. See [Component.isStatic] for what the engine does with that.
  @override
  bool get isStatic => true;

  @override
  List<bool> get diffuseStates => List.filled(stateDim, true);

  @override
  void properPrior(Float64List mean, MatrixBlock covariance) {
    // Every coefficient is diffuse: their whole point is that the data says
    // what they are.
  }

  @override
  String? identifiabilityHint(double from, double to) {
    final silent = [
      for (final regressor in regressors)
        if (regressor.isSilentOver(from, to)) regressor.name
    ];
    if (silent.isEmpty) return null;
    return 'the regressor${silent.length == 1 ? '' : 's'} '
        '${silent.map((n) => '"$n"').join(', ')} '
        '${silent.length == 1 ? 'is' : 'are'} zero everywhere between $from '
        'and $to, so nothing in the data says what '
        '${silent.length == 1 ? 'its coefficient is' : 'their coefficients are'}';
  }

  @override
  Float64List get parameters => Float64List(0);

  @override
  Component withParameters(Float64List theta) => this;

  @override
  String toString() =>
      'RegressionComponent(${regressors.map((r) => r.name).join(', ')})';
}
