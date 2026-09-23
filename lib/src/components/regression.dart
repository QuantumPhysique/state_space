import 'dart:typed_data';

import '../component.dart';
import '../engine/matrix_block.dart';

/// A half-open interval `[from, to)` on the time axis.
typedef Span = ({double from, double to});

/// One column of a regression design: a known function of time whose
/// coefficient is unknown.
///
/// Regressors are plain data rather than closures, so a [StructuralModel]
/// can always be sent to another isolate, printed and compared. A closure can
/// capture something unsendable, and the send would then fail at runtime.
///
/// A regressor is evaluable at any time, not only at the observation times,
/// because an output grid asks for the signal between readings and past them.
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
/// For events that are not periodic: a holiday, an illness, a course of
/// medication. The coefficient is what the event was worth, in signal units,
/// with its standard error.
final class IndicatorRegressor extends Regressor {
  /// A column named [name] that is one inside [spans] and zero elsewhere.
  ///
  /// [spans] must be sorted by start time and must not overlap; both are
  /// checked.
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
  bool operator ==(Object other) =>
      other is IndicatorRegressor &&
      other.name == name &&
      _listEquals(other.spans, spans);

  @override
  int get hashCode =>
      Object.hash(IndicatorRegressor, name, Object.hashAll(spans));

  @override
  String toString() => 'IndicatorRegressor($name, ${spans.length} spans)';
}

/// A covariate that changes at known instants and holds its value in between:
/// a dose, an altitude, a training load.
final class StepRegressor extends Regressor {
  /// A column named [name] that takes `values[i]` from `knots[i]` until the
  /// next knot, and [before] (default zero) before the first.
  ///
  /// [knots] must be sorted ascending and the same length as [values], and
  /// both finite. Both lists are copied.
  StepRegressor(this.name, List<double> knots, List<double> values,
      {this.before = 0})
      : _knots = Float64List.fromList(knots),
        _values = Float64List.fromList(values) {
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

  final Float64List _knots;
  final Float64List _values;

  /// Instants at which the column takes a new value, as a read-only view.
  Float64List get knots => _knots.asUnmodifiableView();

  /// The value taken from each knot until the next, as a read-only view.
  Float64List get values => _values.asUnmodifiableView();

  /// The value before the first knot.
  final double before;

  @override
  double at(double time) {
    var low = 0;
    var high = _knots.length - 1;
    var found = -1;
    while (low <= high) {
      final middle = (low + high) >> 1;
      if (_knots[middle] <= time) {
        found = middle;
        low = middle + 1;
      } else {
        high = middle - 1;
      }
    }
    return found < 0 ? before : _values[found];
  }

  @override
  bool isSilentOver(double from, double to) {
    if (before != 0 && _knots.isNotEmpty && _knots.first > from) return false;
    if (_knots.isEmpty) return before == 0;
    for (var i = 0; i < _knots.length; i++) {
      final until = i + 1 < _knots.length ? _knots[i + 1] : double.infinity;
      if (_values[i] != 0 && _knots[i] < to && until > from) return false;
    }
    return true;
  }

  @override
  bool operator ==(Object other) =>
      other is StepRegressor &&
      other.name == name &&
      other.before == before &&
      _listEquals(other._knots, _knots) &&
      _listEquals(other._values, _values);

  @override
  int get hashCode => Object.hash(StepRegressor, name, before,
      Object.hashAll(_knots), Object.hashAll(_values));

  @override
  String toString() => 'StepRegressor($name, ${_knots.length} knots)';
}

/// Coefficients on known columns, estimated as part of the state.
///
/// ```text
/// y(t) = ... + sum_j beta_j z_j(t) + eps(t)
/// ```
///
/// Each coefficient is one state with `A = I` and `Q = 0` under a flat prior,
/// so the coefficients are estimated by the same recursion as the rest of the
/// posterior, with standard errors, and read from
/// [SmoothingResult.coefficients].
///
/// [parameterCount] is zero: a trend plus twenty indicators is still a
/// one-dimensional search. Each column is still an extra flat direction the
/// filter carries, though, and a forward pass costs roughly the square of the
/// number of flat directions, so twenty columns make each likelihood
/// evaluation tens of times slower than the trend alone.
///
/// {@category Components}
final class RegressionComponent extends Component {
  /// Coefficients on [regressors], which must not be empty.
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

  /// `A = I`, `Q = 0`, every state diffuse. See [Component.isStatic].
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
  String get name => 'RegressionComponent';

  @override
  String? identifiabilityHint(double from, double to, {double resolution = 0}) {
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
  bool operator ==(Object other) =>
      other is RegressionComponent && _listEquals(other.regressors, regressors);

  @override
  int get hashCode =>
      Object.hash(RegressionComponent, Object.hashAll(regressors));

  @override
  String toString() =>
      'RegressionComponent(${regressors.map((r) => r.name).join(', ')})';
}

bool _listEquals<T>(List<T> a, List<T> b) {
  if (a.length != b.length) return false;
  for (var i = 0; i < a.length; i++) {
    if (a[i] != b[i]) return false;
  }
  return true;
}
