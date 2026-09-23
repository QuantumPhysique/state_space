import 'dart:math' as math;
import 'dart:typed_data';

import 'component.dart';
import 'engine/cholesky.dart';
import 'engine/matrix_block.dart';

/// Checks the properties the engine relies on and the type system cannot,
/// and returns one sentence per violation, empty when there are none.
///
/// Run it in a test of any component you write. It checks that:
///
/// * [Component.parameters], [Component.parameterSpecs] and
///   [Component.diffuseStates] have the lengths the component declares;
/// * `withParameters(parameters)` round-trips;
/// * `A(0) = I` and `Q(0) = 0`;
/// * `Q(dt)` is symmetric and positive semi-definite at every gap in [gaps];
/// * the discretisation is consistent across gaps: `A(s + t) = A(t) A(s)` and
///   `Q(s + t) = A(t) Q(s) A(t)' + Q(t)` for each consecutive pair in [gaps];
/// * the prior of the non-diffuse states is symmetric and positive
///   semi-definite;
/// * a component that claims [Component.isStatic] has `A = I`, `Q = 0` and
///   every state diffuse;
/// * [Component.rateStateIndex] is a state of the component.
///
/// [tolerance] is relative to the largest entry involved.
List<String> checkComponent(
  Component component, {
  List<double> gaps = const [0.01, 0.3, 1.0, 1.7, 2.5],
  double tolerance = 1e-9,
}) {
  final problems = <String>[];
  final n = component.stateDim;
  final name = component.name;

  final theta = component.parameters;
  if (theta.length != component.parameterCount) {
    problems.add('$name.parameters has ${theta.length} entries but '
        'parameterCount is ${component.parameterCount}');
  }
  if (component.parameterSpecs.length != component.parameterCount) {
    problems.add('$name.parameterSpecs has '
        '${component.parameterSpecs.length} entries but parameterCount is '
        '${component.parameterCount}');
  }
  if (component.diffuseStates.length != n) {
    problems.add('$name.diffuseStates has ${component.diffuseStates.length} '
        'entries but stateDim is $n');
    return problems;
  }
  if (theta.length == component.parameterCount) {
    final again = component.withParameters(theta).parameters;
    for (var i = 0; i < theta.length; i++) {
      if (!_close(again[i], theta[i], tolerance)) {
        problems.add('$name.withParameters(parameters) does not round-trip: '
            'parameter $i went from ${theta[i]} to ${again[i]}');
        break;
      }
    }
  }

  Float64List a(double dt) {
    final block = MatrixBlock.dense(n, n)..fill(double.nan);
    component.transition(dt, block);
    return block.storage;
  }

  Float64List q(double dt) {
    final block = MatrixBlock.dense(n, n)..fill(double.nan);
    component.processNoise(dt, block);
    return block.storage;
  }

  final identity = Float64List(n * n);
  for (var i = 0; i < n; i++) {
    identity[i * n + i] = 1;
  }
  if (!_allClose(a(0), identity, tolerance)) {
    problems.add('$name.transition(0) is not the identity');
  }
  if (!_allClose(q(0), Float64List(n * n), tolerance)) {
    problems.add('$name.processNoise(0) is not zero');
  }

  for (final dt in gaps) {
    final noise = q(dt);
    if (!_isCovariance(noise, n, tolerance)) {
      problems.add('$name.processNoise($dt) is not symmetric positive '
          'semi-definite');
    }
  }
  for (var k = 1; k < gaps.length; k++) {
    final s = gaps[k - 1];
    final t = gaps[k];
    final as = a(s), at = a(t), ast = a(s + t);
    if (!_allClose(ast, _multiply(at, as, n), tolerance)) {
      problems.add('$name.transition($s + $t) is not '
          'transition($t) * transition($s)');
    }
    final propagated = _multiply(_multiply(at, q(s), n), _transpose(at, n), n);
    final qst = q(s + t);
    final qt = q(t);
    for (var i = 0; i < n * n; i++) {
      propagated[i] += qt[i];
    }
    if (!_allClose(qst, propagated, tolerance)) {
      problems.add('$name.processNoise($s + $t) is not '
          'A($t) Q($s) A($t)\' + Q($t)');
    }
  }

  final mean = Float64List(n);
  final prior = MatrixBlock.dense(n, n);
  component.properPrior(mean, prior);
  final proper = [
    for (var i = 0; i < n; i++)
      if (!component.diffuseStates[i]) i
  ];
  if (proper.isNotEmpty) {
    final m = proper.length;
    final block = Float64List(m * m);
    for (var i = 0; i < m; i++) {
      for (var j = 0; j < m; j++) {
        block[i * m + j] = prior.at(proper[i], proper[j]);
      }
    }
    if (!_isCovariance(block, m, tolerance)) {
      problems.add('$name.properPrior writes a covariance that is not '
          'symmetric positive semi-definite');
    }
  }

  if (component.isStatic) {
    final moves = gaps.any((dt) =>
        !_allClose(a(dt), identity, tolerance) ||
        !_allClose(q(dt), Float64List(n * n), tolerance));
    if (moves || component.diffuseStates.contains(false)) {
      problems.add('$name claims isStatic, but its states move or are not '
          'all diffuse, so it would be smoothed wrongly');
    }
  }

  final rate = component.rateStateIndex;
  if (rate != null && (rate < 0 || rate >= n)) {
    problems.add('$name.rateStateIndex is $rate, outside its $n states');
  }
  return problems;
}

bool _close(double x, double y, double tolerance) {
  final scale = math.max(1.0, math.max(x.abs(), y.abs()));
  return (x - y).abs() <= tolerance * scale;
}

bool _allClose(Float64List x, Float64List y, double tolerance) {
  var scale = 1.0;
  for (var i = 0; i < x.length; i++) {
    if (!x[i].isFinite || !y[i].isFinite) return false;
    scale = math.max(scale, math.max(x[i].abs(), y[i].abs()));
  }
  for (var i = 0; i < x.length; i++) {
    if ((x[i] - y[i]).abs() > tolerance * scale) return false;
  }
  return true;
}

bool _isCovariance(Float64List m, int n, double tolerance) {
  var scale = 0.0;
  for (var i = 0; i < n * n; i++) {
    if (!m[i].isFinite) return false;
    scale = math.max(scale, m[i].abs());
  }
  if (scale == 0) return true;
  for (var i = 0; i < n; i++) {
    for (var j = 0; j < i; j++) {
      if ((m[i * n + j] - m[j * n + i]).abs() > tolerance * scale) return false;
    }
  }
  // Positive semi-definite: a Cholesky factorisation succeeds once the
  // diagonal is nudged by the tolerance.
  final nudged = Float64List.fromList(m);
  for (var i = 0; i < n; i++) {
    nudged[i * n + i] += tolerance * scale;
  }
  return choleskyFactor(nudged, n);
}

Float64List _multiply(Float64List x, Float64List y, int n) {
  final out = Float64List(n * n);
  for (var i = 0; i < n; i++) {
    for (var j = 0; j < n; j++) {
      var sum = 0.0;
      for (var k = 0; k < n; k++) {
        sum += x[i * n + k] * y[k * n + j];
      }
      out[i * n + j] = sum;
    }
  }
  return out;
}

Float64List _transpose(Float64List x, int n) {
  final out = Float64List(n * n);
  for (var i = 0; i < n; i++) {
    for (var j = 0; j < n; j++) {
      out[j * n + i] = x[i * n + j];
    }
  }
  return out;
}
