// Is a linear algebra dependency worth it at these dimensions?
//
//   dart compile exe benchmark/dependency_comparison.dart -o /tmp/deps && /tmp/deps
//
// The operation timed is one covariance prediction, P- = A P A' + Q, which a
// smoothing pass performs once per time step, at 2, 6 and 18 states: a trend,
// a trend plus a weekly seasonal, and a large composite model.
//
// On an M-series Mac:
//
//   hand-rolled  2x2      28 ns      matrices  2x2      87 ns
//   hand-rolled  6x6     457 ns      matrices  6x6     294 ns
//   hand-rolled 18x18  10758 ns      matrices 18x18   3541 ns
//
// The hand-rolled loop wins at 2x2, where per-operation overhead dominates,
// and loses from 6x6 upward, where `matrices` multiplies with Float64x2 SIMD.
// The engine still does not take the dependency, because:
//
//   * A and Q are block diagonal. An 18-state model built from a 2-state
//     trend and four 4-state seasonal blocks does sum(n_i^2) * n arithmetic,
//     not n^3, about a quarter of the work in that example, and `matrices`
//     would multiply the zeros.
//   * There is no in-place or out-parameter path, so a smoothing pass would
//     allocate a result object per operation: roughly twenty per step, or
//     seventy thousand short-lived objects for a decade of daily data.

// ignore_for_file: avoid_print

import 'dart:typed_data';

import 'package:benchmark_harness/benchmark_harness.dart';
import 'package:matrices/matrices.dart';

/// P- = A P A' + Q with flat buffers and no allocation after setup.
/// `BenchmarkBase` scores one call to `exercise()`, which is ten runs. Divide,
/// and report nanoseconds, because a 2x2 prediction takes about twenty of them
/// and microseconds would round it to nothing.
class PerRun implements ScoreEmitter {
  @override
  void emit(String name, double value) {
    print(
      '${name.padRight(20)}${(value * 100).toStringAsFixed(1).padLeft(9)} ns',
    );
  }
}

class HandRolled extends BenchmarkBase {
  HandRolled(this.n) : super('hand-rolled ${n}x$n', emitter: PerRun());

  final int n;
  late Float64List a, p, q, work, out;

  @override
  void setup() {
    a = _filled(n * n, 0.5);
    p = _filled(n * n, 1.5);
    q = _filled(n * n, 0.25);
    work = Float64List(n * n);
    out = Float64List(n * n);
  }

  @override
  void run() {
    // i-k-j order for the first product so that both operands are walked
    // along rows; the naive i-j-k order strides down a column of P and costs
    // roughly three times as much at 18x18.
    work.fillRange(0, n * n, 0);
    for (var i = 0; i < n; i++) {
      for (var k = 0; k < n; k++) {
        final scale = a[i * n + k];
        for (var j = 0; j < n; j++) {
          work[i * n + j] += scale * p[k * n + j];
        }
      }
    }
    for (var i = 0; i < n; i++) {
      for (var j = 0; j < n; j++) {
        var sum = q[i * n + j];
        for (var k = 0; k < n; k++) {
          sum += work[i * n + k] * a[j * n + k];
        }
        out[i * n + j] = sum;
      }
    }
  }
}

/// The same thing written the way the package would look with a dependency.
class WithMatrices extends BenchmarkBase {
  WithMatrices(this.n) : super('matrices   ${n}x$n', emitter: PerRun());

  final int n;
  late Matrix64 a, p, q;

  @override
  void setup() {
    a = Matrix64.fromFlat(_filled(n * n, 0.5), n, n);
    p = Matrix64.fromFlat(_filled(n * n, 1.5), n, n);
    q = Matrix64.fromFlat(_filled(n * n, 0.25), n, n);
  }

  @override
  void run() {
    final predicted = (a * p) as Matrix64;
    final full = (predicted * a.transpose) as Matrix64;
    // Keep the result reachable so nothing gets optimised away.
    sink = full + q;
  }

  Object? sink;
}

Float64List _filled(int length, double seed) {
  final out = Float64List(length);
  for (var i = 0; i < length; i++) {
    out[i] = seed + 0.001 * i;
  }
  return out;
}

void main() {
  // Sanity: the two agree before either is timed.
  const n = 6;
  final hand = HandRolled(n)..setup();
  hand.run();
  final reference = Matrix64.fromFlat(_filled(n * n, 0.5), n, n);
  final state = Matrix64.fromFlat(_filled(n * n, 1.5), n, n);
  final noise = Matrix64.fromFlat(_filled(n * n, 0.25), n, n);
  final product = (reference * state) as Matrix64;
  final expected = ((product * reference.transpose) as Matrix64) + noise;
  for (var i = 0; i < n; i++) {
    for (var j = 0; j < n; j++) {
      final difference = (hand.out[i * n + j] - expected(i, j)).abs();
      if (difference > 1e-9) {
        throw StateError('the two implementations disagree at ($i, $j)');
      }
    }
  }

  for (final size in [2, 6, 18]) {
    HandRolled(size).report();
    WithMatrices(size).report();
  }

  // And the number that actually matters: the same question asked of a whole
  // smoothing pass, where allocation pressure compounds over the series.
  print('');
  print('One covariance prediction, P- = A P A\' + Q, per line.');
}
