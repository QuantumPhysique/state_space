import 'dart:math' as math;

import 'package:state_space/state_space.dart';
import 'package:test/test.dart';

/// `Q(dt) = P_inf - A(dt) P_inf A(dt)'` is a difference of two quantities of
/// order `variance` whose answer is `O(dt^3)` for `nu = 3/2` and `O(dt^5)` for
/// `nu = 5/2`. Written that way it loses every correct digit over a short gap
/// and eventually goes negative, which costs `Q` its positive
/// semi-definiteness.
///
/// Whether a caller reaches that depends on the time unit and not on anything
/// the component can see: `rate * dt` is `1.7e-5` for a length scale of `1e5`
/// seconds sampled every second.
void main() {
  double at(Matern m, double dt, int i, int j) {
    final block = MatrixBlock.dense(m.stateDim, m.stateDim);
    block.fill(0);
    m.processNoise(dt, block);
    return block.at(i, j);
  }

  /// Leading term of `Q(0,0)`, from `integral A(s) q e e' A(s)' ds` with the
  /// exponential set to one: `q dt^(2p-1) / ((p-1)!^2 (2p-1))`.
  double leadingQ00(Matern m, double dt) {
    final u = m.rate * dt;
    return m.variance *
        switch (m.order) {
          MaternOrder.oneHalf => 2 * u,
          MaternOrder.threeHalves => 4 * u * u * u / 3,
          MaternOrder.fiveHalves => 4 * math.pow(u, 5) / 15,
        };
  }

  group('the process noise survives a short gap', () {
    for (final order in MaternOrder.values) {
      test('${order.name}: Q(0,0) keeps its leading term down to dt = 1e-9',
          () {
        final m = Matern(order: order, variance: 1, lengthScale: 10);
        for (final dt in [1e-2, 1e-3, 1e-5, 1e-7, 1e-9]) {
          final q = at(m, dt, 0, 0);
          final leading = leadingQ00(m, dt);
          // The leading term is only the first of a series in u = rate * dt,
          // so allow the next order — and nothing more. Written the old way
          // this was 745 times too large at dt = 1e-3.
          final slack = leading * (3 * m.rate * dt + 1e-12);
          expect(q, closeTo(leading, slack), reason: 'dt = $dt');
        }
      });

      test('${order.name}: Q stays positive semi-definite', () {
        final m = Matern(order: order, variance: 1, lengthScale: 10);
        final n = m.stateDim;
        for (final dt in [1e-1, 1e-3, 1e-6, 1e-9, 1e-12]) {
          final block = MatrixBlock.dense(n, n);
          block.fill(0);
          m.processNoise(dt, block);
          // Leading principal minors, by elimination.
          for (var k = 1; k <= n; k++) {
            final a = [
              for (var i = 0; i < k; i++)
                [for (var j = 0; j < k; j++) block.at(i, j)]
            ];
            var determinant = 1.0;
            for (var c = 0; c < k; c++) {
              if (a[c][c] == 0) {
                determinant = 0;
                break;
              }
              determinant *= a[c][c];
              for (var r = c + 1; r < k; r++) {
                final f = a[r][c] / a[c][c];
                for (var cc = c; cc < k; cc++) {
                  a[r][cc] -= f * a[c][cc];
                }
              }
            }
            expect(determinant, greaterThanOrEqualTo(0.0),
                reason: '${order.name}, dt = $dt, leading minor $k');
          }
        }
      });

      test('${order.name}: the two branches meet at the threshold', () {
        final m = Matern(order: order, variance: 1.7, lengthScale: 7);
        // 0.1 / rate is where processNoise switches; step either side of it.
        final boundary = 0.1 / m.rate;
        for (var i = 0; i < m.stateDim; i++) {
          for (var j = 0; j < m.stateDim; j++) {
            final below = at(m, boundary * (1 - 1e-9), i, j);
            final above = at(m, boundary * (1 + 1e-9), i, j);
            expect(below, closeTo(above, 1e-8 * math.max(above.abs(), 1e-30)),
                reason: 'entry ($i, $j)');
          }
        }
      });
    }

    test('and a filter over microscopic gaps still runs', () {
      // Length scale 1e5, readings one unit apart: rate * dt = 1.7e-5, which
      // is where the closed form had about one correct digit.
      final data = [
        for (var i = 0; i < 200; i++)
          Observation(i.toDouble(), 80 + math.sin(i / 40) + 0.01 * (i % 7))
      ];
      final model = StructuralModel(
        [Matern.threeHalves(variance: 1, lengthScale: 1e5)],
        measurementVariance: 0.01,
      );
      final posterior = model.smooth(data);
      expect(posterior.logMarginalLikelihood.isFinite, isTrue);
      for (var i = 0; i < posterior.length; i++) {
        expect(posterior.levelVariance[i], greaterThanOrEqualTo(0.0));
      }
    });
  });
}
