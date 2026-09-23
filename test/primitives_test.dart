import 'dart:math' as math;
import 'dart:typed_data';

import 'package:state_space/src/components/stationary.dart';
import 'package:state_space/src/engine/cholesky.dart';
import 'package:state_space/src/fit/golden_section.dart';
import 'package:state_space/src/stats/normal.dart';
import 'package:state_space/state_space.dart';
import 'package:test/test.dart';

void main() {
  group('choleskyFactor', () {
    test('reproduces a known factor and solves against it', () {
      // A = L L' with L = [[2, 0, 0], [1, 3, 0], [-1, 2, 1]].
      final a = Float64List.fromList([4, 2, -2, 2, 10, 5, -2, 5, 6]);
      expect(choleskyFactor(a, 3), isTrue);
      expect(a[0], closeTo(2, 1e-15));
      expect(a[3], closeTo(1, 1e-15));
      expect(a[4], closeTo(3, 1e-15));
      expect(a[6], closeTo(-1, 1e-15));
      expect(a[7], closeTo(2, 1e-15));
      expect(a[8], closeTo(1, 1e-15));
      // A x = b for x = (1, -1, 2): b = (-2, 2, 5).
      final b = Float64List.fromList([-2, 2, 5]);
      choleskySolve(a, 3, b, 0);
      expect(b, [closeTo(1, 1e-14), closeTo(-1, 1e-14), closeTo(2, 1e-14)]);
    });

    test('reports an indefinite matrix rather than factoring it', () {
      expect(choleskyFactor(Float64List.fromList([1, 2, 2, 1]), 2), isFalse);
      expect(choleskyFactor(Float64List.fromList([0, 0, 0, 1]), 2), isFalse);
    });
  });

  group('factorInformation', () {
    test('agrees with choleskyFactor on a well-conditioned matrix', () {
      final a = Float64List.fromList([4, 2, -2, 2, 10, 5, -2, 5, 6]);
      final b = Float64List.fromList(a);
      expect(factorInformation(a, 3), isTrue);
      expect(choleskyFactor(b, 3), isTrue);
      for (var i = 0; i < 9; i++) {
        expect(a[i], closeTo(b[i], 1e-14));
      }
    });

    test('decides rank independently of the units of each direction', () {
      // The same singular matrix with one direction scaled by 1e8: the bare
      // pivot is rounding noise whose sign varies, the relative one is not.
      for (final scale in [1e-8, 1.0, 1e8]) {
        final a = Float64List.fromList([
          1,
          scale,
          scale,
          scale * scale * (1 + 1e-15),
        ]);
        expect(factorInformation(a, 2), isFalse, reason: 'scale $scale');
        final b = Float64List.fromList([1, scale, scale, 2 * scale * scale]);
        expect(factorInformation(b, 2), isTrue, reason: 'scale $scale');
      }
    });
  });

  group('normalQuantile', () {
    // Reference values from scipy.stats.norm.ppf.
    const table = [
      (1e-10, -6.361340902404056),
      (0.001, -3.090232306167813),
      (0.01, -2.326347874040841),
      (0.025, -1.959963984540054),
      (0.2, -0.8416212335729143),
      (0.5, 0.0),
      (0.8, 0.8416212335729143),
      (0.975, 1.959963984540054),
      (0.995, 2.5758293035489004),
      (0.9995, 3.2905267314918945),
    ];
    for (final (p, z) in table) {
      test('at $p', () {
        expect(normalQuantile(p), closeTo(z, 2e-9 * math.max(1, z.abs())));
      });
    }

    test('refuses a probability outside (0, 1)', () {
      for (final p in [0.0, 1.0, -0.1, 2.0, double.nan]) {
        expect(() => normalQuantile(p), throwsArgumentError);
        expect(() => twoSidedZ(p), throwsArgumentError);
      }
    });
  });

  test('credibleInterval scales with the coverage asked for', () {
    final data = [
      for (var i = 0; i < 40; i++)
        Observation(i.toDouble(), 80 + 0.01 * i + 0.1 * math.sin(i * 1.7)),
    ];
    final posterior = StructuralModel.localLinearTrend(
      processVariance: 1e-3,
      measurementVariance: 0.01,
    ).smooth(data);
    double width(double coverage) {
      final band = posterior.credibleInterval(10, coverage: coverage);
      return band.hi - band.lo;
    }

    expect(
      width(0.99) / width(0.95),
      closeTo(2.5758293035489004 / 1.959963984540054, 1e-9),
    );
    expect(
      () => posterior.credibleInterval(10, coverage: 1),
      throwsArgumentError,
    );
    final band = posterior.credibleBand(coverage: 0.99);
    final one = posterior.credibleInterval(10, coverage: 0.99);
    expect(band.lo[10], one.lo);
    expect(band.hi[10], one.hi);
  });

  group('golden-section maximise', () {
    test('finds an interior maximum', () {
      final result = maximise((x) => -(x - 0.3) * (x - 0.3), -2, 2);
      expect(result.argument, closeTo(0.3, 1e-4));
      expect(result.converged, isTrue);
    });

    test('finds a maximum at the edge of the bracket', () {
      final result = maximise((x) => x, -1, 1);
      expect(result.argument, closeTo(1, 1e-4));
    });
  });

  test('stationaryWander matches the closed form for an OU process', () {
    // k(tau) = s2 exp(-tau / l): the path spread over T is
    // s2 - (2 s2 l / T^2) (T - l (1 - exp(-T / l))).
    const s2 = 0.7, l = 3.0;
    for (final span in [1.0, 10.0, 100.0]) {
      final exact =
          s2 -
          2 * s2 * l / (span * span) * (span - l * (1 - math.exp(-span / l)));
      expect(
        stationaryWander((tau) => s2 * math.exp(-tau / l), span),
        closeTo(math.sqrt(exact), 1e-6 * math.sqrt(exact)),
      );
    }
  });
}
