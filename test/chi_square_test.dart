import 'dart:math' as math;

import 'package:state_space/src/stats/chi_square.dart';
import 'package:test/test.dart';

/// `(degreesOfFreedom, statistic, upper tail)`, from
/// `scipy.stats.chi2.sf`. The grid straddles both branches of the
/// implementation: the series is used where the statistic is small relative
/// to the degrees of freedom, the continued fraction where it is not.
const List<(int, double, double)> _reference = [
  (1, 0.5, 0.47950012218695337),
  (1, 3.84, 0.050043521248705085),
  (1, 10.0, 0.0015654022580025482),
  (1, 25.0, 5.7330314375838782e-07),
  (1, 60.0, 9.4857375710738541e-15),
  (2, 0.5, 0.77880078307140488),
  (2, 3.84, 0.14660696213035015),
  (2, 25.0, 3.7266531720786718e-06),
  (3, 3.84, 0.2792676171186097),
  (3, 10.0, 0.01856613546304323),
  (3, 60.0, 5.8782307279069194e-13),
  (5, 3.84, 0.57267445983208876),
  (5, 10.0, 0.075235246146512155),
  (5, 25.0, 0.00013933379118562629),
  (10, 3.84, 0.95427630432073585),
  (10, 10.0, 0.44049328506521246),
  (10, 25.0, 0.0053455054871340687),
  (10, 60.0, 3.6243009520614924e-09),
  (20, 10.0, 0.96817194269379514),
  (20, 25.0, 0.20143110494553587),
  (20, 60.0, 7.1217508628155801e-06),
  (50, 25.0, 0.99880755115176834),
  (50, 60.0, 0.15724202723839159),
];

void main() {
  group('the chi-square upper tail', () {
    test('agrees with scipy across both branches', () {
      for (final (df, statistic, expected) in _reference) {
        // Relative, because the far tail values span thirteen decades and an
        // absolute tolerance would be vacuous there.
        final actual = chiSquareUpperTail(statistic, df);
        expect((actual - expected).abs() / expected, lessThan(1e-12),
            reason: 'df $df at $statistic: got $actual, want $expected');
      }
    });

    test('is one at the origin and monotone away from it', () {
      expect(chiSquareUpperTail(0, 4), 1);
      var previous = 1.0;
      for (var x = 0.1; x < 40; x += 0.1) {
        final tail = chiSquareUpperTail(x, 4);
        expect(tail, lessThanOrEqualTo(previous));
        previous = tail;
      }
      expect(previous, lessThan(1e-6));
    });

    test('stays inside [0, 1] deep in either tail', () {
      for (final df in [1, 2, 7, 30]) {
        expect(chiSquareUpperTail(1e-12, df), inInclusiveRange(0, 1));
        expect(chiSquareUpperTail(1e6, df), inInclusiveRange(0, 1));
      }
    });
  });

  group('log gamma', () {
    test('reproduces the values it is pinned to', () {
      // Gamma(1/2) = sqrt(pi), Gamma(n) = (n-1)!, and Gamma is one at both
      // one and two -- enough to catch an off-by-one in the Lanczos shift.
      expect(logGamma(0.5), closeTo(math.log(math.sqrt(math.pi)), 1e-14));
      expect(logGamma(1), closeTo(0, 1e-14));
      expect(logGamma(2), closeTo(0, 1e-14));
      expect(logGamma(6), closeTo(math.log(120), 1e-13));
      expect(logGamma(11), closeTo(math.log(3628800), 1e-12));
    });

    test('refuses arguments it has no expansion for', () {
      expect(() => logGamma(0), throwsArgumentError);
      expect(() => logGamma(-1.5), throwsArgumentError);
    });
  });
}
