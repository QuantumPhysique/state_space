import 'dart:math' as math;

import 'package:state_space/state_space.dart';
import 'package:test/test.dart';

double _gaussian(math.Random random) =>
    math.sqrt(-2 * math.log(random.nextDouble())) *
    math.cos(2 * math.pi * random.nextDouble());

void main() {
  List<Observation> whiteNoise(int n, int seed) {
    final random = math.Random(seed);
    return [
      for (var i = 0; i < n; i++)
        Observation(i.toDouble(), 80 + 0.4 * _gaussian(random))
    ];
  }

  /// A straight line plus noise: nothing for a trend's variance to do, so it
  /// is driven to the bottom of its bracket.
  List<Observation> straightLine(int n, int seed) {
    final random = math.Random(seed);
    return [
      for (var i = 0; i < n; i++)
        Observation(i.toDouble(), 80 + 0.004 * i + 0.3 * _gaussian(random))
    ];
  }

  /// A path actually drawn from a local linear trend, by the same exact
  /// discretisation the filter uses, so the fitted variance has an interior
  /// optimum.
  List<Observation> wanderingTrend(int n, int seed,
      {double q = 1e-4, double r = 0.09}) {
    final random = math.Random(seed);
    var level = 80.0, slope = 0.0;
    return [
      for (var i = 0; i < n; i++)
        () {
          if (i > 0) {
            // Cholesky of q * [[1/3, 1/2], [1/2, 1]] at dt = 1.
            final z1 = _gaussian(random), z2 = _gaussian(random);
            final l11 = math.sqrt(q / 3);
            final l21 = math.sqrt(q) * 0.5 / math.sqrt(1 / 3);
            final l22 = math.sqrt(q * 0.25);
            level += slope + l11 * z1;
            slope += l21 * z1 + l22 * z2;
          }
          return Observation(
              i.toDouble(), level + math.sqrt(r) * _gaussian(random));
        }()
    ];
  }

  FitResult noisyCycle(int seed) => fit(
      StructuralModel([
        const LocalLevel(processVariance: 1e-4),
        StochasticCycle(period: 20, damping: 0.9, stationaryVariance: 0.1),
      ]),
      whiteNoise(400, seed));

  group('a width is only reported in decades where decades mean something', () {
    test('a logit coordinate reports NaN rather than a plausible number', () {
      final fitted = noisyCycle(2);
      // LocalLevel variance, cycle variance, cycle damping (logit), period.
      expect(fitted.parameterSpecs.map((s) => s.label).toList(),
          ['variance', 'variance', 'damping', 'period']);
      expect(fitted.parameterSpecs[2].isLogarithmic, isFalse);
      expect(fitted.plateauDecadesByParameter[2].isNaN, isTrue,
          reason: 'a width in logits is not a width in decades');
      for (final i in [0, 1, 3]) {
        expect(fitted.plateauDecadesByParameter[i].isNaN, isFalse);
      }
    });

    test('the raw width is finite for every parameter', () {
      final fitted = noisyCycle(2);
      expect(fitted.plateauWidthByParameter, hasLength(4));
      for (final width in fitted.plateauWidthByParameter) {
        expect(width.isFinite, isTrue);
        expect(width, greaterThanOrEqualTo(0.0));
      }
    });

    test('decades are the raw width over ln 10 where they are reported', () {
      final fitted = noisyCycle(2);
      for (var i = 0; i < fitted.parameterSpecs.length; i++) {
        if (!fitted.parameterSpecs[i].isLogarithmic) continue;
        expect(fitted.plateauDecadesByParameter[i],
            closeTo(fitted.plateauWidthByParameter[i] / math.ln10, 1e-12));
      }
    });

    test('plateauDecades skips what it cannot compare', () {
      final fitted = noisyCycle(2);
      expect(fitted.plateauDecades.isNaN, isFalse);
      final comparable = [
        for (var i = 0; i < fitted.parameterSpecs.length; i++)
          if (fitted.parameterSpecs[i].isLogarithmic)
            fitted.plateauDecadesByParameter[i]
      ];
      expect(fitted.plateauDecades, comparable.reduce(math.max));
    });

    test('a one-parameter fit is unaffected', () {
      final fitted = fit(StructuralModel.localLinearTrend(processVariance: 1),
          wanderingTrend(300, 4));
      expect(fitted.plateauDecadesByParameter, hasLength(1));
      expect(fitted.plateauDecadesByParameter.first.isNaN, isFalse);
      expect(fitted.plateauDecadesByParameter.first,
          closeTo(fitted.plateauWidthByParameter.first / math.ln10, 1e-12));
    });
  });

  group('warnings say what is wrong with a fit', () {
    test('a well-determined one-component fit has nothing to report', () {
      final fitted = fit(StructuralModel.localLinearTrend(processVariance: 1),
          wanderingTrend(600, 7));
      expect(fitted.atBracketEdge, isFalse);
      expect(fitted.warnings, isEmpty);
    });

    test('a cycle fitted to noise says the period width is conditional', () {
      // The damping is driven to the top of its bracket, where the component
      // is a rigid sinusoid and its likelihood in frequency is as sharp as a
      // periodogram spike. The period then looks pinned to a thousandth of a
      // decade, on data with no cycle in it at all.
      final fitted = noisyCycle(3);
      expect(fitted.plateauDecadesByParameter[3], lessThan(0.05),
          reason: 'the trap: the period reads as superbly determined');

      final warnings = fitted.warnings;
      expect(
          warnings,
          contains(allOf(contains('damping of StochasticCycle'),
              contains('top of its bracket'))));
      expect(
          warnings,
          contains(allOf(
              contains('period of StochasticCycle'),
              contains('damping held on its bound'),
              contains('not how well the data determines it'))));
    });

    test('a parameter shrunk out of the model says so', () {
      // A weekly seasonal on a series with no weekly pattern.
      final fitted = fit(
          StructuralModel([
            const LocalLinearTrend(processVariance: 1e-4),
            TrigonometricSeasonal(period: 7, harmonics: 2, processVariance: 1),
          ]),
          straightLine(400, 11));
      final shrunk = fitted.parameterStatus
          .where((s) => s == ParameterStatus.shrunkToNothing);
      expect(shrunk, isNotEmpty);
      expect(fitted.warnings,
          contains(contains('shrunk to the bottom of its bracket')));
    });

    test('a flat likelihood says the value is a convention', () {
      // Four readings say very little about how fast a trend may wander.
      final fitted = fit(StructuralModel.localLinearTrend(processVariance: 1), [
        Observation(0, 80),
        Observation(1, 80.02),
        Observation(2, 79.98),
        Observation(3, 80.01),
        Observation(4, 80),
      ]);
      if (fitted.plateauDecades > 2 && !fitted.atBracketEdge) {
        expect(fitted.warnings, contains(contains('a convention rather than')));
      }
      expect(fitted.warnings, isNotEmpty);
    });
  });
}
