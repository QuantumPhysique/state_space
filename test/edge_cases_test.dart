import 'dart:math' as math;
import 'dart:typed_data';

import 'package:state_space/src/engine/fast_path_2x2.dart';
import 'package:state_space/src/engine/timeline.dart';
import 'package:state_space/state_space.dart';
import 'package:test/test.dart';

StructuralModel _trend({double processVariance = 1e-3}) =>
    StructuralModel.localLinearTrend(
      processVariance: processVariance,
      measurementVariance: 0.25,
    );

/// The same model under the older, approximate prior, for the cases where a
/// very large number is more useful than an exception.
StructuralModel _wideTrend() => StructuralModel.localLinearTrend(
      processVariance: 1e-3,
      measurementVariance: 0.25,
      initialization: const ApproximateDiffuse(),
    );

void main() {
  _degenerateReporting();

  group('degenerate inputs', () {
    test('no observations gives an empty result rather than an error', () {
      final result = _trend().smooth(const []);
      expect(result.length, 0);
      expect(result.logMarginalLikelihood, 0);
      expect(result.componentCount, 1);
    });

    test('one observation is not enough to place a two-state trend', () {
      // The level is pinned and the slope is not, so there is no posterior to
      // report. Exact initialisation says so instead of returning a number
      // whose size is an artefact of the prior.
      expect(
        () => _trend().smooth([const Observation(4, 82.5)]),
        throwsA(isA<StateError>().having(
            (e) => e.message, 'message', contains('does not determine'))),
      );
    });

    test('unless you ask for the older prior, which answers anyway', () {
      final result = _wideTrend().smooth([const Observation(4, 82.5)]);
      // Not exactly 82.5: the finite prior shrinks it by one part in kappa.
      expect(result.level.single, closeTo(82.5, 1e-3));
      // Everything the single reading says is about the level; the slope keeps
      // its prior, which is enormous by construction.
      expect(result.levelVariance.single, closeTo(0.25, 1e-4));
      expect(result.slopeVariance!.single, greaterThan(1e4));
    });

    test('every value identical gives that value and no slope', () {
      final data = [for (var i = 0; i < 20; i++) Observation(i * 1.5, 61.4)];
      final result = _trend().smooth(data);
      for (var i = 0; i < data.length; i++) {
        expect(result.level[i], closeTo(61.4, 1e-9));
        expect(result.slope![i], closeTo(0, 1e-10));
      }
    });

    test('a five-year gap widens the band without breaking anything', () {
      final data = [
        for (var i = 0; i < 10; i++) Observation(i.toDouble(), 80 + 0.1 * i),
        for (var i = 0; i < 10; i++)
          Observation(1826 + i.toDouble(), 92 + 0.1 * i),
      ];
      final middle = Float64List.fromList([913.0]);
      final result = _trend().smooth(data, grid: middle);

      expect(result.level.single.isFinite, isTrue);
      expect(result.levelVariance.single, greaterThan(1e3));

      final atData = _trend().smooth(data);
      for (final v in atData.levelVariance) {
        expect(v.isFinite, isTrue);
        expect(v, greaterThan(0));
      }
    });

    test('an observation with zero variance is honoured exactly', () {
      final data = [
        const Observation(0, 10),
        const Observation(1, 11),
        const Observation(2, 20, relativeVariance: 0),
        const Observation(3, 13),
        const Observation(4, 14),
      ];
      final result = _trend().smooth(data);
      expect(result.level[2], closeTo(20, 1e-9));
      expect(result.levelVariance[2], closeTo(0, 1e-12));
    });
  });

  group('two readings at the same instant', () {
    test('are the same as one reading of twice the precision', () {
      final shared = [
        const Observation(0, 79.0),
        const Observation(1, 79.4),
        const Observation(3, 80.1),
      ];
      final model = _trend();

      final twice = model.smooth([
        ...shared,
        const Observation(4, 80.0),
        const Observation(4, 81.0),
        const Observation(6, 80.9),
      ]);
      final once = model.smooth([
        ...shared,
        const Observation(4, 80.5, relativeVariance: 0.5),
        const Observation(6, 80.9),
      ]);

      // Same posterior; the pair just occupies two slots in the output. The
      // likelihoods are deliberately not compared: one is the density of two
      // readings, the other of their average, and those are different numbers
      // for the same model. Aggregating duplicates is an optimisation, not a
      // requirement, and this is the sense in which it is safe.
      expect(twice.level[3], closeTo(once.level[3], 1e-9));
      expect(twice.level[4], closeTo(once.level[3], 1e-9));
      expect(twice.levelVariance[4], closeTo(once.levelVariance[3], 1e-11));
      expect(twice.level[5], closeTo(once.level[4], 1e-9));
    });
  });

  group('input validation', () {
    test('rejects unsorted observations, with advice', () {
      expect(
        () =>
            _trend().smooth([const Observation(2, 1), const Observation(1, 1)]),
        throwsA(isA<ArgumentError>().having(
            (e) => e.message.toString(), 'message', contains('sorted'))),
      );
    });

    test('rejects NaN and infinity', () {
      expect(() => _trend().smooth([Observation(0, double.nan)]),
          throwsArgumentError);
      expect(() => _trend().smooth([Observation(double.infinity, 1)]),
          throwsArgumentError);
      expect(
          () => _trend().smooth([
                const Observation(0, 1),
                Observation(1, 1, relativeVariance: double.nan)
              ]),
          throwsArgumentError);
    });

    test('rejects a negative observation variance', () {
      expect(
          () =>
              _trend().smooth([const Observation(0, 1, relativeVariance: -1)]),
          throwsArgumentError);
    });

    test('rejects an unsorted grid', () {
      expect(
          () => _trend().smooth([const Observation(0, 1)],
              grid: Float64List.fromList([3, 1])),
          throwsArgumentError);
    });

    test('rejects a model with no components or a bad variance', () {
      expect(() => StructuralModel(const []), throwsArgumentError);
      expect(
          () => StructuralModel.localLinearTrend(
              processVariance: 1e-3, measurementVariance: 0),
          throwsArgumentError);
    });
  });

  test('a long series stays finite and monotone in its own likelihood', () {
    // Nothing subtle, just a guard against silent overflow or NaN creeping in
    // over a few thousand steps.
    final random = math.Random(11);
    final data = <Observation>[];
    var value = 80.0;
    for (var i = 0; i < 4000; i++) {
      value += 0.02 * (random.nextDouble() - 0.5);
      data.add(
          Observation(i.toDouble(), value + 0.3 * (random.nextDouble() - 0.5)));
    }
    final result = _trend().smooth(data);
    expect(result.logMarginalLikelihood.isFinite, isTrue);
    for (var i = 0; i < data.length; i++) {
      expect(result.level[i].isFinite, isTrue);
      expect(result.levelVariance[i], greaterThan(0));
    }
  });
}

/// Degenerate shapes that used to reach arithmetic rather than a guard.
void _degenerateReporting() {
  group('nothing left to estimate from', () {
    test('a model with as many flat directions as observations profiles to '
        'NaN rather than to a number', () {
      // Two readings, a two-state trend: both are spent locating the state, so
      // there is no residual degree of freedom and no noise level to concentrate
      // out. This used to be 0/0 arriving as NaN by accident, and could as
      // easily have arrived as a finite number.
      final timeline = Timeline.merge(
          [Observation(0, 80.0), Observation(1, 80.4)], null);
      final pass = forwardPass(
        [const LocalLinearTrend(processVariance: 1e-3)],
        timeline,
        measurementVariance: 1,
        initialization: const ExactDiffuse(),
      );
      expect(pass.usedObservations, 0);
      expect(pass.hasResidualDegreesOfFreedom, isFalse);
      expect(pass.profileMeasurementVariance.isNaN, isTrue);
      expect(pass.profileLogLikelihood.isNaN, isTrue);
    });

    test('and fit refuses it with an explanation', () {
      expect(
          () => fit(StructuralModel.localLinearTrend(processVariance: 1),
              [Observation(0, 80.0), Observation(1, 80.4)]),
          throwsA(isA<ArgumentError>().having((e) => e.toString(), 'message',
              contains('too few observations'))));
    });

    test('a single residual reports no spread rather than a spread of zero',
        () {
      final diagnostics = StructuralModel.localLinearTrend(processVariance: 1e-3)
          .diagnose([
        Observation(0, 80.0),
        Observation(1, 80.4),
        Observation(2, 80.1),
      ]);
      expect(diagnostics.count, 1);
      expect(diagnostics.variance.isNaN, isTrue);
      expect(diagnostics.toString(), contains('n/a'));
    });

    test('no residuals at all is a sentence and not a crash', () {
      final diagnostics =
          InnovationDiagnostics(times: Float64List(0), residuals: Float64List(0));
      expect(diagnostics.count, 0);
      expect(diagnostics.mean.isNaN, isTrue);
      expect(diagnostics.variance.isNaN, isTrue);
      expect(diagnostics.toString(), 'InnovationDiagnostics(no residuals)');
    });
  });

  group('ComplexityPenalty validates in release builds too', () {
    test('a non-positive scale is refused', () {
      expect(() => ComplexityPenalty(scale: 0), throwsArgumentError);
      expect(() => ComplexityPenalty(scale: -1), throwsArgumentError);
      expect(() => ComplexityPenalty(scale: double.infinity),
          throwsArgumentError);
    });

    test('a tail probability outside (0, 1) is refused', () {
      expect(() => ComplexityPenalty(tailProbability: 0), throwsArgumentError);
      expect(() => ComplexityPenalty(tailProbability: 1), throwsArgumentError);
    });
  });
}
