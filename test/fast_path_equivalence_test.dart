import 'dart:math' as math;
import 'dart:typed_data';

import 'package:state_space/src/engine/fast_path_2x2.dart';
import 'package:state_space/src/engine/kalman.dart';
import 'package:state_space/src/engine/rts.dart';
import 'package:state_space/src/engine/timeline.dart';
import 'package:state_space/state_space.dart';
import 'package:test/test.dart';

/// The generic engine is the definition of the answer; the fast path is an
/// optimisation of it. So every assertion here is one-sided in spirit: if the
/// two disagree, the fast path is wrong.
final component = LocalLinearTrend(processVariance: 7e-4);
const measurementVariance = 0.06;

/// One flat state and one proper one, which the fast path handles only under
/// an approximate prior.
final damped = DampedLinearTrend(processVariance: 7e-4, timeScale: 9);

/// A deliberately awkward series: uneven gaps, a repeated timestamp, a long
/// hole, unequal weights, and days with no reading at all.
({List<Observation> observations, Float64List missing}) awkwardSeries({
  int seed = 5,
}) {
  final random = math.Random(seed);
  final observations = <Observation>[];
  final missing = <double>[];
  var time = 0.0;

  for (var i = 0; i < 80; i++) {
    if (i == 20) {
      // Two readings at the same instant.
    } else if (i == 45) {
      time += 60; // a two-month hole
    } else {
      time += 0.3 + 2.5 * random.nextDouble();
    }

    if (i % 11 == 7) {
      missing.add(time); // asked about, never measured
      continue;
    }
    observations.add(
      Observation(
        time,
        75 +
            0.02 * time +
            math.sin(time / 12) +
            0.4 * (random.nextDouble() - 0.5),
        relativeVariance: i % 9 == 0 ? 3.0 : 1.0,
      ),
    );
  }
  return (observations: observations, missing: Float64List.fromList(missing));
}

void expectClose(double fast, double generic, double relative, String where) {
  expect(
    fast,
    closeTo(generic, relative * math.max(1, generic.abs())),
    reason: where,
  );
}

void main() {
  final series = awkwardSeries();
  final timeline = Timeline.merge(series.observations, series.missing);

  for (final (component, initialization) in <(Component, Initialization)>[
    (component, const ExactDiffuse()),
    (component, ApproximateDiffuse()),
    (component, ApproximateDiffuse(variance: 1e3)),
    (damped, ApproximateDiffuse()),
    (damped, ApproximateDiffuse(variance: 1e3)),
  ]) {
    group('${component.name}, $initialization', () {
      test('the fast path is chosen at all', () {
        expect(FastPath2x2.handles([component], initialization), isTrue);
      });

      test('forward pass agrees to 1e-12', () {
        final fast = FastPath2x2(
          component,
          measurementVariance: measurementVariance,
          initialization: initialization,
        ).run(timeline, keepHistory: true);
        final generic = KalmanFilter(
          [component],
          measurementVariance: measurementVariance,
          initialization: initialization,
        ).run(timeline, keepHistory: true);

        expect(fast.usedObservations, generic.usedObservations);
        expect(fast.diffuseDim, generic.diffuseDim);
        expectClose(fast.logLikelihood, generic.logLikelihood, 1e-12, 'llf');
        expectClose(
          fast.sumLogInnovationVariance,
          generic.sumLogInnovationVariance,
          1e-12,
          'sum log S',
        );
        expectClose(
          fast.sumWeightedSquares,
          generic.sumWeightedSquares,
          1e-12,
          'residual sum',
        );
        expectClose(
          fast.diffuseLogDeterminant,
          generic.diffuseLogDeterminant,
          1e-12,
          'log|M|',
        );

        for (var t = 0; t < timeline.length; t++) {
          for (var i = 0; i < 2; i++) {
            expectClose(
              fast.stateMean![t * 2 + i],
              generic.stateMean![t * 2 + i],
              1e-12,
              'x[$t][$i]',
            );
            expectClose(
              fast.predictedMean![t * 2 + i],
              generic.predictedMean![t * 2 + i],
              1e-12,
              'x-[$t][$i]',
            );
          }
          for (var e = 0; e < 4; e++) {
            expectClose(
              fast.stateCovariance![t * 4 + e],
              generic.stateCovariance![t * 4 + e],
              1e-12,
              'P[$t][$e]',
            );
            expectClose(
              fast.predictedCovariance![t * 4 + e],
              generic.predictedCovariance![t * 4 + e],
              1e-12,
              'P-[$t][$e]',
            );
          }
        }

        if (fast.diffuseDim > 0) {
          for (var c = 0; c < 2; c++) {
            expectClose(
              fast.diffuseMean![c],
              generic.diffuseMean![c],
              1e-12,
              'dhat[$c]',
            );
          }
          for (var e = 0; e < 4; e++) {
            expectClose(
              fast.diffuseCovariance![e],
              generic.diffuseCovariance![e],
              1e-12,
              'S[$e]',
            );
          }
          for (var t = 0; t < timeline.length; t++) {
            for (var e = 0; e < 4; e++) {
              expectClose(
                fast.stateSensitivity![t * 4 + e],
                generic.stateSensitivity![t * 4 + e],
                1e-12,
                'Xb[$t][$e]',
              );
            }
          }
        }
      });

      test('and so does the posterior once smoothed through it', () {
        // The backward pass is shared, so this checks that the fast path hands
        // it inputs the smoother is equally happy with -- including the
        // sensitivity history the diffuse combination needs.
        Float64List smoothedWith(FilterResult result) {
          RtsSmoother([component])
            ..smoothInPlace(timeline, result)
            ..combineDiffuse(result);
          return result.stateMean!;
        }

        final fast = smoothedWith(
          FastPath2x2(
            component,
            measurementVariance: measurementVariance,
            initialization: initialization,
          ).run(timeline, keepHistory: true),
        );
        final generic = smoothedWith(
          KalmanFilter(
            [component],
            measurementVariance: measurementVariance,
            initialization: initialization,
          ).run(timeline, keepHistory: true),
        );

        for (var i = 0; i < generic.length; i++) {
          expectClose(fast[i], generic[i], 1e-12, 'smoothed[$i]');
        }
      });
    });
  }

  group('when the fast path does not apply', () {
    test('a two-component model goes down the generic path', () {
      expect(
        FastPath2x2.handles([
          component,
          LocalLevel(processVariance: 1e-3),
        ], const ExactDiffuse()),
        isFalse,
      );
    });

    test('so does a damped trend under exact initialisation', () {
      expect(FastPath2x2.handles([damped], const ExactDiffuse()), isFalse);
    });

    test('so does a one-state component', () {
      expect(
        FastPath2x2.handles([
          LocalLevel(processVariance: 1e-3),
        ], const ExactDiffuse()),
        isFalse,
      );
    });

    test('and the dispatcher still returns the generic result for those', () {
      // A trend and a level in one model are confounded -- both contribute a
      // level and nothing distinguishes them -- so under a flat prior the
      // diffuse information matrix really is singular and exact
      // initialisation says so. The wide prior answers anyway, which is what
      // makes it the right choice for checking the dispatch.
      final components = [component, LocalLevel(processVariance: 1e-3)];
      final viaDispatch = forwardPass(
        components,
        timeline,
        measurementVariance: measurementVariance,
        initialization: ApproximateDiffuse(),
      );
      final direct = KalmanFilter(
        components,
        measurementVariance: measurementVariance,
        initialization: ApproximateDiffuse(),
      ).run(timeline);
      expect(viaDispatch.logLikelihood, direct.logLikelihood);
    });

    test(
      'confounded components under a flat prior are refused, not guessed',
      () {
        expect(
          () => forwardPass(
            [component, LocalLevel(processVariance: 1e-3)],
            timeline,
            measurementVariance: measurementVariance,
            initialization: const ExactDiffuse(),
          ),
          throwsA(
            isA<UnderdeterminedModelException>().having(
              (e) => e.message,
              'message',
              contains('does not determine'),
            ),
          ),
        );
      },
    );
  });
}
