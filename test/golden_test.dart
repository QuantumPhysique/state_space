import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:state_space/src/engine/kalman.dart';
import 'package:state_space/src/engine/rts.dart';
import 'package:state_space/src/engine/timeline.dart';
import 'package:state_space/state_space.dart';
import 'package:test/test.dart';

/// A fixture produced by `tool/generate_fixtures.py`.
class Fixture {
  Fixture(Map<String, dynamic> json)
    : name = json['name'] as String,
      description = json['description'] as String,
      initializationName = json['initialization'] as String,
      diffuseObservations = json['diffuseObservations'] as int,
      processVariance = (json['processVariance'] as num).toDouble(),
      measurementVariance = (json['measurementVariance'] as num).toDouble(),
      diffuseVariance = (json['diffuseVariance'] as num).toDouble(),
      times = _doubles(json['times']),
      values = (json['values'] as List)
          .map((v) => v == null ? null : (v as num).toDouble())
          .toList(),
      relativeVariances = _doubles(json['relativeVariances']),
      predictedState = _doubles(json['predictedState']),
      predictedStateCov = _doubles(json['predictedStateCov']),
      filteredState = _doubles(json['filteredState']),
      filteredStateCov = _doubles(json['filteredStateCov']),
      smoothedState = _doubles(json['smoothedState']),
      smoothedStateCov = _doubles(json['smoothedStateCov']),
      loglikelihoodObs = _doubles(json['loglikelihoodObs']);

  static Fixture load(String name) => Fixture(
    jsonDecode(File('test/fixtures/$name.json').readAsStringSync())
        as Map<String, dynamic>,
  );

  static Float64List _doubles(Object? raw) => Float64List.fromList([
    for (final v in raw! as List) (v as num).toDouble(),
  ]);

  final String name;
  final String description;

  /// Which prior statsmodels was given: `approximate` or `diffuse`.
  final String initializationName;

  /// How many leading observations statsmodels spent resolving the flat
  /// directions.
  final int diffuseObservations;
  final double processVariance;
  final double measurementVariance;
  final double diffuseVariance;
  final Float64List times;
  final List<double?> values;
  final Float64List relativeVariances;
  final Float64List predictedState;
  final Float64List predictedStateCov;
  final Float64List filteredState;
  final Float64List filteredStateCov;
  final Float64List smoothedState;
  final Float64List smoothedStateCov;
  final Float64List loglikelihoodObs;

  int get steps => times.length;

  /// The observed points, in time order.
  List<Observation> get observations => [
    for (var t = 0; t < steps; t++)
      if (values[t] != null)
        Observation(
          times[t],
          values[t]!,
          relativeVariance: relativeVariances[t],
        ),
  ];

  /// Whether every step is the same length, which is what decides whether
  /// statsmodels' transition matrix is genuinely time-varying.
  bool get hasConstantStep {
    for (var t = 2; t < steps; t++) {
      if ((times[t] - times[t - 1] - (times[1] - times[0])).abs() > 1e-12) {
        return false;
      }
    }
    return true;
  }

  /// The times with no observation, which the Dart side reproduces by asking
  /// for output there. A step with nothing to update on is a step either way.
  Float64List get missingTimes => Float64List.fromList([
    for (var t = 0; t < steps; t++)
      if (values[t] == null) times[t],
  ]);

  Initialization get initialization => initializationName == 'diffuse'
      ? const ExactDiffuse()
      : ApproximateDiffuse(variance: diffuseVariance);

  StructuralModel get model => StructuralModel.localLinearTrend(
    processVariance: processVariance,
    measurementVariance: measurementVariance,
    initialization: initialization,
  );
}

/// Compares against a reference whose magnitude varies over ten orders of
/// magnitude, from a smoothed slope near 1e-3 to a diffuse prior at 1e5.
void expectClose(
  double actual,
  double expected,
  double relative,
  String where,
) {
  final tolerance = relative * math.max(1, expected.abs());
  expect(actual, closeTo(expected, tolerance), reason: where);
}

/// A local linear trend has two non-stationary states, so the first two
/// observations go entirely into locating them.
const diffuseStates = 2;

void main() {
  for (final name in ['regular', 'irregular', 'missing']) {
    group('statsmodels fixture "$name"', () {
      late Fixture fixture;
      late Timeline timeline;
      late FilterResult forward;

      setUp(() {
        fixture = Fixture.load(name);
        timeline = Timeline.merge(fixture.observations, fixture.missingTimes);
        forward = KalmanFilter(
          fixture.model.components,
          measurementVariance: fixture.measurementVariance,
          initialization: ApproximateDiffuse(variance: fixture.diffuseVariance),
          burnIn: 0,
        ).run(timeline, keepHistory: true);
      });

      test('reproduces the reference timeline', () {
        // Missing observations come back as output-only steps, so the two
        // implementations walk the same sequence of time points.
        expect(timeline.length, fixture.steps);
        expect(timeline.observationCount, fixture.observations.length);
      });

      test('predicted and filtered states', () {
        for (var t = 0; t < fixture.steps; t++) {
          for (var i = 0; i < 2; i++) {
            expectClose(
              forward.predictedMean![t * 2 + i],
              fixture.predictedState[t * 2 + i],
              1e-10,
              'predicted[$t][$i]',
            );
            expectClose(
              forward.stateMean![t * 2 + i],
              fixture.filteredState[t * 2 + i],
              1e-10,
              'filtered[$t][$i]',
            );
          }
          for (var e = 0; e < 4; e++) {
            expectClose(
              forward.predictedCovariance![t * 4 + e],
              fixture.predictedStateCov[t * 4 + e],
              1e-9,
              'P-[$t][$e]',
            );
            expectClose(
              forward.stateCovariance![t * 4 + e],
              fixture.filteredStateCov[t * 4 + e],
              1e-9,
              'P[$t][$e]',
            );
          }
        }
      });

      test('log likelihood, term by term', () {
        var total = 0.0;
        for (final term in fixture.loglikelihoodObs) {
          total += term;
        }
        expectClose(forward.logLikelihood, total, 1e-10, 'llf');
      });

      test('diffuse burn-in drops exactly the first two observations', () {
        final burned = KalmanFilter(
          fixture.model.components,
          measurementVariance: fixture.measurementVariance,
          initialization: ApproximateDiffuse(variance: fixture.diffuseVariance),
        ).run(timeline);

        var total = 0.0;
        var seen = 0;
        for (var t = 0; t < fixture.steps; t++) {
          if (fixture.values[t] == null) continue;
          seen++;
          if (seen > 2) total += fixture.loglikelihoodObs[t];
        }
        expectClose(burned.logLikelihood, total, 1e-10, 'burned llf');
      });

      test('smoothed states', () {
        RtsSmoother(fixture.model.components).smoothInPlace(timeline, forward);
        for (var t = 0; t < fixture.steps; t++) {
          for (var i = 0; i < 2; i++) {
            expectClose(
              forward.stateMean![t * 2 + i],
              fixture.smoothedState[t * 2 + i],
              1e-9,
              'smoothed[$t][$i]',
            );
          }
        }
      });

      test('smoothed covariances, once the diffuse prior has washed out', () {
        RtsSmoother(fixture.model.components).smoothInPlace(timeline, forward);
        for (var t = diffuseStates; t < fixture.steps; t++) {
          for (var e = 0; e < 4; e++) {
            expectClose(
              forward.stateCovariance![t * 4 + e],
              fixture.smoothedStateCov[t * 4 + e],
              1e-11,
              'P^s[$t][$e]',
            );
          }
        }
      });

      test('smoothed covariances over the first two steps, to four digits', () {
        // At the first steps the smoothed covariance is the difference of two
        // quantities of order kappa = 1e6 times the measurement variance, and
        // the answer is of order 1e-3. Nine digits go into that subtraction
        // before either implementation has done anything wrong, and the two
        // do it by different routes: this package uses the RTS gain form,
        // statsmodels the disturbance-smoother form. Exact diffuse
        // initialisation removes the subtraction rather than tightening
        // the tolerance.
        //
        // The point of asserting it at all is that four digits is a floor,
        // not a shrug: a genuine bug in the backward pass shows up here as a
        // gross disagreement, and everywhere else as a failure of the test
        // above.
        RtsSmoother(fixture.model.components).smoothInPlace(timeline, forward);
        for (var t = 0; t < diffuseStates; t++) {
          for (var e = 0; e < 4; e++) {
            expectClose(
              forward.stateCovariance![t * 4 + e],
              fixture.smoothedStateCov[t * 4 + e],
              1e-3,
              'P^s[$t][$e]',
            );
          }
        }
      });
    });
  }

  for (final name in ['regular', 'irregular', 'missing']) {
    group('statsmodels fixture "$name-diffuse"', () {
      late Fixture fixture;
      late Timeline timeline;
      late FilterResult smoothed;

      setUp(() {
        fixture = Fixture.load('$name-diffuse');
        timeline = Timeline.merge(fixture.observations, fixture.missingTimes);
        smoothed = KalmanFilter(
          fixture.model.components,
          measurementVariance: fixture.measurementVariance,
          initialization: const ExactDiffuse(),
        ).run(timeline, keepHistory: true);
        RtsSmoother(fixture.model.components)
          ..smoothInPlace(timeline, smoothed)
          ..combineDiffuse(smoothed);
      });

      test('agrees on how many observations the flat directions cost', () {
        expect(fixture.diffuseObservations, diffuseStates);
        expect(
          smoothed.usedObservations,
          fixture.observations.length - diffuseStates,
        );
      });

      test('smoothed states and covariances, from the very first step', () {
        // No carve-out for the start of the series this time. The four digits
        // the approximate prior lost over the first two steps are simply not
        // lost, and this is the assertion that says so against an independent
        // implementation rather than against our own limit.
        //
        // One exception, and it is statsmodels' rather than ours: with a
        // genuinely time-varying transition matrix its exact diffuse smoother
        // reports a different slope at the very first step. The evidence that
        // the difference is theirs: statsmodels agrees with a dense
        // generalised-least-squares computation to 1e-14 whenever the step is
        // constant -- at unit steps, at 2.5, at 0.5 -- and disagrees only once
        // the steps vary, while this package agrees with the dense form in
        // every case. Our own value is pinned directly against that dense form
        // in dense_gp_reference_test.dart, so skipping the entry here loses no
        // coverage; it just declines to assert somebody else's bug.
        // The exception covers the slope's row and column of the first
        // step's moments -- the mean entry and the three covariance entries
        // that touch it. The level entries agree to 1e-14 even there.
        final trustFirstStepSlope = fixture.hasConstantStep;
        for (var t = 0; t < fixture.steps; t++) {
          for (var i = 0; i < 2; i++) {
            if (t == 0 && i == 1 && !trustFirstStepSlope) continue;
            expectClose(
              smoothed.stateMean![t * 2 + i],
              fixture.smoothedState[t * 2 + i],
              1e-10,
              'smoothed[$t][$i]',
            );
          }
          for (var e = 0; e < 4; e++) {
            if (t == 0 && e != 0 && !trustFirstStepSlope) continue;
            expectClose(
              smoothed.stateCovariance![t * 4 + e],
              fixture.smoothedStateCov[t * 4 + e],
              1e-10,
              'P^s[$t][$e]',
            );
          }
        }
      });
    });
  }
}
