import 'dart:math' as math;
import 'dart:typed_data';

import 'package:state_space/src/engine/kalman.dart';
import 'package:state_space/src/engine/timeline.dart';
import 'package:state_space/state_space.dart';
import 'package:test/test.dart';

import 'support/dense_reference.dart';

/// Irregular readings drifting downwards, with two events on top: a week in
/// March and a fortnight in December.
List<Observation> _series(int n, {int seed = 19}) {
  final random = math.Random(seed);
  final data = <Observation>[];
  var time = 0.0;
  for (var i = 0; i < n; i++) {
    time += 0.6 + 1.4 * random.nextDouble();
    final events = (time >= 80 && time < 87 ? -0.4 : 0.0) +
        (time >= 350 && time < 364 ? 1.2 : 0.0);
    data.add(Observation(
        time, 70 - 0.002 * time + events + 0.25 * (random.nextDouble() - 0.5),
        relativeVariance: 0.7 + random.nextDouble()));
  }
  return data;
}

RegressionComponent _events() => RegressionComponent([
      IndicatorRegressor('conference', [(from: 80, to: 87)]),
      IndicatorRegressor('holiday', [(from: 350, to: 364)]),
    ]);

void main() {
  const processVariance = 2e-6;
  const measurementVariance = 0.02;

  group('regression coefficients against generalised least squares', () {
    // A regression coefficient is a flat direction like any other, so the
    // exact diffuse machinery estimates it as a by-product. The claim being
    // tested is that "as a by-product" means *exactly*, not approximately:
    // the smoothed coefficient is the generalised least-squares estimate a
    // dense computation would produce, and the variance the smoother reports
    // is the corresponding diagonal of `(B' C^-1 B)^-1`.
    final data = _series(330);
    final components = [
      LocalLinearTrend(processVariance: processVariance),
      _events(),
    ];
    final origin = data.first.time;
    final regressors = _events().regressors;

    Basis basis() => (s) => [
          1,
          s,
          for (final regressor in regressors) regressor.at(s + origin),
        ];

    test('four flat directions give the dense restricted likelihood', () {
      final forward = KalmanFilter(
        components,
        measurementVariance: measurementVariance,
        initialization: const ExactDiffuse(),
      ).run(Timeline.merge(data, null));

      final dense = restrictedLikelihood(
        data,
        splineKernel(processVariance),
        basis(),
        measurementVariance: measurementVariance,
      );

      expect(forward.diffuseDim, 4);
      expect(forward.usedObservations, data.length - 4);
      expect(forward.logLikelihood, closeTo(dense.logLikelihood, 1e-9));
    });

    test('the smoothed coefficients are the least-squares estimates', () {
      final dense = restrictedLikelihood(
        data,
        splineKernel(processVariance),
        basis(),
        measurementVariance: measurementVariance,
      );
      final posterior = StructuralModel(
        components,
        measurementVariance: measurementVariance,
      ).smooth(data);

      expect(
          posterior.coefficients.map((c) => c.name), ['conference', 'holiday']);
      for (var j = 0; j < 2; j++) {
        final coefficient = posterior.coefficients[j];
        expect(coefficient.estimate, closeTo(dense.estimate[2 + j], 1e-9),
            reason: coefficient.name);
        expect(coefficient.variance, closeTo(dense.variance[2 + j], 1e-11),
            reason: coefficient.name);
      }
    });

    test('and the level and slope alongside them', () {
      // The same four-column solve produces the trend's own flat directions,
      // so this is the check that the regression columns have not disturbed
      // the ones that were already there.
      final dense = restrictedLikelihood(
        data,
        splineKernel(processVariance),
        basis(),
        measurementVariance: measurementVariance,
      );
      final posterior = StructuralModel(
        components,
        measurementVariance: measurementVariance,
      ).smooth(data);

      // At the first output time the trend's state *is* the pair of flat
      // directions, and its contribution to the signal is the level.
      expect(posterior.componentMean(0)[0], closeTo(dense.estimate[0], 1e-9));
      expect(posterior.trendSlope![0], closeTo(dense.estimate[1], 1e-9));
    });

    test('a coefficient reported the same way at every step', () {
      // A constant state under a flat prior has the same full-data posterior
      // everywhere, which is why one number is the right thing to report.
      // Worth asserting, because the smoother arrives at each step by a
      // different route.
      final model =
          StructuralModel(components, measurementVariance: measurementVariance);
      final plain = model.smooth(data);
      final onAGrid = model.smooth(data,
          grid:
              Float64List.fromList([for (var i = 0; i < 5; i++) data[i].time]));
      for (var j = 0; j < 2; j++) {
        expect(onAGrid.coefficients[j].estimate,
            closeTo(plain.coefficients[j].estimate, 1e-12),
            reason: plain.coefficients[j].name);
        expect(onAGrid.coefficients[j].variance,
            closeTo(plain.coefficients[j].variance, 1e-14),
            reason: plain.coefficients[j].name);
      }
    });
  });
}
