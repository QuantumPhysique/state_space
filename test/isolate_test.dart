import 'dart:isolate';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:state_space/state_space.dart';
import 'package:test/test.dart';

/// What comes back from the worker: enough to prove the answer survived the
/// trip, not just that something did.
typedef Summary = ({
  Float64List level,
  Float64List levelVariance,
  Float64List slope,
  List<double> coefficientEstimates,
  List<String> coefficientNames,
  double logMarginalLikelihood,
  Float64List varianceRatios,
});

/// Runs a whole fit and smooth in the receiving isolate.
///
/// The argument is a record holding the model and the data, so this exercises
/// what `compute()` actually does: send a `StructuralModel` across, use it
/// there, and send a `SmoothingResult`'s contents back.
Summary _worker((StructuralModel, List<Observation>) job) {
  final (model, data) = job;
  final fitted = fit(model, data);
  final posterior = fitted.model.smooth(data);
  return (
    level: posterior.mean,
    levelVariance: posterior.variance,
    slope: posterior.trendSlope!,
    coefficientEstimates: [for (final c in posterior.coefficients) c.estimate],
    coefficientNames: [for (final c in posterior.coefficients) c.name],
    logMarginalLikelihood: posterior.logMarginalLikelihood,
    varianceRatios: fitted.varianceRatios,
  );
}

List<Observation> _data() {
  final random = math.Random(6);
  return [
    for (var day = 0; day < 200; day++)
      Observation(
        day.toDouble(),
        80 -
            0.01 * day +
            0.3 * math.cos(2 * math.pi * day / 7) +
            (day >= 120 && day < 134 ? 0.8 : 0.0) +
            0.2 * (random.nextDouble() - 0.5),
        relativeVariance: day.isEven ? 0.5 : 1.5,
      ),
  ];
}

/// Every component type the package has, in one model, so that adding a
/// closure or a non-transferable field to any of them fails here.
StructuralModel _model() => StructuralModel([
  LocalLinearTrend(processVariance: 1e-4),
  TrigonometricSeasonal(period: 7, harmonics: 2, processVariance: 1e-4),
  RegressionComponent([
    IndicatorRegressor('holiday', [(from: 120, to: 134)]),
    StepRegressor(
      'dose',
      Float64List.fromList([0, 90]),
      Float64List.fromList([0, 1]),
    ),
  ]),
  Matern.oneHalf(variance: 1e-2, lengthScale: 4),
  StochasticCycle(period: 28, damping: 0.95, stationaryVariance: 1e-2),
]);

void main() {
  group('crossing an isolate boundary', () {
    // The package promises that a model can be handed to another isolate and
    // its results handed back, because that is how an application keeps a
    // smoothing pass off the interface thread. Nothing else in the suite
    // exercises that: everything else would pass with a model that throws the
    // moment somebody calls compute().
    //
    // What this actually catches is narrower than it first appears, and the
    // difference is worth writing down. A plain closure in a component sends
    // fine -- since Dart 2.15 closures cross freely within an isolate group,
    // which is what Isolate.run and Flutter's compute() both create. What does
    // not send is anything holding a port, a file handle or another native
    // resource, including a closure that has quietly captured one. So this
    // test guards the promise rather than any particular rule about how to
    // write a component; it fails when something in the model or in a result
    // stops being plain data, whatever the route by which that happened.
    test(
      'a model and its posterior round-trip through Isolate.run unchanged',
      () async {
        final data = _data();
        final here = _worker((_model(), data));
        final there = await Isolate.run(() => _worker((_model(), data)));

        expect(there.level.length, here.level.length);
        for (var i = 0; i < here.level.length; i++) {
          expect(there.level[i], here.level[i], reason: 'level at $i');
          expect(there.levelVariance[i], here.levelVariance[i]);
          expect(there.slope[i], here.slope[i]);
        }
        expect(there.coefficientNames, here.coefficientNames);
        expect(there.coefficientEstimates, here.coefficientEstimates);
        expect(there.logMarginalLikelihood, here.logMarginalLikelihood);
        // Shape parameters have no variance ratio and report NaN, which is
        // compared as NaN rather than left to the matcher's notion of equality.
        expect(there.varianceRatios.length, here.varianceRatios.length);
        for (var i = 0; i < here.varianceRatios.length; i++) {
          if (here.varianceRatios[i].isNaN) {
            expect(there.varianceRatios[i].isNaN, isTrue);
          } else {
            expect(there.varianceRatios[i], here.varianceRatios[i]);
          }
        }
      },
    );

    test('so does a damped trend, which cannot share a model with the trend '
        'above', () async {
      final model = StructuralModel.dampedLinearTrend(
        processVariance: 1e-4,
        timeScale: 30,
      );
      final data = _data();
      final here = _worker((model, data));
      final there = await Isolate.run(() => _worker((model, data)));

      expect(there.level, here.level);
      expect(there.slope, here.slope);
      expect(there.logMarginalLikelihood, here.logMarginalLikelihood);
      expect(there.varianceRatios[0], here.varianceRatios[0]);
      expect(there.varianceRatios[1].isNaN, isTrue);
    });

    test('the model itself survives the trip, not just its output', () async {
      // Sending the model as data rather than rebuilding it inside the worker.
      // This is the half that a closure in a component would break.
      final model = _model();
      final data = _data();
      final returned = await Isolate.run(() => (model, data.length));

      expect(returned.$2, data.length);
      expect(returned.$1.stateDim, model.stateDim);
      expect(returned.$1.parameterCount, model.parameterCount);
      expect(returned.$1.components.length, model.components.length);
      expect(returned.$1.toString(), model.toString());
      // And it still works over there, which is the point of sending it.
      expect(
        returned.$1.logLikelihood(data),
        closeTo(model.logLikelihood(data), 1e-12),
      );
    });

    test('a result is plain numbers, so it can come back on its own', () async {
      // SmoothingResult holds Float64Lists and doubles and nothing else. That
      // is deliberate: it means the posterior can be sent back without being
      // taken apart first.
      final data = _data();
      final posterior = _model().smooth(data);
      final returned = await Isolate.run(() => posterior);

      expect(returned.mean, posterior.mean);
      expect(returned.times, posterior.times);
      expect(returned.componentMean(2), posterior.componentMean(2));
      expect(
        returned.coefficients.map((c) => c.name),
        posterior.coefficients.map((c) => c.name),
      );
      expect(returned.credibleInterval(10), posterior.credibleInterval(10));
    });

    test('and so is a forecast and a set of diagnostics', () async {
      final data = _data();
      final model = _model();
      final horizon = Float64List.fromList([
        for (var d = 200; d < 230; d += 5) d + 0.0,
      ]);

      final forecast = await Isolate.run(() => model.forecast(data, horizon));
      expect(forecast.mean, model.forecast(data, horizon).mean);

      final diagnostics = await Isolate.run(() => model.diagnose(data));
      expect(diagnostics.residuals, model.diagnose(data).residuals);
      expect(
        diagnostics.ljungBox(lags: 10).statistic,
        model.diagnose(data).ljungBox(lags: 10).statistic,
      );
    });
  });
}
