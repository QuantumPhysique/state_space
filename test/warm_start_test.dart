import 'dart:math' as math;

import 'package:state_space/state_space.dart';
import 'package:test/test.dart';

/// Refitting as data arrives should not have to rediscover the same basin.
void main() {
  List<Observation> diary(int n, {int seed = 3}) {
    final random = math.Random(seed);
    double gaussian() =>
        math.sqrt(-2 * math.log(random.nextDouble())) *
        math.cos(2 * math.pi * random.nextDouble());
    return [
      for (var i = 0; i < n; i++)
        Observation(
            i.toDouble(),
            80 +
                0.002 * i +
                0.5 * math.sin(2 * math.pi * i / 7) +
                0.3 * gaussian())
    ];
  }

  StructuralModel template() => StructuralModel([
        const LocalLinearTrend(processVariance: 1e-4),
        TrigonometricSeasonal(period: 7, harmonics: 2, processVariance: 1e-4),
      ]);

  test('a fitted model warm-started on its own data does not move', () {
    final data = diary(500);
    final cold = fit(template(), data);
    final again =
        fit(cold.model, data, start: SearchStart.previousParameters);

    expect(again.logMarginalLikelihood,
        closeTo(cold.logMarginalLikelihood, 1e-6));
    for (var i = 0; i < cold.varianceRatios.length; i++) {
      expect(again.varianceRatios[i],
          closeTo(cold.varianceRatios[i], 1e-6 * cold.varianceRatios[i]));
    }
    expect(again.measurementVariance,
        closeTo(cold.measurementVariance, 1e-9 * cold.measurementVariance));
  });

  test('and reaches the same optimum after one more reading, for less', () {
    final data = diary(500);
    final cold = fit(template(), data);
    final longer = [...data, Observation(500, 81.4)];

    final coldAgain = fit(template(), longer);
    final warm =
        fit(cold.model, longer, start: SearchStart.previousParameters);

    expect(warm.logMarginalLikelihood,
        closeTo(coldAgain.logMarginalLikelihood, 1e-4));
    expect(warm.evaluations, lessThan(coldAgain.evaluations),
        reason: 'the point of it');
  });

  test('the round trip goes through the fitted measurement variance', () {
    // A model whose parameters are absolute variances, warm-started, has to be
    // read as ratios to the noise level it was fitted at, or every variance
    // moves by that factor.
    final data = diary(400);
    final cold = fit(template(), data);
    expect(cold.measurementVariance, isNot(closeTo(1, 0.2)),
        reason: 'otherwise the conversion is untested');
    final warm =
        fit(cold.model, data, start: SearchStart.previousParameters);
    expect(warm.logMarginalLikelihood,
        closeTo(cold.logMarginalLikelihood, 1e-6));
  });

  test('a parameter outside the bracket is clamped rather than refused', () {
    final data = diary(300);
    final wild = StructuralModel([
      const LocalLinearTrend(processVariance: 1e30),
      TrigonometricSeasonal(period: 7, harmonics: 2, processVariance: 1e-40),
    ]);
    final warm =
        fit(wild, data, start: SearchStart.previousParameters);
    expect(warm.logMarginalLikelihood.isFinite, isTrue);
  });

  test('a one-parameter model warm-starts too', () {
    final data = diary(300);
    final cold = fit(StructuralModel.localLinearTrend(processVariance: 1), data);
    final warm =
        fit(cold.model, data, start: SearchStart.previousParameters);
    expect(warm.varianceRatio,
        closeTo(cold.varianceRatio, 1e-6 * cold.varianceRatio));
    expect(warm.evaluations, lessThan(cold.evaluations));
  });

  test('the default is still the scan', () {
    final data = diary(300);
    // Started from a model at the far end of the bracket, the scan still finds
    // the same answer; a local search from there would not.
    final fromNonsense =
        fit(StructuralModel.localLinearTrend(processVariance: 1e18), data);
    final fromSense =
        fit(StructuralModel.localLinearTrend(processVariance: 1e-9), data);
    expect(fromNonsense.varianceRatio,
        closeTo(fromSense.varianceRatio, 1e-6 * fromSense.varianceRatio));
  });
}
