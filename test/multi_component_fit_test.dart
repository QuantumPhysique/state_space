import 'dart:math' as math;
import 'dart:typed_data';

import 'package:state_space/state_space.dart';
import 'package:test/test.dart';

double _gaussian(math.Random random) =>
    math.sqrt(-2 * math.log(1 - random.nextDouble())) *
    math.cos(2 * math.pi * random.nextDouble());

/// Simulates from exactly the model that will be fitted, keeping the paths of
/// the components so the decomposition can be checked and not just the total.
///
/// Both components are stepped by their own exact discretisation: the trend by
/// the correlated level-and-slope increment a Wiener slope implies, the
/// seasonal by the rotation with isotropic noise. A simulator that took the
/// discrete shortcut would be generating from a different model and the
/// recovery test would be measuring the gap between them.
({List<Observation> data, List<double> trend, List<double> seasonal})
    _simulate({
  required double trendVariance,
  required double seasonalVariance,
  required double measurementVariance,
  required int count,
  required int seed,
  double period = 7,
  int harmonics = 2,
}) {
  final random = math.Random(seed);
  final data = <Observation>[];
  final trend = <double>[];
  final seasonal = <double>[];
  final gamma = <double>[0.5, 0.2, -0.2, 0.1];
  var level = 75.0;
  var slope = -0.01;
  var time = 0.0;

  for (var i = 0; i < count; i++) {
    final dt = 0.5 + random.nextDouble();
    final sd = math.sqrt(trendVariance);
    final root = dt * math.sqrt(dt);
    final a = _gaussian(random);
    final b = _gaussian(random);
    level += slope * dt + sd * (root / math.sqrt(3) * a + root / 2 * b);
    slope += sd * math.sqrt(dt) * b;

    final noise = math.sqrt(seasonalVariance * dt);
    for (var j = 1; j <= harmonics; j++) {
      final angle = 2 * math.pi * j / period * dt;
      final c = math.cos(angle);
      final s = math.sin(angle);
      final p = gamma[2 * (j - 1)];
      final q = gamma[2 * (j - 1) + 1];
      gamma[2 * (j - 1)] = c * p + s * q + noise * _gaussian(random);
      gamma[2 * (j - 1) + 1] = -s * p + c * q + noise * _gaussian(random);
    }

    var pattern = 0.0;
    for (var j = 0; j < harmonics; j++) {
      pattern += gamma[2 * j];
    }
    time += dt;
    trend.add(level);
    seasonal.add(pattern);
    data.add(Observation(time,
        level + pattern + math.sqrt(measurementVariance) * _gaussian(random)));
  }
  return (data: data, trend: trend, seasonal: seasonal);
}

StructuralModel _start() => StructuralModel([
      LocalLinearTrend(processVariance: 1e-3),
      TrigonometricSeasonal(period: 7, harmonics: 2, processVariance: 1e-3),
    ]);

double _rootMeanSquare(Float64List fitted, List<double> truth,
    {bool centre = false}) {
  var offset = 0.0;
  if (centre) {
    for (var i = 0; i < truth.length; i++) {
      offset += fitted[i] - truth[i];
    }
    offset /= truth.length;
  }
  var total = 0.0;
  for (var i = 0; i < truth.length; i++) {
    final d = fitted[i] - truth[i] - offset;
    total += d * d;
  }
  return math.sqrt(total / truth.length);
}

void main() {
  const trendVariance = 3e-5;
  const seasonalVariance = 4e-4;
  const measurementVariance = 0.04;

  group('fitting a trend and a seasonal together', () {
    test('recovers both variances and the noise', () {
      for (final seed in [5, 11, 23, 37]) {
        final simulation = _simulate(
          trendVariance: trendVariance,
          seasonalVariance: seasonalVariance,
          measurementVariance: measurementVariance,
          count: 500,
          seed: seed,
        );
        final result = fit(_start(), simulation.data);

        expect(result.converged, isTrue, reason: 'seed $seed');
        expect(result.atBracketEdge, isFalse, reason: 'seed $seed');
        expect(result.varianceRatios.length, 2);

        // One log unit is a factor of e either way. Measured over twelve
        // replications at this sample size the root-mean-square error of the
        // log ratio is 0.37 for the trend and 0.33 for the seasonal, so this
        // is roughly two and a half standard errors.
        final trendError = math.log(
            result.varianceRatios[0] / (trendVariance / measurementVariance));
        final seasonalError = math.log(result.varianceRatios[1] /
            (seasonalVariance / measurementVariance));
        expect(trendError.abs(), lessThan(1.0),
            reason: 'seed $seed trend ratio ${result.varianceRatios[0]}');
        expect(seasonalError.abs(), lessThan(1.0),
            reason: 'seed $seed seasonal ratio ${result.varianceRatios[1]}');

        expect(result.measurementVariance,
            closeTo(measurementVariance, 0.25 * measurementVariance),
            reason: 'seed $seed');
      }
    });

    test('and gets the decomposition right, not just the total', () {
      // The point of the exercise. A model can fit a series beautifully and
      // still attribute the wrong half of it to each component, and nothing
      // in the total fit or the likelihood would show it.
      final simulation = _simulate(
        trendVariance: trendVariance,
        seasonalVariance: seasonalVariance,
        measurementVariance: measurementVariance,
        count: 500,
        seed: 5,
      );
      final fitted = fit(_start(), simulation.data).model;
      final smoothed = fitted.smooth(simulation.data);

      // The split between a trend and a seasonal is identified only up to a
      // constant -- moving a fixed amount from one to the other changes
      // nothing observable -- so the trend is compared after centring. The
      // seasonal has no level of its own and is compared as it stands.
      final seasonal =
          _rootMeanSquare(smoothed.componentMean(1), simulation.seasonal);
      final trend = _rootMeanSquare(smoothed.componentMean(0), simulation.trend,
          centre: true);

      // Measured 0.076 and 0.049 against a measurement standard deviation of
      // 0.2: each component is recovered to about a quarter of the noise on a
      // single reading, which is what five hundred observations should buy.
      expect(seasonal, lessThan(0.12));
      expect(trend, lessThan(0.10));
      expect(seasonal, lessThan(math.sqrt(measurementVariance)));
      expect(trend, lessThan(math.sqrt(measurementVariance)));
    });

    test('leaves nothing behind in the residuals', () {
      final simulation = _simulate(
        trendVariance: trendVariance,
        seasonalVariance: seasonalVariance,
        measurementVariance: measurementVariance,
        count: 500,
        seed: 11,
      );
      final fitted = fit(_start(), simulation.data).model;
      final diagnostics = fitted.diagnose(simulation.data);

      expect(diagnostics.ljungBox(lags: 14, fittedParameters: 2).pValue,
          greaterThan(0.05));
      expect((diagnostics.variance - 1).abs(),
          lessThan(4 * math.sqrt(2 / diagnostics.count)));
    });
  });

  group('a multi-parameter FitResult', () {
    late FitResult result;

    setUpAll(() {
      final simulation = _simulate(
        trendVariance: trendVariance,
        seasonalVariance: seasonalVariance,
        measurementVariance: measurementVariance,
        count: 300,
        seed: 5,
      );
      result = fit(_start(), simulation.data);
    });

    test('refuses to name a single variance ratio', () {
      expect(
          () => result.varianceRatio,
          throwsA(isA<StateError>().having(
              (e) => e.message, 'message', contains('2 variance ratios'))));
    });

    test('gives a plateau width per parameter', () {
      expect(result.plateauDecadesByParameter.length, 2);
      for (final width in result.plateauDecadesByParameter) {
        expect(width, greaterThan(0));
        expect(width, lessThan(1));
      }
      expect(result.plateauDecades,
          result.plateauDecadesByParameter.reduce(math.max));
      expect(result.isFlat, isFalse);
    });

    test('has no penalty term when none was applied', () {
      expect(result.penalty, isA<NoPenalty>());
      expect(result.logPenalty, 0);
    });
  });

  group('the complexity penalty', () {
    test('drives a drift parameter to the floor when there is no drift', () {
      // A weekly pattern that is fixed rather than evolving. Plain maximum
      // likelihood has no reason to prefer zero drift over a little, and
      // sometimes leaves a little; the penalty removes the doubt.
      final simulation = _simulate(
        trendVariance: trendVariance,
        seasonalVariance: 0,
        measurementVariance: measurementVariance,
        count: 200,
        seed: 1,
      );
      final penalised =
          fit(_start(), simulation.data, penalty: ComplexityPenalty());
      final plain = fit(_start(), simulation.data);

      expect(penalised.varianceRatios[1], lessThan(1e-7));
      expect(plain.varianceRatios[1], greaterThan(penalised.varianceRatios[1]));
      expect(penalised.logPenalty, lessThan(0));
      expect(penalised.penalty, isA<ComplexityPenalty>());

      // And the likelihood it reports is the unpenalised one, so it stays
      // comparable with the fit that used no penalty at all.
      expect(penalised.logMarginalLikelihood,
          lessThan(plain.logMarginalLikelihood + 1e-9));
    });

    test('a wider scale penalises less', () {
      final simulation = _simulate(
        trendVariance: trendVariance,
        seasonalVariance: seasonalVariance,
        measurementVariance: measurementVariance,
        count: 200,
        seed: 5,
      );
      final tight = fit(_start(), simulation.data,
          penalty: ComplexityPenalty(scale: 0.25));
      final loose =
          fit(_start(), simulation.data, penalty: ComplexityPenalty(scale: 4));

      expect(tight.logPenalty, lessThan(loose.logPenalty));
      for (var i = 0; i < 2; i++) {
        expect(tight.varianceRatios[i], lessThan(loose.varianceRatios[i]),
            reason: 'parameter $i');
      }
    });
  });
}
