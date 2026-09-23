import 'dart:math' as math;

import 'package:state_space/state_space.dart';
import 'package:test/test.dart';

double _gaussian(math.Random random) =>
    math.sqrt(-2 * math.log(1 - random.nextDouble())) *
    math.cos(2 * math.pi * random.nextDouble());

/// Daily readings with a slow drift, a fixed weekly pattern and a fixed
/// annual one. Nothing here evolves, so the true process variances are all
/// zero and the components are rigid Fourier series.
List<Observation> _withAnnual(int days, {int seed = 7}) {
  final random = math.Random(seed);
  return [
    for (var day = 0; day < days; day++)
      Observation(
          day.toDouble(),
          78 -
              0.001 * day +
              0.4 * math.cos(2 * math.pi * day / 7) +
              0.15 * math.sin(4 * math.pi * day / 7) +
              1.1 * math.cos(2 * math.pi * (day - 30) / 365.25) +
              0.3 * _gaussian(random))
  ];
}

/// The same thing with no annual cycle in it at all.
List<Observation> _withoutAnnual(int days, {int seed = 3}) {
  final random = math.Random(seed);
  return [
    for (var day = 0; day < days; day++)
      Observation(
          day.toDouble(),
          78 -
              0.001 * day +
              0.4 * math.cos(2 * math.pi * day / 7) +
              0.3 * _gaussian(random))
  ];
}

StructuralModel _model() => StructuralModel([
      LocalLinearTrend(processVariance: 1e-5),
      TrigonometricSeasonal(period: 7, harmonics: 2, processVariance: 1e-5),
      TrigonometricSeasonal(
          period: 365.25, harmonics: 2, processVariance: 1e-5),
    ]);

double _amplitude(SmoothingResult posterior, int component) {
  final share = posterior.componentMean(component);
  var lowest = share.first;
  var highest = share.first;
  for (final value in share) {
    if (value < lowest) lowest = value;
    if (value > highest) highest = value;
  }
  return highest - lowest;
}

double _meanStandardDeviation(SmoothingResult posterior, int component) {
  final variance = posterior.componentVariance(component);
  var total = 0.0;
  for (final value in variance) {
    total += math.sqrt(value);
  }
  return total / variance.length;
}

void main() {
  group('an annual cycle alongside a weekly one', () {
    test('three years separates all three components', () {
      // Nothing new is needed for annual seasonality -- it is the same
      // component with a period of 365.25 -- so what is being tested is that
      // two seasonals of very different periods, plus a trend, come apart.
      final data = _withAnnual(3 * 365);
      final fitted = fit(_model(), data);
      final posterior = fitted.model.smooth(data);

      // True amplitudes: 0.85 peak to trough for the weekly pattern, 2.2 for
      // the annual, 1.095 for the drift over three years, 0.3 for the noise.
      expect(_amplitude(posterior, 0), closeTo(1.10, 0.25));
      expect(_amplitude(posterior, 1), closeTo(0.85, 0.15));
      expect(_amplitude(posterior, 2), closeTo(2.20, 0.20));
      expect(math.sqrt(fitted.measurementVariance), closeTo(0.3, 0.03));

      expect(
          fitted.model
              .diagnose(data)
              .ljungBox(lags: 20, fittedParameters: 3)
              .pValue,
          greaterThan(0.05));
    });

    test('and reports the rigid components as shrunk rather than estimated',
        () {
      // The simulated components do not evolve, so a variance of zero is the
      // truth and the fit should say it has hit the boundary rather than
      // quoting a number and a width as though it had found an interior
      // optimum.
      final data = _withAnnual(3 * 365);
      final fitted = fit(_model(), data);

      expect(fitted.parameterStatus, hasLength(3));
      expect(fitted.parameterStatus, contains(ParameterStatus.shrunkToNothing));
      expect(fitted.atBracketEdge, isTrue);
      expect(fitted.parameterStatus,
          isNot(contains(ParameterStatus.beyondBracket)));
    });
  });

  group('an annual component on less than a year of data', () {
    // The roadmap's advice was to always include the annual component and let
    // the marginal likelihood shrink it to nothing, rather than gating on how
    // much history there is. That is not what happens, and the reason is worth
    // knowing: shrinking the variance to zero does not remove the component,
    // it only stops it evolving. What is left is a rigid Fourier series whose
    // starting coefficients have a flat prior that nothing shrinks -- and over
    // less than one period, a rigid sinusoid is very nearly a constant plus a
    // slope.

    test('draws a pattern that is not there', () {
      final data = _withoutAnnual(180);
      final fitted = fit(_model(), data);
      final posterior = fitted.model.smooth(data);

      // Its variance is at the floor, so by the roadmap's reasoning it should
      // be contributing nothing.
      expect(fitted.parameterStatus[2], ParameterStatus.shrunkToNothing);
      // It is contributing more than the weekly pattern that is actually
      // there: about 1.1 peak to trough, on data with no annual cycle in it.
      expect(_amplitude(posterior, 2), greaterThan(0.8));
    });

    test('but says, loudly, that it does not know its own share', () {
      // This is what saves the advice. The component's own posterior is
      // enormously wider than the total signal's, because it is confounded
      // with the trend rather than determined: at 180 days its share carries a
      // standard deviation of about 1.27, larger than the 1.14 amplitude it
      // drew, while the total signal is known to 0.07. Read the band and the
      // spurious pattern is plainly consistent with nothing at all.
      final data = _withoutAnnual(180);
      final posterior = fit(_model(), data).model.smooth(data);

      final annual = _meanStandardDeviation(posterior, 2);
      final weekly = _meanStandardDeviation(posterior, 1);
      var total = 0.0;
      for (final value in posterior.variance) {
        total += math.sqrt(value);
      }
      total /= posterior.length;

      expect(annual, greaterThan(_amplitude(posterior, 2)));
      expect(annual, greaterThan(10 * total));
      expect(annual, greaterThan(10 * weekly));
    });

    test('and stops saying it once there is a year of data', () {
      // 0.065 at one year and 0.026 by two, against 0.045 for the weekly
      // pattern throughout. The confounding is a fact about the window, not
      // about the model, and it goes away when the window does.
      final short = fit(_model(), _withoutAnnual(180)).model;
      final full = fit(_model(), _withoutAnnual(365)).model;

      final atShort =
          _meanStandardDeviation(short.smooth(_withoutAnnual(180)), 2);
      final atFull =
          _meanStandardDeviation(full.smooth(_withoutAnnual(365)), 2);

      expect(atFull, lessThan(atShort / 10));
      expect(atFull, lessThan(0.15));
    });
  });
}
