import 'dart:math' as math;
import 'dart:typed_data';

import 'package:state_space/state_space.dart';
import 'package:test/test.dart';

List<Observation> _series({int n = 60, int seed = 3, double drift = 0.05}) {
  final random = math.Random(seed);
  final data = <Observation>[];
  var time = 0.0;
  for (var i = 0; i < n; i++) {
    time += 0.5 + 2 * random.nextDouble();
    data.add(Observation(time, 70 + drift * time + random.nextDouble() - 0.5));
  }
  return data;
}

/// Ordinary least squares through `(time, value)`.
({double intercept, double slope}) _ols(List<Observation> data) {
  final n = data.length;
  var sumT = 0.0, sumY = 0.0;
  for (final o in data) {
    sumT += o.time;
    sumY += o.value;
  }
  final meanT = sumT / n, meanY = sumY / n;
  var sxy = 0.0, sxx = 0.0;
  for (final o in data) {
    sxy += (o.time - meanT) * (o.value - meanY);
    sxx += (o.time - meanT) * (o.time - meanT);
  }
  final slope = sxy / sxx;
  return (intercept: meanY - slope * meanT, slope: slope);
}

void main() {
  group('limits that have a closed form', () {
    test('a rigid trend is ordinary least squares, in the diffuse limit', () {
      // Send the process variance to zero and the slope can no longer move,
      // so the posterior mean is the best straight line through the data.
      //
      // Only in the limit, though: a finite diffuse prior shrinks the line
      // toward zero by roughly one part in N*kappa. So the test asserts both
      // that the answer is close and that it gets closer as the prior widens,
      // which is a sharper statement than either half alone.
      final data = _series();
      final line = _ols(data);

      double worstErrorAt(double diffuseVariance) {
        final result = StructuralModel.localLinearTrend(
          processVariance: 1e-16,
          measurementVariance: 0.25,
          initialization: ApproximateDiffuse(variance: diffuseVariance),
        ).smooth(data);
        var worst = 0.0;
        for (var i = 0; i < data.length; i++) {
          final expected = line.intercept + line.slope * data[i].time;
          worst = math.max(worst, (result.level[i] - expected).abs());
          expect(result.slope![i], closeTo(line.slope, 1e-5 * line.slope));
        }
        return worst;
      }

      // The discrepancy falls as 1/kappa, cleanly, until around 1e9, where
      // rounding in a covariance of that magnitude takes over and it starts
      // rising again. Two decades of prior buy two decades of agreement.
      final loose = worstErrorAt(1e6);
      final looser = worstErrorAt(1e8);
      expect(loose, lessThan(1e-5));
      expect(looser, lessThan(loose / 50));
    });

    test('a rigid level is the precision-weighted mean', () {
      final data = [
        const Observation(0, 10.0),
        const Observation(1, 12.0, relativeVariance: 0.25),
        const Observation(5, 11.0),
        const Observation(9, 13.0, relativeVariance: 4.0),
      ];
      final result = StructuralModel.localLevel(
        processVariance: 1e-18,
        measurementVariance: 1.0,
        initialization: ApproximateDiffuse(variance: 1e12),
      ).smooth(data);

      var weight = 0.0, weighted = 0.0;
      for (final o in data) {
        weight += 1 / o.relativeVariance;
        weighted += o.value / o.relativeVariance;
      }
      final mean = weighted / weight;

      for (var i = 0; i < data.length; i++) {
        expect(result.level[i], closeTo(mean, 1e-8));
        expect(result.levelVariance[i], closeTo(1 / weight, 1e-8));
      }
    });
  });

  group('invariances the model is supposed to have', () {
    test('shifting every time by a constant changes nothing', () {
      final data = _series();
      final shifted = [
        for (final o in data) Observation(o.time + 1234.5, o.value)
      ];
      final model = StructuralModel.localLinearTrend(processVariance: 1e-3);

      // Not bit-identical: adding 1234.5 to a time near 100 costs a few low
      // bits, so the gaps the recursion sees differ in the last ulp. Nothing
      // else should.
      final a = model.smooth(data);
      final b = model.smooth(shifted);
      for (var i = 0; i < data.length; i++) {
        expect(b.level[i], closeTo(a.level[i], 1e-9));
        expect(b.levelVariance[i], closeTo(a.levelVariance[i], 1e-9));
      }
      expect(b.logMarginalLikelihood, closeTo(a.logMarginalLikelihood, 1e-9));
    });

    test(
        'scaling every variance scales the posterior variance and leaves '
        'the posterior mean alone', () {
      // This is the invariance the profile likelihood in fit() rests on, so
      // it is worth asserting directly rather than trusting that it follows.
      const c = 37.0;
      final data = _series();
      final base = StructuralModel.localLinearTrend(
        processVariance: 2e-3,
        measurementVariance: 0.3,
      );
      final scaled = StructuralModel.localLinearTrend(
        processVariance: 2e-3 * c,
        measurementVariance: 0.3 * c,
      );

      final a = base.smooth(data);
      final b = scaled.smooth(data);
      for (var i = 0; i < data.length; i++) {
        expect(b.level[i], closeTo(a.level[i], 1e-9));
        expect(b.levelVariance[i], closeTo(a.levelVariance[i] * c, 1e-9));
      }
    });

    test('reversing time reverses the answer', () {
      // The cubic spline kernel is symmetric, so smoothing a series backwards
      // has to give the same curve read backwards. This catches an
      // asymmetric backward pass, which a forward-only test cannot.
      final data = _series();
      final span = data.last.time + data.first.time;
      final reversed = [
        for (final o in data.reversed) Observation(span - o.time, o.value)
      ];
      // Symmetric in the diffuse limit only: a finite prior is anchored at
      // whichever end comes first, which is the one thing reversal moves.
      final model = StructuralModel.localLinearTrend(
        processVariance: 5e-4,
        measurementVariance: 0.2,
        initialization: ApproximateDiffuse(variance: 1e8),
      );

      final forward = model.smooth(data);
      final backward = model.smooth(reversed);
      final n = data.length;
      for (var i = 0; i < n; i++) {
        // A prior of 1e8 sits at the sweet spot: wide enough that the
        // asymmetry it causes is down at 1e-8, narrow enough that the
        // arithmetic noise from covariances of that size has not caught up.
        expect(backward.level[n - 1 - i], closeTo(forward.level[i], 1e-6),
            reason: 'level at $i');
        expect(backward.slope![n - 1 - i], closeTo(-forward.slope![i], 1e-6),
            reason: 'slope at $i');
      }
    });
  });

  test('the posterior variance never grows when data is added', () {
    final data = _series(n: 40);
    final model = StructuralModel.localLinearTrend(
      processVariance: 1e-3,
      measurementVariance: 0.25,
    );
    // Look at one fixed time throughout, so what changes is only how much
    // data surrounds it.
    final probe = Float64List.fromList([data[5].time]);

    var previous = double.infinity;
    for (var n = 8; n <= data.length; n += 4) {
      final variance =
          model.smooth(data.sublist(0, n), grid: probe).levelVariance[0];
      expect(variance, lessThanOrEqualTo(previous * (1 + 1e-12)),
          reason: 'variance grew when going to $n observations');
      previous = variance;
    }
  });
}
