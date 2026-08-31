import 'dart:math' as math;
import 'dart:typed_data';

import 'package:state_space/src/engine/kalman.dart';
import 'package:state_space/src/engine/rts.dart';
import 'package:state_space/src/engine/timeline.dart';
import 'package:state_space/state_space.dart';
import 'package:test/test.dart';

/// A local linear trend that starts from a known point instead of a flat
/// prior.
///
/// Transition and process noise are the ordinary ones; only the initial
/// condition differs. This is the reference the sensitivity is checked
/// against, and it is the only thing in the package that exercises
/// [Component.properPrior].
class AnchoredTrend extends Component {
  const AnchoredTrend(this.inner, this.level, this.rate);

  final LocalLinearTrend inner;
  final double level;
  final double rate;

  @override
  int get stateDim => 2;
  @override
  int get parameterCount => 1;
  @override
  void transition(double dt, MatrixBlock out) => inner.transition(dt, out);
  @override
  void processNoise(double dt, MatrixBlock out) => inner.processNoise(dt, out);
  @override
  void observationAt(double time, Float64List out) =>
      inner.observationAt(time, out);
  @override
  double wanderOver(double span) => inner.wanderOver(span);
  @override
  List<bool> get diffuseStates => const [false, false];
  @override
  void properPrior(Float64List mean, MatrixBlock covariance) {
    mean[0] = level;
    mean[1] = rate;
    covariance.fill(0);
  }

  @override
  Float64List get parameters => inner.parameters;
  @override
  Component withParameters(Float64List theta) => AnchoredTrend(
      inner.withParameters(theta) as LocalLinearTrend, level, rate);
  @override
  int? get rateStateIndex => 1;
}

List<Observation> series({int n = 60, int seed = 21}) {
  final random = math.Random(seed);
  final data = <Observation>[];
  var time = 0.0;
  for (var i = 0; i < n; i++) {
    time += 0.4 + 2.2 * random.nextDouble();
    data.add(Observation(time, 74 + 0.03 * time + random.nextDouble() - 0.5));
  }
  return data;
}

const processVariance = 6e-4;
const measurementVariance = 0.08;

FilterResult runFilter(List<Component> components, List<Observation> data,
        Initialization initialization) =>
    KalmanFilter(components,
            measurementVariance: measurementVariance,
            initialization: initialization)
        .run(Timeline.merge(data, null), keepHistory: true);

void main() {
  const trend = LocalLinearTrend(processVariance: processVariance);

  test('the sensitivity really is the derivative of the state', () {
    // The augmentation claims x(t) = xa(t) + Xb(t) d, exactly, for any d. So
    // filtering from a known starting point must reproduce the augmented
    // filter's own answer when d is set to that point. Nothing about this is
    // asymptotic; if the sensitivity is propagated even slightly wrongly, it
    // fails outright.
    const level = 71.5;
    const rate = -0.02;
    final data = series();

    final augmented = runFilter([trend], data, const ExactDiffuse());
    final anchored = runFilter(
        [AnchoredTrend(trend, level, rate)], data, const ExactDiffuse());

    expect(augmented.diffuseDim, 2);
    expect(anchored.diffuseDim, 0);

    final sensitivity = augmented.filteredSensitivity!;
    for (var t = 0; t < augmented.stepCount; t++) {
      for (var i = 0; i < 2; i++) {
        final reconstructed = augmented.filteredMean![t * 2 + i] +
            sensitivity[(t * 2 + i) * 2] * level +
            sensitivity[(t * 2 + i) * 2 + 1] * rate;
        expect(reconstructed, closeTo(anchored.filteredMean![t * 2 + i], 1e-10),
            reason: 'state $i at step $t');
      }
    }
  });

  test('the covariance does not depend on where the state started', () {
    // It cannot: the Riccati recursion never sees the data or the mean. This
    // is what lets the flat directions be integrated out afterwards rather
    // than tracked through a second covariance recursion.
    final data = series();
    final augmented = runFilter([trend], data, const ExactDiffuse());
    final anchored =
        runFilter([AnchoredTrend(trend, 100, 5)], data, const ExactDiffuse());

    for (var i = 0; i < augmented.filteredCovariance!.length; i++) {
      expect(augmented.filteredCovariance![i],
          closeTo(anchored.filteredCovariance![i], 1e-15));
    }
  });

  test('the exact filter is the limit of a widening approximate prior', () {
    // Only at the last step. Combining the sensitivity with the final estimate
    // of the flat directions gives a state conditioned on *all* the data, so
    // it matches an ordinary filtered state only where the two condition on
    // the same thing. Everywhere else the honest comparison is against the
    // smoother, which is what the next test file does.
    final data = series();
    final exact = runFilter([trend], data, const ExactDiffuse());
    final estimate = exact.diffuseMean!;
    final sensitivity = exact.filteredSensitivity!;
    final last = exact.stepCount - 1;

    double worstErrorAt(double variance) {
      final approximate =
          runFilter([trend], data, ApproximateDiffuse(variance: variance));
      var worst = 0.0;
      for (var i = 0; i < 2; i++) {
        final combined = exact.filteredMean![last * 2 + i] +
            sensitivity[(last * 2 + i) * 2] * estimate[0] +
            sensitivity[(last * 2 + i) * 2 + 1] * estimate[1];
        worst = math.max(
            worst, (combined - approximate.filteredMean![last * 2 + i]).abs());
      }
      return worst;
    }

    // Two more decades of prior, two more decades of agreement — the same
    // 1/kappa law the ordinary-least-squares limit obeys.
    final loose = worstErrorAt(1e6);
    final tighter = worstErrorAt(1e8);
    expect(loose, lessThan(1e-3));
    expect(tighter, lessThan(loose / 50));
  });

  group('smoothing', () {
    /// The smoothed posterior, by whichever route.
    ({Float64List mean, Float64List covariance}) posterior(
        List<Observation> data, Initialization initialization) {
      final timeline = Timeline.merge(data, null);
      final forward = KalmanFilter([trend],
              measurementVariance: measurementVariance,
              initialization: initialization)
          .run(timeline, keepHistory: true);
      RtsSmoother([trend])
        ..smoothInPlace(timeline, forward)
        ..combineDiffuse(forward);
      return (
        mean: forward.filteredMean!,
        covariance: forward.filteredCovariance!
      );
    }

    test('the exact posterior is the limit of a widening approximate prior',
        () {
      // Here the comparison is honest at every step: both sides condition on
      // the whole series. This is the claim exact initialisation exists to
      // make, so it is worth watching it converge rather than just checking
      // it once.
      final data = series();
      final exact = posterior(data, const ExactDiffuse());

      ({double mean, double variance}) errorAt(double variance) {
        final approximate =
            posterior(data, ApproximateDiffuse(variance: variance));
        var worstMean = 0.0;
        var worstVariance = 0.0;
        for (var i = 0; i < exact.mean.length; i++) {
          worstMean =
              math.max(worstMean, (exact.mean[i] - approximate.mean[i]).abs());
        }
        for (var i = 0; i < exact.covariance.length; i++) {
          worstVariance = math.max(worstVariance,
              (exact.covariance[i] - approximate.covariance[i]).abs());
        }
        return (mean: worstMean, variance: worstVariance);
      }

      // Two decades of prior buy two decades of agreement, in the mean and
      // in the covariance alike. The window matters: past about 1e7 the
      // covariance stops improving, because the arithmetic noise in numbers
      // of that size has overtaken the approximation being measured, and past
      // about 1e9 the mean does the same. That ceiling is precisely what
      // exact initialisation removes.
      final loose = errorAt(1e4);
      final tighter = errorAt(1e6);
      expect(loose.mean, lessThan(1e-2));
      expect(tighter.mean, lessThan(loose.mean / 50));
      expect(tighter.variance, lessThan(loose.variance / 50));
    });

    test('and it costs the first two steps nothing', () {
      // The four digits the approximate prior threw away at the start of the
      // series are simply not thrown away here: there is no subtraction of
      // two large numbers to lose them in.
      final data = series();
      final exact = posterior(data, const ExactDiffuse());
      final wide = posterior(data, const ApproximateDiffuse(variance: 1e9));

      for (var t = 0; t < 2; t++) {
        for (var e = 0; e < 4; e++) {
          expect(exact.covariance[t * 4 + e],
              closeTo(wide.covariance[t * 4 + e], 1e-6),
              reason: 'P^s[$t][$e]');
        }
      }
    });

    test('the model API produces the same posterior as the generic engine', () {
      // Not bit-identical any more: StructuralModel dispatches a single
      // two-state component to the scalar fast path, which reassociates the
      // same arithmetic. fast_path_equivalence_test.dart is where that is
      // pinned properly; this only checks the wiring reaches the same answer.
      final data = series();
      final model = StructuralModel([trend],
          measurementVariance: measurementVariance,
          initialization: const ExactDiffuse());
      final result = model.smooth(data);
      final engine = posterior(data, const ExactDiffuse());

      for (var t = 0; t < data.length; t++) {
        expect(result.level[t], closeTo(engine.mean[t * 2], 1e-12));
        expect(
            result.levelVariance[t], closeTo(engine.covariance[t * 4], 1e-12));
      }
    });
  });

  test('refuses to invent information it does not have', () {
    // Two diffuse states, one observation: the level is pinned and the slope
    // is not, so the diffuse information matrix is singular and there is no
    // honest answer to give.
    expect(
      () =>
          runFilter([trend], [const Observation(0, 80)], const ExactDiffuse()),
      throwsA(isA<StateError>()
          .having((e) => e.message, 'message', contains('does not determine'))),
    );
  });
}
