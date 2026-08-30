import 'dart:math' as math;
import 'dart:typed_data';

import 'package:state_space/src/engine/kalman.dart';
import 'package:state_space/src/engine/timeline.dart';
import 'package:state_space/src/initialization.dart';
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
