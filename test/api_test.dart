import 'dart:math' as math;
import 'dart:typed_data';

import 'package:state_space/authoring.dart';
import 'package:test/test.dart';

List<Observation> _diary(int days, {int seed = 1}) {
  final random = math.Random(seed);
  double gaussian() =>
      math.sqrt(-2 * math.log(1 - random.nextDouble())) *
      math.cos(2 * math.pi * random.nextDouble());
  var level = 80.0, slope = -0.02;
  return [
    for (var d = 0; d < days; d++)
      () {
        slope += 0.003 * gaussian();
        level += slope;
        return Observation(
          d.toDouble(),
          level + 0.3 * math.sin(2 * math.pi * d / 7) + 0.2 * gaussian(),
        );
      }(),
  ];
}

/// A step offset that says nothing about being static: `A = I`, `Q = 0`.
final class _Offset extends Component {
  const _Offset();
  @override
  int get stateDim => 1;
  @override
  int get parameterCount => 0;
  @override
  void transition(double dt, MatrixBlock out) => out.set(0, 0, 1);
  @override
  void processNoise(double dt, MatrixBlock out) => out.set(0, 0, 0);
  @override
  void observationAt(double time, Float64List out) =>
      out[0] = time < 50 ? 0 : 1;
  @override
  List<bool> get diffuseStates => const [true];
  @override
  void properPrior(Float64List mean, MatrixBlock covariance) {}
  @override
  Float64List get parameters => Float64List(0);
  @override
  Component withParameters(Float64List theta) => this;
}

/// Claims to be static and is not.
final class _LyingLevel extends Component {
  const _LyingLevel();
  @override
  int get stateDim => 1;
  @override
  int get parameterCount => 1;
  @override
  void transition(double dt, MatrixBlock out) => out.set(0, 0, 1);
  @override
  void processNoise(double dt, MatrixBlock out) => out.set(0, 0, 0.1 * dt);
  @override
  void observationAt(double time, Float64List out) => out[0] = 1;
  @override
  List<bool> get diffuseStates => const [true];
  @override
  void properPrior(Float64List mean, MatrixBlock covariance) {}
  @override
  Float64List get parameters => Float64List.fromList([math.log(0.1)]);
  @override
  Component withParameters(Float64List theta) => this;
  @override
  bool get isStatic => true;
}

/// A process noise that does not compose across gaps.
final class _Inconsistent extends Component {
  const _Inconsistent();
  @override
  int get stateDim => 1;
  @override
  int get parameterCount => 0;
  @override
  void transition(double dt, MatrixBlock out) => out.set(0, 0, 1);
  @override
  void processNoise(double dt, MatrixBlock out) => out.set(0, 0, dt * dt);
  @override
  void observationAt(double time, Float64List out) => out[0] = 1;
  @override
  List<bool> get diffuseStates => const [true, false];
  @override
  void properPrior(Float64List mean, MatrixBlock covariance) {}
  @override
  Float64List get parameters => Float64List(0);
  @override
  Component withParameters(Float64List theta) => this;
}

/// A trend with a hidden second state whose process noise is negative: not a
/// covariance, whatever the gap.
final class _Indefinite extends Component {
  const _Indefinite(this.hidden);
  final double hidden;
  @override
  int get stateDim => 2;
  @override
  int get parameterCount => 0;
  @override
  void transition(double dt, MatrixBlock out) {
    out.set(0, 0, 1);
    out.set(0, 1, 0);
    out.set(1, 0, 0);
    out.set(1, 1, 1);
  }

  @override
  void processNoise(double dt, MatrixBlock out) {
    out.set(0, 0, 0.01 * dt);
    out.set(0, 1, 0);
    out.set(1, 0, 0);
    out.set(1, 1, hidden * dt);
  }

  @override
  void observationAt(double time, Float64List out) {
    out[0] = 1;
    out[1] = 0;
  }

  @override
  List<bool> get diffuseStates => const [true, false];
  @override
  void properPrior(Float64List mean, MatrixBlock covariance) {
    covariance.set(1, 1, 1);
  }

  @override
  Float64List get parameters => Float64List(0);
  @override
  Component withParameters(Float64List theta) => this;
}

void main() {
  final data = _diary(120);
  final model = StructuralModel([
    LocalLinearTrend(processVariance: 1e-4),
    TrigonometricSeasonal(period: 7, harmonics: 2, processVariance: 1e-6),
    Matern.threeHalves(variance: 0.02, lengthScale: 4),
  ], measurementVariance: 0.04);

  group('results', () {
    final posterior = model.smooth(data);

    test('hand out read-only arrays', () {
      expect(() => posterior.mean[0] = 0, throwsUnsupportedError);
      expect(() => posterior.variance[0] = 0, throwsUnsupportedError);
      expect(() => posterior.componentMean(0)[0] = 0, throwsUnsupportedError);
      expect(() => posterior.trendSlope![0] = 0, throwsUnsupportedError);
      final fitted = fit(
        StructuralModel.localLinearTrend(processVariance: 1),
        data.take(40).toList(),
      );
      expect(() => fitted.varianceRatios[0] = 42, throwsUnsupportedError);
      final step = StepRegressor('dose', [1, 2], [3, 4]);
      expect(() => step.knots[0] = 30, throwsUnsupportedError);
    });

    test(
      'report the trend slope even beside a component with its own rate',
      () {
        // The Matérn 3/2 carries a derivative state as well.
        expect(posterior.trendIndex, 0);
        expect(posterior.trendSlope, posterior.componentSlope(0));
        expect(posterior.componentSlope(1), isNull);
        expect(posterior.componentSlope(2), isNotNull);
      },
    );

    test('draw a band that matches the pointwise interval', () {
      final band = posterior.predictiveBand();
      for (final i in [0, 60, 119]) {
        final point = posterior.predictiveInterval(i);
        expect(band.lo[i], point.lo);
        expect(band.hi[i], point.hi);
      }
    });
  });

  group('configuration types compare by value', () {
    test('models and their parts', () {
      StructuralModel build() => StructuralModel(
        [
          LocalLinearTrend(processVariance: 1e-4),
          TrigonometricSeasonal(period: 7, harmonics: 2, processVariance: 1),
          Matern.oneHalf(variance: 0.1, lengthScale: 3),
          StochasticCycle(period: 30, damping: 0.9, stationaryVariance: 1),
          RegressionComponent([
            IndicatorRegressor('trip', [(from: 3.0, to: 9.0)]),
            StepRegressor('dose', [1, 5], [2, 3]),
          ]),
        ],
        measurementVariance: 0.5,
        initialization: ApproximateDiffuse(),
      );
      expect(build(), build());
      expect(build().hashCode, build().hashCode);
      expect(build(), isNot(build().withMeasurementVariance(0.4)));
      expect(const Observation(1, 2), const Observation(1, 2));
      expect(const Observation(1, 2), isNot(const Observation(1, 3)));
      expect(ComplexityPenalty(), ComplexityPenalty());
    });
  });

  group('the model checks its components', () {
    test('once, at construction', () {
      expect(
        () => StructuralModel([const _Inconsistent()]),
        throwsArgumentError,
      );
    });

    test('and checkComponent passes every shipped one', () {
      for (final component in [
        LocalLevel(processVariance: 0.3),
        LocalLinearTrend(processVariance: 0.3),
        TrigonometricSeasonal(period: 7, harmonics: 3, processVariance: 0.2),
        Matern.oneHalf(variance: 1, lengthScale: 2),
        Matern.threeHalves(variance: 1, lengthScale: 2),
        Matern.fiveHalves(variance: 1, lengthScale: 2),
        StochasticCycle(period: 12, damping: 0.95, stationaryVariance: 1),
        RegressionComponent([
          IndicatorRegressor('x', [(from: 0.0, to: 1.0)]),
        ]),
        const _Offset(),
      ]) {
        expect(checkComponent(component), isEmpty, reason: component.name);
      }
    });

    test('and names what a broken one gets wrong', () {
      expect(
        checkComponent(const _LyingLevel()),
        contains(contains('claims isStatic')),
      );
      expect(
        checkComponent(const _Inconsistent()),
        contains(contains('diffuseStates has 2 entries')),
      );
    });
  });

  test('a far grid point does not move the answer inside the data', () {
    // The offset is static without saying so, so the smoother has to jitter
    // its covariance at every step.
    final model = StructuralModel([
      LocalLinearTrend(processVariance: 1e-4),
      const _Offset(),
    ], measurementVariance: 0.04);
    final inside = [for (var d = 0; d < 120; d++) d.toDouble()];
    final near = model.smooth(data, grid: inside);
    final far = model.smooth(data, grid: [...inside, 36899]);
    for (var i = 0; i < inside.length; i++) {
      expect(far.mean[i], closeTo(near.mean[i], 1e-9));
      expect(
        far.variance[i],
        closeTo(near.variance[i], 1e-9 * near.variance[i]),
      );
    }
  });

  test('a process noise that is not a covariance fails loudly', () {
    final model = StructuralModel([
      const _Indefinite(-1),
    ], measurementVariance: 0.04);
    expect(
      checkComponent(const _Indefinite(-1)),
      contains(contains('not symmetric positive semi-definite')),
    );
    expect(
      () => model.smooth(data),
      throwsA(isA<NumericalBreakdownException>()),
    );
  });

  group('withEstimatedScale', () {
    test('keeps the ratios and estimates the noise level', () {
      final trend = StructuralModel.localLinearTrend(
        processVariance: math.pow(4, -4).toDouble(),
      );
      final scaled = trend.withEstimatedScale(data);
      final ratio =
          (scaled.components.single as LocalLinearTrend).processVariance /
          scaled.measurementVariance;
      expect(ratio, closeTo(math.pow(4, -4), 1e-12));
      // The same number fit reaches with its bracket shut around the ratio.
      final q = math.log(math.pow(4, -4));
      final pinned = fit(
        trend,
        data,
        lowerLogRatio: q - 1e-9,
        upperLogRatio: q + 1e-9,
      );
      expect(
        scaled.measurementVariance,
        closeTo(pinned.measurementVariance, 1e-9),
      );
    });

    test('respects a floor on the noise and keeps the ratios under it', () {
      final trend = StructuralModel.localLinearTrend(processVariance: 1e-3);
      final floored = trend.withEstimatedScale(
        data.take(10).toList(),
        minimumMeasurementVariance: 4,
      );
      expect(floored.measurementVariance, 4);
      expect(
        (floored.components.single as LocalLinearTrend).processVariance / 4,
        closeTo(1e-3, 1e-15),
      );
    });

    test('refuses data with nothing left over', () {
      expect(
        () => StructuralModel.localLinearTrend(
          processVariance: 1,
        ).withEstimatedScale(data.take(2).toList()),
        throwsA(isA<UnderdeterminedModelException>()),
      );
    });
  });

  group('TimeAxis', () {
    test('counts calendar days as one, across a change of clocks', () {
      for (final origin in [
        DateTime(2026, 3, 1, 7),
        DateTime(2026, 10, 1, 7),
        DateTime.utc(2026, 3, 1, 7),
      ]) {
        final axis = TimeAxis.days(origin);
        for (var d = 0; d < 60; d++) {
          final morning = origin.isUtc
              ? DateTime.utc(origin.year, origin.month, origin.day + d, 7)
              : DateTime(origin.year, origin.month, origin.day + d, 7);
          expect(axis.timeOf(morning), d);
          expect(axis.dateAt(d.toDouble()), morning);
        }
        final evening = origin.add(const Duration(hours: 12));
        expect(axis.timeOf(evening), 0.5);
        expect(axis.dateAt(0.5), evening);
      }
    });
  });

  test('a component named in a warning is named by a literal', () {
    for (final (component, name) in [
      (LocalLevel(processVariance: 1), 'LocalLevel'),
      (LocalLinearTrend(processVariance: 1), 'LocalLinearTrend'),
      (Matern.oneHalf(variance: 1, lengthScale: 1), 'Matern'),
    ]) {
      expect(component.name, name);
    }
  });
}
