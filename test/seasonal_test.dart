import 'dart:math' as math;
import 'dart:typed_data';

import 'package:state_space/authoring.dart';
import 'package:test/test.dart';

MatrixBlock _transition(TrigonometricSeasonal component, double dt) {
  final out = MatrixBlock.dense(component.stateDim, component.stateDim);
  component.transition(dt, out);
  return out;
}

MatrixBlock _product(MatrixBlock left, MatrixBlock right) {
  final out = MatrixBlock.dense(left.rows, right.cols);
  for (var i = 0; i < left.rows; i++) {
    for (var j = 0; j < right.cols; j++) {
      var sum = 0.0;
      for (var k = 0; k < left.cols; k++) {
        sum += left.at(i, k) * right.at(k, j);
      }
      out.set(i, j, sum);
    }
  }
  return out;
}

void main() {
  group('the seasonal transition', () {
    final component = TrigonometricSeasonal(
      period: 7,
      harmonics: 3,
      processVariance: 1e-3,
    );

    test(
      'composes over gaps, which is what makes irregular sampling exact',
      () {
        // A(a) A(b) = A(a + b) is the whole argument for the rotation form.
        // Splitting a step in two, or inserting an output grid point in the
        // middle of one, has to leave the answer alone.
        const first = 1.3;
        const second = 2.9;
        final composed = _product(
          _transition(component, first),
          _transition(component, second),
        );
        final direct = _transition(component, first + second);

        for (var i = 0; i < component.stateDim; i++) {
          for (var j = 0; j < component.stateDim; j++) {
            expect(composed.at(i, j), closeTo(direct.at(i, j), 1e-14));
          }
        }
      },
    );

    test('is the identity after a full period', () {
      final full = _transition(component, 7);
      for (var i = 0; i < component.stateDim; i++) {
        for (var j = 0; j < component.stateDim; j++) {
          expect(full.at(i, j), closeTo(i == j ? 1 : 0, 1e-14));
        }
      }
    });

    test(
      'is orthogonal, so the pattern neither grows nor decays on its own',
      () {
        final a = _transition(component, 0.7);
        final transpose = MatrixBlock.dense(a.rows, a.cols);
        for (var i = 0; i < a.rows; i++) {
          for (var j = 0; j < a.cols; j++) {
            transpose.set(i, j, a.at(j, i));
          }
        }
        final identity = _product(a, transpose);
        for (var i = 0; i < a.rows; i++) {
          for (var j = 0; j < a.cols; j++) {
            expect(identity.at(i, j), closeTo(i == j ? 1 : 0, 1e-15));
          }
        }
      },
    );

    test('leaves a zero gap alone', () {
      final none = _transition(component, 0);
      for (var i = 0; i < component.stateDim; i++) {
        for (var j = 0; j < component.stateDim; j++) {
          expect(none.at(i, j), i == j ? 1 : 0);
        }
      }
    });

    test('drives every state at the same rate', () {
      final q = MatrixBlock.dense(component.stateDim, component.stateDim);
      component.processNoise(2.5, q);
      for (var i = 0; i < component.stateDim; i++) {
        for (var j = 0; j < component.stateDim; j++) {
          expect(q.at(i, j), i == j ? closeTo(1e-3 * 2.5, 1e-18) : 0);
        }
      }
    });

    test('the observation reads the unstarred state of every harmonic', () {
      final h = Float64List(component.stateDim);
      component.observationAt(42, h);
      expect(h, [1, 0, 1, 0, 1, 0]);
    });
  });

  group('the seasonal specification', () {
    test('refuses harmonics at or past the Nyquist frequency of the data', () {
      // Daily readings resolve three harmonics of a weekly period; a fourth
      // sits above pi per day and is an alias of one already present.
      final daily = [
        for (var i = 0; i < 60; i++)
          Observation(i.toDouble(), 80 + math.sin(2 * math.pi * i / 7)),
      ];
      StructuralModel model(int harmonics, double period) => StructuralModel([
        LocalLinearTrend(processVariance: 1e-3),
        TrigonometricSeasonal(
          period: period,
          harmonics: harmonics,
          processVariance: 1e-3,
        ),
      ]);
      final refused = throwsA(
        isA<UnderdeterminedModelException>().having(
          (e) => e.message,
          'message',
          contains('Nyquist'),
        ),
      );
      expect(() => fit(model(4, 7), daily), refused);
      expect(() => fit(model(6, 12), daily), refused);
      expect(() => model(4, 7).smooth(daily), refused);
      expect(fit(model(3, 7), daily).model.components, hasLength(2));
    });

    test('takes the time unit from the data, not from the period', () {
      // An annual pattern with time in years and monthly readings.
      final monthly = [
        for (var i = 0; i < 48; i++)
          Observation(i / 12, 80 + math.sin(2 * math.pi * i / 12)),
      ];
      final annual = StructuralModel([
        LocalLinearTrend(processVariance: 1e-3),
        TrigonometricSeasonal(period: 1, harmonics: 2, processVariance: 1e-3),
      ]);
      expect(fit(annual, monthly).model.components, hasLength(2));
    });

    test('round-trips its parameter', () {
      final component = TrigonometricSeasonal(
        period: 7,
        harmonics: 2,
        processVariance: 4e-3,
      );
      final rebuilt =
          component.withParameters(component.parameters)
              as TrigonometricSeasonal;
      // exp(log(v)) is not v: the round trip costs a couple of ulps, which
      // at this magnitude is a few parts in 1e16.
      expect(rebuilt.processVariance, closeTo(4e-3, 4e-3 * 1e-15));
      expect(rebuilt.period, 7);
      expect(rebuilt.harmonics, 2);
    });

    test('has no rate state to report', () {
      // The derivative of a sinusoid is another sinusoid, not one of the
      // states, so there is nothing here for SmoothingResult.slope to mean.
      expect(
        TrigonometricSeasonal(
          period: 7,
          harmonics: 1,
          processVariance: 1e-3,
        ).rateStateIndex,
        isNull,
      );
    });
  });

  group('a nearly rigid seasonal', () {
    // With the process variance driven down the component is an ordinary
    // Fourier series with unknown coefficients, so the smoother should return
    // the waveform it was given, sampled at whatever times are asked for.
    List<Observation> data() {
      final random = math.Random(4);
      final out = <Observation>[];
      var t = 0.0;
      for (var i = 0; i < 120; i++) {
        t += 0.3 + random.nextDouble();
        final signal =
            2 * math.cos(2 * math.pi * t / 7) +
            0.5 * math.sin(4 * math.pi * t / 7);
        out.add(Observation(t, signal + 0.05 * (random.nextDouble() - 0.5)));
      }
      return out;
    }

    test('recovers the waveform it was generated from', () {
      final observations = data();
      final result = StructuralModel([
        TrigonometricSeasonal(period: 7, harmonics: 2, processVariance: 1e-10),
      ], measurementVariance: 1e-3).smooth(observations);

      for (var i = 0; i < result.length; i++) {
        final t = result.times[i];
        final truth =
            2 * math.cos(2 * math.pi * t / 7) +
            0.5 * math.sin(4 * math.pi * t / 7);
        expect(result.mean[i], closeTo(truth, 0.01));
      }
    });

    test('extrapolates the pattern rather than flattening it', () {
      // A rigid seasonal has nothing to forget, so a forecast a fortnight out
      // is still the same waveform -- unlike a trend, whose band opens up.
      final observations = data();
      final last = observations.last.time;
      final forecast =
          StructuralModel([
            TrigonometricSeasonal(
              period: 7,
              harmonics: 2,
              processVariance: 1e-10,
            ),
          ], measurementVariance: 1e-3).forecast(
            observations,
            Float64List.fromList([last + 3, last + 10, last + 14]),
          );

      for (var i = 0; i < forecast.length; i++) {
        final t = forecast.times[i];
        final truth =
            2 * math.cos(2 * math.pi * t / 7) +
            0.5 * math.sin(4 * math.pi * t / 7);
        expect(forecast.mean[i], closeTo(truth, 0.02));
      }
    });
  });

  group('when the flat directions are not determined', () {
    test('names both the count and the short span when both apply', () {
      // Six flat directions and five readings, spread over three days. The
      // engine knows only that the information matrix came out singular; it
      // knows the count itself, and asks the component about the rest.
      final short = [
        for (var i = 0; i < 5; i++) Observation(i * 0.75, 1.0 + 0.1 * i),
      ];
      expect(
        () => StructuralModel([
          TrigonometricSeasonal(period: 7, harmonics: 3, processVariance: 1e-3),
        ]).smooth(short),
        throwsA(
          isA<UnderdeterminedModelException>()
              .having(
                (e) => e.message,
                'message',
                contains('degree of freedom'),
              )
              .having(
                (e) => e.message,
                'message',
                contains('has not been round once'),
              ),
        ),
      );
    });

    test('two components that supply the same level say that instead', () {
      final data = [
        for (var i = 0; i < 30; i++) Observation(i.toDouble(), 5.0 + 0.1 * i),
      ];
      expect(
        () => StructuralModel([
          LocalLinearTrend(processVariance: 1e-3),
          LocalLevel(processVariance: 1e-3),
        ]).smooth(data),
        throwsA(
          isA<UnderdeterminedModelException>().having(
            (e) => e.message,
            'message',
            contains('the same signal'),
          ),
        ),
      );
    });

    test('a long enough series is fine', () {
      final random = math.Random(2);
      final data = [
        for (var i = 0; i < 40; i++)
          Observation(
            i * 0.75,
            math.cos(2 * math.pi * i * 0.75 / 7) + 0.05 * random.nextDouble(),
          ),
      ];
      final result = StructuralModel([
        TrigonometricSeasonal(period: 7, harmonics: 3, processVariance: 1e-3),
      ]).smooth(data);
      expect(result.logMarginalLikelihood.isFinite, isTrue);
    });
  });
}
