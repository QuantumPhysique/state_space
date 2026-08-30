import 'dart:math' as math;
import 'dart:typed_data';

import 'package:state_space/state_space.dart';
import 'package:test/test.dart';

List<Observation> _data() {
  final random = math.Random(19);
  final data = <Observation>[];
  var time = 0.0;
  for (var i = 0; i < 30; i++) {
    time += 1 + 3 * random.nextDouble();
    data.add(Observation(time, 75 + 0.04 * time + random.nextDouble() - 0.5));
  }
  return data;
}

StructuralModel _model() => StructuralModel.localLinearTrend(
      processVariance: 8e-4,
      measurementVariance: 0.2,
    );

void main() {
  group('output grid', () {
    test('asking for output at the observation times changes nothing', () {
      final data = _data();
      final plain = _model().smooth(data);
      final gridded = _model().smooth(data,
          grid: Float64List.fromList([for (final o in data) o.time]));

      expect(gridded.length, data.length);
      for (var i = 0; i < data.length; i++) {
        expect(gridded.times[i], data[i].time);
        expect(gridded.level[i], closeTo(plain.level[i], 1e-12));
        expect(
            gridded.levelVariance[i], closeTo(plain.levelVariance[i], 1e-14));
        expect(gridded.slope![i], closeTo(plain.slope![i], 1e-14));
      }
      expect(gridded.logMarginalLikelihood,
          closeTo(plain.logMarginalLikelihood, 1e-12));
    });

    test(
        'extra grid points between observations do not disturb the ones on '
        'top of them', () {
      // The interpolated points are steps with nothing to update on, so they
      // carry no information and must leave the rest of the answer alone.
      final data = _data();
      final plain = _model().smooth(data);

      final dense = <double>[];
      for (var i = 0; i < data.length; i++) {
        dense.add(data[i].time);
        if (i + 1 < data.length) {
          dense.add((data[i].time + data[i + 1].time) / 2);
        }
      }
      final gridded = _model().smooth(data, grid: Float64List.fromList(dense));

      // Not bit-identical: splitting A(dt) into A(dt/2) A(dt/2) is the same
      // matrix in exact arithmetic and a slightly different one in floating
      // point, and the difference is amplified while the diffuse prior of
      // order 2e5 is still washing out.
      expect(gridded.length, dense.length);
      for (var i = 0; i < data.length; i++) {
        expect(gridded.level[2 * i], closeTo(plain.level[i], 1e-8));
        expect(gridded.levelVariance[2 * i],
            closeTo(plain.levelVariance[i], 1e-8));
      }
    });

    test('the band bulges over a gap', () {
      // Two dense blocks with seven weeks of nothing between them. Halfway
      // across, the trend is being interpolated rather than measured, and the
      // posterior says so.
      final data = [
        for (var i = 0; i < 12; i++) Observation(i.toDouble(), 75 + 0.04 * i),
        for (var i = 0; i < 12; i++)
          Observation(60 + i.toDouble(), 77 + 0.04 * i),
      ];
      final result =
          _model().smooth(data, grid: Float64List.fromList([11, 35, 60]));

      expect(
          result.levelVariance[1], greaterThan(10 * result.levelVariance[0]));
      expect(
          result.levelVariance[1], greaterThan(10 * result.levelVariance[2]));
    });

    test(
        'the band widens beyond the last observation, at the rate the model '
        'says it should', () {
      final data = _data();
      final last = data.last.time;
      final result = _model().smooth(data,
          grid: Float64List.fromList([last, last + 5, last + 50, last + 400]));

      expect(result.levelVariance[1], greaterThan(result.levelVariance[0]));
      expect(result.levelVariance[2], greaterThan(result.levelVariance[1]));

      // Level variance over a horizon h grows like h^3, so the band grows
      // like h^1.5: eight times the horizon should be about 22.6 times the
      // band. It falls a little short of that because part of the width is
      // uncertainty about where the series ended, which does not grow.
      final growth =
          math.sqrt(result.levelVariance[3] / result.levelVariance[2]);
      expect(growth, greaterThan(15));
      expect(growth, lessThan(math.pow(8, 1.5)));
    });

    test('an empty grid asks for no output at all', () {
      final result = _model().smooth(_data(), grid: Float64List(0));
      expect(result.length, 0);
      expect(result.logMarginalLikelihood.isFinite, isTrue);
    });

    test('a grid with no observations at all has no answer to give', () {
      expect(
        () => _model().smooth(const [], grid: Float64List.fromList([0, 1, 2])),
        throwsA(isA<StateError>()),
      );
    });

    test('under the older prior the same request returns the prior', () {
      final result = StructuralModel.localLinearTrend(
        processVariance: 8e-4,
        measurementVariance: 0.2,
        initialization: const ApproximateDiffuse(),
      ).smooth(const [], grid: Float64List.fromList([0, 1, 2]));
      expect(result.length, 3);
      for (final v in result.levelVariance) {
        expect(v, greaterThan(1e4));
      }
    });
  });
}
