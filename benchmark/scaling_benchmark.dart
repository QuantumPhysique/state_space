// Measures how the smoother scales, against the quadratic method it is meant
// to replace.
//
//   dart compile exe benchmark/scaling_benchmark.dart -o /tmp/scaling
//   /tmp/scaling
//
// Run it AOT-compiled. Under the JIT the first few thousand iterations are
// still being optimised and the small-N numbers are meaningless.

// ignore_for_file: avoid_print

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:state_space/src/engine/kalman.dart';
import 'package:state_space/src/engine/timeline.dart';
import 'package:state_space/state_space.dart';

/// Kernel smoothing, the obvious way: every output point is a weighted mean
/// of every input point. Quadratic, and the reason this package exists.
Float64List nadarayaWatson(List<Observation> data, double bandwidth) {
  final n = data.length;
  final out = Float64List(n);
  for (var i = 0; i < n; i++) {
    var weight = 0.0;
    var weighted = 0.0;
    for (var j = 0; j < n; j++) {
      final u = (data[i].time - data[j].time) / bandwidth;
      final w = math.exp(-0.5 * u * u);
      weight += w;
      weighted += w * data[j].value;
    }
    out[i] = weighted / weight;
  }
  return out;
}

List<Observation> series(int n, math.Random random) {
  final data = <Observation>[];
  var time = 0.0;
  var value = 80.0;
  for (var i = 0; i < n; i++) {
    time += 0.5 + random.nextDouble();
    value += 0.01 * (random.nextDouble() - 0.5);
    data.add(Observation(time, value + 0.3 * (random.nextDouble() - 0.5)));
  }
  return data;
}

/// Median of five timed samples after a warmup, in microseconds per call.
///
/// Each sample repeats the body until it has run for at least 25 ms, because
/// at N = 100 a single pass finishes well inside the clock's resolution and
/// timing it directly measures the clock.
double median(void Function() body, {int warmup = 3, int samples = 5}) {
  for (var i = 0; i < warmup; i++) {
    body();
  }

  var repeats = 1;
  while (true) {
    final probe = Stopwatch()..start();
    for (var i = 0; i < repeats; i++) {
      body();
    }
    probe.stop();
    if (probe.elapsedMicroseconds >= 25000 || repeats >= 1 << 20) break;
    repeats *= 4;
  }

  final times = <double>[];
  for (var s = 0; s < samples; s++) {
    final watch = Stopwatch()..start();
    for (var i = 0; i < repeats; i++) {
      body();
    }
    watch.stop();
    times.add(watch.elapsedMicroseconds / repeats);
  }
  times.sort();
  return times[times.length ~/ 2];
}

void main() {
  final random = math.Random(2026);
  final model = StructuralModel.localLinearTrend(
    processVariance: 1e-3,
    measurementVariance: 0.09,
  );

  print('N        filter      smooth      ns/obs   Nadaraya-Watson');
  for (final n in [100, 1000, 10000, 100000]) {
    final data = series(n, random);

    final filter = median(() => model.logLikelihood(data));
    final smooth = median(() => model.smooth(data));
    // Quadratic time gets out of hand quickly; stop before it becomes rude.
    final quadratic =
        n <= 20000 ? median(() => nadarayaWatson(data, 3), samples: 3) : null;

    print('${n.toString().padRight(9)}'
        '${_ms(filter).padRight(12)}'
        '${_ms(smooth).padRight(12)}'
        '${(smooth * 1000 / n).toStringAsFixed(1).padRight(9)}'
        '${quadratic == null ? '--' : _ms(quadratic)}');
  }

  // What the two-state specialisation is worth on the pass that fit() runs
  // fifty times per call.
  print('');
  print('forward pass only        generic     fast path   speedup');
  for (final n in [1000, 10000, 100000]) {
    final data = series(n, random);
    final timeline = Timeline.merge(data, null);
    final components = model.components;

    final generic = median(() => KalmanFilter(components,
            measurementVariance: model.measurementVariance,
            initialization: model.initialization)
        .run(timeline));
    final fast = median(() => model.logLikelihood(data));

    print('N = ${n.toString().padRight(20)}'
        '${_ms(generic).padRight(12)}'
        '${_ms(fast).padRight(12)}'
        '${(generic / fast).toStringAsFixed(2)}x');
  }
}

String _ms(double micros) => '${(micros / 1000).toStringAsFixed(2)} ms';
