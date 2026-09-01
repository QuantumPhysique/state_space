import 'dart:typed_data';

/// Outcome of a simplex search.
class SimplexResult {
  const SimplexResult({
    required this.argument,
    required this.value,
    required this.evaluations,
    required this.converged,
  });

  /// Location of the best point found.
  final Float64List argument;

  /// The objective there.
  final double value;

  final int evaluations;

  /// Whether the simplex collapsed to within the tolerance rather than the
  /// search running out of evaluations.
  final bool converged;
}

// The classical coefficients: reflect, expand, contract, shrink. There are
// dimension-dependent variants that behave better above ten parameters or so;
// a structural model has three or four, and at that size they make no
// difference worth the extra explanation.
const double _reflection = 1;
const double _expansion = 2;
const double _contraction = 0.5;
const double _shrink = 0.5;

/// Maximises [objective] by Nelder-Mead, starting from [start].
///
/// Derivative-free, which is what a likelihood evaluated by a full Kalman pass
/// wants: a finite-difference gradient would cost `k + 1` passes per step and
/// spend them on differencing noise, since the likelihood is only smooth to
/// the accuracy the recursion delivers.
///
/// The initial simplex is [start] together with one vertex per coordinate,
/// displaced by [step]. In log variance ratios a step of two is about an order
/// of magnitude, which is the right scale for a surface whose interesting
/// region is usually a decade or two wide.
///
/// [steps] overrides that per coordinate, which a model mixing log variances
/// with a period in days needs: the two axes have nothing to do with one
/// another, and one displacement cannot suit both.
///
/// Convergence is on the spread of the objective across the simplex rather
/// than on the size of the simplex itself. For a log-likelihood that spread is
/// in nats and means something: a simplex whose vertices agree to within
/// [tolerance] nats is sitting on a region the data cannot distinguish,
/// however wide it happens to be in parameter space.
///
/// Nelder-Mead can stall at a point that is not a maximum, most often by
/// collapsing the simplex along one direction. The standard insurance is to
/// restart from the answer with a fresh simplex, which the caller does; a
/// second run that finds nothing new is decent evidence the first one
/// finished.
SimplexResult maximiseSimplex(
  double Function(Float64List) objective,
  Float64List start, {
  double step = 2,
  Float64List? steps,
  double tolerance = 1e-4,
  int maxEvaluations = 2000,
}) {
  final k = start.length;
  if (k < 1) {
    throw ArgumentError.value(start, 'start', 'needs at least one dimension');
  }
  if (steps != null && steps.length != k) {
    throw ArgumentError.value(
        steps, 'steps', 'expected one per dimension, got ${steps.length}');
  }

  final vertices = [
    for (var i = 0; i <= k; i++) Float64List.fromList(start),
  ];
  for (var i = 1; i <= k; i++) {
    vertices[i][i - 1] += steps?[i - 1] ?? step;
  }
  final values = Float64List(k + 1);
  var evaluations = 0;
  for (var i = 0; i <= k; i++) {
    values[i] = objective(vertices[i]);
    evaluations++;
  }

  final order = [for (var i = 0; i <= k; i++) i];
  final centroid = Float64List(k);
  final trial = Float64List(k);
  var converged = false;

  void sort() => order.sort((a, b) => values[b].compareTo(values[a]));

  double evaluateAt(Float64List point) {
    evaluations++;
    return objective(point);
  }

  while (evaluations < maxEvaluations) {
    sort();
    final best = order.first;
    final worst = order.last;
    final nextWorst = order[k - 1];

    if ((values[best] - values[worst]).abs() <= tolerance) {
      converged = true;
      break;
    }

    // Centroid of every vertex but the worst one.
    for (var j = 0; j < k; j++) {
      var sum = 0.0;
      for (var i = 0; i < k; i++) {
        sum += vertices[order[i]][j];
      }
      centroid[j] = sum / k;
    }

    for (var j = 0; j < k; j++) {
      trial[j] = centroid[j] + _reflection * (centroid[j] - vertices[worst][j]);
    }
    final reflected = evaluateAt(trial);

    if (reflected > values[best]) {
      // Better than anything so far: try going further in the same direction.
      final expanded = Float64List(k);
      for (var j = 0; j < k; j++) {
        expanded[j] = centroid[j] + _expansion * (trial[j] - centroid[j]);
      }
      final value = evaluateAt(expanded);
      if (value > reflected) {
        vertices[worst] = expanded;
        values[worst] = value;
      } else {
        vertices[worst] = Float64List.fromList(trial);
        values[worst] = reflected;
      }
      continue;
    }

    if (reflected > values[nextWorst]) {
      vertices[worst] = Float64List.fromList(trial);
      values[worst] = reflected;
      continue;
    }

    // The reflection was no help, so pull in — from outside if the reflection
    // at least beat the point it replaced, from inside if it did not.
    final outside = reflected > values[worst];
    final contracted = Float64List(k);
    for (var j = 0; j < k; j++) {
      final target = outside ? trial[j] : vertices[worst][j];
      contracted[j] = centroid[j] + _contraction * (target - centroid[j]);
    }
    final value = evaluateAt(contracted);
    if (value > (outside ? reflected : values[worst])) {
      vertices[worst] = contracted;
      values[worst] = value;
      continue;
    }

    // Nothing worked. Shrink everything towards the best vertex.
    for (var i = 0; i <= k; i++) {
      if (i == best) continue;
      final vertex = vertices[i];
      for (var j = 0; j < k; j++) {
        vertex[j] =
            vertices[best][j] + _shrink * (vertex[j] - vertices[best][j]);
      }
      values[i] = evaluateAt(vertex);
    }
  }

  sort();
  return SimplexResult(
    argument: Float64List.fromList(vertices[order.first]),
    value: values[order.first],
    evaluations: evaluations,
    converged: converged,
  );
}
