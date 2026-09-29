import '../component.dart';

/// Index of the first state of each component in the stacked state vector.
List<int> blockOffsets(List<Component> components) {
  final offsets = <int>[];
  var next = 0;
  for (final c in components) {
    offsets.add(next);
    next += c.stateDim;
  }
  return offsets;
}

/// Index in the stacked state vector of every diffuse state, ascending.
List<int> diffuseStateIndices(List<Component> components) {
  final indices = <int>[];
  var next = 0;
  for (final c in components) {
    for (final flag in c.diffuseStates) {
      if (flag) indices.add(next);
      next++;
    }
  }
  return indices;
}
