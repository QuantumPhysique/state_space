/// Throws an [ArgumentError] naming [name] unless [value] is finite and
/// positive.
void checkPositive(double value, String name) {
  if (!(value > 0) || !value.isFinite) {
    throw ArgumentError.value(value, name, 'must be finite and positive');
  }
}
