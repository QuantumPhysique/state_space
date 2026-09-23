import 'package:state_space/state_space.dart';
import 'package:test/test.dart';

/// The series the README, the library documentation and Getting started open
/// with. They promise a fit the package's own diagnostics are content with.
const _documented = [
  Observation(0, 81.3),
  Observation(2, 81.0),
  Observation(3, 80.5),
  Observation(7, 80.0),
  Observation(8, 80.0),
  Observation(8, 79.8),
  Observation(10, 79.7),
  Observation(11, 79.7),
  Observation(12, 79.5),
  Observation(17, 79.3),
  Observation(18, 79.3),
  Observation(19, 79.2),
  Observation(20, 79.2),
  Observation(21, 79.4),
  Observation(22, 79.2),
  Observation(24, 79.3),
  Observation(27, 79.1),
];

void main() {
  test('the documented first example fits without a warning', () {
    final fitted = fit(
        StructuralModel.localLinearTrend(processVariance: 1e-3), _documented);
    expect(fitted.warnings, isEmpty);
    expect(fitted.parameterStatus, [ParameterStatus.determined]);
    expect(fitted.plateauDecades, lessThan(1.5));
    expect(fitted.model.smooth(_documented).trendSlope, isNotNull);
  });
}
