/// For writing your own [Component]: everything in `state_space.dart`, plus
/// the matrix view components write into, the sampling resolution [fit]
/// narrows brackets by, and a conformance check.
///
/// See [Components](https://github.com/QuantumPhysique/state_space/blob/main/doc/components.md#writing-your-own).
library;

export 'src/conformance.dart';
export 'src/engine/matrix_block.dart' show MatrixBlock;
export 'src/fit/fit.dart' show samplingResolution;
export 'state_space.dart';
