import 'dart:typed_data';

import 'engine/matrix_block.dart';
import 'parameter_spec.dart';

/// One additive block of a structural time-series model.
///
/// A model is a sum of components, `y(t) = sum_i H_i(t) x_i(t) + noise`, where
/// each component owns [stateDim] consecutive states. The engine stacks the
/// blocks and never looks inside them; a component supplies its own transition
/// and process-noise matrices and never sees the filter.
///
/// Everything is continuous-time: [transition] and [processNoise] take a gap
/// `dt` in whatever time unit the caller uses, and must be exact for *any*
/// non-negative gap, including zero (`A(0) = I`, `Q(0) = 0`).
///
/// Implementations are immutable. [withParameters] returns a new instance.
///
/// To write one, extend this class with a `final` or `base` class (`final
/// class MyComponent extends Component`) and implement the nine abstract
/// members:
/// [stateDim], [parameterCount], [transition], [processNoise],
/// [observationAt], [diffuseStates], [properPrior], [parameters] and
/// [withParameters]. The rest have defaults. `checkComponent` in
/// `package:state_space/authoring.dart` tests the properties the engine relies
/// on.
///
/// {@category Components}
/// {@category How it works}
abstract base class Component {
  /// Const, so that components can be.
  const Component();

  /// A name for this component in the sentences [FitResult.warnings] writes.
  ///
  /// Every component in this package returns its class name as a literal.
  /// The default reads `runtimeType`, which an obfuscated build mangles, so a
  /// component of your own should override it.
  String get name => runtimeType.toString();

  /// Number of states this component contributes.
  int get stateDim;

  /// Number of free parameters in the unconstrained vector used by
  /// [withParameters] and returned by [parameters].
  int get parameterCount;

  /// Writes the transition block `A_i(dt)` into [out], which is
  /// [stateDim] x [stateDim] and may contain stale values.
  void transition(double dt, MatrixBlock out);

  /// Writes the process-noise block `Q_i(dt)` into [out], which is
  /// [stateDim] x [stateDim] and may contain stale values. It must be
  /// symmetric and positive semi-definite.
  void processNoise(double dt, MatrixBlock out);

  /// Writes this component's slice of the observation row `H(t)` into [out],
  /// a view of length [stateDim].
  ///
  /// [time] matters only to a component whose row varies with time, such as a
  /// regression column.
  void observationAt(double time, Float64List out);

  /// Which of this component's states have no proper prior, one flag per state.
  ///
  /// Trend and seasonal states are non-stationary and therefore diffuse; a
  /// damped cycle is stationary and has a proper stationary prior instead.
  List<bool> get diffuseStates;

  /// Writes the prior mean and covariance of the non-diffuse states.
  ///
  /// Entries belonging to diffuse states are ignored, so a fully diffuse
  /// component may leave both arguments untouched.
  void properPrior(Float64List mean, MatrixBlock covariance);

  /// The current parameter vector, unconstrained (variances as logs, bounded
  /// quantities as logits), as a fresh copy.
  Float64List get parameters;

  /// A copy of this component with [theta] as its [parameters].
  Component withParameters(Float64List theta);

  /// What each entry of [parameters] is, in the same order.
  ///
  /// Defaults to a log variance for every entry. Override it if any parameter
  /// is a length scale, a period or a damping factor; otherwise [fit] rescales
  /// that parameter by the fitted noise level and searches it over a bracket
  /// meant for variance ratios. See [ParameterSpec].
  List<ParameterSpec> get parameterSpecs =>
      List.filled(parameterCount, const VarianceParameter());

  /// [parameterSpecs], narrowed by what data sampled every [resolution] time
  /// units can resolve.
  ///
  /// [resolution] is the typical gap between readings (see
  /// `samplingResolution`), or zero when the series is too short for that to
  /// mean anything. The default ignores it.
  ///
  /// A shape parameter measured in time units needs this. A Matérn with a
  /// length scale far below the sampling interval is white noise and competes
  /// with the measurement error rather than the trend; on daily readings with
  /// a true noise standard deviation of 0.3, a Matérn allowed down to 0.01
  /// days reports a noise standard deviation of 0.002. The floor may only
  /// raise the lower end of a bracket, never past its upper end.
  List<ParameterSpec> parameterSpecsAt({required double resolution}) =>
      parameterSpecs;

  /// Root-mean-square deviation of this component's contribution from its own
  /// average, over a window of [span] time units of its own driving noise.
  ///
  /// This is the spread of the path,
  ///
  /// ```text
  /// (1/T) integral k(t, t) dt  -  (1/T^2) double integral k(t, s) dt ds
  /// ```
  ///
  /// with the diffuse starting point excluded. It is what [ComplexityPenalty]
  /// measures every component in, so that a trend's variance (per cubed time
  /// unit) and a seasonal's (per time unit) can be penalised on one scale.
  ///
  /// Defaults to 0, which exempts the component from [ComplexityPenalty].
  double wanderOver(double span) => 0;

  /// Why this component's flat directions might not be identifiable on a
  /// series running from [from] to [to] with readings typically [resolution]
  /// apart, or null if nothing about it is suspect.
  ///
  /// Called only to explain an [UnderdeterminedModelException]: the engine
  /// adds whatever each component returns to the message. [resolution] is
  /// zero when the series is too short to have one.
  String? identifiabilityHint(
    double from,
    double to, {
    double resolution = 0,
  }) => null;

  /// Whether this component's states never move: `A(dt) = I` and `Q(dt) = 0`
  /// for every gap, and every state diffuse.
  ///
  /// A regression coefficient is the case. Under exact diffuse initialisation
  /// such a state smooths to its filtered value, so declaring it lets the
  /// backward pass, which is cubic in the state dimension, skip it: a trend
  /// plus twenty indicators smooths two states rather than twenty-two.
  ///
  /// A component that returns true and then moves is silently mis-smoothed,
  /// so the default is false.
  bool get isStatic => false;

  /// Index within this component's block of a state holding the instantaneous
  /// rate of change of the component's contribution, or null if it has none.
  ///
  /// Read only for reporting, by [SmoothingResult.componentSlope].
  int? get rateStateIndex => null;
}
