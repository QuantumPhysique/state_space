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
/// non-negative gap, including zero. That is what makes irregular sampling a
/// non-issue rather than a special case.
///
/// Implementations are immutable. [withParameters] returns a new instance,
/// which is what lets the fitting code evaluate a likelihood surface without
/// mutating the caller's model.
abstract class Component {
  const Component();

  /// Number of states this component contributes.
  int get stateDim;

  /// Number of free parameters in the unconstrained vector used by
  /// [withParameters] and returned by [parameters].
  int get parameterCount;

  /// Writes the transition block `A_i(dt)` into [out], which is
  /// [stateDim] x [stateDim] and may contain stale values.
  void transition(double dt, MatrixBlock out);

  /// Writes the process-noise block `Q_i(dt)` into [out], which is
  /// [stateDim] x [stateDim] and may contain stale values.
  void processNoise(double dt, MatrixBlock out);

  /// Writes this component's slice of the observation row `H(t)` into [out],
  /// a view of length [stateDim].
  ///
  /// Time-varying because a regression component's row holds the covariate
  /// values at [time]. For the components in this release the row is constant
  /// and [time] is ignored.
  void observationAt(double time, Float64List out);

  /// Which of this component's states have no proper prior, one flag per state.
  ///
  /// Trend and seasonal states are non-stationary and therefore diffuse; a
  /// damped cycle is stationary and has a proper stationary prior instead.
  List<bool> get diffuseStates;

  /// Writes the prior mean and covariance of the non-diffuse states.
  ///
  /// Called once per fit. Entries belonging to diffuse states are ignored, so
  /// a fully diffuse component may leave both arguments untouched.
  void properPrior(Float64List mean, MatrixBlock covariance);

  /// The current parameter vector, unconstrained (variances as logs, bounded
  /// quantities as logits). Fitting works in this space so that no optimiser
  /// ever has to respect a constraint.
  Float64List get parameters;

  /// A copy of this component with [theta] as its [parameters].
  Component withParameters(Float64List theta);

  /// What each entry of [parameters] is, in the same order.
  ///
  /// Defaults to a log variance for every one, which is what every component
  /// shipped before v0.5 had and all it needed. A component with a length
  /// scale, a period or a damping factor among its parameters must override
  /// this, or [fit] will rescale that parameter by the fitted noise level and
  /// search it over a bracket meant for variance ratios. See [ParameterSpec].
  List<ParameterSpec> get parameterSpecs =>
      List.filled(parameterCount, const VarianceParameter());

  /// The same, narrowed by what the sampling of the data can actually resolve.
  ///
  /// [resolution] is the median gap between consecutive distinct observation
  /// times, or zero when the series is too short for that to mean anything.
  /// The default ignores it, which is right for every parameter whose meaning
  /// does not involve the time axis.
  ///
  /// It exists because a *shape* parameter measured in time units has a limit
  /// below which it stops being a different model. A Matérn with `nu = 1/2`
  /// and a length scale far below the sampling interval is white noise, so it
  /// competes with the measurement error rather than with the trend, and the
  /// likelihood is happy to let it win: on daily readings with a true noise
  /// standard deviation of 0.3, a Matérn allowed down to a length scale of
  /// 0.01 days takes the noise for itself and the fit reports a measurement
  /// standard deviation of 0.002. The reported noise level, the credible band
  /// and the trend curve are all then wrong together, and only
  /// [FitResult.atBracketEdge] says anything is amiss.
  ///
  /// This is the same kind of refusal as `TrigonometricSeasonal` rejecting a
  /// harmonic at or past the Nyquist frequency, moved to where the limit
  /// depends on the data rather than on the component alone. A caller's own
  /// bracket is still respected — the floor can only raise the lower end, and
  /// never past what the caller allowed.
  List<ParameterSpec> parameterSpecsAt({required double resolution}) =>
      parameterSpecs;

  /// Root-mean-square deviation of this component's contribution from its own
  /// average, over a window of [span] time units of its own driving noise.
  ///
  /// This exists so that a penalty can be stated in one unit for every
  /// component. A trend's variance is per cubed time unit and a seasonal's per
  /// time unit, so their raw parameters are not comparable and no single scale
  /// could serve both. How much each one contributes to the spread of the data
  /// is comparable, and is a quantity a modeller has an opinion about.
  ///
  /// It is deliberately the spread of the *path* and not the standard
  /// deviation the component reaches at the end,
  ///
  /// ```text
  /// (1/T) integral k(t, t) dt  -  (1/T^2) double integral k(t, s) dt ds
  /// ```
  ///
  /// because that is what the sample standard deviation of the observations
  /// estimates, and the two differ by a constant that is not the same for
  /// every component — a factor of sqrt(10) for a trend against sqrt(2) for a
  /// seasonal. Using the terminal figure would silently penalise trends about
  /// two and a half times harder than seasonals for the same visible
  /// contribution.
  ///
  /// The diffuse starting point is excluded. Where the component began is for
  /// the data to say; the penalty is about how much it moves afterwards.
  double wanderOver(double span);

  /// Why this component's flat directions might not be identifiable on a
  /// series running from [from] to [to], or null if nothing about it is
  /// suspect.
  ///
  /// The engine knows nothing about seasonal periods or holiday calendars and
  /// is not about to start. When the diffuse system comes out singular it asks
  /// each component whether it can explain itself and pastes together whatever
  /// comes back, which keeps the diagnosis where the knowledge is.
  ///
  /// Both endpoints are passed rather than the duration, because a component
  /// may care where the series sits on the time axis and not only how long it
  /// is: an event indicator whose occurrences all fall outside the data
  /// contributes a column of zeros, and that is the most common way for this
  /// to be reached at all.
  String? identifiabilityHint(double from, double to) => null;

  /// Whether this component's states never move: `A(dt) = I` and `Q(dt) = 0`
  /// for every gap, and every state diffuse.
  ///
  /// A regression coefficient is the case, and it is worth the engine knowing
  /// because such a state has *no dynamics to smooth*. Under exact diffuse
  /// initialisation its covariance conditional on the flat directions is
  /// identically zero at every step, so the backward pass's gain has zero rows
  /// and zero columns there: its smoothed moments equal its filtered ones, and
  /// it contributes nothing to anyone else's. Everything such a state is worth
  /// is carried in `dx/dd` by the forward pass and folded back in at the end.
  ///
  /// Declaring it lets the smoother work on the states that actually move,
  /// which for a trend plus twenty holiday indicators is two rather than
  /// twenty-two — and the backward pass is cubic in that number.
  ///
  /// A component that returns true and then moves would be silently
  /// mis-smoothed, so the default is false and the claim is opt-in.
  bool get isStatic => false;

  /// Index within this component's block of a state holding the instantaneous
  /// rate of change of the component's contribution, or null if it has none.
  ///
  /// Purely a convenience for reporting: it is how [stateDim]-agnostic result
  /// objects find a slope to expose. The engine never reads it.
  int? get rateStateIndex => null;
}
