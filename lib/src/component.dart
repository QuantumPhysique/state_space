import 'dart:typed_data';

import 'engine/matrix_block.dart';

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
  /// series spanning [span] time units, or null if nothing about it is
  /// suspect.
  ///
  /// The engine knows nothing about seasonal periods and is not about to
  /// start. When the diffuse system comes out singular it asks each component
  /// whether it can explain itself and pastes together whatever comes back,
  /// which keeps the diagnosis where the knowledge is.
  String? identifiabilityHint(double span) => null;

  /// Index within this component's block of a state holding the instantaneous
  /// rate of change of the component's contribution, or null if it has none.
  ///
  /// Purely a convenience for reporting: it is how [stateDim]-agnostic result
  /// objects find a slope to expose. The engine never reads it.
  int? get rateStateIndex => null;
}
