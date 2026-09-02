import 'dart:math' as math;
import 'dart:typed_data';

import '../component.dart';
import '../engine/matrix_block.dart';
import '../parameter_spec.dart';
import 'stationary.dart';

/// How many times the process is differentiable, in the usual `nu` notation.
///
/// Only the half-integer orders have a finite-dimensional state-space form, and
/// only these three are small enough to be worth having: `nu = 7/2` costs a
/// fourth state to buy a smoothness nobody can tell apart from the fifth-order
/// one on real data.
enum MaternOrder {
  /// `nu = 1/2`: the Ornstein-Uhlenbeck process. Continuous, nowhere
  /// differentiable, and the exact continuous-time analogue of an AR(1).
  oneHalf(1),

  /// `nu = 3/2`: once differentiable. The default choice for Gaussian process
  /// smoothing in most of the literature.
  threeHalves(2),

  /// `nu = 5/2`: twice differentiable, and about as smooth as anything short of
  /// the squared exponential.
  fiveHalves(3);

  const MaternOrder(this.stateDim);

  /// Number of states the order needs, which is `nu + 1/2`.
  final int stateDim;
}

/// A stationary Matérn process: the standard Gaussian process kernel, in
/// state-space form.
///
/// ```text
/// k(tau) = variance * (1 + a) * exp(-a),               nu = 3/2
///          variance * (1 + a + a^2 / 3) * exp(-a),     nu = 5/2
///          variance * exp(-a),                         nu = 1/2
///
/// a = sqrt(2 nu) |tau| / lengthScale
/// ```
///
/// This is the first component in the package that is **stationary**, and the
/// distinction is not a technicality. A trend or a seasonal has no prior of its
/// own: where the level sits, and where the pattern sits in its cycle, are
/// questions only the data can answer, and the engine handles them as flat
/// directions. A Matérn process has an answer built in — it hovers around zero
/// with variance [variance] and forgets where it has been over about
/// [lengthScale] — so it needs no diffuse states at all and contributes none.
///
/// That makes it the right shape for a component that is a *deviation* rather
/// than a level: put it alongside a `LocalLinearTrend` and it absorbs the
/// correlated wobble the trend should not be chasing, leaving the trend to be a
/// trend. Water retention in a body-weight series is the motivating case, and
/// `nu = 1/2` is exactly the AR(1) such a series wants.
///
/// The order also decides how rough the curve is allowed to be, which is the
/// modelling choice a reader should actually make. The cubic spline of
/// `LocalLinearTrend` assumes a trend with a continuous derivative;
/// `nu = 1/2` assumes nothing of the sort and will happily produce a path with
/// visible corners. Body weight is arguably closer to the second, which is
/// worth knowing before reaching automatically for the spline.
///
/// Two parameters, and only one of them is a variance: see [parameterSpecs].
/// [lengthScale] is in the caller's time unit, so the bracket it is searched
/// over depends on that unit and can be set with [lengthScaleBounds].
class Matern extends Component {
  /// Validates rather than asserts, for the same reason
  /// `TrigonometricSeasonal` does: a length scale far outside its bracket
  /// produces a confident wrong answer rather than an obvious failure.
  Matern({
    required this.order,
    required this.variance,
    required this.lengthScale,
    this.lengthScaleBounds = const (lower: 1e-2, upper: 1e4),
  }) {
    if (!(variance > 0) || !variance.isFinite) {
      throw ArgumentError.value(
          variance, 'variance', 'must be finite and positive');
    }
    if (!(lengthScale > 0) || !lengthScale.isFinite) {
      throw ArgumentError.value(
          lengthScale, 'lengthScale', 'must be finite and positive');
    }
    final (:lower, :upper) = lengthScaleBounds;
    if (!(lower > 0) || !(lower < upper) || !upper.isFinite) {
      throw ArgumentError.value(lengthScaleBounds, 'lengthScaleBounds',
          'must be a positive increasing range');
    }
  }

  /// An Ornstein-Uhlenbeck process: `nu = 1/2`, one state.
  factory Matern.oneHalf({
    required double variance,
    required double lengthScale,
    ({
      double lower,
      double upper
    }) lengthScaleBounds = const (lower: 1e-2, upper: 1e4),
  }) =>
      Matern(
        order: MaternOrder.oneHalf,
        variance: variance,
        lengthScale: lengthScale,
        lengthScaleBounds: lengthScaleBounds,
      );

  /// `nu = 3/2`, two states: once differentiable, and the usual default.
  factory Matern.threeHalves({
    required double variance,
    required double lengthScale,
    ({
      double lower,
      double upper
    }) lengthScaleBounds = const (lower: 1e-2, upper: 1e4),
  }) =>
      Matern(
        order: MaternOrder.threeHalves,
        variance: variance,
        lengthScale: lengthScale,
        lengthScaleBounds: lengthScaleBounds,
      );

  /// `nu = 5/2`, three states: twice differentiable.
  factory Matern.fiveHalves({
    required double variance,
    required double lengthScale,
    ({
      double lower,
      double upper
    }) lengthScaleBounds = const (lower: 1e-2, upper: 1e4),
  }) =>
      Matern(
        order: MaternOrder.fiveHalves,
        variance: variance,
        lengthScale: lengthScale,
        lengthScaleBounds: lengthScaleBounds,
      );

  /// Which half-integer smoothness this is.
  final MaternOrder order;

  /// Marginal variance of the process, `k(0)`, in squared signal units.
  final double variance;

  /// Distance over which the process forgets where it has been, in the
  /// caller's time unit.
  ///
  /// Not a correlation *time* in the half-life sense: the convention
  /// `a = sqrt(2 nu) |tau| / lengthScale` makes the three orders comparable, so
  /// a length scale of ten days means about the same amount of memory whichever
  /// order is used, which is the point of the convention.
  final double lengthScale;

  /// Range the fit searches [lengthScale] over, in the caller's time unit.
  ///
  /// The default spans six decades, which for daily data is a quarter of an
  /// hour to twenty-seven years. Time measured in seconds needs a different
  /// one, exactly as the variance bracket does.
  ///
  /// The bottom of it is only a request. [fit] raises it to the median gap
  /// between readings, because below that a Matérn is measurement noise under
  /// another name and the likelihood will take it — see [parameterSpecsAt].
  /// The top is honoured as given.
  final ({double lower, double upper}) lengthScaleBounds;

  /// The inverse length scale `sqrt(2 nu) / lengthScale`, which is where every
  /// formula in this class actually starts.
  double get rate =>
      switch (order) {
        MaternOrder.oneHalf => 1.0,
        MaternOrder.threeHalves => math.sqrt(3),
        MaternOrder.fiveHalves => math.sqrt(5),
      } /
      lengthScale;

  /// The covariance function, `k(tau)`.
  ///
  /// Exposed because it is the object a reader coming from the Gaussian
  /// process literature is looking for, and because the whole claim of the
  /// package is that the recursion computes the same thing. The reference test
  /// builds an `O(N^3)` dense posterior from this and checks it agrees.
  double covariance(double lag) {
    final a = rate * lag.abs();
    return variance *
        switch (order) {
          MaternOrder.oneHalf => 1.0,
          MaternOrder.threeHalves => 1 + a,
          MaternOrder.fiveHalves => 1 + a + a * a / 3,
        } *
        math.exp(-a);
  }

  @override
  int get stateDim => order.stateDim;

  @override
  int get parameterCount => 2;

  @override
  List<ParameterSpec> get parameterSpecs => [
        const VarianceParameter(),
        ShapeParameter(
          lower: math.log(lengthScaleBounds.lower),
          upper: math.log(lengthScaleBounds.upper),
        ),
      ];

  /// Raises the length-scale bracket to the sampling interval.
  ///
  /// Below one gap between readings a Matérn is indistinguishable from white
  /// noise, and — this is the part worth stating — the likelihood *prefers*
  /// that corner, because taking the measurement error for itself explains the
  /// data slightly better than leaving it alone. See
  /// [Component.parameterSpecsAt] for the measurement.
  ///
  /// One gap rather than two: a Matérn of length scale equal to the sampling
  /// interval still has a correlation of `exp(-sqrt(2 nu))` between
  /// neighbouring readings, which is a real thing the data can see, unlike a
  /// cycle at the Nyquist period.
  @override
  List<ParameterSpec> parameterSpecsAt({required double resolution}) {
    if (!(resolution > 0)) return parameterSpecs;
    final floor = math.max(lengthScaleBounds.lower, resolution);
    final upper = lengthScaleBounds.upper;
    if (!(floor < upper)) {
      throw ArgumentError('a Matern length scale is bracketed at '
          '[${lengthScaleBounds.lower}, $upper], but the readings are '
          '$resolution apart, and a length scale below one sampling interval '
          'is measurement noise rather than a separate component. Widen '
          'lengthScaleBounds or drop the component.');
    }
    return [
      const VarianceParameter(),
      ShapeParameter(lower: math.log(floor), upper: math.log(upper)),
    ];
  }

  // The state is (f, f', f'') truncated to the order's dimension, and the
  // transition is the matrix exponential of the companion form of
  // (d/dt + rate)^(nu + 1/2). All three eigenvalues coincide at -rate, so the
  // exponential is e^(-rate dt) times a polynomial in dt of degree one less
  // than the state dimension, and no eigen-decomposition is needed.
  //
  // Writing it in the scaled coordinate u = rate * dt turns every entry into a
  // pure function of u times a power of the rate, which is the form both this
  // and processNoise use: entry (i, j) carries rate^(i - j).
  @override
  void transition(double dt, MatrixBlock out) {
    final u = rate * dt;
    final decay = math.exp(-u);
    final r = rate;
    switch (order) {
      case MaternOrder.oneHalf:
        out.set(0, 0, decay);
      case MaternOrder.threeHalves:
        out.set(0, 0, decay * (1 + u));
        out.set(0, 1, decay * u / r);
        out.set(1, 0, decay * -u * r);
        out.set(1, 1, decay * (1 - u));
      case MaternOrder.fiveHalves:
        final h = u * u / 2;
        out.set(0, 0, decay * (1 + u + h));
        out.set(0, 1, decay * (u + u * u) / r);
        out.set(0, 2, decay * h / (r * r));
        out.set(1, 0, decay * -h * r);
        out.set(1, 1, decay * (1 + u - u * u));
        out.set(1, 2, decay * (u - h) / r);
        out.set(2, 0, decay * (h - u) * r * r);
        out.set(2, 1, decay * (u * u - 3 * u) * r);
        out.set(2, 2, decay * (1 - 2 * u + h));
    }
  }

  /// `Q(dt) = P_inf - A(dt) P_inf A(dt)'`, which is what "stationary" means:
  /// the covariance the process loses by being propagated is exactly the
  /// covariance the driving noise has to put back.
  ///
  /// In the scaled coordinate of [transition] the whole product collapses to a
  /// function of `u = rate * dt` times `variance * rate^(i + j)`, so there is
  /// no matrix work to do and nothing to allocate.
  ///
  /// Over a very short gap the leading terms cancel — `Q` is `O(dt^3)` for
  /// `nu = 3/2` while the terms it is built from are `O(1)` — and the result
  /// loses relative precision. It is left as it is, because the *absolute*
  /// error is a few units in the last place of `variance`, which is exactly the
  /// rounding the covariance propagation around it already carries.
  @override
  void processNoise(double dt, MatrixBlock out) {
    final u = rate * dt;
    final decay = math.exp(-2 * u);
    final r = rate;
    switch (order) {
      case MaternOrder.oneHalf:
        out.set(0, 0, variance * (1 - decay));
      case MaternOrder.threeHalves:
        // Scaled transition rows, and the identity for the scaled stationary
        // covariance.
        final a0 = 1 + u, a1 = u, b0 = -u, b1 = 1 - u;
        out.set(0, 0, variance * (1 - decay * (a0 * a0 + a1 * a1)));
        out.set(0, 1, variance * r * -decay * (a0 * b0 + a1 * b1));
        out.set(1, 0, out.at(0, 1));
        out.set(1, 1, variance * r * r * (1 - decay * (b0 * b0 + b1 * b1)));
      case MaternOrder.fiveHalves:
        final h = u * u / 2;
        final a0 = 1 + u + h, a1 = u + u * u, a2 = h;
        final b0 = -h, b1 = 1 + u - u * u, b2 = u - h;
        final c0 = h - u, c1 = u * u - 3 * u, c2 = 1 - 2 * u + h;
        // The scaled stationary covariance is [[1, 0, -1/3], [0, 1/3, 0],
        // [-1/3, 0, 1]]: the derivative has variance rate^2 / 3, and a value
        // and its second derivative are negatively correlated, which is what
        // stops a twice-differentiable path from curving away forever.
        double form(double x0, double x1, double x2, double y0, double y1,
                double y2) =>
            x0 * y0 + x1 * y1 / 3 + x2 * y2 - (x0 * y2 + x2 * y0) / 3;
        final g00 = 1 - decay * form(a0, a1, a2, a0, a1, a2);
        final g01 = -decay * form(a0, a1, a2, b0, b1, b2);
        final g02 = -1 / 3 - decay * form(a0, a1, a2, c0, c1, c2);
        final g11 = 1 / 3 - decay * form(b0, b1, b2, b0, b1, b2);
        final g12 = -decay * form(b0, b1, b2, c0, c1, c2);
        final g22 = 1 - decay * form(c0, c1, c2, c0, c1, c2);
        final r2 = r * r;
        out.set(0, 0, variance * g00);
        out.set(0, 1, variance * r * g01);
        out.set(0, 2, variance * r2 * g02);
        out.set(1, 0, out.at(0, 1));
        out.set(1, 1, variance * r2 * g11);
        out.set(1, 2, variance * r2 * r * g12);
        out.set(2, 0, out.at(0, 2));
        out.set(2, 1, out.at(1, 2));
        out.set(2, 2, variance * r2 * r2 * g22);
    }
  }

  @override
  void observationAt(double time, Float64List out) {
    out[0] = 1;
    for (var i = 1; i < stateDim; i++) {
      out[i] = 0;
    }
  }

  /// Nothing here is diffuse. The process has a stationary distribution, and
  /// that distribution is the prior.
  @override
  List<bool> get diffuseStates => List.filled(stateDim, false);

  @override
  void properPrior(Float64List mean, MatrixBlock covariance) {
    mean.fillRange(0, stateDim, 0);
    covariance.fill(0);
    final r2 = rate * rate;
    switch (order) {
      case MaternOrder.oneHalf:
        covariance.set(0, 0, variance);
      case MaternOrder.threeHalves:
        covariance.set(0, 0, variance);
        covariance.set(1, 1, variance * r2);
      case MaternOrder.fiveHalves:
        covariance.set(0, 0, variance);
        covariance.set(0, 2, -variance * r2 / 3);
        covariance.set(1, 1, variance * r2 / 3);
        covariance.set(2, 0, -variance * r2 / 3);
        covariance.set(2, 2, variance * r2 * r2);
    }
  }

  /// For a stationary component this is the exact quantity `Component.wanderOver`
  /// defines, and it has a closed form only for `nu = 1/2`. It is integrated
  /// numerically instead — a few hundred kernel evaluations, once per penalty
  /// evaluation, against a forward pass that costs far more.
  @override
  double wanderOver(double span) => stationaryWander(covariance, span);

  /// An unobserved [order] beyond `nu = 1/2` carries derivatives, and the
  /// first of them is the rate of change of the component's own contribution.
  ///
  /// `nu = 1/2` has none to report, and that is not an oversight: an
  /// Ornstein-Uhlenbeck path is nowhere differentiable, so there is no slope
  /// for the smoother to estimate.
  @override
  int? get rateStateIndex => order == MaternOrder.oneHalf ? null : 1;

  @override
  Float64List get parameters =>
      Float64List.fromList([math.log(variance), math.log(lengthScale)]);

  @override
  Component withParameters(Float64List theta) => Matern(
        order: order,
        variance: math.exp(theta[0]),
        lengthScale: math.exp(theta[1]),
        lengthScaleBounds: lengthScaleBounds,
      );

  @override
  String toString() => 'Matern(order: ${order.name}, variance: $variance, '
      'lengthScale: $lengthScale)';
}
