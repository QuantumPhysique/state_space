import 'dart:math' as math;
import 'dart:typed_data';

import '../arguments.dart';
import '../component.dart';
import '../engine/matrix_block.dart';
import '../exceptions.dart';
import '../parameter_spec.dart';
import 'stationary.dart';

/// How many times the process is differentiable, in the usual `nu` notation.
///
/// Only the half-integer orders have a finite-dimensional state-space form;
/// these are the three smallest.
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
/// The class is spelled without the accent because Dart identifiers are
/// ASCII.
///
/// The process is **stationary**: it hovers around zero with variance
/// [variance] and forgets where it has been over about [lengthScale], so it
/// has a proper prior and no diffuse states. Alongside a `LocalLinearTrend` it
/// absorbs short-lived correlated deviations, such as water retention in a
/// body-weight series, that the trend would otherwise chase; `nu = 1/2` is the
/// continuous-time AR(1).
///
/// The order sets how rough the path may be: `nu = 1/2` produces visible
/// corners, `nu = 5/2` a curve with two continuous derivatives.
///
/// Beside a trend it can also take over the measurement noise: when the
/// readings carry correlated day-to-day variation, the likelihood can prefer
/// a Matérn with a large variance and a fitted noise level near zero. Pass
/// `minimumMeasurementVariance` to [fit] at what the instrument can resolve
/// whenever a Matérn is in the model; [FitResult.warnings] says when this has
/// happened.
///
/// Two parameters, and only one of them is a variance: see [parameterSpecs].
/// [lengthScale] is in the caller's time unit, so the bracket it is searched
/// over depends on that unit and can be set with [lengthScaleBounds].
///
/// {@category Components}
final class Matern extends Component {
  /// A Matérn process of the given [order], [variance] and [lengthScale].
  ///
  /// [variance] and [lengthScale] must be finite and positive, and
  /// [lengthScaleBounds] a positive increasing range: the bracket [fit]
  /// searches the length scale over.
  Matern({
    required this.order,
    required this.variance,
    required this.lengthScale,
    this.lengthScaleBounds = const (lower: 1e-2, upper: 1e4),
  }) {
    checkPositive(variance, 'variance');
    checkPositive(lengthScale, 'lengthScale');
    final (:lower, :upper) = lengthScaleBounds;
    if (!(lower > 0) || !(lower < upper) || !upper.isFinite) {
      throw ArgumentError.value(
        lengthScaleBounds,
        'lengthScaleBounds',
        'must be a positive increasing range',
      );
    }
  }

  /// An Ornstein-Uhlenbeck process: `nu = 1/2`, one state.
  factory Matern.oneHalf({
    required double variance,
    required double lengthScale,
    ({double lower, double upper}) lengthScaleBounds = const (
      lower: 1e-2,
      upper: 1e4,
    ),
  }) => Matern(
    order: MaternOrder.oneHalf,
    variance: variance,
    lengthScale: lengthScale,
    lengthScaleBounds: lengthScaleBounds,
  );

  /// `nu = 3/2`, two states: once differentiable, and the usual default.
  factory Matern.threeHalves({
    required double variance,
    required double lengthScale,
    ({double lower, double upper}) lengthScaleBounds = const (
      lower: 1e-2,
      upper: 1e4,
    ),
  }) => Matern(
    order: MaternOrder.threeHalves,
    variance: variance,
    lengthScale: lengthScale,
    lengthScaleBounds: lengthScaleBounds,
  );

  /// `nu = 5/2`, three states: twice differentiable.
  factory Matern.fiveHalves({
    required double variance,
    required double lengthScale,
    ({double lower, double upper}) lengthScaleBounds = const (
      lower: 1e-2,
      upper: 1e4,
    ),
  }) => Matern(
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
  /// order is used.
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
      label: 'length scale',
      lower: math.log(lengthScaleBounds.lower),
      upper: math.log(lengthScaleBounds.upper),
    ),
  ];

  /// Raises the length-scale bracket to the sampling interval.
  ///
  /// Below one gap between readings a Matérn is indistinguishable from white
  /// noise, and the likelihood prefers that corner; see
  /// [Component.parameterSpecsAt]. At one gap neighbouring readings still
  /// correlate by `exp(-sqrt(2 nu))`, which the data can see.
  @override
  List<ParameterSpec> parameterSpecsAt({required double resolution}) {
    if (!(resolution > 0)) return parameterSpecs;
    final floor = math.max(lengthScaleBounds.lower, resolution);
    final upper = lengthScaleBounds.upper;
    if (!(floor < upper)) {
      throw UnderdeterminedModelException(
        'a Matern length scale is bracketed at '
        '[${lengthScaleBounds.lower}, $upper], but the readings are '
        '$resolution apart, and a length scale below one sampling interval '
        'is measurement noise rather than a separate component. Widen '
        'lengthScaleBounds or drop the component.',
      );
    }
    return [
      const VarianceParameter(),
      ShapeParameter(
        label: 'length scale',
        lower: math.log(floor),
        upper: math.log(upper),
      ),
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
  /// Over a very short gap the two terms very nearly cancel — `Q(0,0)` is
  /// `O(dt^3)` for `nu = 3/2` and `O(dt^5)` for `nu = 5/2`, while the terms it
  /// is built from are `O(1)` — so below [_seriesBelow] the integral form is
  /// summed directly instead. Computed the other way `Q(0,0)` has no correct
  /// digits once `rate * dt` falls under about `6e-6` for `nu = 3/2` or `6e-4`
  /// for `nu = 5/2`, and it comes out *negative* not much further down, which
  /// costs `Q` its positive semi-definiteness.
  ///
  /// Which gaps are small enough to reach depends on the caller's time unit
  /// and not on anything the component can see — `rate * dt` is `1.7e-5` for a
  /// length scale of `1e5` seconds sampled every second, which is an ordinary
  /// thing to ask for.
  @override
  void processNoise(double dt, MatrixBlock out) {
    final u = rate * dt;
    if (u < _seriesBelow) {
      _seriesNoise(u, out);
      return;
    }
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
        double form(
          double x0,
          double x1,
          double x2,
          double y0,
          double y1,
          double y2,
        ) => x0 * y0 + x1 * y1 / 3 + x2 * y2 - (x0 * y2 + x2 * y0) / 3;
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

  /// `Q(dt)` from the integral it is defined by, summed as a power series.
  ///
  /// `Q = integral_0^dt A(s) q e e' A(s)' ds`, and in the scaled coordinate
  /// `x = rate * s` every entry is a fixed polynomial combination of the
  /// moments `integral_0^u x^m exp(-2x) dx`. Summing those directly has no
  /// cancellation in it, so the leading behaviour survives however small the
  /// gap. The polynomials are the last column of the scaled transition, which
  /// is where the driving noise enters the companion form:
  ///
  /// ```text
  /// nu = 1/2   1
  /// nu = 3/2   x,        1 - x
  /// nu = 5/2   x^2 / 2,  x - x^2 / 2,  1 - 2x + x^2 / 2
  /// ```
  ///
  /// and `q` is the spectral intensity `2 variance sqrt(pi) rate^(2p-1)
  /// Gamma(p) / Gamma(p - 1/2)`, which comes to `2 variance rate`,
  /// `4 variance rate^3` and `16 variance rate^5 / 3` for the three orders.
  void _seriesNoise(double u, MatrixBlock out) {
    final v = variance;
    final r = rate;
    switch (order) {
      case MaternOrder.oneHalf:
        out.set(0, 0, 2 * v * _moment(0, u));
      case MaternOrder.threeHalves:
        final m0 = _moment(0, u), m1 = _moment(1, u), m2 = _moment(2, u);
        final scale = 4 * v;
        final offDiagonal = scale * r * (m1 - m2);
        out.set(0, 0, scale * m2);
        out.set(0, 1, offDiagonal);
        out.set(1, 0, offDiagonal);
        out.set(1, 1, scale * r * r * (m0 - 2 * m1 + m2));
      case MaternOrder.fiveHalves:
        final m0 = _moment(0, u),
            m1 = _moment(1, u),
            m2 = _moment(2, u),
            m3 = _moment(3, u),
            m4 = _moment(4, u);
        final scale = 4 * v / 3;
        final r2 = r * r;
        final q00 = scale * m4;
        final q01 = 2 * scale * r * (m3 - m4 / 2);
        final q02 = 2 * scale * r2 * (m2 - 2 * m3 + m4 / 2);
        final q11 = 4 * scale * r2 * (m2 - m3 + m4 / 4);
        final q12 = 4 * scale * r2 * r * (m1 - 2.5 * m2 + 1.5 * m3 - m4 / 4);
        final q22 =
            4 * scale * r2 * r2 * (m0 - 4 * m1 + 5 * m2 - 2 * m3 + m4 / 4);
        out.set(0, 0, q00);
        out.set(0, 1, q01);
        out.set(0, 2, q02);
        out.set(1, 0, q01);
        out.set(1, 1, q11);
        out.set(1, 2, q12);
        out.set(2, 0, q02);
        out.set(2, 1, q12);
        out.set(2, 2, q22);
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

  /// Integrated numerically from [covariance]; only `nu = 1/2` has a closed
  /// form.
  @override
  double wanderOver(double span) => stationaryWander(covariance, span);

  /// The derivative state for `nu = 3/2` and `5/2`. An Ornstein-Uhlenbeck
  /// path (`nu = 1/2`) is nowhere differentiable, so it has none.
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
  String get name => 'Matern';

  @override
  bool operator ==(Object other) =>
      other is Matern &&
      other.order == order &&
      other.variance == variance &&
      other.lengthScale == lengthScale &&
      other.lengthScaleBounds == lengthScaleBounds;

  @override
  int get hashCode =>
      Object.hash(Matern, order, variance, lengthScale, lengthScaleBounds);

  @override
  String toString() =>
      'Matern(order: ${order.name}, variance: $variance, '
      'lengthScale: $lengthScale)';
}

/// Where the closed form stops being the better of the two.
///
/// Both are accurate on either side of this for some way, so the exact value
/// is not delicate; it was chosen as the middle of the band where the two agree
/// to the last few bits for all three orders.
const double _seriesBelow = 0.1;

/// `integral_0^u x^m exp(-2x) dx`, as its power series
/// `sum_n (-2)^n u^(m+n+1) / (n! (m + n + 1))`.
///
/// Only ever called for `u` below [_seriesBelow], where successive terms fall
/// off like `(2u)^n / n!` and a handful suffice.
double _moment(int m, double u) {
  var coefficient = math.pow(u, m + 1).toDouble();
  var sum = coefficient / (m + 1);
  for (var n = 1; n <= 24; n++) {
    coefficient *= -2 * u / n;
    final term = coefficient / (m + n + 1);
    sum += term;
    if (term.abs() <= sum.abs() * 1e-18) break;
  }
  return sum;
}
