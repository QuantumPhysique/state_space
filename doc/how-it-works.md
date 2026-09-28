# How it works

## Three descriptions of one model

The same model can be written as a Gaussian process, as a stochastic
differential equation, or as a state-space model for a Kalman filter. The
package computes the third, which gives the same answer as the first.

### As a Gaussian process

Put a prior `f ~ GP(0, k)` on the signal, observe it with noise, and the
posterior is the textbook expression:

```text
E[f(t*) | y] = k*' C^-1 y,     Var[f(t*) | y] = k(t*,t*) - k*' C^-1 k*
log p(y) = -0.5 (y' C^-1 y + log|C| + N log 2 pi),     C = K + sigma_eps^2 I
```

This is exact, and costs `O(N³)` time and `O(N²)` memory. It is the reference
the tests compare against; see [Validation](https://github.com/QuantumPhysique/state_space/blob/main/doc/validation.md).

### As a stochastic differential equation

For a Markovian kernel the same prior is a linear SDE. Let the rate of change be
a Wiener process and the level be its integral:

```text
d(mu) = nu dt,   d(nu) = sigma dB
```

which discretises **exactly**, over any gap `dt`, to

```text
A(dt) = [[1, dt],      Q(dt) = sigma^2 [[dt^3/3, dt^2/2],
         [0,  1]]                      [dt^2/2, dt    ]]
```

The off-diagonal term in `Q` is what matters for irregular sampling: over a
gap, uncertainty about the slope integrates into uncertainty about the level,
and the two become correlated. A discrete local linear trend with a diagonal
`Q` drops that term and is exact only for unit steps, which is why time here is
a `double` and not an index.

### As a Kalman filter and RTS smoother

These compute the same posterior and the same log marginal likelihood in `O(N)`
time and memory, and give the slope and the per-component decomposition along
the way.

The implied kernel for `LocalLinearTrend` is the cubic spline kernel
`k(t,t') = σ²(m³/3 + m²|t−t'|/2)` with `m = min(t, t')`, so **the trend curve is
a natural cubic smoothing spline** (Wahba 1978). The smoothing parameter is
`lambda = measurementVariance / processVariance`, and `fit` estimates it by
maximum marginal likelihood rather than by cross-validation.

## Exact diffuse initialisation

A trend or a seasonal has no proper prior: the level of a random walk has no
stationary distribution. The flat prior can be approximated with a very wide
proper one, or the flat directions can be handled exactly. The package does the
latter by default.

It uses **augmentation** rather than a second set of recursions. Write the
initial state as `x(0) = a + B d`, with `d` unknown and flat. Everything
downstream is affine in `d`, so the filter carries the sensitivity `dx/dd`
alongside the state (one extra mean propagation per flat direction, no extra
covariance work), and the flat directions are integrated out in closed form at
the end of the pass. The smoother reuses the same gains and recombines by the
law of total variance.

The integration leaves a `log|M|` term, which makes the result exactly the
**restricted** likelihood. It is comparable across models only when the diffuse
structure matches; see
[Choosing a model](https://github.com/QuantumPhysique/state_space/blob/main/doc/choosing-a-model.md#comparing-models).

Because it is exact, the data has to determine those directions. A two-state
trend needs readings at two distinct times; given one, the package throws an
`UnderdeterminedModelException` instead of returning a variance whose size
comes from the prior, and the message names the component responsible where it
can. The same goes for two components that produce the same signal,
such as a trend beside a level, or one event entered twice.

Whether `M` is singular is decided after scaling it to unit diagonal, so the
decision does not depend on the units of each direction: a direction is refused
when less than one part in `1e10` of its information is not already carried by
the others. A matrix that is singular in exact arithmetic leaves a pivot of
rounding size, about `1e-16`, whose sign is an accident; the most
ill-conditioned determined models in the test suite stay above `1e-8`.

`ApproximateDiffuse` uses a very wide proper prior instead. It gives every flat
direction the same prior variance whatever its units, so it only works when
time is measured in a unit that keeps rates of change near order one, such as
days for a daily series. With time in seconds or milliseconds a slope's
prior is many orders of magnitude too wide next to a level's, and the curve can
be off by a sizeable fraction of the noise with a band of zero width, whatever
`kappa` is. Exact initialisation does not depend on the time unit at all.

## The filter

**The measurement update is in Joseph form.** For a scalar observation it expands
to a rank-two symmetric update costing `O(n²)` rather than `O(n³)`, and it stays
symmetric and positive semi-definite for *any* gain, including one degraded by
rounding. The textbook `P = (I − KH)P⁻` is algebraically equal but numerically
worse.

**`A` and `Q` are block diagonal**, so a prediction costs `2n · Σnᵢ²` rather than
`2n³`. The covariance is dense, since the gain is a rank-one update across every
state, but the transition is not.

**A two-state single-component model has a scalar fast path**, with the loops
unrolled and the state in local doubles. It is a little over three times faster
than the generic engine on a forward pass. The reference tests validate the
generic engine, and `fast_path_equivalence_test.dart` checks that the two agree
to 1e-12.

## The smoother

**It solves `P⁻G' = A P` by Cholesky** rather than forming an inverse. When a
predicted covariance is not quite positive definite it adds a small multiple of
its own mean diagonal and tries again, and after six escalations it throws a
`NumericalBreakdownException`. The multiple that worked is tried first at the
following steps, as a fraction of each step's scale, so a grid point far past
the data does not change the answer inside it.

**It runs only over the states that can move.** A coefficient with `A = I` and
`Q = 0` under a flat prior has covariance identically zero conditional on the flat
directions, so its smoother gain has zero rows *and* zero columns: it smooths to
its filtered value and contributes nothing to anyone else's. Since the backward
pass is cubic in the state dimension, a trend plus twenty holiday indicators is
the difference between two states and twenty-two. `Component.isStatic` is how a
component declares the property.

**The results overwrite the filtered moments in place.** Each smoothed step is
read once, by the step before it, so nothing is lost, and a decade of daily
data at sixteen states is a few megabytes per array.

## The fit

**The measurement variance is concentrated out analytically.** Scaling every
covariance in the model by a constant leaves the gains and every innovation
unchanged and scales every innovation variance by that constant, so the noise
level has a closed form given the ratios. A `k`-component model is therefore a
`k`-dimensional search rather than `k + 1`.

**The search has three stages.** A coordinate scan sweeps each parameter across
its whole bracket, because a likelihood that is flat over decades, or
multimodal, can leave a local search stuck. Then golden section for one
parameter, or Nelder–Mead with a restart for several. Finally each parameter is
probed along its own axis to see how far it can move before the objective
falls by half a nat.

**`SearchStart.previousParameters`** skips the scan and starts from the model you
passed in, for refitting as data arrives. With one parameter the golden-section
window moves along for as long as the optimum lands on its edge.

**A stationary component can take over the measurement noise.** When a Matérn's
or a cycle's variance finishes at the top of its bracket, `fit` searches again
from starts that hand the noise a larger share, and keeps whichever optimum is
higher.

**Data a model explains exactly**, such as identical readings under a trend,
profiles to a noise level of zero. The concentrated noise variance is floored at
(1e-9 times the largest reading)², so the fit still returns a model.

**Shape parameters measured in time get a floor from the data**, because below
the sampling interval they no longer describe a different model. See
[Components](https://github.com/QuantumPhysique/state_space/blob/main/doc/components.md#a-floor-from-the-sampling).

## What the fit tells you afterwards

* **`parameterStatus`**: whether each parameter was estimated, shrunk out at the
  bottom of its bracket, or pushed past an edge (the top, or for a shape
  parameter either end). A parameter counts as on a bound only when the
  likelihood at the bound is within half a nat of the optimum. A variance of
  zero is the edge of the parameter space rather than an interior point, so the
  width reported for such a parameter is one-sided and is not an error bar.
* **`plateauDecadesByParameter`**: how far each parameter can move before the fit
  loses half a nat. `NaN` where the coordinate is not a logarithm;
  `plateauWidthByParameter` has the raw number.
* **`warnings`**: the above in sentences, including the case where a width was
  measured while another parameter sat on a bound.

## The residuals

Standardised residuals are the **recursive** ones. Under a flat prior the
innovation is an affine function of the unknown starting point, and substituting
the final estimate would condition every residual on the whole series,
including its own future. Instead the starting point is re-estimated from what
came strictly *before* each observation, which makes the errors exactly
independent under the model.

Their sum of squares reproduces the likelihood's quadratic form to 6e-11
relative, as it should: the restricted likelihood factorises into these
predictive densities. There are `N − d` of them, as many as the likelihood
charges for.

## The penalty that is off by default

`ComplexityPenalty` puts a penalised-complexity penalty on the variances and is
off by default. It makes no measurable difference to the trend/seasonal
decomposition and is worse at recovering the variances; what it does reliably
is push a component's drift parameter to the floor when there is no drift. The measurements are in
[Validation](https://github.com/QuantumPhysique/state_space/blob/main/doc/validation.md#the-complexity-penalty).
