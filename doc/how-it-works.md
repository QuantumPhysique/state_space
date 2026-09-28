# How it works

## One object, three descriptions

The package computes one thing under three descriptions, and its whole value is
that they agree. It computes the third to get the first.

### As a Gaussian process

Put a prior `f ~ GP(0, k)` on the signal, observe it with noise, and the
posterior is the textbook expression:

```text
E[f(t*) | y] = k*' C^-1 y,     Var[f(t*) | y] = k(t*,t*) - k*' C^-1 k*
log p(y) = -0.5 (y' C^-1 y + log|C| + N log 2 pi),     C = K + sigma_eps^2 I
```

Exact, and `O(N³)` in time with `O(N²)` in memory. **This is the specification.**
It is also what the tests compare against — see [Validation](https://github.com/QuantumPhysique/state_space/blob/main/doc/validation.md).

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

The off-diagonal term in `Q` is the whole story about irregular sampling: over a
gap, uncertainty about the slope integrates into uncertainty about the level, and
the two come out correlated. A discrete local linear trend with a diagonal `Q`
silently drops it and is exact only for unit steps. That difference is why this
package takes a `double` time and not an index.

### As a Kalman filter and RTS smoother

Which computes the same posterior and the same log marginal likelihood in `O(N)`
time and memory, and hands you the slope and the per-component decomposition on
the way past.

The implied kernel for `LocalLinearTrend` is the cubic spline kernel
`k(t,t') = σ²(m³/3 + m²|t−t'|/2)` with `m = min(t, t')`, so **the trend curve is
a natural cubic smoothing spline, computed as the exact posterior of a Gaussian
process, in linear time** (Wahba 1978). The smoothing parameter is
`lambda = measurementVariance / processVariance`, and `fit` estimates it by
maximum marginal likelihood rather than by cross-validation.

## Exact diffuse initialisation

A trend or a seasonal has no proper prior: the level of a random walk has no
stationary distribution. The standard answers are to approximate the flat prior
with a very wide proper one, or to handle the flat directions exactly. This
package does the second by default.

It does it by **augmentation** rather than by a second set of recursions. Write
the initial state as `x(0) = a + B d`, with `d` unknown and flat. Everything
downstream is affine in `d`, so the filter carries the sensitivity `dx/dd`
alongside the state — one extra mean propagation per flat direction and no extra
covariance work at all — and the flat directions are integrated out in closed
form at the end of the pass. The smoother reuses the same gains and recombines by
the law of total variance.

What survives the integration is a `log|M|` term, which makes the result exactly
the **restricted** likelihood. That is the right thing to maximise, and it is
comparable across models only when the diffuse structure matches — see
[Choosing a model](https://github.com/QuantumPhysique/state_space/blob/main/doc/choosing-a-model.md#how-to-tell-whether-it-earned-its-place--and-how-not-to).

Because it is exact, the data has to actually determine those directions. A
two-state trend needs readings at two distinct times; given one, the package
throws an `UnderdeterminedModelException` rather than returning a variance whose
size is an artefact of a prior, and the message names whichever component can
explain itself. The same goes for two components that produce the same signal,
such as a trend beside a level, or one event entered twice.

Whether `M` is singular is decided after scaling it to unit diagonal, so the
decision does not depend on the units of each direction: a direction is refused
when less than one part in `1e10` of its information is not already carried by
the others. A matrix that is singular in exact arithmetic leaves a pivot of
rounding size, about `1e-16`, whose sign is an accident; the most
ill-conditioned determined models in the test suite stay above `1e-8`.

`ApproximateDiffuse` answers instead with a very wide proper prior. It gives
every flat direction the same prior variance whatever its units, so it is sound
only when time is measured in a unit that keeps rates of change near order one,
such as days for a daily series. With time in seconds or milliseconds a slope's
prior is many orders of magnitude too wide next to a level's, and the curve can
be off by a sizeable fraction of the noise with a band of zero width, whatever
`kappa` is. Exact initialisation does not depend on the time unit at all.

## The filter

**The measurement update is in Joseph form.** For a scalar observation it expands
to a rank-two symmetric update costing `O(n²)` rather than `O(n³)`, and it is
symmetric and positive semi-definite by construction for *any* gain — including
one degraded by rounding. The textbook `P = (I − KH)P⁻` is algebraically the same
expression and numerically worse.

**`A` and `Q` are block diagonal** and the code exploits it, so a prediction costs
`2n · Σnᵢ²` rather than `2n³`. The covariance itself is dense — the gain is a
rank-one update spanning every state — but the transition is not.

**A two-state single-component model has a scalar fast path**, with the loops
unrolled and the state in local doubles. It is a little over three times faster
than the generic engine on a forward pass, and it is never load-bearing: the
generic engine is what the reference tests validate and remains the definition
of the answer. `fast_path_equivalence_test.dart` holds the two together to
1e-12.

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
read exactly once, by the step before it, so nothing is lost — and a decade of
daily data at sixteen states is a few megabytes per array, which is worth not
doubling.

## The fit

**The measurement variance is concentrated out analytically.** Scaling every
covariance in the model by a constant leaves the gains and every innovation
untouched and scales every innovation variance by that constant, so the noise
level has a closed form given the ratios. A `k`-component model is therefore a
`k`-dimensional search rather than `k + 1`, and the saving is exact and does not
shrink as components are added.

**The search is three stages.** A coordinate scan sweeps each parameter across
its whole bracket — a likelihood flat over decades will strand a local search
wherever it started, and a multimodal one will strand it in the wrong mode. Then
golden section for one parameter, or Nelder–Mead with a restart for several.
Finally each parameter is probed along its own axis to see how far it can move
before the objective falls half a nat.

**`SearchStart.previousParameters`** skips the scan and starts from the model you
passed in, for refitting as data arrives. With one parameter the golden-section
window moves along for as long as the optimum lands on its edge.

**A stationary component can take over the measurement noise.** When a Matérn's
or a cycle's variance finishes at the top of its bracket, `fit` searches again
from starts that hand the noise a larger share, and keeps whichever optimum is
higher.

**Data a model explains exactly**, such as identical readings under a trend,
profiles to a noise level of zero. The concentrated noise variance is floored at
one part in a billion of the largest reading, squared, so the fit returns
instead of building a model with no noise.

**Shape parameters measured in time get a floor from the data**, because below
the sampling interval they stop being a different model. See
[Components](https://github.com/QuantumPhysique/state_space/blob/main/doc/components.md#both-stationary-components-have-a-floor-you-did-not-set).

## What the fit tells you afterwards

* **`parameterStatus`** — whether each parameter was estimated, shrunk out at the
  bottom of its bracket, or pushed past an edge: the top, or for a shape
  parameter either end. A parameter counts as on a bound only when the
  likelihood at the bound is within half a nat of the optimum. A variance of
  zero is the edge of the parameter space rather than an interior point, so the
  width reported for such a parameter is one-sided and is not an error bar.
* **`plateauDecadesByParameter`** — how far each parameter can move before the fit
  loses half a nat. `NaN` where the coordinate is not a logarithm;
  `plateauWidthByParameter` has the raw number.
* **`warnings`** — the above in sentences, including the case where a width was
  measured beside another parameter sitting on a bound.

## The residuals

Standardised residuals are the **recursive** ones. Under a flat prior the
innovation is an affine function of the unknown starting point rather than a
number, and substituting the final estimate would condition every residual on the
whole series — including its own future. Instead the starting point is
re-estimated from what came strictly *before* each observation, which gives errors
that are exactly independent under the model rather than approximately so.

Their sum of squares reproduces the likelihood's own quadratic form to 6e-11
relative, which it must: the restricted likelihood factorises into precisely these
predictive densities. There are `N − d` of them, exactly as many as the likelihood
charges for.

## The penalty that is off by default

`ComplexityPenalty` puts a penalised-complexity penalty on the variances and is
off. It makes no measurable difference to the trend/seasonal decomposition at any
sample size and is worse at recovering the variances themselves; what it does
reliably is drive a component's drift parameter to the floor when there is no
drift to find. The measurements are in
[Validation](https://github.com/QuantumPhysique/state_space/blob/main/doc/validation.md#the-complexity-penalty).
