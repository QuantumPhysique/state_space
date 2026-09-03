# Validation

Computing the right thing is a claim that can be tested, and this is the part of
the package worth reading before trusting it. There are several independent
checks and none of them subsumes the others.

The shape of the argument throughout: the `O(N³)` Gaussian process is the
*specification*, the linear-time recursion is the *implementation*, and the tests
compare them in-language on every commit. That validates the model and not merely
the code.

## The linear-time answer equals the cubic-time definition

`test/dense_gp_reference_test.dart` builds the `N × N` spline kernel from the
equations in [How it works](https://github.com/QuantumPhysique/state_space/blob/main/doc/how-it-works.md), adds the noise, and computes the
textbook `O(N³)` posterior with a Cholesky factorisation from the `matrices`
package — a dev dependency used for nothing else.

The smoothed mean and the log likelihood agree to 1e-9, and the posterior variance
to 1e-9 absolute, at observation times and at grid points between them.

## The seasonal component equals its own closed-form kernel

The rotation is orthogonal and its driving noise isotropic, so `A(t−u) Q A(t'−u)'`
does not depend on `u` at all and the integral over the noise collapses to

```text
k(t, t') = sigma^2 min(t, t') sum_j cos(lambda_j (t - t'))
```

— Brownian motion multiplied by a comb of cosines.
`test/seasonal_reference_test.dart` builds that matrix densely and checks the
filter against it, then does the same for a trend plus two seasonals of different
periods and checks each component's share against the dense per-component
posterior. Confounding between a trend and a weekly pattern leaves the total fit
intact and shows up only in the decomposition, which is what you would want.

## A regression coefficient equals its generalised least-squares estimate

A coefficient is a flat direction like any other, so the exact diffuse machinery
estimates it as a by-product. `test/regression_reference_test.dart` is what makes
"by-product" mean exactly rather than approximately: the smoothed coefficient
matches the dense GLS estimate to 1e-9 and its variance matches the corresponding
diagonal of `(B' C⁻¹ B)⁻¹` to 1e-11, with the trend's own level and slope checked
from the same solve to show the regression columns have not disturbed them.

## The diffuse likelihood equals the restricted likelihood

Exact diffuse initialisation leaves a `log|M|` term behind when the flat prior is
integrated out. Get its sign or scale wrong and every likelihood shifts by a
constant that nothing else would notice — fits still converge, to the same place,
but the number reported is not the restricted likelihood it claims to be.

Densely the same quantity is REML, so the test computes

```text
-2 log L = (N-d) log 2pi + log|C| + log|B' C^-1 B| + y' P y
```

and checks both the total and `log|M|` on its own.

**Being REML is also the limit of what the number is good for.**
`comparability_test.dart` pins both directions
in which it fails to be a general model-comparison statistic: writing a regression
column in grams rather than kilograms shifts it by exactly `log 1000`, and
measuring time in half-days rather than days shifts it by exactly `log 2` per
diffuse direction carrying a time dimension — with the fit, the posterior and the
coefficient unchanged in both cases. The same test checks that a *proper* prior
has no such freedom and does not move. See
[Choosing a model](https://github.com/QuantumPhysique/state_space/blob/main/doc/choosing-a-model.md#how-to-tell-whether-it-earned-its-place--and-how-not-to).

## Cross-language golden fixtures, for both initialisations

`tool/generate_fixtures.py` builds the same model in statsmodels as an `MLEModel`
with time-varying system matrices, once with a wide proper prior and once with
`initialization='diffuse'`, and dumps the filtered, predicted and smoothed states,
their covariances and the per-observation likelihood to JSON. Script and both
fixture sets are committed, so the claim that exact initialisation changes the
first few steps and nothing else is testable rather than asserted.

Under the exact prior everything agrees to 1e-10 at every step including the
first. Under the wide one the first two steps agree to about four digits, because
there the smoothed covariance is the difference of two quantities of order 1e5
giving an answer of order 1e-3, which is asserted as a floor.

Two things worth knowing about the reference:

* `UnobservedComponents(level='local linear trend')` is the *discrete* model and
  is not what this package implements, so comparing against it would be comparing
  against a different model.
* statsmodels' exact diffuse smoother disagrees with a dense generalised
  least-squares computation about the smoothed slope at the very first step when
  the transition matrix is genuinely time-varying. It agrees to 1e-14 whenever
  the step is constant — at unit steps, at 2.5, at 0.5 — and diverges only once
  the steps vary, while this package agrees with the dense form in every case.
  Those four numbers are pinned against the dense form instead, which is a
  sharper test anyway.

## The fast path is held to the engine it specialises

A single two-state component runs its forward pass in unrolled scalars.
`fast_path_equivalence_test.dart` asserts agreement with the generic engine to
1e-12 across irregular gaps, a repeated timestamp, a two-month hole, missing
observations, unequal weights and all three initialisations. If the two ever
disagree, the fast path is wrong.

## The reduced backward pass is held to the full one

`static_state_test.dart` does the same job for the smoother's static-state
reduction, through a regression component that declines to admit it is static, so
the reference runs the full `n × n` recursion over states the reduced path skips.
They agree to nine significant figures across three output grids; the residual is
the reference's own jitter, which the reduced path never incurs.

## Analytic limits

* Sending the process variance to zero reproduces ordinary least squares for the
  two-state model and the precision-weighted mean for the one-state model — and
  the agreement improves as `1/kappa` with the width of the diffuse prior, which
  is asserted rather than assumed.
* Shifting all times by a constant changes nothing.
* Scaling every variance by `c` scales the posterior variance by `c` and leaves
  the posterior mean alone. That is the invariance the profile likelihood rests
  on, so it is tested directly.
* Reversing time reverses the answer.
* Posterior variance never grows when data is added.

## Numerical limits

`matern_noise_test.dart` checks that the Matérn process noise keeps its leading
asymptotic down to `dt = 1e-9` and that every leading principal minor stays
non-negative down to `dt = 1e-12`, because `Q = P∞ − A P∞ A'` is a difference of
two quantities of order `variance` whose answer is `O(dt³)` or `O(dt⁵)` and loses
every digit if computed that way.

## Parameter recovery

Simulate from known variances using the exact discretisation, fit, and check both
that the estimate is close and that it gets closer as the series grows.

## Edge cases

No observations, one observation, repeated timestamps, a five-year gap, constant
data, zero-variance readings, a model with as many flat directions as
observations, and rejection of NaN, infinities and unsorted input.

---

# Performance

`benchmark/scaling_benchmark.dart`, AOT-compiled, median of five samples, on an
M-series Mac. `smooth` is filter plus smoother plus the reported posterior.

| N | `logLikelihood` | `smooth` | per observation | `O(N²)` kernel smoother |
|---|---|---|---|---|
| 100 | 0.00 ms | 0.02 ms | 206 ns | 0.04 ms |
| 1 000 | 0.02 ms | 0.21 ms | 208 ns | 3.67 ms |
| 10 000 | 0.23 ms | 2.21 ms | 221 ns | 368 ms |
| 100 000 | 2.41 ms | 23.21 ms | 232 ns | — |

Flat cost per observation across three orders of magnitude, which is what linear
means. Ten years of daily readings smooth in about a millisecond.

The `logLikelihood` column is where the two-state fast path shows up — about four
times faster than the generic engine, steadily, from a thousand points to a
hundred thousand:

| N | generic engine | fast path | |
|---|---|---|---|
| 1 000 | 0.09 ms | 0.02 ms | 4.0x |
| 10 000 | 0.93 ms | 0.23 ms | 4.1x |
| 100 000 | 9.37 ms | 2.43 ms | 3.9x |

That is the half worth specialising: `fit` runs a forward pass per likelihood
evaluation, some fifty per call, while the backward pass runs once.

## Models with many components

The backward pass is cubic in the state dimension, so what it costs depends on how
many states actually move. Smoothing 20 000 points:

| model | states | `smooth` |
|---|---|---|
| `LocalLevel` | 1 | 2.5 ms |
| `LocalLinearTrend` | 2 | 4.4 ms |
| trend + Matérn 5/2 | 5 | 19.3 ms |
| trend + weekly seasonal | 6 | 38.5 ms |
| trend + 1 indicator | 3 | 7.2 ms |
| trend + 20 indicators | 22 | 663 ms |

The last two are cheaper than their state count suggests because a regression
coefficient never moves and is skipped by the backward pass — see
[How it works](https://github.com/QuantumPhysique/state_space/blob/main/doc/how-it-works.md#the-smoother).

## Why the linear algebra is hand-rolled

`A` and `Q` are block diagonal, which a general matrix type has no way to express
and would multiply the zeros of, and the Dart candidate offers no in-place or
out-parameter path — a pass would allocate a result object per operation, some
seventy thousand short-lived ones for a decade of daily data. On raw dense
products it is the faster of the two from 6×6 upward;
`benchmark/dependency_comparison.dart` has those numbers.
