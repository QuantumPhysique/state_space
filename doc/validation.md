# Validation

What the tests check, against what, and how closely. The `O(N³)` Gaussian
process is the reference and the linear-time recursion the implementation, and
the tests compare the two in Dart on every commit. The checks below are
independent, and none of them covers the others.

## The linear-time answer equals the cubic-time definition

`test/dense_gp_reference_test.dart` builds the `N × N` spline kernel from the
equations in [How it works](https://github.com/QuantumPhysique/state_space/blob/main/doc/how-it-works.md), adds the noise, and computes the
textbook `O(N³)` posterior with a Cholesky factorisation from the `matrices`
package, a dev dependency used only here.

The smoothed mean and the log likelihood agree to 1e-9, and the posterior variance
to 1e-9 absolute, at observation times and at grid points between them.

## The seasonal component equals its own closed-form kernel

The rotation is orthogonal and its driving noise isotropic, so `A(t−u) Q A(t'−u)'`
does not depend on `u` at all and the integral over the noise collapses to

```text
k(t, t') = sigma^2 min(t, t') sum_j cos(lambda_j (t - t'))
```

which is Brownian motion multiplied by a comb of cosines.
`test/seasonal_reference_test.dart` builds that matrix densely and checks the
filter against it, then does the same for a trend plus two seasonals of different
periods and checks each component's share against the dense per-component
posterior. Confounding between a trend and a weekly pattern leaves the total fit
intact and shows up only in the decomposition.

## A regression coefficient equals its generalised least-squares estimate

A coefficient is a flat direction like any other, so exact diffuse
initialisation estimates it along the way. `test/regression_reference_test.dart`
checks that the smoothed coefficient matches the dense GLS estimate to 1e-9 and its variance matches the corresponding
diagonal of `(B' C⁻¹ B)⁻¹` to 1e-11, with the trend's own level and slope checked
from the same solve to show the regression columns have not disturbed them.

## The diffuse likelihood equals the restricted likelihood

Exact diffuse initialisation leaves a `log|M|` term behind when the flat prior is
integrated out. With its sign or scale wrong, every likelihood would shift by a
constant: fits would still converge to the same place, but the reported number
would not be the restricted likelihood.

Densely the same quantity is REML, so the test computes

```text
-2 log L = (N-d) log 2pi + log|C| + log|B' C^-1 B| + y' P y
```

and checks both the total and `log|M|` on its own.

Being REML also limits what the number can be used for.
`comparability_test.dart` pins the two changes of unit that shift it while
leaving the fit, the posterior and the coefficient alone, and checks that the
likelihood under a *proper* prior does not move. What that means for
comparing models is in
[Choosing a model](https://github.com/QuantumPhysique/state_space/blob/main/doc/choosing-a-model.md#comparing-models).

## Cross-language golden fixtures, for both initialisations

`tool/generate_fixtures.py` builds the same model in statsmodels as an `MLEModel`
with time-varying system matrices, once with a wide proper prior and once with
`initialization='diffuse'`, and dumps the filtered, predicted and smoothed states,
their covariances and the per-observation likelihood to JSON. The script and
both fixture sets are committed, which lets the tests check that exact
initialisation changes the first few steps and nothing else.

Under the exact prior everything agrees to 1e-10 at every step including the
first. Under the wide one the first two steps agree to about four digits, because
there the smoothed covariance is the difference of two quantities of order 1e5
giving an answer of order 1e-3, which is asserted as a floor.

Two notes on the reference:

* `UnobservedComponents(level='local linear trend')` is the *discrete* model,
  not the one this package implements, so the fixtures do not use it.
* statsmodels' exact diffuse smoother (0.15.0) disagrees with a dense generalised
  least-squares computation about the smoothed slope at the very first step when
  the transition matrix is time-varying. It agrees to 1e-14 whenever the step is
  constant, at unit steps, at 2.5 and at 0.5. With unequal steps its first slope
  is the dense one multiplied by the second gap over the first, exactly, and its
  variance by that ratio squared, while the level agrees to every digit. This
  package agrees with the dense form in every case, and those four fixture
  numbers are pinned against the dense form instead.
  `tool/statsmodels_diffuse_repro.py` reproduces it in under a hundred lines:

  ```text
  steps 1, 2, 1, 2, ...   slope  statsmodels 0.113392860740  dense 0.056696430370
  irregular steps         slope  statsmodels 0.177785553235  dense 0.059261851078
  ```

## The fast path is held to the engine it specialises

A single two-state component runs its forward pass in unrolled scalars.
`fast_path_equivalence_test.dart` asserts agreement with the generic engine to
1e-12 across irregular gaps, a repeated timestamp, a two-month hole, missing
observations, unequal weights and all three initialisations. Under a very wide
approximate prior on a series with extreme gaps, 1e-9 and 1000 time units in the
same series, the two agree only to a few parts in a million in the likelihood,
which is the rounding that prior costs either engine; exact initialisation
agrees to 1e-12 there too.

## The reduced backward pass is held to the full one

`static_state_test.dart` does the same for the smoother's static-state
reduction, using a regression component that does not declare itself static, so
the reference runs the full `n × n` recursion over states the reduced path skips.
They agree to nine significant figures across three output grids; the residual is
the reference's own jitter, which the reduced path never incurs.

## Analytic limits

* Sending the process variance to zero reproduces ordinary least squares for the
  two-state model and the precision-weighted mean for the one-state model, and
  the tests check that the agreement improves as `1/kappa` with the width of
  the diffuse prior.
* Shifting all times by a constant changes nothing.
* Scaling every variance by `c` scales the posterior variance by `c` and leaves
  the posterior mean alone. The profile likelihood depends on this.
* Reversing time reverses the answer.
* Posterior variance never grows when data is added.

## Numerical limits

`matern_noise_test.dart` checks that the Matérn process noise keeps its leading
asymptotic down to `dt = 1e-9` and that every leading principal minor stays
non-negative down to `dt = 1e-12`. `Q = P∞ − A P∞ A'` is a difference of two
quantities of order `variance` with a result of order `dt³` or `dt⁵`, so
computing it that way loses every digit.

## Parameter recovery

Simulate from known variances using the exact discretisation, fit, and check both
that the estimate is close and that it gets closer as the series grows.

## Identifiability

`identifiability_test.dart` builds models the data cannot determine: a trend
beside a level, the same event entered twice, a step that switches on before
the first reading, and one reading under a two-state trend with the output grid
on either side of it. Each is refused, at integer days and at random times of
day, over twenty seeds. Ill-conditioned but determined models, an annual
seasonal on sixty days and time in milliseconds since the epoch, are not.

## Edge cases

No observations, one observation, repeated timestamps, a five-year gap, constant
data, data a trend fits exactly, zero-variance readings, a model with as many
flat directions as observations, and rejection of NaN, infinities and unsorted
input.

## The complexity penalty

`ComplexityPenalty` is off by default. A trend plus a weekly seasonal simulated
at known variances, fitted with and without the penalty, and compared against
the paths that generated it, over twelve replications:

```text
           seasonal RMSE              trend RMSE
         penalised    plain       penalised    plain
N =  60     0.0867   0.0870          0.0556   0.0545
N = 120     0.0824   0.0826          0.0483   0.0464
N = 500     0.0759   0.0759          0.0487   0.0480
```

The penalty makes no difference to the decomposition at any sample size and is
slightly worse for the trend. On recovering the variances it is clearly worse:
at N = 500 the root-mean-square error of the log variance ratio goes from 0.37
to 0.59 for the trend, and at N = 60 from 6.06 to 7.42, and the spread of the
estimates is larger too. What it does reliably is put a seasonal's drift
variance at the bottom of its bracket, 1e-9, when the pattern is fixed, where
plain maximum likelihood sometimes leaves it at 1e-4 or higher.

---

# Performance

Measured on an Apple M4 Pro with Dart 3.13.1, AOT-compiled, on an otherwise
lightly loaded machine. Absolute times move by a fifth or so between machines
and SDK releases; ratios move much less.

`benchmark/scaling_benchmark.dart`, median of five samples. `smooth` is filter
plus smoother plus the reported posterior, for a `LocalLinearTrend`.

| N | `logLikelihood` | `smooth` | per observation | `O(N²)` kernel smoother |
|---|---|---|---|---|
| 100 | 0.00 ms | 0.02 ms | 173 ns | 0.04 ms |
| 1 000 | 0.03 ms | 0.17 ms | 167 ns | 3.7 ms |
| 10 000 | 0.28 ms | 1.75 ms | 175 ns | 371 ms |
| 100 000 | 2.86 ms | 18.7 ms | 187 ns | — |

The cost per observation stays flat across three orders of magnitude. Ten years
of daily readings smooth in under a millisecond.

The `logLikelihood` column is where the two-state fast path shows up, a little
over three times faster than the generic engine on the same forward pass:

| N | generic engine | fast path | |
|---|---|---|---|
| 1 000 | 0.10 ms | 0.03 ms | 3.4x |
| 10 000 | 0.96 ms | 0.28 ms | 3.5x |
| 100 000 | 9.5 ms | 2.9 ms | 3.3x |

## Models with many components

`benchmark/components_benchmark.dart`. The backward pass is cubic in the state
dimension, so what it costs depends on how many states actually move. Smoothing
20 000 daily points:

| model | states | `smooth` |
|---|---|---|
| `LocalLevel` | 1 | 3.4 ms |
| `LocalLinearTrend` | 2 | 4.2 ms |
| trend + Matérn 5/2 | 5 | 21 ms |
| trend + weekly seasonal | 6 | 40 ms |
| trend + 1 indicator | 3 | 7.5 ms |
| trend + 20 indicators | 22 | 690 ms |

The indicators are cheaper than their state count suggests because a
regression coefficient never moves and is skipped by the backward pass; see
[How it works](https://github.com/QuantumPhysique/state_space/blob/main/doc/how-it-works.md#the-smoother).
They are still flat directions in the forward pass: one likelihood evaluation of
a trend and a weekly seasonal on 730 daily readings takes 0.41 ms, 0.53 ms with
one indicator, and 20 ms with twenty.

## What a fit costs

`fit` runs one forward pass per likelihood evaluation, including the scan and
the plateau probes, and the backward pass not at all. The same benchmark, on
daily readings:

| model | parameters | passes | 1 year | 3 years | 5 years |
|---|---|---|---|---|---|
| trend | 1 | 73–77 | 1 ms | 2 ms | 3 ms |
| trend + weekly | 2 | 214–221 | 43 ms | 136 ms | 225 ms |
| trend + weekly + Matérn | 4 | 526–706 | 132 ms | 494 ms | 887 ms |

A warm refit after one more reading, `SearchStart.previousParameters`, on two
years of trend + weekly + Matérn: 232 passes and 118 ms against 573 passes and
289 ms cold, with the same likelihood to 1e-4.

## Why the linear algebra is hand-rolled

`A` and `Q` are block diagonal, which a general matrix type has no way to express
and would multiply the zeros of, and the Dart candidate offers no in-place or
out-parameter path: a pass would allocate a result object per operation, some
seventy thousand short-lived ones for a decade of daily data. On raw dense
products it is the faster of the two from 6×6 upward;
`benchmark/dependency_comparison.dart` has those numbers.
