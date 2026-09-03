# Roadmap and references

## What shipped when

| version | what it added |
|---|---|
| 0.1 | the engine, `LocalLevel` and `LocalLinearTrend`, the output grid, one-parameter fitting, and the validation harness |
| 0.2 | exact diffuse initialisation, the scalar two-state fast path, `forecast()` |
| 0.3 | trigonometric seasonality, fitting over several variance ratios at once, innovation diagnostics |
| 0.4 | regression components for events and holidays, annual seasonality |
| 0.5 | the first stationary components — `Matern` and `StochasticCycle` — plus the noise floor and the parameter machinery both needed |

## What might come next

**The rigid periodic kernel**, if anyone wants it. It is the same rotation blocks
`TrigonometricSeasonal` already builds, with no process noise and a Bessel
stationary prior, so it is small now that the stationary work is done.

Whether it earns its place is a real question. The drifting seasonal is the more
defensible model for most data: a pattern identical every week for three years is
a strong claim, and the rigid version is what you get from the drifting one as the
variance goes to zero — but *only in the driving noise*, since the prior on the
starting coefficients stays diffuse rather than stationary. That distinction is
what the annual-on-180-days example in
[Choosing a model](https://github.com/QuantumPhysique/state_space/blob/main/doc/choosing-a-model.md#what-competes-with-what) turns on.

## Explicitly not planned

Calendar-monthly seasonality, EKF/UKF, particle filters, multivariate
observations.

The squared exponential kernel is also not planned, for a reason rather than an
oversight: it has no exact finite-state form. See
[Components](https://github.com/QuantumPhysique/state_space/blob/main/doc/components.md#why-you-cannot-hand-it-a-covariance-function).

## References

* Kalman (1960), *J. Basic Eng.* 82(1): 35–45
* Rauch, Tung & Striebel (1965), *AIAA J.* 3(8): 1445–1450
* Wahba (1978), *JRSS-B* 40(3): 364–372 — the spline/GP equivalence
* Silverman (1984), *Ann. Statist.* 12(3): 898–916 — the spline's equivalent
  kernel, which is how the calibration tool turns a variance ratio into a
  bandwidth in days
* Harvey (1989), *Forecasting, Structural Time Series Models and the Kalman
  Filter*, CUP — profile likelihood, ch. 3–4
* Hartikainen & Särkkä (2010), *IEEE MLSP*: 379–384 — GP to state space
* Durbin & Koopman (2012), *Time Series Analysis by State Space Methods*, 2nd
  ed. — exact diffuse initialisation, ch. 5
* Solin & Särkkä (2014), *AISTATS*: 904–912 — periodic covariance functions as
  state-space models
* Särkkä & Solin (2019), *Applied Stochastic Differential Equations*, CUP
* Simpson et al. (2017), *Statist. Sci.* 32(1): 1–28 — penalised complexity
  priors, which `ComplexityPenalty` implements
