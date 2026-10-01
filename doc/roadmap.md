# Roadmap and references

## What might come next

**The rigid periodic kernel.** The same rotation blocks `TrigonometricSeasonal`
builds, with no process noise and a Bessel stationary prior. It differs from a
seasonal with its variance at zero in the prior on the starting coefficients,
which is stationary rather than diffuse; that is what the annual-on-180-days
example in
[Choosing a model](https://github.com/QuantumPhysique/state_space/blob/main/doc/choosing-a-model.md#what-competes-with-what)
turns on.

**A robust fit.** Down-weighting readings with large standardised residuals
through `relativeVariance`, iterated, as a built-in option. Today
`FitResult.largestResidual` names the worst reading and the caller decides.

## Explicitly not planned

Calendar-monthly seasonality, EKF/UKF, particle filters, multivariate
observations.

The squared exponential kernel: it has no exact finite-state form. See
[Components](https://github.com/QuantumPhysique/state_space/blob/main/doc/components.md#supported-kernels).

## References

* Kalman (1960), *J. Basic Eng.* 82(1): 35–45
* Rauch, Tung & Striebel (1965), *AIAA J.* 3(8): 1445–1450
* Wahba (1978), *JRSS-B* 40(3): 364–372: the spline/GP equivalence
* Silverman (1984), *Ann. Statist.* 12(3): 898–916: the spline's equivalent
  kernel, used by the calibration tool to turn a variance ratio into a
  bandwidth in days
* Gardner & McKenzie (1985), *Management Science* 31(10): 1237–1246: the
  damped trend
* Harvey (1989), *Forecasting, Structural Time Series Models and the Kalman
  Filter*, CUP: profile likelihood, ch. 3–4
* Taylor, Cumberland & Sy (1994), *JASA* 89(427): 727–736: the integrated
  Ornstein–Uhlenbeck process
* Hartikainen & Särkkä (2010), *IEEE MLSP*: 379–384: GP to state space
* Durbin & Koopman (2012), *Time Series Analysis by State Space Methods*, 2nd
  ed.: exact diffuse initialisation, ch. 5
* Solin & Särkkä (2014), *AISTATS*: 904–912: periodic covariance functions as
  state-space models
* Särkkä & Solin (2019), *Applied Stochastic Differential Equations*, CUP
* Simpson et al. (2017), *Statist. Sci.* 32(1): 1–28: penalised complexity
  priors, which `ComplexityPenalty` implements
