"""A standalone reproduction of statsmodels' exact diffuse smoother
disagreeing with a dense generalised-least-squares computation.

    uv run --with numpy --with statsmodels tool/statsmodels_diffuse_repro.py

The model is a continuous-time local linear trend observed at the times given,
with the exact discretisation over each gap:

    A(dt) = [[1, dt], [0, 1]],   Q(dt) = s2 * [[dt^3/3, dt^2/2], [dt^2/2, dt]]

and a flat (exact diffuse) prior on the first state. Under that prior the
smoothed first state is the GLS estimate of x0 from y = B x0 + u, where
B_i = [1, t_i - t_0] and u has the covariance of the integrated Wiener process
plus the measurement noise. That is computed densely below and compared with
statsmodels' `initialize_diffuse()` smoother.

For evenly spaced times the two agree to rounding. For unevenly spaced times the
smoothed slope at the first step disagrees.
"""

from __future__ import annotations

import numpy as np
import statsmodels.api as sm


class Trend(sm.tsa.statespace.MLEModel):
    def __init__(self, values, times, s2, noise):
        super().__init__(values, k_states=2, k_posdef=2)
        n = len(values)
        gaps = np.append(np.diff(times), 0.0)
        self.ssm["design"] = np.array([[1.0, 0.0]])
        self.ssm["selection"] = np.eye(2)
        self.ssm["obs_cov"] = np.array([[noise]])
        self.ssm["transition"] = np.zeros((2, 2, n))
        self.ssm["state_cov"] = np.zeros((2, 2, n))
        for t, dt in enumerate(gaps):
            # statsmodels' transition[:, :, t] is the step out of t.
            self.ssm["transition", :, :, t] = [[1.0, dt], [0.0, 1.0]]
            self.ssm["state_cov", :, :, t] = s2 * np.array(
                [[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]]
            )
        self.ssm.initialize_diffuse()

    @property
    def start_params(self):
        return np.array([])


def dense_first_state(values, times, s2, noise):
    tau = times - times[0]
    m = np.minimum.outer(tau, tau)
    lag = np.abs(np.subtract.outer(tau, tau))
    cov = s2 * (m**3 / 3 + m**2 * lag / 2) + noise * np.eye(len(tau))
    design = np.column_stack([np.ones_like(tau), tau])
    weighted = np.linalg.solve(cov, design)
    information = design.T @ weighted
    estimate = np.linalg.solve(information, weighted.T @ values)
    return estimate, np.linalg.inv(information)


def compare(label, times, seed=0):
    rng = np.random.default_rng(seed)
    times = np.asarray(times, dtype=float)
    values = 80 + 0.05 * (times - times[0]) + 0.3 * rng.standard_normal(len(times))
    s2, noise = 1e-3, 0.09
    result = Trend(values, times, s2, noise).smooth([])
    dense, dense_cov = dense_first_state(values, times, s2, noise)
    print(f"{label:<28} level  statsmodels {result.smoothed_state[0, 0]: .12f}"
          f"  dense {dense[0]: .12f}")
    print(f"{'':<28} slope  statsmodels {result.smoothed_state[1, 0]: .12f}"
          f"  dense {dense[1]: .12f}"
          f"  difference {result.smoothed_state[1, 0] - dense[1]: .2e}")
    print(f"{'':<28} var(slope) statsmodels {result.smoothed_state_cov[1, 1, 0]:.6e}"
          f"  dense {dense_cov[1, 1]:.6e}")


if __name__ == "__main__":
    print("statsmodels", sm.__version__, "numpy", np.__version__)
    compare("unit steps", np.arange(20))
    compare("steps of 2.5", 2.5 * np.arange(20))
    compare("steps of 0.5", 0.5 * np.arange(20))
    compare("steps 1, 2, 1, 2, ...", np.cumsum([0] + [1 + (i % 2) for i in range(19)]))
    compare("irregular steps", np.cumsum([0, 1, 3, 1, 1, 7, 2, 1, 1, 4, 1, 2, 1, 1, 5, 1, 1, 2, 1, 3]))
