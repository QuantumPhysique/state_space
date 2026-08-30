"""Generate golden fixtures for the Dart filter from statsmodels.

Run with:

    uv run --with numpy --with statsmodels tool/generate_fixtures.py

The fixtures and this script are both committed. Regenerate them only when the
model changes, and say so in the changelog when you do.

Two things are worth knowing before reading this.

First, statsmodels' `UnobservedComponents(level='local linear trend')` is *not*
the model this package implements. That one is the discrete local linear trend,
with a diagonal state covariance and an implicit unit time step. Ours is the
continuous-time model, whose state covariance has the off-diagonal term that
couples slope uncertainty into level uncertainty over a gap. Comparing against
it would be comparing against a different model, so we build the model
explicitly with time-varying system matrices instead.

Second, statsmodels writes the recursion as

    y_t       = Z_t alpha_t + eps_t
    alpha_t+1 = T_t alpha_t + R_t eta_t

so `transition[:, :, t]` is the step *out of* t, not the step into it. The gap
stored at index t is therefore times[t+1] - times[t]. Getting this backwards
produces a filter that looks almost right and is wrong by one step everywhere.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import statsmodels.api as sm

OUT = pathlib.Path(__file__).resolve().parent.parent / "test" / "fixtures"


class ContinuousLocalLinearTrend(sm.tsa.statespace.MLEModel):
    """The exact discretisation of an integrated Wiener process.

    A(dt) = [[1, dt], [0, 1]]
    Q(dt) = sigma2 * [[dt^3/3, dt^2/2], [dt^2/2, dt]]
    """

    def __init__(self, values, times, sigma2, sigma_eps2, weights, kappa):
        super().__init__(values, k_states=2, k_posdef=2)
        n = len(values)
        gaps = np.empty(n)
        gaps[:-1] = np.diff(times)
        gaps[-1] = 0.0  # never used to filter; only the final prediction

        self.ssm["design"] = np.array([[1.0, 0.0]])
        self.ssm["selection"] = np.eye(2)
        self.ssm["transition"] = np.zeros((2, 2, n))
        self.ssm["state_cov"] = np.zeros((2, 2, n))
        self.ssm["obs_cov"] = np.zeros((1, 1, n))

        for t, dt in enumerate(gaps):
            self.ssm["transition", :, :, t] = [[1.0, dt], [0.0, 1.0]]
            self.ssm["state_cov", :, :, t] = sigma2 * np.array(
                [[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]]
            )
        for t, w in enumerate(weights):
            self.ssm["obs_cov", :, :, t] = sigma_eps2 * w

        # The Dart side scales its diffuse prior by the measurement variance so
        # that the whole model is scale-equivariant. Match that here.
        self.ssm.initialize_approximate_diffuse(kappa * sigma_eps2)

    @property
    def start_params(self):
        return np.array([])


def fixture(name, times, values, weights, sigma2, sigma_eps2, kappa=1e6):
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    model = ContinuousLocalLinearTrend(values, times, sigma2, sigma_eps2, weights, kappa)
    res = model.smooth([])

    payload = {
        "name": name,
        "description": DESCRIPTIONS[name],
        "generator": "statsmodels " + sm.__version__,
        "processVariance": sigma2,
        "measurementVariance": sigma_eps2,
        "diffuseVariance": kappa,
        "times": times.tolist(),
        "values": [None if np.isnan(v) else float(v) for v in values],
        "relativeVariances": weights.tolist(),
        # statsmodels stores states column-wise; flatten row-major per step so
        # the Dart side can read them with the same indexing it uses inside.
        "predictedState": res.predicted_state[:, : len(times)].T.ravel().tolist(),
        "predictedStateCov": np.moveaxis(
            res.predicted_state_cov[:, :, : len(times)], 2, 0
        ).ravel().tolist(),
        "filteredState": res.filtered_state.T.ravel().tolist(),
        "filteredStateCov": np.moveaxis(res.filtered_state_cov, 2, 0).ravel().tolist(),
        "smoothedState": res.smoothed_state.T.ravel().tolist(),
        "smoothedStateCov": np.moveaxis(res.smoothed_state_cov, 2, 0).ravel().tolist(),
        "loglikelihoodObs": [float(v) for v in res.llf_obs],
    }

    path = OUT / f"{name}.json"
    path.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {path.relative_to(OUT.parent.parent)} ({len(times)} steps)")


DESCRIPTIONS = {
    "regular": "40 unit-spaced observations, nothing missing",
    "irregular": "60 observations with gaps from 0 to 40 time units, including "
    "a repeated timestamp",
    "missing": "50 unit-spaced steps with a block of missing observations and "
    "per-observation noise weights",
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260830)

    # 1. The easy case: regular sampling, every point observed.
    times = np.arange(40.0)
    signal = 80 + 0.03 * times - 2 * np.sin(times / 9)
    fixture(
        "regular",
        times,
        signal + rng.normal(0, 0.25, times.size),
        np.ones(times.size),
        sigma2=2e-3,
        sigma_eps2=0.0625,
    )

    # 2. Irregular gaps, including a same-day repeat (dt = 0) and a long gap.
    steps = rng.gamma(shape=2.0, scale=1.5, size=59)
    steps[10] = 0.0
    steps[30] = 40.0
    times = np.concatenate([[0.0], np.cumsum(steps)])
    signal = 80 + 0.01 * times - 3 * np.cos(times / 30)
    fixture(
        "irregular",
        times,
        signal + rng.normal(0, 0.3, times.size),
        np.ones(times.size),
        sigma2=5e-4,
        sigma_eps2=0.09,
    )

    # 3. Missing observations and unequal measurement weights.
    times = np.arange(50.0)
    signal = 80 + 0.05 * times
    values = signal + rng.normal(0, 0.2, times.size)
    values[20:28] = np.nan
    values[41] = np.nan
    weights = np.where(np.arange(50) % 7 == 0, 4.0, 1.0)
    fixture(
        "missing",
        times,
        values,
        weights,
        sigma2=1e-3,
        sigma_eps2=0.04,
    )


if __name__ == "__main__":
    main()
