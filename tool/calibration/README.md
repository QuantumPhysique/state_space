# Calibration

The test suite checks the package against dense Gaussian process references to
nine or ten digits. This tool looks at how the models behave on weight diaries:
whether the fitted smoothing is stable enough to show someone, whether a
weekly component is needed, and whether the reported noise level is
believable.

## Running it

```sh
dart run tool/calibration/calibrate.dart --all           # synthetic diaries
dart run tool/calibration/calibrate.dart --all export.txt
```

With no file it builds four synthetic diaries. Pass one or more files and it
uses those instead: comment lines starting with `#`, then an ISO 8601
timestamp, a space and a weight in kilograms per line. Commas and semicolons
work as separators too, so an ordinary two-column CSV needs no conversion.

Everything runs locally. A file passed on the command line is only read and
summarised.

| option | |
|---|---|
| `--stability` | refit at growing history lengths |
| `--floor[=SD]` | show where a noise floor of the given precision in kg would bind |
| `--all` | both |
| `--today=DATE` | pin the calendar the synthetic diaries are built against |

## What the tables say

**The model table** fits four candidates to each diary: a trend alone, and then
the same trend with a weekly seasonal, with a Matérn deviation for hydration,
and with both. The Ljung–Box degrees of freedom are 14 lags less the model's
`parameterCount`.

Read the log likelihoods only down a run of equal `diff`: the trend-only and
trend-plus-hydration models have two flat directions each and can be compared,
and neither can be compared with the ones carrying a weekly component, which have
six. The fitted noise level and the Ljung–Box p-value are comparable across all
four, and are the way to choose. [Choosing a model](https://github.com/QuantumPhysique/state_space/blob/main/doc/choosing-a-model.md#comparing-models)
explains why.

**The stability table** refits at 14, 21, 30, 45, 60, 90, 120 and 186 days, to
see whether the curve keeps its character as the diary grows. The bracketed
figure is how many decades the trend's variance can move before the fit is
half a nat worse. Under half a decade is a real estimate; over two (starred)
means the printed bandwidth is not determined by the data.

Bandwidth is in days, via Silverman's equivalent kernel: a smoothing spline with
parameter `lambda` on data sampled `f` times per unit time behaves like a kernel
smoother of bandwidth `(lambda / f)^(1/4)`. With daily readings and `lambda` the
reciprocal of the fitted variance ratio, that is the ratio to the power of minus
a quarter. So a factor of sixteen in the variance is only a factor of two on
the chart, and a user setting in days is easier to work with than one in
variance.

**The floor table** shows where a noise floor at the scale's precision changes
the answer. A floor that does not bind costs nothing; one that binds costs one
more fit. Where it binds it usually changes the result a lot, because a model
with a measurement noise of a thousandth of a kilogram is fitting the readings
rather than the weight.

## The synthetic diaries

Four courses (two losing at different rates, one steady, one gaining) with
hydration that persists across days, a weekday pattern, reading error,
rounding to the 0.1 kg the display shows, missed mornings that cluster into
lapses, and a trip with a gap and a rebound. There are four so that you can see
whether a fit behaves the same way on all of them.

Every constant in `synthetic.dart` has a comment on where it came from, and
some are guesses. A real export is the better test, above all for whether a
weekly pattern is present, which on this data decides whether the fitted
smoothing is stable.
