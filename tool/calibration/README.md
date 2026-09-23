# Calibration

Does the right answer help?

The test suite pins this package to dense Gaussian process references to nine
or ten digits. That says it computes the right thing. It says nothing about
whether the right thing is any use on a real weight diary — whether the fitted
smoothing is stable enough to show someone, whether a model needs a weekly
component, whether the noise level it reports is believable. This directory
answers those with numbers.

## Running it

```sh
dart run tool/calibration/calibrate.dart --all           # synthetic diaries
dart run tool/calibration/calibrate.dart --all export.txt
```

With no file it builds four synthetic diaries. Pass one or more files and it
uses those instead: comment lines starting with `#`, then an ISO 8601
timestamp, a space and a weight in kilograms per line. Commas and semicolons
work as separators too, so an ordinary two-column CSV needs no conversion.

Nothing leaves the machine. A file passed on the command line is read, turned
into numbers, and the numbers are printed.

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
four, and are the way to choose. [Choosing a model](https://github.com/QuantumPhysique/state_space/blob/main/doc/choosing-a-model.md#how-to-tell-whether-it-earned-its-place--and-how-not-to)
explains why.

**The stability table** refits at 14, 21, 30, 45, 60, 90, 120 and 186 days, and
asks the question a user would: does the curve keep its character as the diary
grows? The bracketed figure is how many decades the trend's own variance can
move before the fit is half a nat worse, so it is a width and not a value: under
half a decade is a real estimate, and over two — starred — means the printed
bandwidth is a convention.

Bandwidth is in days, via Silverman's equivalent kernel: a smoothing spline with
parameter `lambda` on data sampled `f` times per unit time behaves like a kernel
smoother of bandwidth `(lambda / f)^(1/4)`. With daily readings and `lambda` the
reciprocal of the fitted variance ratio, that is the ratio to the power of minus
a quarter. The exponent is why a factor of sixteen in the variance is only a
factor of two on the chart, and why a setting expressed in days is a much better
thing to hand a user than one expressed in variance.

**The floor table** shows where asserting the scale's own precision changes the
answer. A floor that does not bind costs nothing; one that binds costs one more
fit. Where it binds it tends to change everything, because a model that has
driven the measurement noise to a thousandth of a kilogram is fitting the
readings rather than the weight.

## The synthetic diaries

Four courses — two losing at different scales, one holding steady, one gaining —
with hydration that persists across days rather than scattering, a weekday
pattern, reading error, rounding to the tenth of a kilogram the display shows,
missed mornings that clump into lapses, and a trip with a gap and a rebound.
Four rather than one because the interesting question is whether a fit behaves
the same way for all of them, and one series cannot answer that.

They are a stand-in and not the point. Every constant in `synthetic.dart` says
where it came from, and several say they are guesses. A real export settles
things that a generator written by the same person who wrote the model never
can — most of all whether a weekly pattern is there at all, which on this data
is the single thing that decides whether the fitted smoothing is stable.
