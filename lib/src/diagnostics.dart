import 'dart:typed_data';

import 'stats/chi_square.dart';

/// Outcome of a Ljung-Box test for autocorrelation in the residuals.
typedef LjungBoxResult = ({
  double statistic,
  int lags,
  int degreesOfFreedom,
  double pValue,
});

/// What the one-step-ahead prediction errors say about the model that
/// produced them.
///
/// A structural model asserts that everything systematic in the series has
/// been absorbed by its components, leaving independent Gaussian noise. That
/// assertion is testable, and this is the object that tests it: the residuals
/// should have mean zero, unit variance, and no autocorrelation at any lag. A
/// weekly pattern left out of the model shows up here as a spike in the
/// autocorrelation at seven days' worth of observations, long before it shows
/// up as a visibly bad fit.
///
/// The residuals are the recursive ones described in `recursive_residuals.dart`
/// — each conditioned only on what came before it — so they are genuinely
/// independent under the model rather than approximately so, and there are
/// `N - d` of them for `d` flat directions, exactly as many as the likelihood
/// charges for.
class InnovationDiagnostics {
  /// Built by `StructuralModel.diagnose`.
  InnovationDiagnostics({
    required this.times,
    required this.residuals,
  }) {
    if (times.length != residuals.length) {
      throw ArgumentError('times and residuals must have the same length');
    }
  }

  /// Time of each residual, ascending. The first few observations are absent:
  /// they are spent locating the flat directions.
  final Float64List times;

  /// Standardised one-step-ahead prediction errors. Standard normal under a
  /// model that is telling the truth.
  final Float64List residuals;

  /// How many residuals there are.
  int get count => residuals.length;

  /// Their sample mean, which should be zero to within `1/sqrt(count)`.
  ///
  /// A mean that is reliably away from zero means the model is biased: the
  /// series is systematically above or below what it predicts one step ahead.
  double get mean {
    var total = 0.0;
    for (final r in residuals) {
      total += r;
    }
    return total / count;
  }

  /// Their sample variance, which should be one to within
  /// `sqrt(2 / count)`.
  ///
  /// Reliably above one means the model is more confident than it has earned
  /// — the measurement variance, or a process variance, is too small. Below
  /// one means the opposite.
  ///
  /// **This says nothing at all about a model that has just been fitted.**
  /// Fitting concentrates the measurement variance out, which is to say it
  /// chooses the noise level that makes this number one, so it will be one
  /// whether or not the model is any good. It is informative for a model whose
  /// variances were set by hand, and for a fitted model the thing to look at
  /// is the fitted noise level itself: a model missing a component has to
  /// explain that component as noise, and reports a scale far noisier than it
  /// is.
  double get variance {
    final centre = mean;
    var total = 0.0;
    for (final r in residuals) {
      final d = r - centre;
      total += d * d;
    }
    return total / count;
  }

  /// Sample autocorrelation of the residuals at [lag].
  ///
  /// Note what a lag is here. It counts *observations*, not time: lag one is
  /// the previous reading, whenever that happened to be. That is the right
  /// notion for this test — under the model the residuals are independent
  /// however unevenly they are spaced, so any dependence between neighbours
  /// is a defect regardless of the gap between them — but it does mean a
  /// weekly pattern shows up at "seven observations" only when the sampling
  /// is roughly daily. On thinner data, look for it wherever a period's worth
  /// of readings falls.
  double autocorrelation(int lag) {
    if (lag < 1 || lag >= count) {
      throw ArgumentError.value(
          lag, 'lag', 'must be between 1 and ${count - 1}');
    }
    final centre = mean;
    var cross = 0.0;
    var total = 0.0;
    for (var t = 0; t < count; t++) {
      final d = residuals[t] - centre;
      total += d * d;
      if (t >= lag) cross += d * (residuals[t - lag] - centre);
    }
    return cross / total;
  }

  /// The Ljung-Box portmanteau test for autocorrelation up to [lags].
  ///
  /// ```text
  /// Q = n (n + 2) sum_k r_k^2 / (n - k)
  /// ```
  ///
  /// which is chi-square with `lags - fittedParameters` degrees of freedom
  /// under the hypothesis that the residuals are independent. A small p-value
  /// says there is structure the model has not accounted for; it does not say
  /// what, and the individual [autocorrelation] values are usually more
  /// informative about that than the statistic is.
  ///
  /// Pass [fittedParameters] when the variances were estimated on the same
  /// data. Each estimated parameter costs a degree of freedom, and ignoring
  /// that makes the test optimistic — it will fail to reject models it should
  /// reject.
  ///
  /// [lags] defaults to ten, the usual choice for a non-seasonal series. With
  /// a seasonal component, twice the number of observations in a period is
  /// the better one, so that the test can see a pattern at the period itself.
  LjungBoxResult ljungBox({int lags = 10, int fittedParameters = 0}) {
    if (lags < 1) {
      throw ArgumentError.value(lags, 'lags', 'must be at least 1');
    }
    if (lags >= count) {
      throw ArgumentError.value(lags, 'lags',
          'must be fewer than the $count residuals there are to test');
    }
    if (fittedParameters < 0) {
      throw ArgumentError.value(
          fittedParameters, 'fittedParameters', 'cannot be negative');
    }
    final degreesOfFreedom = lags - fittedParameters;
    if (degreesOfFreedom < 1) {
      throw ArgumentError('$lags lags cannot absorb $fittedParameters fitted '
          'parameters; ask for more lags than parameters, or the test has '
          'nothing left to measure');
    }

    final centre = mean;
    var total = 0.0;
    for (final r in residuals) {
      final d = r - centre;
      total += d * d;
    }

    var statistic = 0.0;
    for (var k = 1; k <= lags; k++) {
      var cross = 0.0;
      for (var t = k; t < count; t++) {
        cross += (residuals[t] - centre) * (residuals[t - k] - centre);
      }
      final r = cross / total;
      statistic += r * r / (count - k);
    }
    statistic *= count * (count + 2);

    return (
      statistic: statistic,
      lags: lags,
      degreesOfFreedom: degreesOfFreedom,
      pValue: chiSquareUpperTail(statistic, degreesOfFreedom),
    );
  }

  @override
  String toString() => 'InnovationDiagnostics(count: $count, mean: '
      '${mean.toStringAsFixed(4)}, variance: '
      '${variance.toStringAsFixed(4)}, lag-1 autocorrelation: '
      '${count > 1 ? autocorrelation(1).toStringAsFixed(4) : "n/a"})';
}
