import 'dart:math' as math;

import '../arguments.dart';

/// A normal prior on the natural logarithm of a positive parameter, for
/// [fit].
///
/// [median] is where it is centred and [logSd] the standard deviation of the
/// log, so a [logSd] of `ln 2` makes a factor of two either way one standard
/// deviation.
///
/// On a variance it is a prior on the variance itself. One on a standard
/// deviation, centred on `m` with a log standard deviation `s`, is
/// `LogNormalPrior(median: m * m, logSd: 2 * s)` on the variance.
///
/// {@category Choosing a model}
final class LogNormalPrior {
  /// A prior centred on [median], with [logSd] the standard deviation of the
  /// natural log. Both must be finite and positive.
  LogNormalPrior({required this.median, required this.logSd}) {
    checkPositive(median, 'median');
    checkPositive(logSd, 'logSd');
  }

  /// The value the prior is centred on, in the parameter's own units.
  final double median;

  /// The standard deviation of the natural log of the value.
  final double logSd;

  /// The log density of the prior at a value whose natural log is
  /// [logValue], as a density over that log.
  double logDensity(double logValue) {
    final z = (logValue - math.log(median)) / logSd;
    return -0.5 * z * z - math.log(logSd * math.sqrt(2 * math.pi));
  }

  @override
  bool operator ==(Object other) =>
      other is LogNormalPrior && other.median == median && other.logSd == logSd;

  @override
  int get hashCode => Object.hash(LogNormalPrior, median, logSd);

  @override
  String toString() => 'LogNormalPrior(median: $median, logSd: $logSd)';
}
