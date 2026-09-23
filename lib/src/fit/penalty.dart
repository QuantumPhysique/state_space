import 'dart:math' as math;

import '../component.dart';

/// A term added to the log-likelihood before it is maximised.
///
/// With several variances to estimate on a short history, the likelihood can
/// have a long flat ridge between, say, a wandering trend with a rigid
/// seasonal and a rigid trend with a wandering seasonal. A penalty tilts that
/// ridge towards the simpler end.
///
/// {@category Choosing a model}
sealed class Penalty {
  const Penalty();

  /// The penalty at [components] for a series spanning [span] time units.
  ///
  /// [spreadUnit] converts a component's [Component.wanderOver], which is in
  /// units of the measurement standard deviation, into units of the observed
  /// standard deviation of the series: it is the fitted measurement standard
  /// deviation divided by the sample standard deviation of the data. The
  /// ratio does not depend on the noise level, so the penalty survives
  /// concentrating the measurement variance out.
  double at(List<Component> components, double span, double spreadUnit);
}

/// Plain maximum likelihood.
final class NoPenalty extends Penalty {
  /// No penalty; the default for [fit].
  const NoPenalty();

  @override
  double at(List<Component> components, double span, double spreadUnit) => 0;

  @override
  bool operator ==(Object other) => other is NoPenalty;

  @override
  int get hashCode => (NoPenalty).hashCode;

  @override
  String toString() => 'NoPenalty()';
}

/// A penalised-complexity penalty (Simpson et al. 2017) on how much of the
/// data's own spread each component may claim.
///
/// ```text
/// penalty = -rate * sum_i wander_i,   rate = -log(tailProbability) / scale
/// ```
///
/// where `wander_i` is [Component.wanderOver] in units of the series'
/// standard deviation. The defaults make a component that alone accounts for
/// the whole spread of the data a one-in-a-hundred surprise.
///
/// Off by default: in simulation it did not improve the recovered
/// trend/seasonal decomposition at any sample size and made the variance
/// estimates worse (see the Validation guide). What it does reliably is drive
/// a drift variance to the bottom of its bracket when there is no drift, so
/// use it when the question is whether a component moves at all. It shrinks
/// the drift, not the component: a rigid pattern keeps its amplitude.
///
/// It is a penalty rather than a prior, so the result is a penalised maximum
/// likelihood estimate, not a posterior mode, and it can shrink a variance all
/// the way to the bottom of its bracket.
final class ComplexityPenalty extends Penalty {
  /// A penalty making it a [tailProbability] surprise for one component to
  /// account for [scale] times the series' standard deviation.
  ///
  /// [scale] must be finite and positive and [tailProbability] strictly
  /// between 0 and 1.
  ComplexityPenalty({this.scale = 1, this.tailProbability = 0.01}) {
    if (!(scale > 0) || !scale.isFinite) {
      throw ArgumentError.value(scale, 'scale', 'must be finite and positive');
    }
    if (!(tailProbability > 0) || !(tailProbability < 1)) {
      throw ArgumentError.value(
        tailProbability,
        'tailProbability',
        'must lie strictly between 0 and 1',
      );
    }
  }

  /// How much of the series' own standard deviation a single component may
  /// account for before it is considered surprising. One means all of it.
  final double scale;

  /// How surprising: the probability the penalty assigns to exceeding
  /// [scale].
  final double tailProbability;

  /// The exponential rate the two parameters imply.
  double get rate => -math.log(tailProbability) / scale;

  @override
  double at(List<Component> components, double span, double spreadUnit) {
    final lambda = rate * spreadUnit;
    var total = 0.0;
    for (final component in components) {
      total -= lambda * component.wanderOver(span);
    }
    return total;
  }

  @override
  bool operator ==(Object other) =>
      other is ComplexityPenalty &&
      other.scale == scale &&
      other.tailProbability == tailProbability;

  @override
  int get hashCode => Object.hash(ComplexityPenalty, scale, tailProbability);

  @override
  String toString() =>
      'ComplexityPenalty(scale: $scale, tailProbability: '
      '$tailProbability)';
}
