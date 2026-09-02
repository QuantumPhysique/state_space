import 'dart:math' as math;

import '../component.dart';

/// A term added to the log-likelihood before it is maximised.
///
/// With one variance to estimate, plain maximum likelihood is well behaved and
/// there is nothing here worth having. With three or four it can fail
/// differently: on a short history the data barely distinguishes a wandering
/// trend with a rigid seasonal from a rigid trend with a wandering seasonal,
/// and the likelihood has a long flat ridge joining the two. Left alone, the
/// optimiser picks a point on that ridge more or less arbitrarily, and the
/// point it picks moves when one more observation arrives.
sealed class Penalty {
  const Penalty();

  /// The penalty at [components] for a series spanning [span] time units.
  ///
  /// [spreadUnit] converts a component's [Component.wanderOver] — which the
  /// profiling machinery reports in units of the measurement standard
  /// deviation — into units of the observed standard deviation of the series.
  /// It is the fitted measurement standard deviation divided by the sample
  /// standard deviation of the data.
  ///
  /// The conversion is what makes the whole thing scale-free. A component's
  /// wander in noise units depends on the noise level; the observed spread
  /// depends on it in the same way; the ratio does not, so the penalty
  /// survives concentrating the measurement variance out and needs no
  /// separate estimate of the noise.
  double at(List<Component> components, double span, double spreadUnit);
}

/// Plain maximum likelihood.
final class NoPenalty extends Penalty {
  const NoPenalty();

  @override
  double at(List<Component> components, double span, double spreadUnit) => 0;

  @override
  String toString() => 'NoPenalty()';
}

/// A penalised-complexity penalty on how much of the data's own spread each
/// component may claim.
///
/// Simpson et al. (2017) measure the complexity of a component by the
/// Kullback-Leibler divergence from the simpler model it collapses to — here,
/// the component being absent. For a Gaussian component that distance is
/// proportional to its standard deviation, and an exponential prior on the
/// distance gives a penalty linear in it:
///
/// ```text
/// penalty = -rate * sum_i wander_i,   rate = -log(tailProbability) / scale
/// ```
///
/// Wandering is measured in fractions of the observed standard deviation of
/// the series, which is what makes a trend and a seasonal comparable when
/// their raw variances are not. The defaults then say: a component that on its
/// own accounts for the entire spread of the data would be a surprise at the
/// one per cent level. That is a mild statement — a single-component model is
/// *supposed* to account for the whole spread, and lands near a penalty of
/// four and a half nats — and it becomes a real one only when several
/// components each want to claim the same variation.
///
/// **It is off by default, and the measurement is why.** Simulating a trend
/// plus a weekly seasonal at known variances, fitting both ways, and comparing
/// the fitted decomposition against the paths that generated it, over twelve
/// replications:
///
/// ```text
///            seasonal RMSE              trend RMSE
///          penalised    plain       penalised    plain
/// N =  60     0.0867   0.0870          0.0556   0.0545
/// N = 120     0.0824   0.0826          0.0483   0.0464
/// N = 500     0.0759   0.0759          0.0487   0.0480
/// ```
///
/// The penalty makes no difference to the decomposition at any sample size,
/// which is the job it was brought in to do, and it is slightly worse for the
/// trend. On recovering the variances themselves it is clearly worse: at
/// N = 500 the root-mean-square error of the log variance ratio goes from 0.37
/// to 0.59 for the trend, and at N = 60 from 6.06 to 7.42. Nor does it buy the
/// usual consolation of shrinkage, a reduction in variance — the spread of the
/// estimates is larger too.
///
/// What it does do, reliably, is drive a component's drift parameter to the
/// floor when there is no drift to find: on a series whose weekly pattern is
/// fixed rather than evolving, the penalised fit puts the seasonal variance
/// ratio at 1e-9 where plain maximum likelihood sometimes leaves it at 1e-4 or
/// higher. If what you want is a decision about whether a component is moving,
/// rather than the best estimate of how fast, that is worth having. Note that
/// it shrinks the *drift* and not the component: the seasonal's starting shape
/// is set by a flat prior that nothing here penalises, so a rigid pattern
/// survives at full amplitude.
///
/// This is a penalty, not a prior, and the difference is deliberate. Written
/// as a prior in the log-variance parameterisation the search actually uses,
/// there would be a Jacobian term running to minus infinity as a component
/// shrinks to nothing — forbidding the very outcome the penalty exists to
/// allow. Dropping it means the objective is not a posterior density and the
/// answer is not a MAP estimate. It is a penalised maximum likelihood
/// estimate, and it can shrink a component all the way out of the model,
/// which is what shrinkage towards a simpler model has to mean.
final class ComplexityPenalty extends Penalty {
  /// Validates rather than asserts, for the reason the components give: a rate
  /// built from a bad scale produces a plausible-looking fit rather than an
  /// obvious failure, and that is not a thing to ship to release builds.
  ComplexityPenalty({this.scale = 1, this.tailProbability = 0.01}) {
    if (!(scale > 0) || !scale.isFinite) {
      throw ArgumentError.value(scale, 'scale', 'must be finite and positive');
    }
    if (!(tailProbability > 0) || !(tailProbability < 1)) {
      throw ArgumentError.value(tailProbability, 'tailProbability',
          'must lie strictly between 0 and 1');
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
  String toString() => 'ComplexityPenalty(scale: $scale, tailProbability: '
      '$tailProbability)';
}
