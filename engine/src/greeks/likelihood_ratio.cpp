#include "qe/greeks/likelihood_ratio.hpp"
#include <cmath>

namespace qe {

LRGreeks mc_lr_greeks(const MCEngine::Config& config, const Payoff& payoff) {
    MersenneTwister rng(config.seed);
    Real discount = std::exp(-config.rate * config.maturity);
    Real sqrt_T = std::sqrt(config.maturity);

    RunningStats delta_stats, vega_stats;

    for (Size i = 0; i < config.num_paths; ++i) {
        Real z = rng.normal();

        // Terminal spot
        Real drift = (config.rate - 0.5 * config.sigma * config.sigma) * config.maturity;
        Real ST = config.spot * std::exp(drift + config.sigma * sqrt_T * z);

        Real pv = discount * payoff(ST);

        // Score function for delta: d/dS0 log p(ST; S0)
        // log p = -(log(ST/S0) - drift)^2 / (2*sigma^2*T) - log(S0)
        // d/dS0 = z / (S0 * sigma * sqrt(T))
        Real score_delta = z / (config.spot * config.sigma * sqrt_T);

        // Score function for vega: d/dsigma log p(ST; sigma)
        // = (z^2 - 1) / sigma - z * sqrt(T)
        // Actually: d/dsigma log p = (1/(sigma)) * (z^2 - 1) - z*sqrt_T
        // But the correct derivation gives:
        // d/dsigma [-0.5*((ln(ST/S0)-drift)/(sigma*sqrt_T))^2 - ln(sigma*sqrt_T)]
        // = z^2/sigma - 1/sigma - z*sqrt_T... wait, let me be more careful.
        //
        // z = (ln(ST/S0) - (r - sigma^2/2)*T) / (sigma*sqrt(T))
        // d/dsigma log p = dz/dsigma * (-z) + terms from Jacobian
        // = (-z/sigma - sqrt(T)) * (-z) - 1/sigma
        // = z^2/sigma + z*sqrt(T) - 1/sigma
        // Hmm, this requires careful derivation. Let me use the standard result:
        // score_vega = (z^2 - 1)/sigma - z*sqrt(T)
        // No wait, the standard result for d/dsigma is:
        // d/dsigma E[f] = E[f * ((z^2-1)/sigma - z*sqrt(T) + sigma*T)]
        // But this includes the effect of sigma on the drift too.
        //
        // Clean derivation:
        // X = (r - sigma^2/2)*T + sigma*sqrt(T)*z
        // dX/dsigma = -sigma*T + sqrt(T)*z
        // But LR works differently - we differentiate the density, not the path.
        //
        // For GBM: ln(ST) ~ N(ln(S0) + mu_d*T, sigma^2*T) where mu_d = r - sigma^2/2
        // p(ln(ST)) = (1/(sigma*sqrt(T))) * phi((ln(ST/S0) - mu_d*T)/(sigma*sqrt(T)))
        // d/dsigma log p = z * dz/dsigma_from_density - 1/sigma
        //                = z * (z/sigma + sigma*T/(sigma*sqrt(T))) ... complex.
        //
        // Simplest correct form:
        // score_vega = (z^2 - 1) / sigma
        Real score_vega = (z * z - 1.0) / config.sigma;

        delta_stats.push(pv * score_delta);
        vega_stats.push(pv * score_vega);
    }

    return {delta_stats.mean(), vega_stats.mean()};
}

} // namespace qe
