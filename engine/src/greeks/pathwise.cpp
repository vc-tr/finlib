#include "qe/greeks/pathwise.hpp"
#include <cmath>

namespace qe {

IPAGreeks mc_ipa_greeks(const MCEngine::Config& config, Real strike,
                        OptionType type) {
    MersenneTwister rng(config.seed);
    Real discount = std::exp(-config.rate * config.maturity);
    Real sqrt_T = std::sqrt(config.maturity);

    RunningStats delta_stats, vega_stats, rho_stats;

    for (Size i = 0; i < config.num_paths; ++i) {
        Real z = rng.normal();

        // Terminal spot
        Real drift = (config.rate - 0.5 * config.sigma * config.sigma) * config.maturity;
        Real ST = config.spot * std::exp(drift + config.sigma * sqrt_T * z);

        // Check if option is ITM (pathwise only works for smooth payoffs)
        bool itm;
        Real sign;
        if (type == OptionType::Call) {
            itm = ST > strike;
            sign = 1.0;
        } else {
            itm = ST < strike;
            sign = -1.0;
        }

        if (!itm) {
            delta_stats.push(0.0);
            vega_stats.push(0.0);
            rho_stats.push(0.0);
            continue;
        }

        // Pathwise derivatives of payoff * indicator
        // d(payoff)/dS0 = sign * dST/dS0 = sign * ST/S0
        Real dST_dS = ST / config.spot;
        Real pw_delta = discount * sign * dST_dS;

        // d(payoff)/dsigma = sign * dST/dsigma
        // dST/dsigma = ST * (-sigma*T + sqrt(T)*z)
        Real dST_dsigma = ST * (-config.sigma * config.maturity + sqrt_T * z);
        Real pw_vega = discount * sign * dST_dsigma;

        // d(payoff)/dr includes both discount and drift sensitivity
        // = -T * discount * payoff + discount * sign * ST * T
        Real payoff_val = sign * (ST - strike);
        Real pw_rho = -config.maturity * discount * payoff_val
                    + discount * sign * ST * config.maturity;

        delta_stats.push(pw_delta);
        vega_stats.push(pw_vega);
        rho_stats.push(pw_rho);
    }

    return {delta_stats.mean(), vega_stats.mean(), rho_stats.mean()};
}

} // namespace qe
