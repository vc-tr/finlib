#pragma once

#include "qe/core/types.hpp"
#include "qe/math/random.hpp"
#include "qe/math/statistics.hpp"
#include "qe/montecarlo/engine.hpp"

namespace qe {

// Pathwise (IPA) Greeks for European options under GBM
// Differentiates through the payoff: d/dtheta E[f(S(T))] = E[f'(S(T)) * dS/dtheta]
// Works for smooth payoffs (vanilla calls/puts) but NOT for digital options
struct IPAGreeks {
    Real delta;
    Real vega;
    Real rho;
};

IPAGreeks mc_ipa_greeks(const MCEngine::Config& config, Real strike,
                        OptionType type);

} // namespace qe
