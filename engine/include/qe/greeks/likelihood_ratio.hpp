#pragma once

#include "qe/core/types.hpp"
#include "qe/math/random.hpp"
#include "qe/math/statistics.hpp"
#include "qe/instruments/payoff.hpp"
#include "qe/montecarlo/engine.hpp"

namespace qe {

// Likelihood ratio method (score function) for MC Greeks
// d/dtheta E[f(X)] = E[f(X) * d/dtheta log p(X; theta)]
// Works for ALL payoffs including discontinuous ones (digital)
struct LRGreeks {
    Real delta;
    Real vega;
};

LRGreeks mc_lr_greeks(const MCEngine::Config& config, const Payoff& payoff);

} // namespace qe
