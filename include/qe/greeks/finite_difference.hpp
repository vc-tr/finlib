#pragma once

#include "qe/core/types.hpp"
#include "qe/montecarlo/engine.hpp"
#include "qe/instruments/payoff.hpp"

namespace qe {

// Bump-and-reprice Greeks via central finite differences
// Works for any model/payoff but requires multiple MC runs
struct FDGreeks {
    Real delta;      // dV/dS
    Real gamma;      // d2V/dS2
    Real vega;       // dV/dsigma
    Real theta;      // dV/dT (per year)
    Real rho;        // dV/dr
};

FDGreeks mc_fd_greeks(const MCEngine::Config& base_config,
                      const Payoff& payoff,
                      Real bump_spot = 1.0,
                      Real bump_sigma = 0.01,
                      Real bump_time = 1.0 / 252.0,
                      Real bump_rate = 0.001);

} // namespace qe
