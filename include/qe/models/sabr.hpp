#pragma once

#include "qe/core/types.hpp"

namespace qe {

// SABR stochastic volatility model (Hagan et al., 2002)
// dF = alpha * F^beta * dW_1
// dalpha = nu * alpha * dW_2
// Corr(dW_1, dW_2) = rho
//
// The implied volatility approximation is widely used for
// interest rate options and FX options.
struct SABRParams {
    Real alpha;   // initial vol level
    Real beta;    // CEV exponent (0 = normal, 1 = lognormal)
    Real rho;     // correlation between forward and vol
    Real nu;      // vol of vol
};

// Hagan et al. (2002) implied volatility approximation
// Returns the Black implied volatility for a given forward, strike, maturity
Real sabr_implied_vol(const SABRParams& params, Real forward, Real strike, Real T);

// Price a European call using SABR implied vol + Black-Scholes
Real sabr_call(const SABRParams& params, Real forward, Real strike,
               Real rate, Real T);

// Price a European put
Real sabr_put(const SABRParams& params, Real forward, Real strike,
              Real rate, Real T);

} // namespace qe
