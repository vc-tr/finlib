#pragma once

#include "qe/core/types.hpp"

namespace qe {

// Implied volatility solver using Newton-Raphson with BS vega
// Given a market price, finds sigma such that BS(sigma) = market_price
struct ImpliedVolResult {
    Real vol;
    Size iterations;
    bool converged;
};

ImpliedVolResult implied_vol(Real market_price, Real S, Real K, Real r, Real T,
                             OptionType type, Real tol = 1e-8, Size max_iter = 100);

} // namespace qe
