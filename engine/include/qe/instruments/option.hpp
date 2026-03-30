#pragma once

#include "qe/core/types.hpp"

namespace qe {

// Option contract descriptor
struct OptionSpec {
    Real spot;       // Current underlying price
    Real strike;     // Strike price
    Real rate;       // Risk-free rate (annualized)
    Real sigma;      // Volatility (annualized)
    Real maturity;   // Time to expiration (years)
    OptionType type; // Call or Put
};

} // namespace qe
