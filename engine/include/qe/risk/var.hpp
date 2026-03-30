#pragma once

#include "qe/core/types.hpp"
#include <span>

namespace qe {

// Value-at-Risk: the loss threshold at a given confidence level
// VaR(alpha) = -quantile(returns, 1 - alpha)
// e.g., VaR(95%) is the 5th percentile loss

// Historical VaR: directly from sorted P&L data
Real historical_var(std::span<const Real> pnl, Real confidence = 0.95);

// Parametric (Gaussian) VaR: assumes normal distribution
// VaR = -mu + z_alpha * sigma
Real parametric_var(Real mean, Real std_dev, Real confidence = 0.95);

// From portfolio returns
Real parametric_var(std::span<const Real> returns, Real confidence = 0.95);

// Conditional VaR (Expected Shortfall / TVaR)
// CVaR(alpha) = E[L | L >= VaR(alpha)]
// Subadditive (unlike VaR), making it a coherent risk measure
Real historical_cvar(std::span<const Real> pnl, Real confidence = 0.95);

// Parametric CVaR assuming normality
Real parametric_cvar(Real mean, Real std_dev, Real confidence = 0.95);

} // namespace qe
