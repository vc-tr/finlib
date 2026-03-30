#pragma once

#include "qe/core/types.hpp"
#include "qe/risk/stress.hpp"
#include "qe/models/black_scholes.hpp"
#include <vector>

namespace qe {

// A single option position in the portfolio
struct OptionPosition {
    Real spot;
    Real strike;
    Real rate;
    Real sigma;
    Real maturity;
    OptionType type;
    Real quantity;   // positive = long, negative = short
};

// Portfolio risk report
struct PortfolioRisk {
    Real total_value;
    Real total_delta;
    Real total_gamma;
    Real total_vega;
    Real total_theta;
    Real total_rho;
};

// Compute aggregate portfolio risk
PortfolioRisk portfolio_risk(const std::vector<OptionPosition>& positions);

// Stress test portfolio against scenarios
std::vector<StressResult> stress_test_portfolio(
    const std::vector<OptionPosition>& positions,
    const std::vector<StressScenario>& scenarios);

} // namespace qe
