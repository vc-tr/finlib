#pragma once

#include "qe/core/types.hpp"
#include <string>
#include <vector>

namespace qe {

// A stress scenario defines shocks to market factors
struct StressScenario {
    std::string name;
    Real spot_shock;     // multiplicative (0.8 = -20%)
    Real vol_shock;      // additive (0.1 = +10 vol points)
    Real rate_shock;     // additive (-0.02 = -200bp)
};

// Result of stress testing a position
struct StressResult {
    std::string scenario_name;
    Real base_value;
    Real stressed_value;
    Real pnl;
    Real pnl_pct;
};

// Predefined stress scenarios
std::vector<StressScenario> predefined_scenarios();

} // namespace qe
