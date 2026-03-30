#include "qe/risk/stress.hpp"

namespace qe {

std::vector<StressScenario> predefined_scenarios() {
    return {
        {"Market Crash (-25%)", 0.75, 0.15, -0.01},
        {"Moderate Decline (-10%)", 0.90, 0.05, 0.0},
        {"Vol Spike", 1.0, 0.20, 0.0},
        {"Rate Hike (+200bp)", 1.0, 0.0, 0.02},
        {"Rate Cut (-100bp)", 1.0, 0.0, -0.01},
        {"Stagflation (-15%, +vol, +rates)", 0.85, 0.10, 0.015},
        {"Recovery (+15%, -vol)", 1.15, -0.05, 0.0},
    };
}

} // namespace qe
