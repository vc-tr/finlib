#include "qe/risk/portfolio.hpp"
#include <cmath>

namespace qe {

PortfolioRisk portfolio_risk(const std::vector<OptionPosition>& positions) {
    PortfolioRisk risk{};

    for (const auto& pos : positions) {
        Real price = bs_price(pos.spot, pos.strike, pos.rate,
                              pos.sigma, pos.maturity, pos.type);
        auto greeks = bs_greeks(pos.spot, pos.strike, pos.rate,
                                pos.sigma, pos.maturity, pos.type);

        risk.total_value += pos.quantity * price;
        risk.total_delta += pos.quantity * greeks.delta;
        risk.total_gamma += pos.quantity * greeks.gamma;
        risk.total_vega += pos.quantity * greeks.vega;
        risk.total_theta += pos.quantity * greeks.theta;
        risk.total_rho += pos.quantity * greeks.rho;
    }

    return risk;
}

std::vector<StressResult> stress_test_portfolio(
    const std::vector<OptionPosition>& positions,
    const std::vector<StressScenario>& scenarios) {

    // Base portfolio value
    Real base_value = 0.0;
    for (const auto& pos : positions) {
        base_value += pos.quantity * bs_price(pos.spot, pos.strike, pos.rate,
                                              pos.sigma, pos.maturity, pos.type);
    }

    std::vector<StressResult> results;

    for (const auto& scenario : scenarios) {
        Real stressed_value = 0.0;

        for (const auto& pos : positions) {
            Real s_spot = pos.spot * scenario.spot_shock;
            Real s_vol = std::max(pos.sigma + scenario.vol_shock, 0.01);
            Real s_rate = pos.rate + scenario.rate_shock;

            stressed_value += pos.quantity * bs_price(s_spot, pos.strike, s_rate,
                                                       s_vol, pos.maturity, pos.type);
        }

        Real pnl = stressed_value - base_value;
        Real pnl_pct = (base_value != 0.0) ? pnl / std::abs(base_value) : 0.0;

        results.push_back({scenario.name, base_value, stressed_value, pnl, pnl_pct});
    }

    return results;
}

} // namespace qe
