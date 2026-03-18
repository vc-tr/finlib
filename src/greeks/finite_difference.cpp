#include "qe/greeks/finite_difference.hpp"

namespace qe {

FDGreeks mc_fd_greeks(const MCEngine::Config& base_config,
                      const Payoff& payoff,
                      Real bump_spot,
                      Real bump_sigma,
                      Real bump_time,
                      Real bump_rate) {
    FDGreeks greeks{};

    // Delta: dV/dS via central difference
    {
        auto cfg_up = base_config; cfg_up.spot += bump_spot;
        auto cfg_dn = base_config; cfg_dn.spot -= bump_spot;
        Real v_up = MCEngine(cfg_up).price(payoff).price;
        Real v_dn = MCEngine(cfg_dn).price(payoff).price;
        greeks.delta = (v_up - v_dn) / (2.0 * bump_spot);
    }

    // Gamma: d2V/dS2
    {
        auto cfg_up = base_config; cfg_up.spot += bump_spot;
        auto cfg_dn = base_config; cfg_dn.spot -= bump_spot;
        Real v_up = MCEngine(cfg_up).price(payoff).price;
        Real v_mid = MCEngine(base_config).price(payoff).price;
        Real v_dn = MCEngine(cfg_dn).price(payoff).price;
        greeks.gamma = (v_up - 2.0 * v_mid + v_dn) / (bump_spot * bump_spot);
    }

    // Vega: dV/dsigma
    {
        auto cfg_up = base_config; cfg_up.sigma += bump_sigma;
        auto cfg_dn = base_config; cfg_dn.sigma -= bump_sigma;
        Real v_up = MCEngine(cfg_up).price(payoff).price;
        Real v_dn = MCEngine(cfg_dn).price(payoff).price;
        greeks.vega = (v_up - v_dn) / (2.0 * bump_sigma);
    }

    // Theta: dV/dT (negative convention: shorter maturity)
    {
        auto cfg_up = base_config; cfg_up.maturity += bump_time;
        auto cfg_dn = base_config; cfg_dn.maturity -= bump_time;
        if (cfg_dn.maturity > 0.0) {
            Real v_up = MCEngine(cfg_up).price(payoff).price;
            Real v_dn = MCEngine(cfg_dn).price(payoff).price;
            greeks.theta = (v_dn - v_up) / (2.0 * bump_time);  // negative: value decays
        }
    }

    // Rho: dV/dr
    {
        auto cfg_up = base_config; cfg_up.rate += bump_rate;
        auto cfg_dn = base_config; cfg_dn.rate -= bump_rate;
        Real v_up = MCEngine(cfg_up).price(payoff).price;
        Real v_dn = MCEngine(cfg_dn).price(payoff).price;
        greeks.rho = (v_up - v_dn) / (2.0 * bump_rate);
    }

    return greeks;
}

} // namespace qe
