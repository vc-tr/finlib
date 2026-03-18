#pragma once

#include "qe/core/types.hpp"
#include "qe/instruments/option.hpp"
#include "qe/models/black_scholes.hpp"

namespace qe {

// European option pricer — wraps analytical Black-Scholes
class EuropeanOption {
public:
    explicit EuropeanOption(const OptionSpec& spec) : spec_(spec) {}

    // Analytical price
    Real price() const {
        return bs_price(spec_.spot, spec_.strike, spec_.rate,
                       spec_.sigma, spec_.maturity, spec_.type);
    }

    // All Greeks
    BSGreeks greeks() const {
        return bs_greeks(spec_.spot, spec_.strike, spec_.rate,
                        spec_.sigma, spec_.maturity, spec_.type);
    }

    // Put-call parity: C - P = S - K*exp(-rT)
    // Returns the other option's price given this one
    Real parity_price() const {
        Real pv_strike = spec_.strike * std::exp(-spec_.rate * spec_.maturity);
        Real this_price = price();

        if (spec_.type == OptionType::Call) {
            // P = C - S + K*exp(-rT)
            return this_price - spec_.spot + pv_strike;
        }
        // C = P + S - K*exp(-rT)
        return this_price + spec_.spot - pv_strike;
    }

    const OptionSpec& spec() const { return spec_; }

private:
    OptionSpec spec_;
};

} // namespace qe
