#pragma once

#include "qe/core/types.hpp"
#include <algorithm>

namespace qe {

// Abstract payoff base class
class Payoff {
public:
    virtual ~Payoff() = default;
    virtual Real operator()(Real spot) const = 0;
};

// Vanilla call/put payoff: max(S - K, 0) or max(K - S, 0)
class VanillaPayoff : public Payoff {
public:
    VanillaPayoff(Real strike, OptionType type)
        : strike_(strike), type_(type) {}

    Real operator()(Real spot) const override {
        if (type_ == OptionType::Call)
            return std::max(spot - strike_, 0.0);
        return std::max(strike_ - spot, 0.0);
    }

    Real strike() const { return strike_; }
    OptionType type() const { return type_; }

private:
    Real strike_;
    OptionType type_;
};

// Digital (binary) payoff: pays 1 if ITM, 0 otherwise
class DigitalPayoff : public Payoff {
public:
    DigitalPayoff(Real strike, OptionType type, Real payout = 1.0)
        : strike_(strike), type_(type), payout_(payout) {}

    Real operator()(Real spot) const override {
        if (type_ == OptionType::Call)
            return spot > strike_ ? payout_ : 0.0;
        return spot < strike_ ? payout_ : 0.0;
    }

private:
    Real strike_;
    OptionType type_;
    Real payout_;
};

} // namespace qe
