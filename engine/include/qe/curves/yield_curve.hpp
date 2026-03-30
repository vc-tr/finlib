#pragma once

#include "qe/core/types.hpp"
#include "qe/math/interpolation.hpp"
#include <memory>

namespace qe {

// Yield curve providing discount factors, zero rates, and forward rates
class YieldCurve {
public:
    // Build from zero rates at given tenors
    YieldCurve(const Vec& tenors, const Vec& zero_rates);

    // Discount factor P(0, T) = exp(-r(T) * T)
    Real discount(Real T) const;

    // Continuously compounded zero rate r(T)
    Real zero_rate(Real T) const;

    // Instantaneous forward rate f(T) = -d/dT ln(P(0,T))
    Real forward_rate(Real T) const;

    // Forward rate between T1 and T2
    Real forward_rate(Real T1, Real T2) const;

    const Vec& tenors() const { return tenors_; }
    const Vec& rates() const { return rates_; }

private:
    Vec tenors_;
    Vec rates_;
    std::unique_ptr<CubicSplineInterpolator> interp_;
};

// Bootstrap yield curve from market instruments
struct DepositRate {
    Real tenor;   // in years
    Real rate;    // simple rate (annualized)
};

struct SwapRate {
    Real tenor;       // in years
    Real rate;        // par swap rate
    Real frequency;   // payments per year (2 = semi-annual)
};

// Bootstrap a yield curve from deposit rates and swap rates
YieldCurve bootstrap_curve(const std::vector<DepositRate>& deposits,
                           const std::vector<SwapRate>& swaps);

} // namespace qe
