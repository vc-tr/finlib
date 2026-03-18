#pragma once

#include "qe/core/types.hpp"

namespace qe {

// Standard normal CDF (Abramowitz & Stegun rational approximation)
Real norm_cdf(Real x);

// Standard normal PDF
Real norm_pdf(Real x);

// Black-Scholes d1 and d2
Real bs_d1(Real S, Real K, Real r, Real sigma, Real T);
Real bs_d2(Real S, Real K, Real r, Real sigma, Real T);

// Black-Scholes closed-form pricing
Real bs_price(Real S, Real K, Real r, Real sigma, Real T, OptionType type);
Real bs_call(Real S, Real K, Real r, Real sigma, Real T);
Real bs_put(Real S, Real K, Real r, Real sigma, Real T);

// Analytical Greeks
struct BSGreeks {
    Real delta;
    Real gamma;
    Real vega;
    Real theta;
    Real rho;
};

BSGreeks bs_greeks(Real S, Real K, Real r, Real sigma, Real T, OptionType type);

// Individual Greeks
Real bs_delta(Real S, Real K, Real r, Real sigma, Real T, OptionType type);
Real bs_gamma(Real S, Real K, Real r, Real sigma, Real T);
Real bs_vega(Real S, Real K, Real r, Real sigma, Real T);
Real bs_theta(Real S, Real K, Real r, Real sigma, Real T, OptionType type);
Real bs_rho(Real S, Real K, Real r, Real sigma, Real T, OptionType type);

} // namespace qe
