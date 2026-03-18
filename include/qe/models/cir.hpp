#pragma once

#include "qe/core/types.hpp"
#include "qe/math/random.hpp"

namespace qe {

// Cox-Ingersoll-Ross (1985) short rate model:
// dr = kappa*(theta - r)*dt + sigma*sqrt(r)*dW
//
// Mean-reverting square-root diffusion, guarantees r >= 0
// when 2*kappa*theta >= sigma^2 (Feller condition)
struct CIRParams {
    Real r0;       // initial short rate
    Real kappa;    // mean reversion speed
    Real theta;    // long-run mean rate
    Real sigma;    // volatility
};

// Exact simulation using non-central chi-squared distribution
// CIR has a known transition density: r(t+dt) | r(t) ~ scaled non-central chi^2
Real cir_exact_step(const CIRParams& params, Real r_t, Real dt,
                    MersenneTwister& rng);

// Generate a path of short rates
Vec cir_path(const CIRParams& params, Real T, Size n_steps,
             MersenneTwister& rng);

// Zero-coupon bond price P(0, T) under CIR (closed-form)
Real cir_bond_price(const CIRParams& params, Real T);

// Zero rate R(0, T) = -ln(P(0,T)) / T
Real cir_zero_rate(const CIRParams& params, Real T);

// Check Feller condition: 2*kappa*theta >= sigma^2
bool cir_feller_satisfied(const CIRParams& params);

} // namespace qe
