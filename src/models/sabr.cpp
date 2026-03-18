#include "qe/models/sabr.hpp"
#include "qe/models/black_scholes.hpp"
#include <cmath>

namespace qe {

Real sabr_implied_vol(const SABRParams& p, Real F, Real K, Real T) {
    if (T <= 0.0) return p.alpha;

    Real eps = 1e-10;

    // ATM case
    if (std::abs(F - K) < eps * F) {
        Real FK_mid = F;  // = K
        Real FK_beta = std::pow(FK_mid, 1.0 - p.beta);

        Real term1 = p.alpha / FK_beta;
        Real term2 = 1.0 + T * (
            ((1.0 - p.beta) * (1.0 - p.beta) * p.alpha * p.alpha)
                / (24.0 * FK_beta * FK_beta)
            + (p.rho * p.beta * p.nu * p.alpha)
                / (4.0 * FK_beta)
            + (2.0 - 3.0 * p.rho * p.rho) * p.nu * p.nu / 24.0
        );

        return term1 * term2;
    }

    // General case
    Real FK = F * K;
    Real FK_beta_half = std::pow(FK, (1.0 - p.beta) / 2.0);
    Real logFK = std::log(F / K);

    // z and x(z) for the main correction
    Real z = (p.nu / p.alpha) * FK_beta_half * logFK;
    Real x_z;

    if (std::abs(z) < eps) {
        x_z = 1.0;  // limit as z -> 0
    } else {
        Real sqrt_term = std::sqrt(1.0 - 2.0 * p.rho * z + z * z);
        x_z = z / std::log((sqrt_term + z - p.rho) / (1.0 - p.rho));
    }

    // Denominator correction
    Real one_minus_beta = 1.0 - p.beta;
    Real denom = FK_beta_half * (
        1.0
        + one_minus_beta * one_minus_beta * logFK * logFK / 24.0
        + std::pow(one_minus_beta, 4) * std::pow(logFK, 4) / 1920.0
    );

    // Numerator correction
    Real numer = p.alpha * x_z;

    // Higher order T correction
    Real correction = 1.0 + T * (
        one_minus_beta * one_minus_beta * p.alpha * p.alpha
            / (24.0 * std::pow(FK, one_minus_beta))
        + p.rho * p.beta * p.nu * p.alpha
            / (4.0 * FK_beta_half)
        + (2.0 - 3.0 * p.rho * p.rho) * p.nu * p.nu / 24.0
    );

    return (numer / denom) * correction;
}

Real sabr_call(const SABRParams& params, Real forward, Real strike,
               Real rate, Real T) {
    Real vol = sabr_implied_vol(params, forward, strike, T);
    // Use Black formula (forward version of BS)
    Real discount = std::exp(-rate * T);
    return discount * bs_call(forward, strike, 0.0, vol, T);
}

Real sabr_put(const SABRParams& params, Real forward, Real strike,
              Real rate, Real T) {
    Real vol = sabr_implied_vol(params, forward, strike, T);
    Real discount = std::exp(-rate * T);
    return discount * bs_put(forward, strike, 0.0, vol, T);
}

} // namespace qe
