#include "qe/models/black_scholes.hpp"
#include "qe/core/constants.hpp"
#include <cmath>
#include <stdexcept>

namespace qe {

// ===========================================================================
// Normal CDF: Abramowitz & Stegun approximation (formula 26.2.17)
// Maximum absolute error: 7.5e-8
// ===========================================================================
Real norm_cdf(Real x) {
    // Abramowitz & Stegun formula 7.1.26 via error function
    // Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))
    // Higher-order rational approximation with max |eps| < 3e-7

    if (x > 8.0) return 1.0;
    if (x < -8.0) return 0.0;

    // Use symmetry: Phi(-x) = 1 - Phi(x)
    if (x < 0.0) return 1.0 - norm_cdf(-x);

    // Map to erf argument: z = x / sqrt(2)
    Real z = x / SQRT2;

    // A&S 7.1.26: 5-term rational approximation for erfc
    constexpr Real a1 =  0.254829592;
    constexpr Real a2 = -0.284496736;
    constexpr Real a3 =  1.421413741;
    constexpr Real a4 = -1.453152027;
    constexpr Real a5 =  1.061405429;
    constexpr Real p  =  0.3275911;

    Real t = 1.0 / (1.0 + p * z);
    Real t2 = t * t;
    Real t3 = t2 * t;
    Real t4 = t3 * t;
    Real t5 = t4 * t;

    Real erfc_approx = (a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * std::exp(-z * z);

    // Phi(x) = 0.5 * (1 + erf(z)) = 0.5 * (1 + (1 - erfc(z))) = 1 - 0.5 * erfc(z)
    return 1.0 - 0.5 * erfc_approx;
}

Real norm_pdf(Real x) {
    return INV_SQRT2PI * std::exp(-0.5 * x * x);
}

// ===========================================================================
// Black-Scholes formulas
// ===========================================================================

Real bs_d1(Real S, Real K, Real r, Real sigma, Real T) {
    return (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
}

Real bs_d2(Real S, Real K, Real r, Real sigma, Real T) {
    return bs_d1(S, K, r, sigma, T) - sigma * std::sqrt(T);
}

Real bs_call(Real S, Real K, Real r, Real sigma, Real T) {
    if (T <= 0.0) return std::max(S - K, 0.0);
    Real d1 = bs_d1(S, K, r, sigma, T);
    Real d2 = d1 - sigma * std::sqrt(T);
    return S * norm_cdf(d1) - K * std::exp(-r * T) * norm_cdf(d2);
}

Real bs_put(Real S, Real K, Real r, Real sigma, Real T) {
    if (T <= 0.0) return std::max(K - S, 0.0);
    Real d1 = bs_d1(S, K, r, sigma, T);
    Real d2 = d1 - sigma * std::sqrt(T);
    return K * std::exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1);
}

Real bs_price(Real S, Real K, Real r, Real sigma, Real T, OptionType type) {
    return (type == OptionType::Call) ? bs_call(S, K, r, sigma, T)
                                     : bs_put(S, K, r, sigma, T);
}

// ===========================================================================
// Greeks
// ===========================================================================

Real bs_delta(Real S, Real K, Real r, Real sigma, Real T, OptionType type) {
    if (T <= 0.0) {
        if (type == OptionType::Call) return S > K ? 1.0 : 0.0;
        return S < K ? -1.0 : 0.0;
    }
    Real d1 = bs_d1(S, K, r, sigma, T);
    if (type == OptionType::Call) return norm_cdf(d1);
    return norm_cdf(d1) - 1.0;
}

Real bs_gamma(Real S, Real K, Real r, Real sigma, Real T) {
    if (T <= 0.0) return 0.0;
    Real d1 = bs_d1(S, K, r, sigma, T);
    return norm_pdf(d1) / (S * sigma * std::sqrt(T));
}

Real bs_vega(Real S, Real K, Real r, Real sigma, Real T) {
    if (T <= 0.0) return 0.0;
    Real d1 = bs_d1(S, K, r, sigma, T);
    return S * norm_pdf(d1) * std::sqrt(T);
}

Real bs_theta(Real S, Real K, Real r, Real sigma, Real T, OptionType type) {
    if (T <= 0.0) return 0.0;
    Real d1 = bs_d1(S, K, r, sigma, T);
    Real d2 = d1 - sigma * std::sqrt(T);
    Real sqrt_T = std::sqrt(T);

    Real common = -(S * norm_pdf(d1) * sigma) / (2.0 * sqrt_T);

    if (type == OptionType::Call) {
        return common - r * K * std::exp(-r * T) * norm_cdf(d2);
    }
    return common + r * K * std::exp(-r * T) * norm_cdf(-d2);
}

Real bs_rho(Real S, Real K, Real r, Real sigma, Real T, OptionType type) {
    if (T <= 0.0) return 0.0;
    Real d2 = bs_d2(S, K, r, sigma, T);

    if (type == OptionType::Call) {
        return K * T * std::exp(-r * T) * norm_cdf(d2);
    }
    return -K * T * std::exp(-r * T) * norm_cdf(-d2);
}

BSGreeks bs_greeks(Real S, Real K, Real r, Real sigma, Real T, OptionType type) {
    return {
        bs_delta(S, K, r, sigma, T, type),
        bs_gamma(S, K, r, sigma, T),
        bs_vega(S, K, r, sigma, T),
        bs_theta(S, K, r, sigma, T, type),
        bs_rho(S, K, r, sigma, T, type)
    };
}

} // namespace qe
