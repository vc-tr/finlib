#include "qe/volatility/implied_vol.hpp"
#include "qe/models/black_scholes.hpp"
#include <cmath>

namespace qe {

ImpliedVolResult implied_vol(Real market_price, Real S, Real K, Real r, Real T,
                             OptionType type, Real tol, Size max_iter) {
    // Initial guess using Brenner & Subrahmanyam (1988) approximation
    Real sigma = std::sqrt(2.0 * 3.14159 / T) * market_price / S;
    sigma = std::max(sigma, 0.01);
    sigma = std::min(sigma, 5.0);

    for (Size i = 0; i < max_iter; ++i) {
        Real price = bs_price(S, K, r, sigma, T, type);
        Real vega = bs_vega(S, K, r, sigma, T);

        Real diff = price - market_price;

        if (std::abs(diff) < tol) {
            return {sigma, i + 1, true};
        }

        if (std::abs(vega) < 1e-20) {
            // Vega too small — try bisection instead
            break;
        }

        Real d_sigma = diff / vega;
        sigma -= d_sigma;

        // Keep sigma in reasonable bounds
        sigma = std::max(sigma, 0.001);
        sigma = std::min(sigma, 10.0);
    }

    // Fallback: bisection
    Real lo = 0.001, hi = 5.0;
    for (Size i = 0; i < max_iter; ++i) {
        Real mid = (lo + hi) / 2.0;
        Real price = bs_price(S, K, r, mid, T, type);

        if (std::abs(price - market_price) < tol) {
            return {mid, i + 1, true};
        }

        if (price > market_price) {
            hi = mid;
        } else {
            lo = mid;
        }
    }

    return {(lo + hi) / 2.0, max_iter, false};
}

} // namespace qe
