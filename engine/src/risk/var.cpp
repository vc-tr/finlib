#include "qe/risk/var.hpp"
#include "qe/math/statistics.hpp"
#include "qe/models/black_scholes.hpp"
#include "qe/core/constants.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <numeric>

namespace qe {

Real historical_var(std::span<const Real> pnl, Real confidence) {
    if (pnl.empty()) throw std::invalid_argument("historical_var: empty data");

    // VaR = negative of the (1-alpha) quantile of P&L
    Real q = quantile(pnl, 1.0 - confidence);
    return -q;
}

Real parametric_var(Real mean_val, Real std_dev_val, Real confidence) {
    // z_alpha: quantile of standard normal at confidence level
    // For 95%: z = 1.6449, for 99%: z = 2.3263
    // Use inverse normal CDF approximation

    // Bisect to find z such that Phi(z) = confidence
    Real z = 0.0;
    Real lo = 0.0, hi = 5.0;
    for (Size i = 0; i < 100; ++i) {
        z = (lo + hi) / 2.0;
        Real cdf = norm_cdf(z);
        if (cdf < confidence) lo = z;
        else hi = z;
    }

    return -(mean_val - z * std_dev_val);
}

Real parametric_var(std::span<const Real> returns, Real confidence) {
    return parametric_var(mean(returns), std_dev(returns), confidence);
}

Real historical_cvar(std::span<const Real> pnl, Real confidence) {
    if (pnl.empty()) throw std::invalid_argument("historical_cvar: empty data");

    Vec sorted(pnl.begin(), pnl.end());
    std::sort(sorted.begin(), sorted.end());

    // CVaR = average of losses beyond VaR threshold
    Size cutoff = static_cast<Size>(std::ceil(static_cast<Real>(sorted.size()) * (1.0 - confidence)));
    if (cutoff == 0) cutoff = 1;

    Real sum = 0.0;
    for (Size i = 0; i < cutoff; ++i) {
        sum += sorted[i];
    }

    return -(sum / static_cast<Real>(cutoff));
}

Real parametric_cvar(Real mean_val, Real std_dev_val, Real confidence) {
    // CVaR_normal = -mean + sigma * phi(z_alpha) / (1 - alpha)
    // where phi is the standard normal PDF

    // Find z_alpha
    Real z = 0.0;
    Real lo = 0.0, hi = 5.0;
    for (Size i = 0; i < 100; ++i) {
        z = (lo + hi) / 2.0;
        Real cdf = norm_cdf(z);
        if (cdf < confidence) lo = z;
        else hi = z;
    }

    Real pdf_z = norm_pdf(z);
    return -(mean_val - std_dev_val * pdf_z / (1.0 - confidence));
}

} // namespace qe
