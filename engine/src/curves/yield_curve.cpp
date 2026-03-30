#include "qe/curves/yield_curve.hpp"
#include "qe/math/root_finding.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace qe {

YieldCurve::YieldCurve(const Vec& tenors, const Vec& zero_rates)
    : tenors_(tenors), rates_(zero_rates) {
    if (tenors.size() < 2) throw std::invalid_argument("YieldCurve: need at least 2 tenors");
    if (tenors.size() != zero_rates.size()) throw std::invalid_argument("YieldCurve: size mismatch");

    interp_ = std::make_unique<CubicSplineInterpolator>(tenors_, rates_);
}

Real YieldCurve::zero_rate(Real T) const {
    if (T <= 0.0) return rates_.front();
    if (T <= tenors_.front()) return rates_.front();
    if (T >= tenors_.back()) return rates_.back();
    return (*interp_)(T);
}

Real YieldCurve::discount(Real T) const {
    if (T <= 0.0) return 1.0;
    return std::exp(-zero_rate(T) * T);
}

Real YieldCurve::forward_rate(Real T) const {
    // f(T) = r(T) + T * r'(T)
    Real eps = 0.001;
    Real r_up = zero_rate(T + eps);
    Real r_dn = zero_rate(std::max(T - eps, 0.001));
    Real r_mid = zero_rate(T);
    Real dr_dT = (r_up - r_dn) / (2.0 * eps);
    return r_mid + T * dr_dT;
}

Real YieldCurve::forward_rate(Real T1, Real T2) const {
    if (T2 <= T1) throw std::invalid_argument("forward_rate: T2 must be > T1");
    Real df1 = discount(T1);
    Real df2 = discount(T2);
    return -std::log(df2 / df1) / (T2 - T1);
}

// ============================================================================
// Bootstrapping
// ============================================================================

YieldCurve bootstrap_curve(const std::vector<DepositRate>& deposits,
                           const std::vector<SwapRate>& swaps) {
    Vec tenors;
    Vec zero_rates;

    // Step 1: Deposit rates → zero rates (direct conversion)
    // Simple rate R over tenor T: P = 1 / (1 + R*T)
    // Zero rate: r = -ln(P) / T = ln(1 + R*T) / T
    for (const auto& dep : deposits) {
        tenors.push_back(dep.tenor);
        Real df = 1.0 / (1.0 + dep.rate * dep.tenor);
        zero_rates.push_back(-std::log(df) / dep.tenor);
    }

    // Sort by tenor
    // (assuming deposits are already sorted)

    // Step 2: Swap rates → zero rates (iterative bootstrapping)
    // Par swap: sum(c * P(0, t_i)) + P(0, T_n) = 1
    // where c = swap_rate / frequency
    for (const auto& swap : swaps) {
        Real c = swap.rate / swap.frequency;
        Size n_payments = static_cast<Size>(swap.tenor * swap.frequency);
        Real dt = 1.0 / swap.frequency;

        // Sum discount factors for all but the last payment
        Real pv_coupons = 0.0;
        for (Size i = 1; i < n_payments; ++i) {
            Real t_i = static_cast<Real>(i) * dt;

            // Interpolate zero rate at t_i from already-known points
            Real r_i;
            if (t_i <= tenors.back()) {
                // Linear interpolation from known points
                auto it = std::lower_bound(tenors.begin(), tenors.end(), t_i);
                if (it == tenors.begin()) {
                    r_i = zero_rates.front();
                } else {
                    Size idx = static_cast<Size>(it - tenors.begin());
                    Real t_lo = tenors[idx - 1];
                    Real t_hi = tenors[idx];
                    Real r_lo = zero_rates[idx - 1];
                    Real r_hi = zero_rates[idx];
                    Real frac = (t_i - t_lo) / (t_hi - t_lo);
                    r_i = r_lo + frac * (r_hi - r_lo);
                }
            } else {
                r_i = zero_rates.back();  // flat extrapolation
            }

            pv_coupons += c * std::exp(-r_i * t_i);
        }

        // Solve for the last discount factor:
        // pv_coupons + (c + 1) * P(0, T_n) = 1
        Real df_n = (1.0 - pv_coupons) / (1.0 + c);
        Real T_n = swap.tenor;
        Real r_n = -std::log(df_n) / T_n;

        tenors.push_back(T_n);
        zero_rates.push_back(r_n);
    }

    return YieldCurve(tenors, zero_rates);
}

} // namespace qe
