#include "qe/models/cir.hpp"
#include <cmath>
#include <random>
#include <algorithm>

namespace qe {

bool cir_feller_satisfied(const CIRParams& p) {
    return 2.0 * p.kappa * p.theta >= p.sigma * p.sigma;
}

// Exact simulation via non-central chi-squared
// r(t+dt) = (sigma^2 * (1-e^{-kappa*dt})) / (4*kappa) * chi^2_d(lambda)
// where d = 4*kappa*theta / sigma^2
// lambda = 4*kappa*e^{-kappa*dt} / (sigma^2*(1-e^{-kappa*dt})) * r(t)
Real cir_exact_step(const CIRParams& p, Real r_t, Real dt,
                    MersenneTwister& rng) {
    Real c = p.sigma * p.sigma * (1.0 - std::exp(-p.kappa * dt))
             / (4.0 * p.kappa);
    Real d = 4.0 * p.kappa * p.theta / (p.sigma * p.sigma);
    Real lambda = r_t * std::exp(-p.kappa * dt) / c;

    // Simulate non-central chi-squared with d degrees of freedom
    // and non-centrality parameter lambda
    // Method: X = sum of (Z_i + sqrt(lambda/d))^2 for i=1..d (integer d)
    // For non-integer d, use Poisson mixture of central chi-squared

    // Simple approximation: use normal approximation for large d
    // E[chi^2_d(lambda)] = d + lambda
    // Var[chi^2_d(lambda)] = 2*(d + 2*lambda)

    // For better accuracy, use the Poisson mixture:
    // chi^2_d(lambda) ~ chi^2_{d+2N} where N ~ Poisson(lambda/2)
    std::mt19937_64 gen(static_cast<uint64_t>(r_t * 1e8 + dt * 1e6));
    std::poisson_distribution<int> poisson(std::max(lambda / 2.0, 0.0));
    int N = poisson(gen);

    // chi^2_{d+2N}: sum of d + 2N independent standard normals squared
    Real df = d + 2.0 * static_cast<Real>(N);
    Real chi2_val = 0.0;

    if (df > 0) {
        // For large df, use normal approximation: chi^2_df ≈ df*(1 + sqrt(2/df)*Z)^2
        // But more accurately, sum of squares
        Size int_df = static_cast<Size>(std::max(std::round(df), 1.0));
        for (Size i = 0; i < int_df; ++i) {
            Real z = rng.normal();
            chi2_val += z * z;
        }
    }

    return std::max(c * chi2_val, 0.0);
}

Vec cir_path(const CIRParams& p, Real T, Size n_steps,
             MersenneTwister& rng) {
    Real dt = T / static_cast<Real>(n_steps);
    Vec path(n_steps + 1);
    path[0] = p.r0;

    for (Size i = 0; i < n_steps; ++i) {
        path[i + 1] = cir_exact_step(p, path[i], dt, rng);
    }

    return path;
}

// Closed-form zero-coupon bond price under CIR
Real cir_bond_price(const CIRParams& p, Real T) {
    Real gamma = std::sqrt(p.kappa * p.kappa + 2.0 * p.sigma * p.sigma);

    Real exp_gamma_T = std::exp(gamma * T);
    Real denom = (gamma + p.kappa) * (exp_gamma_T - 1.0) + 2.0 * gamma;

    Real A = std::pow(2.0 * gamma * std::exp((p.kappa + gamma) * T / 2.0) / denom,
                      2.0 * p.kappa * p.theta / (p.sigma * p.sigma));

    Real B = 2.0 * (exp_gamma_T - 1.0) / denom;

    return A * std::exp(-B * p.r0);
}

Real cir_zero_rate(const CIRParams& p, Real T) {
    if (T <= 0.0) return p.r0;
    return -std::log(cir_bond_price(p, T)) / T;
}

} // namespace qe
