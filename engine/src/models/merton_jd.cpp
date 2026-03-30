#include "qe/models/merton_jd.hpp"
#include <cmath>
#include <random>

namespace qe {

// ============================================================================
// Semi-analytical: Poisson-weighted sum of BS prices
// ============================================================================
static Real log_factorial(Size n) {
    Real result = 0.0;
    for (Size i = 2; i <= n; ++i) {
        result += std::log(static_cast<Real>(i));
    }
    return result;
}

Real merton_call(const MertonParams& p, Real K, Size n_terms) {
    // No jumps: reduce to Black-Scholes
    if (p.lambda <= 0.0) {
        return bs_call(p.spot, K, p.rate, p.sigma, p.maturity);
    }

    Real k = std::exp(p.mu_j + 0.5 * p.sigma_j * p.sigma_j) - 1.0;
    Real lambda_prime = p.lambda * (1.0 + k);
    Real T = p.maturity;

    Real price = 0.0;

    for (Size n = 0; n < n_terms; ++n) {
        // Adjusted parameters for n jumps
        Real sigma_n_sq = p.sigma * p.sigma
                        + static_cast<Real>(n) * p.sigma_j * p.sigma_j / T;
        Real sigma_n = std::sqrt(sigma_n_sq);

        Real r_n = p.rate - p.lambda * k
                 + static_cast<Real>(n) * (p.mu_j + 0.5 * p.sigma_j * p.sigma_j) / T;

        // Poisson weight: exp(-lambda'*T) * (lambda'*T)^n / n!
        Real log_weight = -lambda_prime * T
                        + static_cast<Real>(n) * std::log(lambda_prime * T)
                        - log_factorial(n);
        Real weight = std::exp(log_weight);

        price += weight * bs_call(p.spot, K, r_n, sigma_n, T);

        if (weight < 1e-15 && n > 5) break;  // convergence
    }

    return price;
}

Real merton_put(const MertonParams& p, Real K, Size n_terms) {
    Real call = merton_call(p, K, n_terms);
    Real discount = std::exp(-p.rate * p.maturity);
    return call - p.spot + K * discount;  // put-call parity
}

// ============================================================================
// Monte Carlo
// ============================================================================
MCResult merton_mc(const MertonParams& p, Real strike, OptionType type,
                   Size num_paths, Size num_steps, uint64_t seed) {
    auto start = std::chrono::steady_clock::now();

    MersenneTwister rng(seed);
    std::mt19937_64 poisson_gen(seed + 1);

    Real T = p.maturity;
    Real dt = T / static_cast<Real>(num_steps);
    Real discount = std::exp(-p.rate * T);
    Real k = std::exp(p.mu_j + 0.5 * p.sigma_j * p.sigma_j) - 1.0;

    // Compensated drift
    Real drift = (p.rate - 0.5 * p.sigma * p.sigma - p.lambda * k) * dt;
    Real vol = p.sigma * std::sqrt(dt);

    std::poisson_distribution<int> poisson(p.lambda * dt);

    RunningStats stats;

    for (Size i = 0; i < num_paths; ++i) {
        Real S = p.spot;

        for (Size j = 0; j < num_steps; ++j) {
            Real z = rng.normal();

            // Diffusion
            Real log_return = drift + vol * z;

            // Jumps: N ~ Poisson(lambda * dt)
            int n_jumps = poisson(poisson_gen);
            for (int jj = 0; jj < n_jumps; ++jj) {
                Real jump_z = rng.normal();
                log_return += p.mu_j + p.sigma_j * jump_z;
            }

            S *= std::exp(log_return);
        }

        Real payoff;
        if (type == OptionType::Call) {
            payoff = std::max(S - strike, 0.0);
        } else {
            payoff = std::max(strike - S, 0.0);
        }
        stats.push(discount * payoff);
    }

    auto end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    Real se = stats.standard_error();

    return {stats.mean(), se, stats.mean() - 1.96 * se,
            stats.mean() + 1.96 * se, stats.count(), elapsed};
}

} // namespace qe
