#include "qe/montecarlo/engine.hpp"
#include <cmath>

namespace qe {

MCEngine::MCEngine(const Config& config) : config_(config) {}

Real MCEngine::terminal_spot(Real z) const {
    // GBM exact solution: S(T) = S(0) * exp((r - sigma^2/2)*T + sigma*sqrt(T)*z)
    Real drift = (config_.rate - 0.5 * config_.sigma * config_.sigma) * config_.maturity;
    Real diffusion = config_.sigma * std::sqrt(config_.maturity) * z;
    return config_.spot * std::exp(drift + diffusion);
}

MCResult MCEngine::build_result(const RunningStats& stats,
                                 std::chrono::steady_clock::time_point start) const {
    auto end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();

    Real se = stats.standard_error();
    return {
        stats.mean(),
        se,
        stats.mean() - 1.96 * se,
        stats.mean() + 1.96 * se,
        stats.count(),
        elapsed
    };
}

// ============================================================================
// Standard MC (no variance reduction)
// ============================================================================
MCResult MCEngine::price(const Payoff& payoff) const {
    auto start = std::chrono::steady_clock::now();
    MersenneTwister rng(config_.seed);
    Real discount = std::exp(-config_.rate * config_.maturity);

    RunningStats stats;
    for (Size i = 0; i < config_.num_paths; ++i) {
        Real z = rng.normal();
        Real ST = terminal_spot(z);
        Real pv = discount * payoff(ST);
        stats.push(pv);
    }

    return build_result(stats, start);
}

// ============================================================================
// Antithetic variates: use both z and -z
// Reduces variance by ~50% for monotonic payoffs
// ============================================================================
MCResult MCEngine::price_antithetic(const Payoff& payoff) const {
    auto start = std::chrono::steady_clock::now();
    MersenneTwister rng(config_.seed);
    Real discount = std::exp(-config_.rate * config_.maturity);

    RunningStats stats;
    for (Size i = 0; i < config_.num_paths / 2; ++i) {
        Real z = rng.normal();

        Real ST_pos = terminal_spot(z);
        Real ST_neg = terminal_spot(-z);

        // Average the two payoffs — this is the antithetic estimator
        Real pv = discount * 0.5 * (payoff(ST_pos) + payoff(ST_neg));
        stats.push(pv);
    }

    return build_result(stats, start);
}

// ============================================================================
// Control variate: E[Y] = E[Y - beta*(C - E[C])]
// Uses a correlated variable with known expectation to reduce variance
// ============================================================================
MCResult MCEngine::price_control_variate(const Payoff& payoff,
                                          const std::function<Real(Real)>& control_fn,
                                          Real control_mean) const {
    auto start = std::chrono::steady_clock::now();
    MersenneTwister rng(config_.seed);
    Real discount = std::exp(-config_.rate * config_.maturity);

    // First pass: estimate optimal beta = Cov(Y, C) / Var(C)
    Vec payoff_vals(config_.num_paths);
    Vec control_vals(config_.num_paths);

    for (Size i = 0; i < config_.num_paths; ++i) {
        Real z = rng.normal();
        Real ST = terminal_spot(z);
        payoff_vals[i] = discount * payoff(ST);
        control_vals[i] = discount * control_fn(ST);
    }

    Real beta = covariance(payoff_vals, control_vals) / variance(control_vals);

    // Second pass: compute adjusted estimator
    RunningStats stats;
    Real disc_control_mean = discount * control_mean;
    for (Size i = 0; i < config_.num_paths; ++i) {
        Real adjusted = payoff_vals[i] - beta * (control_vals[i] - disc_control_mean);
        stats.push(adjusted);
    }

    return build_result(stats, start);
}

// ============================================================================
// Importance sampling: shift the drift to sample rare events more efficiently
// Useful for deep OTM options
// ============================================================================
MCResult MCEngine::price_importance_sampling(const Payoff& payoff,
                                              Real drift_shift) const {
    auto start = std::chrono::steady_clock::now();
    MersenneTwister rng(config_.seed);
    Real discount = std::exp(-config_.rate * config_.maturity);
    Real sqrt_T = std::sqrt(config_.maturity);

    RunningStats stats;
    for (Size i = 0; i < config_.num_paths; ++i) {
        Real z = rng.normal();

        // Sample from shifted distribution: z' = z + shift
        Real z_shifted = z + drift_shift;

        // Compute terminal spot using shifted normal
        Real drift = (config_.rate - 0.5 * config_.sigma * config_.sigma) * config_.maturity;
        Real ST = config_.spot * std::exp(drift + config_.sigma * sqrt_T * z_shifted);

        // Likelihood ratio (Radon-Nikodym derivative)
        // L = exp(-drift_shift * z - 0.5 * drift_shift^2)
        Real log_lr = -drift_shift * z - 0.5 * drift_shift * drift_shift;
        Real lr = std::exp(log_lr);

        Real pv = discount * payoff(ST) * lr;
        stats.push(pv);
    }

    return build_result(stats, start);
}

} // namespace qe
