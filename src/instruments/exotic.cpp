#include "qe/instruments/exotic.hpp"
#include "qe/math/linalg.hpp"
#include <chrono>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace qe {

Vec ExoticPricer::generate_path(MersenneTwister& rng) const {
    Size n = config_.num_steps;
    Real dt = config_.maturity / static_cast<Real>(n);
    Real drift = (config_.rate - 0.5 * config_.sigma * config_.sigma) * dt;
    Real vol = config_.sigma * std::sqrt(dt);

    Vec path(n + 1);
    path[0] = config_.spot;

    for (Size i = 0; i < n; ++i) {
        Real z = rng.normal();
        path[i + 1] = path[i] * std::exp(drift + vol * z);
    }

    return path;
}

// ============================================================================
// Barrier Options
// ============================================================================
ExoticPricer::BarrierResult ExoticPricer::price_barrier(
    Real strike, OptionType opt_type,
    Real barrier, BarrierType bar_type) const
{
    MersenneTwister rng(config_.seed);
    Real discount = std::exp(-config_.rate * config_.maturity);

    RunningStats stats;
    Size knock_count = 0;

    for (Size i = 0; i < config_.num_paths; ++i) {
        Vec path = generate_path(rng);

        // Check barrier condition
        bool hit = false;
        for (Size j = 1; j <= config_.num_steps; ++j) {
            if ((bar_type == BarrierType::DownAndOut || bar_type == BarrierType::DownAndIn)
                && path[j] <= barrier) {
                hit = true;
                break;
            }
            if ((bar_type == BarrierType::UpAndOut || bar_type == BarrierType::UpAndIn)
                && path[j] >= barrier) {
                hit = true;
                break;
            }
        }

        if (hit) ++knock_count;

        Real ST = path[config_.num_steps];
        Real payoff = (opt_type == OptionType::Call)
            ? std::max(ST - strike, 0.0)
            : std::max(strike - ST, 0.0);

        // Apply barrier logic
        Real value = 0.0;
        switch (bar_type) {
            case BarrierType::DownAndOut:
            case BarrierType::UpAndOut:
                value = hit ? 0.0 : payoff;
                break;
            case BarrierType::DownAndIn:
            case BarrierType::UpAndIn:
                value = hit ? payoff : 0.0;
                break;
        }

        stats.push(discount * value);
    }

    Real se = stats.standard_error();
    Real knock_pct = static_cast<Real>(knock_count) / static_cast<Real>(config_.num_paths);

    return {stats.mean(), se, knock_pct};
}

// ============================================================================
// Asian Options
// ============================================================================
MCResult ExoticPricer::price_asian(Real strike, OptionType opt_type,
                                    AverageType avg_type) const {
    auto start = std::chrono::steady_clock::now();
    MersenneTwister rng(config_.seed);
    Real discount = std::exp(-config_.rate * config_.maturity);

    RunningStats stats;

    for (Size i = 0; i < config_.num_paths; ++i) {
        Vec path = generate_path(rng);

        Real avg;
        if (avg_type == AverageType::Arithmetic) {
            // Arithmetic average of monitoring points (excluding t=0)
            avg = 0.0;
            for (Size j = 1; j <= config_.num_steps; ++j) {
                avg += path[j];
            }
            avg /= static_cast<Real>(config_.num_steps);
        } else {
            // Geometric average
            Real log_sum = 0.0;
            for (Size j = 1; j <= config_.num_steps; ++j) {
                log_sum += std::log(path[j]);
            }
            avg = std::exp(log_sum / static_cast<Real>(config_.num_steps));
        }

        Real payoff = (opt_type == OptionType::Call)
            ? std::max(avg - strike, 0.0)
            : std::max(strike - avg, 0.0);

        stats.push(discount * payoff);
    }

    auto end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    Real se = stats.standard_error();

    return {stats.mean(), se, stats.mean() - 1.96 * se,
            stats.mean() + 1.96 * se, stats.count(), elapsed};
}

// Closed-form geometric Asian call price (Kemna & Vorst, 1990)
Real ExoticPricer::geometric_asian_call(Real S, Real K, Real r,
                                         Real sigma, Real T, Size n) {
    Real n_r = static_cast<Real>(n);

    // Adjusted parameters for geometric average
    Real sigma_a = sigma * std::sqrt((2.0 * n_r + 1.0) / (6.0 * (n_r + 1.0)));
    Real mu_a = 0.5 * (r - 0.5 * sigma * sigma + sigma_a * sigma_a);

    // Price using BS with adjusted parameters
    Real d1 = (std::log(S / K) + (mu_a + 0.5 * sigma_a * sigma_a) * T)
              / (sigma_a * std::sqrt(T));
    Real d2 = d1 - sigma_a * std::sqrt(T);

    return std::exp(-r * T) * (S * std::exp(mu_a * T) * norm_cdf(d1)
                                - K * norm_cdf(d2));
}

// ============================================================================
// Lookback Options (floating strike)
// ============================================================================
MCResult ExoticPricer::price_lookback(OptionType type) const {
    auto start = std::chrono::steady_clock::now();
    MersenneTwister rng(config_.seed);
    Real discount = std::exp(-config_.rate * config_.maturity);

    RunningStats stats;

    for (Size i = 0; i < config_.num_paths; ++i) {
        Vec path = generate_path(rng);

        Real ST = path[config_.num_steps];
        Real path_min = *std::min_element(path.begin(), path.end());
        Real path_max = *std::max_element(path.begin(), path.end());

        Real payoff;
        if (type == OptionType::Call) {
            // Floating strike lookback call: S(T) - min(S)
            payoff = ST - path_min;
        } else {
            // Floating strike lookback put: max(S) - S(T)
            payoff = path_max - ST;
        }

        stats.push(discount * payoff);
    }

    auto end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    Real se = stats.standard_error();

    return {stats.mean(), se, stats.mean() - 1.96 * se,
            stats.mean() + 1.96 * se, stats.count(), elapsed};
}

// ============================================================================
// Basket Options
// ============================================================================
MCResult ExoticPricer::price_basket(const BasketConfig& basket, Real strike,
                                     OptionType type) const {
    auto start = std::chrono::steady_clock::now();
    Size n_assets = basket.spots.size();

    // Cholesky decomposition of correlation matrix
    Mat L = cholesky(basket.correlation);

    MersenneTwister rng(config_.seed);
    Real discount = std::exp(-config_.rate * config_.maturity);
    Real sqrt_T = std::sqrt(config_.maturity);

    RunningStats stats;

    for (Size i = 0; i < config_.num_paths; ++i) {
        // Generate correlated normals
        Vec z(n_assets);
        for (Size a = 0; a < n_assets; ++a) {
            z[a] = rng.normal();
        }
        Vec corr_z = mat_vec_mul(L, z);

        // Simulate terminal prices and compute basket value
        Real basket_value = 0.0;
        for (Size a = 0; a < n_assets; ++a) {
            Real drift = (config_.rate - 0.5 * basket.sigmas[a] * basket.sigmas[a])
                         * config_.maturity;
            Real ST = basket.spots[a] * std::exp(drift + basket.sigmas[a] * sqrt_T * corr_z[a]);
            basket_value += basket.weights[a] * ST;
        }

        Real payoff = (type == OptionType::Call)
            ? std::max(basket_value - strike, 0.0)
            : std::max(strike - basket_value, 0.0);

        stats.push(discount * payoff);
    }

    auto end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    Real se = stats.standard_error();

    return {stats.mean(), se, stats.mean() - 1.96 * se,
            stats.mean() + 1.96 * se, stats.count(), elapsed};
}

} // namespace qe
