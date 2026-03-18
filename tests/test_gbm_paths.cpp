#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "qe/processes/gbm.hpp"
#include "qe/processes/discretization.hpp"
#include "qe/math/random.hpp"
#include "qe/math/statistics.hpp"
#include <cmath>

using namespace qe;
using Catch::Matchers::WithinAbs;

TEST_CASE("GBM exact evolution matches expected distribution", "[gbm]") {
    // Under risk-neutral measure with r=0.05, sigma=0.2:
    // E[S(T)] = S(0) * exp(r*T)
    Real mu = 0.05, sigma = 0.2;
    Real S0 = 100.0, T = 1.0;

    GBM gbm(mu, sigma);
    MersenneTwister rng(42);

    Size n_sims = 100000;
    Vec terminal(n_sims);

    for (Size i = 0; i < n_sims; ++i) {
        Real z = rng.normal();
        terminal[i] = gbm.evolve(0.0, S0, T, z);
    }

    Real expected_mean = S0 * std::exp(mu * T);
    Real sample_mean = mean(terminal);

    // Mean should be close to S0 * exp(mu * T)
    REQUIRE_THAT(sample_mean, WithinAbs(expected_mean, 1.0));
}

TEST_CASE("GBM path stays positive", "[gbm]") {
    GBM gbm(0.05, 0.3);
    MersenneTwister rng(123);

    Size n_steps = 252;
    Vec normals(n_steps);
    rng.fill_normal(normals);

    Vec path = generate_path(gbm, 100.0, 1.0, n_steps, normals);

    REQUIRE(path.size() == n_steps + 1);
    for (Real s : path) {
        REQUIRE(s > 0.0);
    }
}

TEST_CASE("Euler-Maruyama converges to exact for GBM", "[discretization]") {
    // With enough steps, Euler should be close to exact evolution
    GBM gbm(0.05, 0.2);
    MersenneTwister rng(42);
    Real S0 = 100.0, T = 1.0;

    Size n_sims = 50000;
    Size n_steps = 500;  // fine discretization
    Real dt = T / static_cast<Real>(n_steps);

    Vec euler_terminal(n_sims);
    Vec exact_terminal(n_sims);

    for (Size i = 0; i < n_sims; ++i) {
        Real s_euler = S0;
        Real w_total = 0.0;

        for (Size j = 0; j < n_steps; ++j) {
            Real z = rng.normal();
            s_euler = EulerMaruyama::step(gbm, static_cast<Real>(j) * dt, s_euler, dt, z);
            w_total += z;
        }

        euler_terminal[i] = s_euler;
        // Exact: use total Brownian increment
        exact_terminal[i] = gbm.evolve(0.0, S0, T, w_total / std::sqrt(static_cast<Real>(n_steps)));
    }

    // Means should be close
    REQUIRE_THAT(mean(euler_terminal), WithinAbs(mean(exact_terminal), 2.0));
}

TEST_CASE("Milstein has higher accuracy than Euler for GBM", "[discretization]") {
    // For GBM, Milstein correction is: + 0.5 * sigma * S * sigma * (dW^2 - dt)
    // = 0.5 * sigma^2 * S * (dW^2 - dt)
    // With few steps, Milstein should be closer to exact
    GBM gbm(0.05, 0.3);
    MersenneTwister rng(42);
    Real S0 = 100.0, T = 1.0;
    Size n_steps = 10;  // coarse discretization to see difference
    Real dt = T / static_cast<Real>(n_steps);

    Size n_sims = 10000;
    Real euler_err = 0.0, milstein_err = 0.0;

    for (Size i = 0; i < n_sims; ++i) {
        rng.seed(static_cast<uint64_t>(i));
        Vec normals(n_steps);
        rng.fill_normal(normals);

        Real s_euler = S0, s_milstein = S0;
        Real w_sum = 0.0;

        for (Size j = 0; j < n_steps; ++j) {
            Real t = static_cast<Real>(j) * dt;
            s_euler = EulerMaruyama::step(gbm, t, s_euler, dt, normals[j]);
            s_milstein = Milstein::step(gbm, t, s_milstein, dt, normals[j]);
            w_sum += std::sqrt(dt) * normals[j];
        }

        // Exact terminal value using total Brownian motion
        Real s_exact = S0 * std::exp((gbm.mu() - 0.5 * gbm.sigma() * gbm.sigma()) * T
                                      + gbm.sigma() * w_sum);

        euler_err += std::abs(s_euler - s_exact);
        milstein_err += std::abs(s_milstein - s_exact);
    }

    euler_err /= static_cast<Real>(n_sims);
    milstein_err /= static_cast<Real>(n_sims);

    // Milstein should have smaller average error
    REQUIRE(milstein_err < euler_err);
}

TEST_CASE("generate_path produces correct size", "[discretization]") {
    GBM gbm(0.05, 0.2);
    Vec normals(100);
    MersenneTwister rng(42);
    rng.fill_normal(normals);

    auto path = generate_path<EulerMaruyama>(gbm, 100.0, 1.0, 100, normals);
    REQUIRE(path.size() == 101);
    REQUIRE_THAT(path[0], WithinAbs(100.0, 1e-12));
}
