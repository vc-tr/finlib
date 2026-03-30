#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "qe/montecarlo/engine.hpp"
#include "qe/montecarlo/convergence.hpp"
#include "qe/models/black_scholes.hpp"
#include "qe/instruments/payoff.hpp"

using namespace qe;
using Catch::Matchers::WithinAbs;

static const Real S = 100.0, K = 100.0, r = 0.05, sigma = 0.2, T = 1.0;

TEST_CASE("MC European call converges to Black-Scholes", "[montecarlo]") {
    Real bs = bs_call(S, K, r, sigma, T);
    VanillaPayoff payoff(K, OptionType::Call);

    auto engine = MCEngineBuilder()
        .spot(S).rate(r).sigma(sigma).maturity(T)
        .num_paths(500000).seed(42)
        .build();

    auto result = engine.price(payoff);

    REQUIRE_THAT(result.price, WithinAbs(bs, 0.1));
    REQUIRE(result.std_error < 0.1);
    REQUIRE(result.ci_lo < bs);
    REQUIRE(result.ci_hi > bs);
}

TEST_CASE("MC European put converges to Black-Scholes", "[montecarlo]") {
    Real bs = bs_put(S, K, r, sigma, T);
    VanillaPayoff payoff(K, OptionType::Put);

    auto engine = MCEngineBuilder()
        .spot(S).rate(r).sigma(sigma).maturity(T)
        .num_paths(500000).seed(123)
        .build();

    auto result = engine.price(payoff);
    REQUIRE_THAT(result.price, WithinAbs(bs, 0.1));
}

TEST_CASE("Antithetic variates reduce standard error", "[montecarlo]") {
    VanillaPayoff payoff(K, OptionType::Call);

    auto engine = MCEngineBuilder()
        .spot(S).rate(r).sigma(sigma).maturity(T)
        .num_paths(100000).seed(42)
        .build();

    auto plain = engine.price(payoff);
    auto anti = engine.price_antithetic(payoff);

    // Antithetic should have lower standard error
    REQUIRE(anti.std_error < plain.std_error);

    // And still converge to correct price
    Real bs = bs_call(S, K, r, sigma, T);
    REQUIRE_THAT(anti.price, WithinAbs(bs, 0.2));
}

TEST_CASE("Control variate reduces variance", "[montecarlo]") {
    VanillaPayoff payoff(K, OptionType::Call);

    auto engine = MCEngineBuilder()
        .spot(S).rate(r).sigma(sigma).maturity(T)
        .num_paths(100000).seed(42)
        .build();

    // Use the underlying as control variate
    // E[S(T)] = S(0) * exp(r*T) under risk-neutral measure
    Real control_mean = S * std::exp(r * T);
    auto control_fn = [](Real ST) -> Real { return ST; };

    auto plain = engine.price(payoff);
    auto cv = engine.price_control_variate(payoff, control_fn, control_mean);

    // Control variate should have lower standard error
    REQUIRE(cv.std_error < plain.std_error);

    // Still converges
    Real bs = bs_call(S, K, r, sigma, T);
    REQUIRE_THAT(cv.price, WithinAbs(bs, 0.2));
}

TEST_CASE("Importance sampling for deep OTM call", "[montecarlo]") {
    // Deep OTM: K = 150 with S = 100
    Real K_otm = 150.0;
    VanillaPayoff payoff(K_otm, OptionType::Call);
    Real bs = bs_call(S, K_otm, r, sigma, T);

    auto engine = MCEngineBuilder()
        .spot(S).rate(r).sigma(sigma).maturity(T)
        .num_paths(200000).seed(42)
        .build();

    // Shift drift to sample more paths near K_otm
    // Optimal shift ≈ (ln(K/S) - (r - sigma^2/2)*T) / (sigma * sqrt(T))
    Real optimal_shift = (std::log(K_otm / S) - (r - 0.5 * sigma * sigma) * T)
                         / (sigma * std::sqrt(T));

    auto plain = engine.price(payoff);
    auto is = engine.price_importance_sampling(payoff, optimal_shift);

    // IS should give lower standard error for deep OTM
    REQUIRE(is.std_error < plain.std_error);

    // Both should be close to BS (IS should be closer)
    REQUIRE_THAT(is.price, WithinAbs(bs, 0.1));
}

TEST_CASE("MC builder pattern works", "[montecarlo]") {
    auto engine = MCEngineBuilder()
        .spot(50.0).rate(0.1).sigma(0.3).maturity(0.5)
        .num_paths(1000).num_steps(1).seed(99)
        .build();

    REQUIRE(engine.config().spot == 50.0);
    REQUIRE(engine.config().rate == 0.1);
    REQUIRE(engine.config().num_paths == 1000);
}

TEST_CASE("Convergence monitor tracks progress", "[montecarlo]") {
    Real bs = bs_call(S, K, r, sigma, T);
    ConvergenceMonitor monitor(bs);

    MersenneTwister rng(42);
    Real discount = std::exp(-r * T);

    for (Size i = 0; i < 1024; ++i) {
        Real z = rng.normal();
        Real ST = S * std::exp((r - 0.5 * sigma * sigma) * T + sigma * std::sqrt(T) * z);
        Real pv = discount * std::max(ST - K, 0.0);
        monitor.push(pv);
    }
    monitor.finalize();

    auto table = monitor.table();
    REQUIRE(table.size() >= 3);  // at least checkpoints at 64, 128, 256, 512, 1024

    // Standard error should decrease
    REQUIRE(table.back().std_error < table.front().std_error);

    // Final estimate should be reasonable
    REQUIRE_THAT(table.back().estimate, WithinAbs(bs, 2.0));
}
