#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "qe/pde/fdm_solver.hpp"
#include "qe/models/black_scholes.hpp"

using namespace qe;
using Catch::Matchers::WithinAbs;

static FDMConfig make_config(OptionType type) {
    return {
        .spot = 100.0,
        .strike = 100.0,
        .rate = 0.05,
        .sigma = 0.2,
        .maturity = 1.0,
        .type = type,
        .n_spot = 400,
        .n_time = 2000,
        .spot_min_mult = 0.1,
        .spot_max_mult = 4.0
    };
}

TEST_CASE("Crank-Nicolson matches BS call price", "[fdm]") {
    auto config = make_config(OptionType::Call);
    FDMSolver solver(config);

    auto result = solver.solve(FDMScheme::CrankNicolson);
    Real bs = bs_call(100.0, 100.0, 0.05, 0.2, 1.0);

    REQUIRE_THAT(result.price, WithinAbs(bs, 0.25));
}

TEST_CASE("Crank-Nicolson matches BS put price", "[fdm]") {
    auto config = make_config(OptionType::Put);
    FDMSolver solver(config);

    auto result = solver.solve(FDMScheme::CrankNicolson);
    Real bs = bs_put(100.0, 100.0, 0.05, 0.2, 1.0);

    REQUIRE_THAT(result.price, WithinAbs(bs, 0.15));
}

TEST_CASE("Implicit scheme matches BS call price", "[fdm]") {
    auto config = make_config(OptionType::Call);
    FDMSolver solver(config);

    auto result = solver.solve(FDMScheme::Implicit);
    Real bs = bs_call(100.0, 100.0, 0.05, 0.2, 1.0);

    REQUIRE_THAT(result.price, WithinAbs(bs, 0.2));
}

TEST_CASE("Explicit scheme matches BS call price", "[fdm]") {
    // Use finer grid to satisfy CFL condition
    FDMConfig config = {
        .spot = 100.0,
        .strike = 100.0,
        .rate = 0.05,
        .sigma = 0.2,
        .maturity = 1.0,
        .type = OptionType::Call,
        .n_spot = 100,
        .n_time = 10000,  // many time steps for stability
        .spot_min_mult = 0.1,
        .spot_max_mult = 4.0
    };
    FDMSolver solver(config);

    auto result = solver.solve(FDMScheme::Explicit);
    Real bs = bs_call(100.0, 100.0, 0.05, 0.2, 1.0);

    REQUIRE_THAT(result.price, WithinAbs(bs, 0.5));
}

TEST_CASE("FDM delta matches BS delta", "[fdm]") {
    auto config = make_config(OptionType::Call);
    FDMSolver solver(config);

    auto result = solver.solve(FDMScheme::CrankNicolson);
    Real bs_d = bs_delta(100.0, 100.0, 0.05, 0.2, 1.0, OptionType::Call);

    REQUIRE_THAT(result.delta, WithinAbs(bs_d, 0.02));
}

TEST_CASE("FDM gamma matches BS gamma", "[fdm]") {
    auto config = make_config(OptionType::Call);
    FDMSolver solver(config);

    auto result = solver.solve(FDMScheme::CrankNicolson);
    Real bs_g = bs_gamma(100.0, 100.0, 0.05, 0.2, 1.0);

    REQUIRE_THAT(result.gamma, WithinAbs(bs_g, 0.005));
}

TEST_CASE("All schemes agree on ITM call", "[fdm]") {
    FDMConfig config = {
        .spot = 120.0,
        .strike = 100.0,
        .rate = 0.05,
        .sigma = 0.2,
        .maturity = 1.0,
        .type = OptionType::Call,
        .n_spot = 200,
        .n_time = 5000,
        .spot_min_mult = 0.1,
        .spot_max_mult = 3.0
    };
    FDMSolver solver(config);

    auto expl = solver.solve(FDMScheme::Explicit);
    auto impl = solver.solve(FDMScheme::Implicit);
    auto cn = solver.solve(FDMScheme::CrankNicolson);
    Real bs = bs_call(120.0, 100.0, 0.05, 0.2, 1.0);

    // All should be close to BS
    REQUIRE_THAT(cn.price, WithinAbs(bs, 0.5));
    REQUIRE_THAT(impl.price, WithinAbs(bs, 0.5));
    REQUIRE_THAT(expl.price, WithinAbs(bs, 1.0));
}

TEST_CASE("FDM produces full option value curve", "[fdm]") {
    auto config = make_config(OptionType::Call);
    FDMSolver solver(config);

    auto result = solver.solve(FDMScheme::CrankNicolson);

    REQUIRE(result.spot_grid.size() == config.n_spot + 1);
    REQUIRE(result.option_values.size() == config.n_spot + 1);

    // Option values should be non-negative
    for (Real v : result.option_values) {
        REQUIRE(v >= -0.01);  // allow tiny numerical noise
    }
}
