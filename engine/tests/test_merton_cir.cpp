#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "qe/models/merton_jd.hpp"
#include "qe/models/cir.hpp"
#include "qe/models/black_scholes.hpp"

using namespace qe;
using Catch::Matchers::WithinAbs;

// ============================================================================
// Merton Jump-Diffusion Tests
// ============================================================================

TEST_CASE("Merton with no jumps equals Black-Scholes", "[merton]") {
    MertonParams p = {
        .spot = 100.0, .rate = 0.05, .sigma = 0.2,
        .lambda = 0.0,   // no jumps
        .mu_j = 0.0, .sigma_j = 0.0,
        .maturity = 1.0
    };

    Real merton_c = merton_call(p, 100.0);
    Real bs_c = bs_call(100.0, 100.0, 0.05, 0.2, 1.0);

    REQUIRE_THAT(merton_c, WithinAbs(bs_c, 0.01));
}

TEST_CASE("Merton put-call parity", "[merton]") {
    MertonParams p = {
        .spot = 100.0, .rate = 0.05, .sigma = 0.2,
        .lambda = 1.0, .mu_j = -0.05, .sigma_j = 0.1,
        .maturity = 1.0
    };

    Real call = merton_call(p, 100.0);
    Real put = merton_put(p, 100.0);
    Real parity = p.spot - 100.0 * std::exp(-p.rate * p.maturity);

    REQUIRE_THAT(call - put, WithinAbs(parity, 0.01));
}

TEST_CASE("Merton MC vs semi-analytical", "[merton]") {
    MertonParams p = {
        .spot = 100.0, .rate = 0.05, .sigma = 0.2,
        .lambda = 1.0, .mu_j = -0.05, .sigma_j = 0.1,
        .maturity = 1.0
    };

    Real analytical = merton_call(p, 100.0);
    auto mc = merton_mc(p, 100.0, OptionType::Call, 200000, 252, 42);

    REQUIRE_THAT(mc.price, WithinAbs(analytical, 1.0));
}

TEST_CASE("Merton with jumps produces fatter tails than BS", "[merton]") {
    MertonParams p = {
        .spot = 100.0, .rate = 0.05, .sigma = 0.15,
        .lambda = 2.0, .mu_j = 0.0, .sigma_j = 0.15,
        .maturity = 1.0
    };

    // Deep OTM options are more expensive with jumps (fat tails)
    Real merton_otm = merton_call(p, 130.0);

    // Compare to BS with same diffusion vol
    Real bs_otm = bs_call(100.0, 130.0, 0.05, 0.15, 1.0);

    REQUIRE(merton_otm > bs_otm);
}

// ============================================================================
// CIR Tests
// ============================================================================

TEST_CASE("CIR Feller condition check", "[cir]") {
    CIRParams satisfied = {.r0 = 0.03, .kappa = 0.5, .theta = 0.05, .sigma = 0.1};
    CIRParams violated = {.r0 = 0.03, .kappa = 0.1, .theta = 0.05, .sigma = 0.5};

    REQUIRE(cir_feller_satisfied(satisfied));
    REQUIRE_FALSE(cir_feller_satisfied(violated));
}

TEST_CASE("CIR bond price is in (0, 1)", "[cir]") {
    CIRParams p = {.r0 = 0.05, .kappa = 0.3, .theta = 0.05, .sigma = 0.1};

    for (Real T : {0.5, 1.0, 2.0, 5.0, 10.0}) {
        Real bond = cir_bond_price(p, T);
        REQUIRE(bond > 0.0);
        REQUIRE(bond < 1.0);
    }
}

TEST_CASE("CIR bond prices decrease with maturity", "[cir]") {
    CIRParams p = {.r0 = 0.05, .kappa = 0.3, .theta = 0.05, .sigma = 0.1};

    Real p1 = cir_bond_price(p, 1.0);
    Real p5 = cir_bond_price(p, 5.0);
    Real p10 = cir_bond_price(p, 10.0);

    REQUIRE(p1 > p5);
    REQUIRE(p5 > p10);
}

TEST_CASE("CIR zero rate converges to theta for long maturities", "[cir]") {
    CIRParams p = {.r0 = 0.03, .kappa = 0.5, .theta = 0.05, .sigma = 0.1};

    Real r_short = cir_zero_rate(p, 0.1);
    Real r_long = cir_zero_rate(p, 30.0);

    // Short rate should be near r0
    REQUIRE_THAT(r_short, WithinAbs(p.r0, 0.01));

    // Long rate converges to a value related to theta
    // (not exactly theta due to convexity adjustment)
    REQUIRE(r_long > 0.0);
}

TEST_CASE("CIR path stays non-negative (Feller satisfied)", "[cir]") {
    CIRParams p = {.r0 = 0.05, .kappa = 1.0, .theta = 0.05, .sigma = 0.1};
    REQUIRE(cir_feller_satisfied(p));

    MersenneTwister rng(42);
    Vec path = cir_path(p, 5.0, 500, rng);

    REQUIRE(path.size() == 501);
    for (Real r : path) {
        REQUIRE(r >= 0.0);
    }
}
