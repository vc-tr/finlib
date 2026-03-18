#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "qe/models/sabr.hpp"
#include "qe/models/black_scholes.hpp"

using namespace qe;
using Catch::Matchers::WithinAbs;

TEST_CASE("SABR ATM implied vol equals alpha when beta=1, nu=0", "[sabr]") {
    // With no vol-of-vol and beta=1 (lognormal), ATM vol ≈ alpha
    SABRParams p = {.alpha = 0.2, .beta = 1.0, .rho = 0.0, .nu = 0.001};

    Real vol = sabr_implied_vol(p, 100.0, 100.0, 1.0);
    REQUIRE_THAT(vol, WithinAbs(0.2, 0.01));
}

TEST_CASE("SABR implied vol is positive", "[sabr]") {
    SABRParams p = {.alpha = 0.2, .beta = 0.5, .rho = -0.3, .nu = 0.4};

    for (Real K : {80.0, 90.0, 100.0, 110.0, 120.0}) {
        Real vol = sabr_implied_vol(p, 100.0, K, 1.0);
        REQUIRE(vol > 0.0);
    }
}

TEST_CASE("SABR produces a smile with negative rho", "[sabr]") {
    SABRParams p = {.alpha = 0.2, .beta = 0.5, .rho = -0.5, .nu = 0.5};

    Real vol_otm_put = sabr_implied_vol(p, 100.0, 85.0, 1.0);
    Real vol_atm = sabr_implied_vol(p, 100.0, 100.0, 1.0);
    Real vol_otm_call = sabr_implied_vol(p, 100.0, 115.0, 1.0);

    // Negative rho creates a skew: left side higher
    REQUIRE(vol_otm_put > vol_atm);
}

TEST_CASE("SABR call price is positive and reasonable", "[sabr]") {
    SABRParams p = {.alpha = 0.2, .beta = 0.7, .rho = -0.3, .nu = 0.4};

    Real call = sabr_call(p, 100.0, 100.0, 0.05, 1.0);
    Real put = sabr_put(p, 100.0, 100.0, 0.05, 1.0);

    REQUIRE(call > 0.0);
    REQUIRE(put > 0.0);

    // Put-call parity
    Real discount = std::exp(-0.05 * 1.0);
    Real parity = 100.0 * discount - 100.0 * discount;  // F = K for ATM fwd
    REQUIRE_THAT(call - put, WithinAbs(parity, 1.0));
}

TEST_CASE("SABR nu=0 reduces to constant vol", "[sabr]") {
    SABRParams p = {.alpha = 0.25, .beta = 1.0, .rho = 0.0, .nu = 0.0};

    // All strikes should have same implied vol
    Real vol_80 = sabr_implied_vol(p, 100.0, 80.0, 1.0);
    Real vol_100 = sabr_implied_vol(p, 100.0, 100.0, 1.0);
    Real vol_120 = sabr_implied_vol(p, 100.0, 120.0, 1.0);

    REQUIRE_THAT(vol_80, WithinAbs(vol_100, 0.02));
    REQUIRE_THAT(vol_120, WithinAbs(vol_100, 0.02));
}
