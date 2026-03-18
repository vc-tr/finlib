#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "qe/models/black_scholes.hpp"
#include <cmath>

using namespace qe;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

TEST_CASE("Normal CDF", "[black_scholes]") {
    SECTION("known values") {
        REQUIRE_THAT(norm_cdf(0.0), WithinAbs(0.5, 1e-7));
        REQUIRE_THAT(norm_cdf(1.0), WithinAbs(0.8413447, 1e-5));
        REQUIRE_THAT(norm_cdf(-1.0), WithinAbs(0.1586553, 1e-5));
        REQUIRE_THAT(norm_cdf(2.0), WithinAbs(0.9772499, 1e-5));
        REQUIRE_THAT(norm_cdf(-2.0), WithinAbs(0.0227501, 1e-5));
    }

    SECTION("symmetry: Phi(x) + Phi(-x) = 1") {
        for (Real x = -3.0; x <= 3.0; x += 0.5) {
            REQUIRE_THAT(norm_cdf(x) + norm_cdf(-x), WithinAbs(1.0, 1e-7));
        }
    }

    SECTION("extreme values") {
        REQUIRE_THAT(norm_cdf(10.0), WithinAbs(1.0, 1e-10));
        REQUIRE_THAT(norm_cdf(-10.0), WithinAbs(0.0, 1e-10));
    }
}

TEST_CASE("Black-Scholes call price (textbook example)", "[black_scholes]") {
    // S=100, K=100, r=5%, sigma=20%, T=1
    // Expected call price ≈ 10.4506
    Real S = 100.0, K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    Real call = bs_call(S, K, r, sigma, T);
    REQUIRE_THAT(call, WithinAbs(10.4506, 0.01));
}

TEST_CASE("Black-Scholes put price (textbook example)", "[black_scholes]") {
    Real S = 100.0, K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    Real put = bs_put(S, K, r, sigma, T);
    // put ≈ 5.5735
    REQUIRE_THAT(put, WithinAbs(5.5735, 0.01));
}

TEST_CASE("Put-call parity: C - P = S - K*exp(-rT)", "[black_scholes]") {
    Real S = 100.0, K = 105.0, r = 0.05, sigma = 0.25, T = 0.5;
    Real call = bs_call(S, K, r, sigma, T);
    Real put = bs_put(S, K, r, sigma, T);
    Real parity = S - K * std::exp(-r * T);

    REQUIRE_THAT(call - put, WithinAbs(parity, 1e-10));
}

TEST_CASE("Put-call parity holds for various parameters", "[black_scholes]") {
    for (Real S : {80.0, 100.0, 120.0}) {
        for (Real K : {90.0, 100.0, 110.0}) {
            for (Real T : {0.25, 0.5, 1.0, 2.0}) {
                Real r = 0.05, sigma = 0.3;
                Real call = bs_call(S, K, r, sigma, T);
                Real put = bs_put(S, K, r, sigma, T);
                Real parity = S - K * std::exp(-r * T);
                REQUIRE_THAT(call - put, WithinAbs(parity, 1e-8));
            }
        }
    }
}

TEST_CASE("Black-Scholes at expiration", "[black_scholes]") {
    REQUIRE_THAT(bs_call(110.0, 100.0, 0.05, 0.2, 0.0), WithinAbs(10.0, 1e-12));
    REQUIRE_THAT(bs_call(90.0, 100.0, 0.05, 0.2, 0.0), WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(bs_put(90.0, 100.0, 0.05, 0.2, 0.0), WithinAbs(10.0, 1e-12));
    REQUIRE_THAT(bs_put(110.0, 100.0, 0.05, 0.2, 0.0), WithinAbs(0.0, 1e-12));
}

TEST_CASE("Black-Scholes Greeks", "[black_scholes]") {
    Real S = 100.0, K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;

    SECTION("call delta in (0, 1)") {
        Real delta = bs_delta(S, K, r, sigma, T, OptionType::Call);
        REQUIRE(delta > 0.0);
        REQUIRE(delta < 1.0);
        // ATM call delta ≈ 0.6368
        REQUIRE_THAT(delta, WithinAbs(0.6368, 0.005));
    }

    SECTION("put delta in (-1, 0)") {
        Real delta = bs_delta(S, K, r, sigma, T, OptionType::Put);
        REQUIRE(delta > -1.0);
        REQUIRE(delta < 0.0);
    }

    SECTION("put-call delta parity: delta_call - delta_put = 1") {
        Real dc = bs_delta(S, K, r, sigma, T, OptionType::Call);
        Real dp = bs_delta(S, K, r, sigma, T, OptionType::Put);
        REQUIRE_THAT(dc - dp, WithinAbs(1.0, 1e-5));
    }

    SECTION("gamma is same for call and put") {
        Real gamma = bs_gamma(S, K, r, sigma, T);
        REQUIRE(gamma > 0.0);
    }

    SECTION("vega is positive") {
        Real vega = bs_vega(S, K, r, sigma, T);
        REQUIRE(vega > 0.0);
    }

    SECTION("call theta is negative (time decay)") {
        Real theta = bs_theta(S, K, r, sigma, T, OptionType::Call);
        REQUIRE(theta < 0.0);
    }

    SECTION("call rho is positive") {
        Real rho = bs_rho(S, K, r, sigma, T, OptionType::Call);
        REQUIRE(rho > 0.0);
    }

    SECTION("put rho is negative") {
        Real rho = bs_rho(S, K, r, sigma, T, OptionType::Put);
        REQUIRE(rho < 0.0);
    }
}

TEST_CASE("Greeks numerical verification", "[black_scholes]") {
    // Verify analytical Greeks match central finite differences
    Real S = 100.0, K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;

    SECTION("delta ≈ dPrice/dS") {
        Real eps = 0.01;
        Real price_up = bs_call(S + eps, K, r, sigma, T);
        Real price_dn = bs_call(S - eps, K, r, sigma, T);
        Real fd_delta = (price_up - price_dn) / (2.0 * eps);
        Real an_delta = bs_delta(S, K, r, sigma, T, OptionType::Call);
        REQUIRE_THAT(fd_delta, WithinAbs(an_delta, 1e-3));
    }

    SECTION("gamma ≈ d2Price/dS2") {
        Real eps = 1.0;  // larger eps for 2nd derivative stability
        Real price_up = bs_call(S + eps, K, r, sigma, T);
        Real price_mid = bs_call(S, K, r, sigma, T);
        Real price_dn = bs_call(S - eps, K, r, sigma, T);
        Real fd_gamma = (price_up - 2.0 * price_mid + price_dn) / (eps * eps);
        Real an_gamma = bs_gamma(S, K, r, sigma, T);
        REQUIRE_THAT(fd_gamma, WithinAbs(an_gamma, 1e-3));
    }

    SECTION("vega ≈ dPrice/dSigma") {
        Real eps = 0.001;
        Real price_up = bs_call(S, K, r, sigma + eps, T);
        Real price_dn = bs_call(S, K, r, sigma - eps, T);
        Real fd_vega = (price_up - price_dn) / (2.0 * eps);
        Real an_vega = bs_vega(S, K, r, sigma, T);
        REQUIRE_THAT(fd_vega, WithinAbs(an_vega, 0.1));
    }
}
