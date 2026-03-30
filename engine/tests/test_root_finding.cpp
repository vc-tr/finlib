#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "qe/math/root_finding.hpp"
#include <cmath>

using namespace qe;
using Catch::Matchers::WithinAbs;

TEST_CASE("Newton-Raphson finds sqrt(2)", "[root_finding]") {
    // f(x) = x^2 - 2, f'(x) = 2x
    auto f = [](Real x) { return x * x - 2.0; };
    auto df = [](Real x) { return 2.0 * x; };

    auto result = newton_raphson(f, df, 1.5);
    REQUIRE(result.converged);
    REQUIRE_THAT(result.root, WithinAbs(std::sqrt(2.0), 1e-12));
}

TEST_CASE("Newton-Raphson finds cos root", "[root_finding]") {
    // f(x) = cos(x), root at pi/2
    auto f = [](Real x) { return std::cos(x); };
    auto df = [](Real x) { return -std::sin(x); };

    auto result = newton_raphson(f, df, 1.0);
    REQUIRE(result.converged);
    REQUIRE_THAT(result.root, WithinAbs(M_PI / 2.0, 1e-12));
}

TEST_CASE("Brent's method finds cubic root", "[root_finding]") {
    // f(x) = x^3 - x - 2, real root near 1.521
    auto f = [](Real x) { return x * x * x - x - 2.0; };

    auto result = brent(f, 1.0, 2.0);
    REQUIRE(result.converged);
    REQUIRE_THAT(f(result.root), WithinAbs(0.0, 1e-10));
}

TEST_CASE("Brent's method throws for same-sign brackets", "[root_finding]") {
    auto f = [](Real x) { return x * x + 1.0; };  // always positive
    REQUIRE_THROWS(brent(f, 0.0, 1.0));
}

TEST_CASE("Bisection finds simple root", "[root_finding]") {
    auto f = [](Real x) { return x * x - 4.0; };  // roots at +/- 2

    auto result = bisection(f, 0.0, 3.0);
    REQUIRE(result.converged);
    REQUIRE_THAT(result.root, WithinAbs(2.0, 1e-10));
}

TEST_CASE("Bisection throws for same-sign brackets", "[root_finding]") {
    auto f = [](Real x) { return x * x + 1.0; };
    REQUIRE_THROWS(bisection(f, 0.0, 1.0));
}

TEST_CASE("All methods agree on exp(x) - 3", "[root_finding]") {
    // root at ln(3) ≈ 1.0986
    auto f = [](Real x) { return std::exp(x) - 3.0; };
    auto df = [](Real x) { return std::exp(x); };
    Real expected = std::log(3.0);

    auto nr = newton_raphson(f, df, 1.0);
    auto br = brent(f, 0.0, 2.0);
    auto bi = bisection(f, 0.0, 2.0);

    REQUIRE(nr.converged);
    REQUIRE(br.converged);
    REQUIRE(bi.converged);

    REQUIRE_THAT(nr.root, WithinAbs(expected, 1e-10));
    REQUIRE_THAT(br.root, WithinAbs(expected, 1e-10));
    REQUIRE_THAT(bi.root, WithinAbs(expected, 1e-10));

    // Newton should converge fastest (quadratic)
    REQUIRE(nr.iterations <= br.iterations);
}
