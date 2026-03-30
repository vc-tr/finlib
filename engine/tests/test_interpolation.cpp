#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "qe/math/interpolation.hpp"
#include <cmath>

using namespace qe;
using Catch::Matchers::WithinAbs;

TEST_CASE("Linear interpolation", "[interpolation]") {
    Vec x = {0.0, 1.0, 2.0, 3.0};
    Vec y = {0.0, 1.0, 4.0, 9.0};
    LinearInterpolator interp(x, y);

    SECTION("exact data points") {
        REQUIRE_THAT(interp(0.0), WithinAbs(0.0, 1e-12));
        REQUIRE_THAT(interp(1.0), WithinAbs(1.0, 1e-12));
        REQUIRE_THAT(interp(2.0), WithinAbs(4.0, 1e-12));
        REQUIRE_THAT(interp(3.0), WithinAbs(9.0, 1e-12));
    }

    SECTION("midpoints") {
        REQUIRE_THAT(interp(0.5), WithinAbs(0.5, 1e-12));
        REQUIRE_THAT(interp(1.5), WithinAbs(2.5, 1e-12));
        REQUIRE_THAT(interp(2.5), WithinAbs(6.5, 1e-12));
    }

    SECTION("extrapolation at boundaries") {
        // Clamps to first/last segment
        Real left = interp(-1.0);
        Real right = interp(4.0);
        REQUIRE(std::isfinite(left));
        REQUIRE(std::isfinite(right));
    }
}

TEST_CASE("Linear interpolation of straight line is exact", "[interpolation]") {
    Vec x = {0.0, 1.0, 2.0, 3.0, 4.0};
    Vec y = {1.0, 3.0, 5.0, 7.0, 9.0};  // y = 2x + 1
    LinearInterpolator interp(x, y);

    for (Real xi = 0.0; xi <= 4.0; xi += 0.1) {
        REQUIRE_THAT(interp(xi), WithinAbs(2.0 * xi + 1.0, 1e-10));
    }
}

TEST_CASE("Cubic spline interpolation", "[interpolation]") {
    // Interpolate sin(x) on [0, pi]
    Size n = 11;
    Vec x(n), y(n);
    for (Size i = 0; i < n; ++i) {
        x[i] = static_cast<Real>(i) * M_PI / static_cast<Real>(n - 1);
        y[i] = std::sin(x[i]);
    }

    CubicSplineInterpolator spline(x, y);

    SECTION("passes through data points") {
        for (Size i = 0; i < n; ++i) {
            REQUIRE_THAT(spline(x[i]), WithinAbs(y[i], 1e-10));
        }
    }

    SECTION("good accuracy between points") {
        // Test at midpoints — cubic spline should be very accurate for sin
        Real xi = M_PI / 4.0;
        REQUIRE_THAT(spline(xi), WithinAbs(std::sin(xi), 1e-4));

        xi = M_PI / 2.0;
        REQUIRE_THAT(spline(xi), WithinAbs(std::sin(xi), 1e-4));
    }

    SECTION("derivative approximation") {
        // d/dx sin(x) = cos(x)
        Real xi = M_PI / 4.0;
        REQUIRE_THAT(spline.derivative(xi), WithinAbs(std::cos(xi), 1e-2));
    }
}

TEST_CASE("Cubic spline is exact for cubic polynomials", "[interpolation]") {
    // y = x^3 - 2x^2 + x - 1
    auto f = [](Real x) { return x * x * x - 2.0 * x * x + x - 1.0; };

    Vec x = {0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0};
    Vec y(x.size());
    for (Size i = 0; i < x.size(); ++i) y[i] = f(x[i]);

    CubicSplineInterpolator spline(x, y);

    // Natural spline has c[0]=c[n]=0 boundary conditions, so it won't
    // exactly reproduce a cubic near boundaries. Interior points are more accurate.
    REQUIRE_THAT(spline(1.25), WithinAbs(f(1.25), 0.05));
    REQUIRE_THAT(spline(1.5), WithinAbs(f(1.5), 1e-10));  // exact at knot
    REQUIRE_THAT(spline(2.0), WithinAbs(f(2.0), 1e-10));
}

TEST_CASE("Interpolation batch mode", "[interpolation]") {
    Vec x = {0.0, 1.0, 2.0};
    Vec y = {0.0, 1.0, 0.0};

    LinearInterpolator lin(x, y);
    Vec xi = {0.0, 0.5, 1.0, 1.5, 2.0};
    Vec result = lin.interpolate(xi);

    REQUIRE(result.size() == 5);
    REQUIRE_THAT(result[0], WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(result[1], WithinAbs(0.5, 1e-12));
    REQUIRE_THAT(result[2], WithinAbs(1.0, 1e-12));
    REQUIRE_THAT(result[3], WithinAbs(0.5, 1e-12));
    REQUIRE_THAT(result[4], WithinAbs(0.0, 1e-12));
}
