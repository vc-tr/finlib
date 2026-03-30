#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "qe/math/statistics.hpp"
#include <cmath>

using namespace qe;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

TEST_CASE("Mean computation", "[statistics]") {
    Vec data = {1.0, 2.0, 3.0, 4.0, 5.0};
    REQUIRE_THAT(mean(data), WithinAbs(3.0, 1e-12));
}

TEST_CASE("Variance and std_dev", "[statistics]") {
    Vec data = {2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0};
    // Sample variance = 4.571428...
    REQUIRE_THAT(variance(data), WithinAbs(4.571428571428571, 1e-10));
    REQUIRE_THAT(std_dev(data), WithinAbs(std::sqrt(4.571428571428571), 1e-10));
}

TEST_CASE("Quantile computation", "[statistics]") {
    Vec data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};

    SECTION("median") {
        REQUIRE_THAT(quantile(data, 0.5), WithinAbs(5.5, 1e-12));
    }

    SECTION("quartiles") {
        REQUIRE_THAT(quantile(data, 0.0), WithinAbs(1.0, 1e-12));
        REQUIRE_THAT(quantile(data, 1.0), WithinAbs(10.0, 1e-12));
        REQUIRE_THAT(quantile(data, 0.25), WithinAbs(3.25, 1e-12));
        REQUIRE_THAT(quantile(data, 0.75), WithinAbs(7.75, 1e-12));
    }
}

TEST_CASE("Empirical CDF", "[statistics]") {
    Vec data = {3.0, 1.0, 2.0};
    auto ecdf = empirical_cdf(data);

    REQUIRE(ecdf.x.size() == 3);
    REQUIRE_THAT(ecdf.x[0], WithinAbs(1.0, 1e-12));
    REQUIRE_THAT(ecdf.x[1], WithinAbs(2.0, 1e-12));
    REQUIRE_THAT(ecdf.x[2], WithinAbs(3.0, 1e-12));

    REQUIRE_THAT(ecdf.cdf[0], WithinAbs(1.0 / 3.0, 1e-12));
    REQUIRE_THAT(ecdf.cdf[1], WithinAbs(2.0 / 3.0, 1e-12));
    REQUIRE_THAT(ecdf.cdf[2], WithinAbs(1.0, 1e-12));
}

TEST_CASE("Covariance and correlation", "[statistics]") {
    Vec x = {1.0, 2.0, 3.0, 4.0, 5.0};
    Vec y = {2.0, 4.0, 6.0, 8.0, 10.0};

    SECTION("perfect positive correlation") {
        REQUIRE_THAT(correlation(x, y), WithinAbs(1.0, 1e-12));
    }

    SECTION("covariance of y = 2x") {
        // Cov(x, 2x) = 2 * Var(x) = 2 * 2.5 = 5.0
        REQUIRE_THAT(covariance(x, y), WithinAbs(5.0, 1e-12));
    }
}

TEST_CASE("Descriptive statistics", "[statistics]") {
    Vec data = {1.0, 2.0, 3.0, 4.0, 5.0};
    auto s = describe(data);

    REQUIRE(s.count == 5);
    REQUIRE_THAT(s.mean, WithinAbs(3.0, 1e-12));
    REQUIRE_THAT(s.min, WithinAbs(1.0, 1e-12));
    REQUIRE_THAT(s.max, WithinAbs(5.0, 1e-12));
}

TEST_CASE("RunningStats (Welford)", "[statistics]") {
    RunningStats rs;
    Vec data = {1.0, 2.0, 3.0, 4.0, 5.0};

    for (Real x : data) {
        rs.push(x);
    }

    REQUIRE(rs.count() == 5);
    REQUIRE_THAT(rs.mean(), WithinAbs(3.0, 1e-12));
    REQUIRE_THAT(rs.variance(), WithinAbs(variance(data), 1e-12));
    REQUIRE_THAT(rs.standard_error(), WithinAbs(std_dev(data) / std::sqrt(5.0), 1e-12));
}

TEST_CASE("Edge cases throw", "[statistics]") {
    Vec empty;
    Vec single = {1.0};

    REQUIRE_THROWS(mean(empty));
    REQUIRE_THROWS(variance(single));
    REQUIRE_THROWS(quantile(empty, 0.5));
}
