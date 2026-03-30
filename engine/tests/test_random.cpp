#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "qe/math/random.hpp"
#include "qe/math/statistics.hpp"
#include <cmath>

using namespace qe;
using Catch::Matchers::WithinAbs;

TEST_CASE("MersenneTwister uniform distribution", "[random]") {
    MersenneTwister rng(42);

    SECTION("generates values in [0, 1)") {
        for (int i = 0; i < 10000; ++i) {
            Real u = rng.uniform();
            REQUIRE(u >= 0.0);
            REQUIRE(u < 1.0);
        }
    }

    SECTION("mean is approximately 0.5") {
        Vec samples(100000);
        rng.fill_uniform(samples);
        Real m = mean(samples);
        REQUIRE_THAT(m, WithinAbs(0.5, 0.01));
    }
}

TEST_CASE("MersenneTwister normal distribution (Box-Muller)", "[random]") {
    MersenneTwister rng(123);

    SECTION("mean is approximately 0") {
        Vec samples(100000);
        rng.fill_normal(samples);
        Real m = mean(samples);
        REQUIRE_THAT(m, WithinAbs(0.0, 0.02));
    }

    SECTION("std dev is approximately 1") {
        Vec samples(100000);
        rng.fill_normal(samples);
        Real s = std_dev(samples);
        REQUIRE_THAT(s, WithinAbs(1.0, 0.02));
    }
}

TEST_CASE("MersenneTwister reproducibility", "[random]") {
    MersenneTwister rng1(42);
    MersenneTwister rng2(42);

    for (int i = 0; i < 100; ++i) {
        REQUIRE(rng1.normal() == rng2.normal());
    }
}

TEST_CASE("SobolSequence 1D", "[random]") {
    SobolSequence sobol(1);

    SECTION("first point is origin") {
        auto pt = sobol.next();
        REQUIRE(pt.size() == 1);
        REQUIRE_THAT(pt[0], WithinAbs(0.0, 1e-10));
    }

    SECTION("generates low-discrepancy sequence") {
        auto points = sobol.generate(1024);
        REQUIRE(points.size() == 1024);

        // Check uniformity: mean should be close to 0.5
        Vec vals(points.size());
        for (Size i = 0; i < points.size(); ++i) {
            vals[i] = points[i][0];
        }
        // Skip the first point (origin)
        Vec vals_skip(vals.begin() + 1, vals.end());
        Real m = mean(vals_skip);
        REQUIRE_THAT(m, WithinAbs(0.5, 0.02));
    }
}

TEST_CASE("SobolSequence multi-dimensional", "[random]") {
    SobolSequence sobol(3);

    auto points = sobol.generate(256);
    REQUIRE(points.size() == 256);
    REQUIRE(points[0].size() == 3);

    // All values should be in [0, 1)
    for (const auto& pt : points) {
        for (Real v : pt) {
            REQUIRE(v >= 0.0);
            REQUIRE(v < 1.0);
        }
    }
}

TEST_CASE("SobolSequence reset", "[random]") {
    SobolSequence sobol(2);
    auto points1 = sobol.generate(10);
    sobol.reset();
    auto points2 = sobol.generate(10);

    for (Size i = 0; i < 10; ++i) {
        for (Size d = 0; d < 2; ++d) {
            REQUIRE(points1[i][d] == points2[i][d]);
        }
    }
}
