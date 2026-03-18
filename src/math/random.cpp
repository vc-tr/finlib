#include "qe/math/random.hpp"
#include "qe/core/constants.hpp"
#include <cmath>
#include <stdexcept>

namespace qe {

// ============================================================================
// MersenneTwister
// ============================================================================

MersenneTwister::MersenneTwister(uint64_t seed) : gen_(seed) {}

Real MersenneTwister::uniform() {
    return std::uniform_real_distribution<Real>(0.0, 1.0)(gen_);
}

Real MersenneTwister::normal() {
    // Box-Muller transform: generates two independent standard normals
    // from two uniform random variables. Cache the spare for next call.
    if (has_spare_) {
        has_spare_ = false;
        return spare_;
    }

    Real u1, u2;
    do {
        u1 = uniform();
        u2 = uniform();
    } while (u1 <= 0.0);  // u1 must be > 0 for log

    Real mag = std::sqrt(-2.0 * std::log(u1));
    Real angle = 2.0 * PI * u2;

    spare_ = mag * std::sin(angle);
    has_spare_ = true;

    return mag * std::cos(angle);
}

void MersenneTwister::fill_normal(Vec& out) {
    for (auto& x : out) {
        x = normal();
    }
}

void MersenneTwister::fill_uniform(Vec& out) {
    for (auto& x : out) {
        x = uniform();
    }
}

void MersenneTwister::seed(uint64_t s) {
    gen_.seed(s);
    has_spare_ = false;
}

// ============================================================================
// SobolSequence
// ============================================================================

SobolSequence::SobolSequence(Size dimension) : dimension_(dimension), index_(0) {
    if (dimension_ == 0) {
        throw std::invalid_argument("Sobol dimension must be >= 1");
    }
    if (dimension_ > 21) {
        throw std::invalid_argument("Sobol implementation supports up to 21 dimensions");
    }
    init_direction_numbers();
}

void SobolSequence::init_direction_numbers() {
    // Joe-Kuo direction numbers for dimensions 1-21
    // Dimension 1 uses the Van der Corput sequence (bit-reversal)
    // Higher dimensions use primitive polynomials and initial direction numbers

    // Primitive polynomial degrees and coefficients (s, a) for dimensions 2-21
    // From Joe & Kuo (2010)
    struct PolyInfo {
        uint32_t s;    // degree
        uint32_t a;    // coefficients (binary)
        std::vector<uint32_t> m;  // initial direction numbers
    };

    std::vector<PolyInfo> polys = {
        {1, 0, {1}},                          // dim 2
        {2, 1, {1, 1}},                       // dim 3
        {3, 1, {1, 1, 1}},                    // dim 4
        {3, 2, {1, 3, 1}},                    // dim 5
        {4, 1, {1, 1, 1, 1}},                 // dim 6
        {4, 4, {1, 3, 5, 1}},                 // dim 7
        {5, 2, {1, 1, 5, 3, 1}},              // dim 8
        {5, 4, {1, 3, 1, 7, 5}},              // dim 9
        {5, 7, {1, 3, 7, 7, 1}},              // dim 10
        {5, 11, {1, 1, 3, 3, 9}},             // dim 11
        {5, 13, {1, 3, 5, 13, 1}},            // dim 12
        {5, 14, {1, 1, 7, 11, 1}},            // dim 13
        {6, 1, {1, 1, 1, 1, 1, 1}},           // dim 14
        {6, 13, {1, 3, 5, 5, 11, 1}},         // dim 15
        {6, 16, {1, 1, 7, 3, 29, 1}},         // dim 16
        {6, 19, {1, 3, 7, 13, 3, 1}},         // dim 17
        {6, 22, {1, 1, 1, 9, 5, 21}},         // dim 18
        {6, 25, {1, 3, 1, 3, 11, 27}},        // dim 19
        {7, 1, {1, 1, 1, 1, 1, 1, 1}},        // dim 20
        {7, 4, {1, 3, 3, 5, 7, 11, 1}},       // dim 21
    };

    constexpr uint32_t BITS = 32;
    direction_numbers_.resize(dimension_);

    // Dimension 1: Van der Corput sequence
    direction_numbers_[0].resize(BITS);
    for (uint32_t i = 0; i < BITS; ++i) {
        direction_numbers_[0][i] = uint32_t(1) << (BITS - 1 - i);
    }

    // Higher dimensions: use primitive polynomials
    for (Size d = 1; d < dimension_; ++d) {
        const auto& poly = polys[d - 1];
        uint32_t s = poly.s;

        direction_numbers_[d].resize(BITS);

        // Set initial direction numbers (shifted to high bits)
        for (uint32_t i = 0; i < s; ++i) {
            direction_numbers_[d][i] = poly.m[i] << (BITS - 1 - i);
        }

        // Generate remaining direction numbers via recurrence
        for (uint32_t i = s; i < BITS; ++i) {
            uint32_t v = direction_numbers_[d][i - s];
            v ^= (v >> s);

            uint32_t a = poly.a;
            for (uint32_t k = 1; k < s; ++k) {
                if (a & (uint32_t(1) << (s - 1 - k))) {
                    v ^= direction_numbers_[d][i - k];
                }
            }
            direction_numbers_[d][i] = v;
        }
    }
}

Real SobolSequence::to_real(uint32_t val) const {
    return static_cast<Real>(val) / static_cast<Real>(uint64_t(1) << 32);
}

Vec SobolSequence::next() {
    Vec point(dimension_);

    if (index_ == 0) {
        // First point is origin
        for (Size d = 0; d < dimension_; ++d) {
            point[d] = 0.0;
        }
    } else {
        for (Size d = 0; d < dimension_; ++d) {
            // XOR with direction number
            // We need to maintain state; use index-based generation
            // Recompute from scratch for simplicity (could optimize with state)
            uint32_t result = 0;
            uint32_t idx = index_;
            for (uint32_t bit = 0; idx != 0; ++bit, idx >>= 1) {
                if (idx & 1u) {
                    result ^= direction_numbers_[d][bit];
                }
            }
            point[d] = to_real(result);
        }
    }

    ++index_;
    return point;
}

Mat SobolSequence::generate(Size n) {
    Mat points(n);
    for (Size i = 0; i < n; ++i) {
        points[i] = next();
    }
    return points;
}

void SobolSequence::reset() {
    index_ = 0;
}

} // namespace qe
