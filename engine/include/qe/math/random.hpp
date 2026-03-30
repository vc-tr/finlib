#pragma once

#include "qe/core/types.hpp"
#include <cstdint>
#include <random>

namespace qe {

// Thin wrapper around std::mt19937_64 with convenient normal/uniform generation
class MersenneTwister {
public:
    explicit MersenneTwister(uint64_t seed = 42);

    // Uniform [0, 1)
    Real uniform();

    // Standard normal via Box-Muller transform
    Real normal();

    // Fill vector with standard normal deviates
    void fill_normal(Vec& out);

    // Fill vector with uniform deviates
    void fill_uniform(Vec& out);

    void seed(uint64_t s);

private:
    std::mt19937_64 gen_;
    bool has_spare_ = false;
    Real spare_ = 0.0;
};

// Sobol quasi-random sequence generator (dimension 1)
// Implements the Joe-Kuo direction numbers for low-discrepancy sequences
class SobolSequence {
public:
    explicit SobolSequence(Size dimension = 1);

    // Get next point in [0, 1)^dimension
    Vec next();

    // Get next N points
    Mat generate(Size n);

    void reset();

private:
    Size dimension_;
    uint32_t index_;

    // Direction numbers for each dimension
    std::vector<std::vector<uint32_t>> direction_numbers_;

    void init_direction_numbers();
    Real to_real(uint32_t val) const;
};

} // namespace qe
