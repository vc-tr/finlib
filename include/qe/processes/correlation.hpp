#pragma once

#include "qe/core/types.hpp"
#include "qe/math/random.hpp"

namespace qe {

// Generate two correlated standard normal deviates from independent normals
// W1 = Z1, W2 = rho*Z1 + sqrt(1-rho^2)*Z2
struct CorrelatedNormals {
    Real w1;
    Real w2;
};

inline CorrelatedNormals correlate(Real z1, Real z2, Real rho) {
    return {z1, rho * z1 + std::sqrt(1.0 - rho * rho) * z2};
}

} // namespace qe
