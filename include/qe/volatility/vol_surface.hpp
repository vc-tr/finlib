#pragma once

#include "qe/core/types.hpp"
#include <vector>

namespace qe {

// Volatility surface: implied vol as a function of (strike, maturity)
// Constructed from a grid of implied vols
class VolSurface {
public:
    // strikes: vector of strikes (columns)
    // maturities: vector of maturities (rows)
    // vols: matrix of implied vols, vols[i][j] = vol(maturity_i, strike_j)
    VolSurface(const Vec& strikes, const Vec& maturities, const Mat& vols);

    // Interpolated implied vol at (K, T) using bilinear interpolation
    Real vol(Real K, Real T) const;

    // Get smile at a fixed maturity
    Vec smile(Real T) const;

    const Vec& strikes() const { return strikes_; }
    const Vec& maturities() const { return maturities_; }

private:
    Vec strikes_;
    Vec maturities_;
    Mat vols_;

    // Find bracketing indices
    std::pair<Size, Real> find_bracket(const Vec& grid, Real x) const;
};

} // namespace qe
