#include "qe/volatility/vol_surface.hpp"
#include <algorithm>
#include <stdexcept>
#include <cmath>

namespace qe {

VolSurface::VolSurface(const Vec& strikes, const Vec& maturities, const Mat& vols)
    : strikes_(strikes), maturities_(maturities), vols_(vols) {
    if (maturities.size() != vols.size())
        throw std::invalid_argument("VolSurface: maturity/vol size mismatch");
    for (const auto& row : vols) {
        if (row.size() != strikes.size())
            throw std::invalid_argument("VolSurface: strike/vol size mismatch");
    }
}

std::pair<Size, Real> VolSurface::find_bracket(const Vec& grid, Real x) const {
    if (x <= grid.front()) return {0, 0.0};
    if (x >= grid.back()) return {grid.size() - 2, 1.0};

    auto it = std::lower_bound(grid.begin(), grid.end(), x);
    Size idx = static_cast<Size>(it - grid.begin());
    if (idx == 0) idx = 1;
    Real frac = (x - grid[idx - 1]) / (grid[idx] - grid[idx - 1]);
    return {idx - 1, frac};
}

Real VolSurface::vol(Real K, Real T) const {
    auto [ki, kf] = find_bracket(strikes_, K);
    auto [ti, tf] = find_bracket(maturities_, T);

    // Bilinear interpolation
    Real v00 = vols_[ti][ki];
    Real v01 = vols_[ti][ki + 1];
    Real v10 = vols_[ti + 1][ki];
    Real v11 = vols_[ti + 1][ki + 1];

    Real v0 = v00 * (1.0 - kf) + v01 * kf;
    Real v1 = v10 * (1.0 - kf) + v11 * kf;

    return v0 * (1.0 - tf) + v1 * tf;
}

Vec VolSurface::smile(Real T) const {
    auto [ti, tf] = find_bracket(maturities_, T);
    Vec result(strikes_.size());

    for (Size j = 0; j < strikes_.size(); ++j) {
        result[j] = vols_[ti][j] * (1.0 - tf) + vols_[ti + 1][j] * tf;
    }
    return result;
}

} // namespace qe
