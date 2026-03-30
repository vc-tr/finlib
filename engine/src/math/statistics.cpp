#include "qe/math/statistics.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace qe {

Real mean(std::span<const Real> data) {
    if (data.empty()) throw std::invalid_argument("mean: empty data");
    return std::accumulate(data.begin(), data.end(), 0.0) / static_cast<Real>(data.size());
}

Real variance(std::span<const Real> data) {
    if (data.size() < 2) throw std::invalid_argument("variance: need at least 2 data points");
    Real m = mean(data);
    Real sum_sq = 0.0;
    for (Real x : data) {
        Real diff = x - m;
        sum_sq += diff * diff;
    }
    return sum_sq / static_cast<Real>(data.size() - 1);
}

Real std_dev(std::span<const Real> data) {
    return std::sqrt(variance(data));
}

Real quantile(std::span<const Real> data, Real q) {
    if (data.empty()) throw std::invalid_argument("quantile: empty data");
    if (q < 0.0 || q > 1.0) throw std::invalid_argument("quantile: q must be in [0, 1]");

    Vec sorted(data.begin(), data.end());
    std::sort(sorted.begin(), sorted.end());

    if (q == 0.0) return sorted.front();
    if (q == 1.0) return sorted.back();

    Real pos = q * static_cast<Real>(sorted.size() - 1);
    auto lo = static_cast<Size>(std::floor(pos));
    auto hi = static_cast<Size>(std::ceil(pos));
    Real frac = pos - static_cast<Real>(lo);

    if (lo == hi) return sorted[lo];
    return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

EmpiricalCDF empirical_cdf(std::span<const Real> data) {
    if (data.empty()) throw std::invalid_argument("empirical_cdf: empty data");

    EmpiricalCDF result;
    result.x.assign(data.begin(), data.end());
    std::sort(result.x.begin(), result.x.end());

    Size n = result.x.size();
    result.cdf.resize(n);
    for (Size i = 0; i < n; ++i) {
        result.cdf[i] = static_cast<Real>(i + 1) / static_cast<Real>(n);
    }
    return result;
}

Real covariance(std::span<const Real> x, std::span<const Real> y) {
    if (x.size() != y.size()) throw std::invalid_argument("covariance: mismatched sizes");
    if (x.size() < 2) throw std::invalid_argument("covariance: need at least 2 data points");

    Real mx = mean(x);
    Real my = mean(y);
    Real sum = 0.0;
    for (Size i = 0; i < x.size(); ++i) {
        sum += (x[i] - mx) * (y[i] - my);
    }
    return sum / static_cast<Real>(x.size() - 1);
}

Real correlation(std::span<const Real> x, std::span<const Real> y) {
    Real cov = covariance(x, y);
    Real sx = std_dev(x);
    Real sy = std_dev(y);
    if (sx == 0.0 || sy == 0.0) return 0.0;
    return cov / (sx * sy);
}

DescriptiveStats describe(std::span<const Real> data) {
    if (data.size() < 2) throw std::invalid_argument("describe: need at least 2 data points");

    DescriptiveStats s{};
    s.count = data.size();
    s.mean = qe::mean(data);
    s.variance = qe::variance(data);
    s.std_dev = std::sqrt(s.variance);

    auto [min_it, max_it] = std::minmax_element(data.begin(), data.end());
    s.min = *min_it;
    s.max = *max_it;

    // Skewness and kurtosis (sample, excess)
    Real n = static_cast<Real>(s.count);
    Real sum3 = 0.0, sum4 = 0.0;
    for (Real x : data) {
        Real z = (x - s.mean) / s.std_dev;
        Real z2 = z * z;
        sum3 += z2 * z;
        sum4 += z2 * z2;
    }
    // Adjusted Fisher-Pearson skewness
    s.skewness = (n / ((n - 1.0) * (n - 2.0))) * sum3;
    // Excess kurtosis
    s.kurtosis = ((n * (n + 1.0)) / ((n - 1.0) * (n - 2.0) * (n - 3.0))) * sum4
                 - (3.0 * (n - 1.0) * (n - 1.0)) / ((n - 2.0) * (n - 3.0));

    return s;
}

// ============================================================================
// RunningStats (Welford's online algorithm)
// ============================================================================

void RunningStats::push(Real x) {
    ++n_;
    Real delta = x - mean_;
    mean_ += delta / static_cast<Real>(n_);
    Real delta2 = x - mean_;
    m2_ += delta * delta2;
}

Real RunningStats::mean() const {
    if (n_ == 0) throw std::runtime_error("RunningStats: no data");
    return mean_;
}

Real RunningStats::variance() const {
    if (n_ < 2) throw std::runtime_error("RunningStats: need at least 2 data points");
    return m2_ / static_cast<Real>(n_ - 1);
}

Real RunningStats::std_dev() const {
    return std::sqrt(variance());
}

Real RunningStats::standard_error() const {
    return std_dev() / std::sqrt(static_cast<Real>(n_));
}

} // namespace qe
