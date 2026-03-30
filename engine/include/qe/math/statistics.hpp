#pragma once

#include "qe/core/types.hpp"
#include <span>

namespace qe {

// Descriptive statistics computed from data
struct DescriptiveStats {
    Real mean;
    Real variance;
    Real std_dev;
    Real skewness;
    Real kurtosis;
    Real min;
    Real max;
    Size count;
};

// Compute sample mean
Real mean(std::span<const Real> data);

// Compute sample variance (Bessel-corrected, ddof=1)
Real variance(std::span<const Real> data);

// Compute sample standard deviation
Real std_dev(std::span<const Real> data);

// Compute quantile using linear interpolation (quantile in [0, 1])
Real quantile(std::span<const Real> data, Real q);

// Compute empirical CDF: returns sorted values and their CDF values
struct EmpiricalCDF {
    Vec x;     // sorted data points
    Vec cdf;   // corresponding CDF values
};
EmpiricalCDF empirical_cdf(std::span<const Real> data);

// Compute covariance between two series
Real covariance(std::span<const Real> x, std::span<const Real> y);

// Compute Pearson correlation
Real correlation(std::span<const Real> x, std::span<const Real> y);

// Full descriptive statistics
DescriptiveStats describe(std::span<const Real> data);

// Welford's online algorithm for numerically stable running statistics
class RunningStats {
public:
    void push(Real x);
    Size count() const { return n_; }
    Real mean() const;
    Real variance() const;
    Real std_dev() const;
    Real standard_error() const;

private:
    Size n_ = 0;
    Real mean_ = 0.0;
    Real m2_ = 0.0;
};

} // namespace qe
