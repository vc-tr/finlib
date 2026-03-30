#pragma once

#include <numbers>

namespace qe {

inline constexpr double PI = std::numbers::pi;
inline constexpr double SQRT2 = std::numbers::sqrt2;
inline constexpr double INV_SQRT2PI = 0.3989422804014327;  // 1 / sqrt(2*pi)
inline constexpr double TRADING_DAYS_PER_YEAR = 252.0;
inline constexpr double DAYS_PER_YEAR = 365.25;

} // namespace qe
