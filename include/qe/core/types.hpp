#pragma once

#include <cstddef>
#include <vector>

namespace qe {

using Real = double;
using Vec = std::vector<Real>;
using Mat = std::vector<Vec>;
using Size = std::size_t;

enum class OptionType { Call, Put };

} // namespace qe
