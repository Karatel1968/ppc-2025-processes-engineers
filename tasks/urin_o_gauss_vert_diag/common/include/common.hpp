#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace urin_o_gauss_vert_diag {

using InType = int;  // Расширенная матрица n x (n+1)
using OutType = int;              // Решение СЛАУ
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace urin_o_gauss_vert_diag
