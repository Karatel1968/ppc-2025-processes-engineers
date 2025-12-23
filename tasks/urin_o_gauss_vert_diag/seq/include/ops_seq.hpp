#pragma once

#include <cstddef>
#include <vector>

#include "task/include/task.hpp"
#include "urin_o_gauss_vert_diag/common/include/common.hpp"

namespace urin_o_gauss_vert_diag {

class UrinOGaussVertDiagSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit UrinOGaussVertDiagSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void GenerateRandomMatrix(size_t size, std::vector<std::vector<double>> &matrix, std::vector<double> &rhs);
  static bool SolveGaussian(const std::vector<std::vector<double>> &matrix, const std::vector<double> &rhs,
                            std::vector<double> &solution);
  static bool ForwardElimination(std::vector<std::vector<double>> &augmented);
  static void EliminateRows(std::size_t pivot, std::vector<std::vector<double>> &augmented);
  static void BackSubstitution(const std::vector<std::vector<double>> &augmented, std::vector<double> &solution);
};

}  // namespace urin_o_gauss_vert_diag
