#pragma once

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

  static void GenerateRandomMatrix(size_t size, std::vector<std::vector<double>> &matrix, std::vector<double> &vector);
  bool SolveGaussian(const std::vector<std::vector<double>> &a, const std::vector<double> &b, std::vector<double> &x);
};

}  // namespace urin_o_gauss_vert_diag
