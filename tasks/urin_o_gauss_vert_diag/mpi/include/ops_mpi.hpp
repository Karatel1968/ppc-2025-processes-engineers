#pragma once

#include <cstddef>
#include <vector>

#include "task/include/task.hpp"
#include "urin_o_gauss_vert_diag/common/include/common.hpp"

namespace urin_o_gauss_vert_diag {

class UrinOGaussVertDiagMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit UrinOGaussVertDiagMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  // Вспомогательные методы
  static void GenerateRandomMatrix(std::size_t size, std::vector<double> &augmented);

  static void CalculateColumnDistribution(std::size_t columns, int process_count, std::vector<int> &counts,
                                          std::vector<int> &displacements);

  static int FindOwner(std::size_t global_row, const std::vector<int> &displs, const std::vector<int> &rows_per_proc);
};

}  // namespace urin_o_gauss_vert_diag
