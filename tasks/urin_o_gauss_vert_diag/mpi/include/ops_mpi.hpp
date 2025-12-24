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

  // static void CalculateColumnDistribution(std::size_t columns, int process_count, std::vector<int> &counts,
  // std::vector<int> &displacements);

  static int FindOwner(std::size_t global_row, const std::vector<int> &displs, const std::vector<int> &rows_per_proc);

  static void EliminateLocalRows(std::vector<double> &local, const std::vector<double> &pivot_row,
                                 std::size_t local_rows, std::size_t width, std::size_t k, int rank,
                                 const std::vector<int> &displs);

  static void NormalizePivotRow(std::vector<double> &local, std::vector<double> &pivot_row, std::size_t local_k,
                                std::size_t k, std::size_t width);

  static void DistributeRows(int proc_count, std::size_t size, std::vector<int> &rows_per_proc,
                             std::vector<int> &displs);

  static OutType BackSubstitutionMPI(const std::vector<double> &full_matrix, std::size_t size, std::size_t row_width);
};

}  // namespace urin_o_gauss_vert_diag
